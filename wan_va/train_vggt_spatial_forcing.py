import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from . import train_vggt as base_train
from .configs import VA_CONFIGS as BASE_CONFIGS
from .configs.va_robotwin_train_vggt_spatial_forcing_cfg import (
    va_robotwin_train_vggt_spatial_forcing_cfg,
)
from .configs.va_robotwin_train_vggt_spatial_forcing_debug_cfg import (
    va_robotwin_train_vggt_spatial_forcing_debug_cfg,
)
from .distributed.util import dist_max, dist_mean, init_distributed
from .modules.utils import load_vae
from .modules.utils_spatial_forcing import load_transformer as load_transformer_spatial_forcing
from .utils import init_logger, logger, warmup_constant_lambda


base_train.load_transformer = load_transformer_spatial_forcing


class TokenAlignProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.norm = nn.BatchNorm1d(in_dim) if use_bn else nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens):
        original_shape = tokens.shape
        flat_tokens = tokens.reshape(-1, original_shape[-1]).float()
        flat_tokens = self.norm(flat_tokens)
        flat_tokens = self.proj(flat_tokens)
        return flat_tokens.reshape(*original_shape[:-1], flat_tokens.shape[-1])


class VGGTSpatialForcingTrainer(base_train.VGGTTrainer):
    def __init__(self, config):
        self.vggt_align_mode = getattr(config, 'vggt_align_mode', 'hybrid')
        self.vggt_align_layer_idx = int(getattr(config, 'vggt_align_layer_idx', 24))
        self.vggt_align_weight = float(getattr(config, 'vggt_align_weight', 0.1))
        self.vggt_align_use_bn = bool(getattr(config, 'vggt_align_use_bn', True))
        self.vggt_align_proj_dim = int(getattr(config, 'vggt_align_proj_dim', 2048))
        self.vggt_align_start_step = int(getattr(config, 'vggt_align_start_step', 0))
        self.vggt_teacher_layer_idx = int(getattr(config, 'vggt_teacher_layer_idx', -1))
        self.use_depth_point_loss = self.vggt_align_mode in ('depth_point', 'hybrid')
        self.use_feature_align_loss = self.vggt_align_mode in ('feature', 'hybrid')
        self.feature_align_head = None
        self.optimizer_align_head = None
        self.lr_scheduler_align = None

        super().__init__(config)

        if self.use_feature_align_loss and (self.vae is None or self.vggt is None):
            vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'vae')
            self.vae = load_vae(vae_path, torch_dtype=self.dtype, torch_device=self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)

            from vggt.models.vggt import VGGT as VGGTModel

            self.vggt = VGGTModel.from_pretrained("facebook/VGGT-1B")
            del self.vggt.camera_head
            del self.vggt.track_head
            self.vggt.camera_head = None
            self.vggt.track_head = None
            self.vggt.eval()
            self.vggt.requires_grad_(False)
            self.vggt.to(dtype=self.dtype, device=self.device)

        if self.use_feature_align_loss and self.vggt_align_weight > 0:
            student_dim = int(self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim)
            teacher_dim = int(self.vggt.aggregator.camera_token.shape[-1] * 2)
            self.feature_align_head = TokenAlignProjectionHead(
                in_dim=student_dim,
                hidden_dim=self.vggt_align_proj_dim,
                out_dim=teacher_dim,
                use_bn=self.vggt_align_use_bn,
            ).to(self.device)

            if getattr(config, 'resume_from', None):
                align_head_path = Path(config.resume_from) / 'feature_align_head.safetensors'
                if align_head_path.exists():
                    self.feature_align_head.load_state_dict(load_file(str(align_head_path)))
                    logger.info(f"Loaded feature alignment head from {align_head_path}")

            # 与 self.optimizer（仅 FSDP transformer）分开：DTensor 与普通 Tensor 不能进同一个 fused AdamW。
            self.optimizer_align_head = torch.optim.AdamW(
                [p for p in self.feature_align_head.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=False,
                foreach=False,
            )
            self.lr_scheduler_align = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_align_head,
                lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
            )
            logger.info(
                "Using separate AdamW for feature_align_head (fused=False); "
                "transformer optimizer unchanged (FSDP / DTensor)."
            )

    def _get_vggt_frame_indices(self, num_frames):
        num_sample = min(self.vggt_num_sample_frames, num_frames)
        if num_frames <= num_sample:
            return list(range(num_frames))
        return torch.linspace(0, num_frames - 1, num_sample).long().tolist()

    def _get_sigma_weights(self, sigmas, frame_indices):
        frame_sigmas = sigmas[:, frame_indices].detach()
        sigma_weights = (1.0 - frame_sigmas).clamp(min=0.05)
        return frame_sigmas, sigma_weights

    @torch.no_grad()
    def _compute_vggt_teacher_tokens(self, pixel_frames):
        bsz, num_frames, channels, height, width = pixel_frames.shape
        target_h, target_w = 518, 518
        frames_resized = F.interpolate(
            pixel_frames.reshape(bsz * num_frames, channels, height, width),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        ).reshape(bsz, num_frames, channels, target_h, target_w)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(frames_resized)

        teacher_layer_idx = self.vggt_teacher_layer_idx
        if teacher_layer_idx < 0:
            teacher_layer_idx = len(aggregated_tokens_list) + teacher_layer_idx
        teacher_tokens = aggregated_tokens_list[teacher_layer_idx][:, :, patch_start_idx:]
        patch_size = int(getattr(self.vggt.aggregator, 'patch_size', 14))
        grid_h = target_h // patch_size
        grid_w = target_w // patch_size
        return teacher_tokens.reshape(bsz, num_frames, grid_h, grid_w, teacher_tokens.shape[-1])

    def compute_vggt_feature_align_loss(self, align_hidden_states, input_dict):
        if align_hidden_states is None or self.feature_align_head is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        latent_dict = input_dict['latent_dict']
        gt_clean = latent_dict['latent']
        _, _, num_frames, latent_h, latent_w = gt_clean.shape
        frame_indices = self._get_vggt_frame_indices(num_frames)
        sigmas = latent_dict['timesteps'] / self.train_scheduler_latent.num_train_timesteps
        _, sigma_weights = self._get_sigma_weights(sigmas, frame_indices)

        with torch.no_grad():
            gt_pixels = self._decode_latent_to_pixels(gt_clean, frame_indices, enable_grad=False)
            gt_pixels = self._select_vggt_supervision_pixels(gt_pixels)
            teacher_tokens = self._compute_vggt_teacher_tokens(gt_pixels)

        token_h = latent_h // self.patch_size[1]
        token_w = latent_w // self.patch_size[2]
        student_tokens = align_hidden_states.reshape(
            align_hidden_states.shape[0],
            num_frames,
            token_h,
            token_w,
            align_hidden_states.shape[-1],
        )[:, frame_indices]
        student_tokens = self.feature_align_head(student_tokens)

        teacher_tokens = rearrange(teacher_tokens, 'b f h w c -> (b f) c h w')
        teacher_tokens = F.interpolate(
            teacher_tokens,
            size=(token_h, token_w),
            mode='bilinear',
            align_corners=False,
        )
        teacher_tokens = rearrange(
            teacher_tokens,
            '(b f) c h w -> b f h w c',
            b=align_hidden_states.shape[0],
            f=len(frame_indices),
        )

        student_tokens = F.normalize(student_tokens.float(), dim=-1)
        teacher_tokens = F.normalize(teacher_tokens.float(), dim=-1)
        cosine_sim = (student_tokens * teacher_tokens).sum(dim=-1)
        weights = sigma_weights[:, :, None, None].expand_as(cosine_sim)
        align_loss = ((1.0 - cosine_sim) * weights).sum() / (weights.sum() + 1e-6)
        align_cosine = (cosine_sim * weights).sum() / (weights.sum() + 1e-6)
        return align_loss, align_cosine.detach()

    def _sync_feature_align_grads(self):
        if not dist.is_initialized() or self.feature_align_head is None:
            return
        world_size = dist.get_world_size()
        for param in self.feature_align_head.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        need_align_hidden_states = (
            self.use_feature_align_loss
            and self.vggt_align_weight > 0
            and self.step >= self.vggt_align_start_step
            and self.step % self.vggt_loss_interval == 0
        )
        transformer_input = input_dict
        if need_align_hidden_states:
            transformer_input = dict(input_dict)
            transformer_input['return_hidden_layers'] = True
            transformer_input['align_layer_idx'] = self.vggt_align_layer_idx
            transformer_input['align_video_tokens_only'] = True

        output = self.transformer(transformer_input, train_mode=True)
        align_hidden_states = None
        if isinstance(output, dict):
            align_hidden_states = output.get('align_hidden_states')
            output = output['pred']

        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        vggt_loss_val = torch.tensor(0.0, device=self.device)
        vggt_depth_loss_val = torch.tensor(0.0, device=self.device)
        vggt_point_loss_val = torch.tensor(0.0, device=self.device)
        feature_align_loss_val = torch.tensor(0.0, device=self.device)
        feature_align_cosine_val = torch.tensor(0.0, device=self.device)

        should_run_vggt = self.step % self.vggt_loss_interval == 0
        if (
            self.use_depth_point_loss
            and self.vggt_loss_weight > 0
            and self.step >= self.vggt_loss_start_step
            and should_run_vggt
        ):
            vggt_input = latent_pred if self.vggt_grad_enabled else latent_pred.detach()
            vggt_loss_val, vggt_depth_loss_val, vggt_point_loss_val = self.compute_vggt_loss(vggt_input, input_dict)
            loss = loss + self.vggt_loss_weight * vggt_loss_val / self.gradient_accumulation_steps

        if (
            self.use_feature_align_loss
            and self.vggt_align_weight > 0
            and self.step >= self.vggt_align_start_step
            and should_run_vggt
        ):
            feature_align_loss_val, feature_align_cosine_val = self.compute_vggt_feature_align_loss(
                align_hidden_states=align_hidden_states,
                input_dict=input_dict,
            )
            loss = loss + self.vggt_align_weight * feature_align_loss_val / self.gradient_accumulation_steps

        loss.backward()
        self._sync_feature_align_grads()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'vggt_loss': vggt_loss_val.detach(),
            'vggt_depth_loss': vggt_depth_loss_val,
            'vggt_point_loss': vggt_point_loss_val,
            'feature_align_loss': feature_align_loss_val.detach(),
            'feature_align_cosine': feature_align_cosine_val.detach(),
        }

        if should_sync:
            # FSDP2 下 transformer 梯度多为 DTensor；feature_align_head 为普通 Tensor。
            # 合并进一次 clip_grad_norm_ 会触发 foreach 路径混合类型报错。
            max_norm = 2.0
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), max_norm, foreach=False
            )
            if self.feature_align_head is not None:
                h_norm = torch.nn.utils.clip_grad_norm_(
                    self.feature_align_head.parameters(), max_norm, foreach=False
                )
                total_norm = (total_norm * total_norm + h_norm * h_norm).sqrt()
            self.optimizer.step()
            if self.optimizer_align_head is not None:
                self.optimizer_align_head.step()
            self.lr_scheduler.step()
            if self.lr_scheduler_align is not None:
                self.lr_scheduler_align.step()
            self.optimizer.zero_grad()
            if self.optimizer_align_head is not None:
                self.optimizer_align_head.zero_grad()
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self):
        super().save_checkpoint()
        if self.config.rank == 0 and self.feature_align_head is not None:
            checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
            align_head_file = checkpoint_dir / "feature_align_head.safetensors"
            save_file(
                {
                    k: v.detach().cpu().to(torch.bfloat16 if v.is_floating_point() else v.dtype)
                    for k, v in self.feature_align_head.state_dict().items()
                },
                str(align_head_file),
            )

    def train(self):
        logger.info(f"Starting VGGT spatial forcing training for {self.config.num_steps} steps...")
        logger.info(
            f"VGGT loss weight: {self.vggt_loss_weight}, align_mode: {self.vggt_align_mode}, "
            f"align_layer: {self.vggt_align_layer_idx}, align_weight: {self.vggt_align_weight}"
        )
        self.transformer.train()
        if self.feature_align_head is not None:
            self.feature_align_head.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (SpatialForcing)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        if self.optimizer_align_head is not None:
            self.optimizer_align_head.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_vggt_losses = []
        accumulated_vggt_depth_losses = []
        accumulated_vggt_point_losses = []
        accumulated_feature_align_losses = []
        accumulated_feature_align_cosines = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)

            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_vggt_losses.append(losses['vggt_loss'])
            accumulated_vggt_depth_losses.append(losses['vggt_depth_loss'])
            accumulated_vggt_point_losses.append(losses['vggt_point_loss'])
            accumulated_feature_align_losses.append(losses['feature_align_loss'])
            accumulated_feature_align_cosines.append(losses['feature_align_cosine'])
            step_in_accumulation += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                vggt_loss_show = dist_mean(torch.stack(accumulated_vggt_losses).sum()).detach().cpu().item()
                vggt_depth_loss_show = dist_mean(torch.stack(accumulated_vggt_depth_losses).sum()).detach().cpu().item()
                vggt_point_loss_show = dist_mean(torch.stack(accumulated_vggt_point_losses).sum()).detach().cpu().item()
                feature_align_loss_show = dist_mean(torch.stack(accumulated_feature_align_losses).sum()).detach().cpu().item()
                feature_align_cosine_show = dist_mean(torch.stack(accumulated_feature_align_cosines).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_vggt_losses = []
                accumulated_vggt_depth_losses = []
                accumulated_vggt_point_losses = []
                accumulated_feature_align_losses = []
                accumulated_feature_align_cosines = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        {
                            'lat': f'{latent_loss_show:.4f}',
                            'act': f'{action_loss_show:.4f}',
                            'vggt': f'{vggt_loss_show:.4f}',
                            'vF': f'{feature_align_loss_show:.4f}',
                            'cos': f'{feature_align_cosine_show:.3f}',
                            'step': self.step,
                            'gn': f'{total_norm.item():.2f}',
                            'lr': f'{lr:.2e}',
                        }
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action', action_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/vggt', vggt_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/vggt_depth', vggt_depth_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/vggt_point', vggt_point_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/vggt_feature_align', feature_align_loss_show, self.step)
                        self.tb_writer.add_scalar('train/vggt_feature_align_cosine', feature_align_cosine_show, self.step)
                        self.tb_writer.add_scalar('loss/video_max', max_latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action_max', max_action_loss_show, self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "loss/video": latent_loss_show,
                                "loss/action": action_loss_show,
                                "loss/vggt": vggt_loss_show,
                                "loss/vggt_depth": vggt_depth_loss_show,
                                "loss/vggt_point": vggt_point_loss_show,
                                "loss/vggt_feature_align": feature_align_loss_show,
                                "train/vggt_feature_align_cosine": feature_align_cosine_show,
                                "loss/video_max": max_latent_loss_show,
                                "loss/action_max": max_action_loss_show,
                                "train/grad_norm": total_norm.item(),
                                "train/lr": lr,
                            },
                            step=self.step,
                        )

                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()
        logger.info("Training completed!")


SPATIAL_FORCING_CONFIGS = dict(BASE_CONFIGS)
SPATIAL_FORCING_CONFIGS['robotwin_train_vggt_spatial_forcing'] = va_robotwin_train_vggt_spatial_forcing_cfg
SPATIAL_FORCING_CONFIGS['robotwin_train_vggt_spatial_forcing_debug'] = va_robotwin_train_vggt_spatial_forcing_debug_cfg


def run(args):
    config = SPATIAL_FORCING_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    debug_seed = getattr(config, 'debug_seed', None)
    if debug_seed is not None:
        seed = int(debug_seed) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if args.save_root is not None:
        config.save_root = args.save_root
    if getattr(args, "enable_wandb", False):
        config.enable_wandb = True
    if getattr(args, "wandb_run_name", None):
        config.wandb_run_name = args.wandb_run_name

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = VGGTSpatialForcingTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train WAN model with VGGT spatial forcing alignment loss")
    parser.add_argument("--config-name", type=str, default='robotwin_train_vggt_spatial_forcing')
    parser.add_argument("--save-root", type=str, default=None, help="Root directory for saving checkpoints")
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name (overrides config.wandb_run_name)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
