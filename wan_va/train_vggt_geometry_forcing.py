# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# Geometry Forcing training for Wan-VA.
#
# Paper: "Geometry Forcing: Marrying Video Diffusion and 3D Representation
#         for Consistent World Modeling" (Wu et al., 2025)
#
# Idea: align multiple intermediate layers of the Wan DiT with multiple
#       layers of a frozen VGGT aggregator, via two complementary objectives:
#       * Angular Alignment  (cosine,   paper default lambda=0.5)
#       * Scale   Alignment  (MSE on unit-normalized student,
#                             paper default lambda=0.05)
#
# This file is a *separate entry point* from train_vggt_spatial_forcing.py
# and does not modify anything in the original training code.
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from . import train_vggt as base_train
from .configs import VA_CONFIGS as BASE_CONFIGS
from .configs.va_robotwin_train_vggt_geometry_forcing_cfg import (
    va_robotwin_train_vggt_geometry_forcing_cfg,
)
from .configs.va_robotwin_train_vggt_geometry_forcing_debug_cfg import (
    va_robotwin_train_vggt_geometry_forcing_debug_cfg,
)
from .distributed.util import dist_max, dist_mean, init_distributed
from .modules.geometry_forcing_head import (
    GeometryForcingHead,
    angular_loss,
    scale_loss,
)
from .modules.utils import load_vae
from .modules.utils_geometry_forcing import (
    load_transformer as load_transformer_geometry_forcing,
)
from .utils import init_logger, logger, warmup_constant_lambda


# Ensure the base VGGT trainer uses a transformer that exposes hidden layers
# (same backbone as model_spatial_forcing; see modules/utils_geometry_forcing.py).
base_train.load_transformer = load_transformer_geometry_forcing


class GeometryForcingTrainer(base_train.VGGTTrainer):
    """Wan-VA trainer with Geometry Forcing alignment losses.

    Inherits from VGGTTrainer:
      * flow-matching latent loss + action loss (`compute_loss`)
      * optional pixel-level VGGT depth/point loss (`compute_vggt_loss`)
        -- only active in `gf_mode='hybrid'`.

    Adds:
      * Multi-layer Angular alignment (f_phi per layer)
      * Multi-layer Scale   alignment (g_phi per layer)
    """

    # ------------------------------------------------------------------ init
    def __init__(self, config):
        # Geometry-Forcing-specific config (read before super().__init__,
        # because super() builds the optimizer etc. and we may want to know
        # whether to load VAE / VGGT for pure mode as well).
        self.gf_mode = str(getattr(config, 'gf_mode', 'pure')).lower()
        assert self.gf_mode in ('pure', 'hybrid'), (
            f"gf_mode must be 'pure' or 'hybrid', got {self.gf_mode}"
        )

        self.gf_student_layers = list(getattr(config, 'gf_student_layers', [29]))
        self.gf_teacher_layers = list(getattr(config, 'gf_teacher_layers', [-1]))
        assert len(self.gf_student_layers) == len(self.gf_teacher_layers), (
            "gf_student_layers and gf_teacher_layers must have equal length"
        )

        self.gf_lambda_angular = float(getattr(config, 'gf_lambda_angular', 0.5))
        self.gf_lambda_scale = float(getattr(config, 'gf_lambda_scale', 0.05))
        self.gf_proj_hidden_dim = int(getattr(config, 'gf_proj_hidden_dim', 2048))
        self.gf_use_bn = bool(getattr(config, 'gf_use_bn', True))
        self.gf_start_step = int(getattr(config, 'gf_start_step', 0))
        self.gf_use_sigma_weight = bool(getattr(config, 'gf_use_sigma_weight', True))

        # In pure mode we still need VAE + VGGT for teacher features, but we
        # *disable* the base VGGT pixel-level depth/point loss by zeroing its
        # weight. In hybrid mode we keep whatever vggt_loss_weight the config
        # inherits from va_robotwin_train_vggt_cfg.
        if self.gf_mode == 'pure':
            config.vggt_loss_weight = 0.0

        # Placeholders populated below / during training.
        self.gf_head = None
        self.optimizer_gf = None
        self.lr_scheduler_gf = None

        super().__init__(config)

        # Ensure VAE + VGGT are loaded even when vggt_loss_weight == 0 (pure mode).
        if self.vae is None or self.vggt is None:
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

        # ------------------------------------------------------------------
        # Build Geometry-Forcing heads.
        # ------------------------------------------------------------------
        student_dim = int(
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        teacher_dim = int(self.vggt.aggregator.camera_token.shape[-1] * 2)

        if self.gf_lambda_angular > 0 or self.gf_lambda_scale > 0:
            self.gf_head = GeometryForcingHead(
                student_dims=[student_dim] * len(self.gf_student_layers),
                teacher_dims=[teacher_dim] * len(self.gf_teacher_layers),
                hidden_dim=self.gf_proj_hidden_dim,
                use_bn=self.gf_use_bn,
                use_scale=self.gf_lambda_scale > 0,
            ).to(self.device)

            if getattr(config, 'resume_from', None):
                gf_head_path = Path(config.resume_from) / 'geometry_forcing_head.safetensors'
                if gf_head_path.exists():
                    self.gf_head.load_state_dict(load_file(str(gf_head_path)))
                    logger.info(f"Loaded Geometry Forcing head from {gf_head_path}")

            # Keep GF head in its own AdamW: the transformer uses FSDP/DTensor
            # + fused=True, which cannot be mixed with plain Tensors.
            self.optimizer_gf = torch.optim.AdamW(
                [p for p in self.gf_head.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=False,
                foreach=False,
            )
            self.lr_scheduler_gf = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_gf,
                lr_lambda=lambda step: warmup_constant_lambda(
                    step, warmup_steps=config.warmup_steps
                ),
            )
            logger.info(
                "GeometryForcing head built: "
                f"{len(self.gf_student_layers)} layer pair(s), "
                f"student_dim={student_dim}, teacher_dim={teacher_dim}, "
                f"lambda_A={self.gf_lambda_angular}, lambda_S={self.gf_lambda_scale}, "
                f"mode={self.gf_mode}"
            )

    # ------------------------------------------------------------------
    # Frame / sigma helpers (same logic as spatial_forcing trainer).
    # ------------------------------------------------------------------
    def _get_vggt_frame_indices(self, num_frames):
        num_sample = min(self.vggt_num_sample_frames, num_frames)
        if num_frames <= num_sample:
            return list(range(num_frames))
        return torch.linspace(0, num_frames - 1, num_sample).long().tolist()

    def _get_sigma_weights(self, sigmas, frame_indices):
        frame_sigmas = sigmas[:, frame_indices].detach()
        sigma_weights = (1.0 - frame_sigmas).clamp(min=0.05)
        return frame_sigmas, sigma_weights

    # _select_vggt_supervision_pixels is inherited from base_train.VGGTTrainer.

    # ------------------------------------------------------------------
    # Teacher features: run frozen VGGT aggregator on GT-decoded pixels.
    # Returns dict: {teacher_layer_idx: (B, F, H, W, D)} for every requested
    # teacher layer.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_vggt_teacher_tokens_multi(self, pixel_frames, teacher_layer_indices):
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

        num_vggt_layers = len(aggregated_tokens_list)
        patch_size = int(getattr(self.vggt.aggregator, 'patch_size', 14))
        grid_h = target_h // patch_size
        grid_w = target_w // patch_size

        out = {}
        for t_idx in teacher_layer_indices:
            real_idx = t_idx if t_idx >= 0 else num_vggt_layers + t_idx
            if real_idx < 0 or real_idx >= num_vggt_layers:
                raise IndexError(
                    f"teacher layer {t_idx} out of range [0,{num_vggt_layers})"
                )
            tokens = aggregated_tokens_list[real_idx][:, :, patch_start_idx:]
            tokens = tokens.reshape(bsz, num_frames, grid_h, grid_w, tokens.shape[-1])
            out[t_idx] = tokens
        return out

    # ------------------------------------------------------------------
    # Geometry Forcing loss computation.
    # align_hidden_states: dict {student_layer_idx: tensor(B, L, D)} captured
    # from the Wan DiT backbone via return_hidden_layers=True.
    # ------------------------------------------------------------------
    def compute_geometry_forcing_loss(self, align_hidden_states, input_dict):
        zero = torch.tensor(0.0, device=self.device)
        if (
            align_hidden_states is None
            or self.gf_head is None
            or not isinstance(align_hidden_states, dict)
            or len(align_hidden_states) == 0
        ):
            return zero, zero, zero

        latent_dict = input_dict['latent_dict']
        gt_clean = latent_dict['latent']
        _, _, num_frames, latent_h, latent_w = gt_clean.shape
        frame_indices = self._get_vggt_frame_indices(num_frames)

        sigmas = latent_dict['timesteps'] / self.train_scheduler_latent.num_train_timesteps
        _, sigma_w = self._get_sigma_weights(sigmas, frame_indices)  # (B, Fn)

        # Teacher features (GT RGB -> VGGT) for all requested teacher layers.
        with torch.no_grad():
            gt_pixels = self._decode_latent_to_pixels(
                gt_clean, frame_indices, enable_grad=False
            )
            gt_pixels = self._select_vggt_supervision_pixels(gt_pixels)
            teacher_tokens_by_layer = self._compute_vggt_teacher_tokens_multi(
                gt_pixels, self.gf_teacher_layers
            )

        token_h = latent_h // self.patch_size[1]
        token_w = latent_w // self.patch_size[2]
        n_sample = len(frame_indices)
        bsz = gt_clean.shape[0]

        total_ang = zero.clone()
        total_scl = zero.clone()
        total_cos = zero.clone()
        n_pairs = 0

        for s_idx, t_idx in zip(self.gf_student_layers, self.gf_teacher_layers):
            captured = align_hidden_states.get(int(s_idx))
            if captured is None:
                # Silently skip if backbone did not return this layer.
                continue
            n_pairs += 1

            # (B, F*Hs*Ws, D) -> (B, F_sel, Hs, Ws, D)
            student = captured.reshape(
                captured.shape[0], num_frames, token_h, token_w, captured.shape[-1]
            )[:, frame_indices]

            teacher = teacher_tokens_by_layer[t_idx]  # (B, F, Hg, Wg, Dt)

            # Resize teacher token grid to student grid via bilinear interp.
            teacher = rearrange(teacher, 'b f h w c -> (b f) c h w')
            teacher = F.interpolate(
                teacher,
                size=(token_h, token_w),
                mode='bilinear',
                align_corners=False,
            )
            teacher = rearrange(
                teacher, '(b f) c h w -> b f h w c', b=bsz, f=n_sample
            )

            # f_phi(h) -> teacher_dim
            student_proj = self.gf_head.project(student, layer_idx=n_pairs - 1)

            # Per-token weights.
            if self.gf_use_sigma_weight:
                w = sigma_w[:, :, None, None].expand(bsz, n_sample, token_h, token_w)
            else:
                w = torch.ones(
                    (bsz, n_sample, token_h, token_w),
                    device=self.device,
                    dtype=torch.float32,
                )

            # Angular loss.
            ang, cos_val = angular_loss(student_proj, teacher, w)
            total_ang = total_ang + ang
            total_cos = total_cos + cos_val

            # Scale loss (on normalize(student_proj) -> teacher, un-normalized).
            if self.gf_head.use_scale and self.gf_lambda_scale > 0:
                hat = F.normalize(student_proj.float(), dim=-1)
                y_tilde = self.gf_head.predict_scale(hat, layer_idx=n_pairs - 1)
                total_scl = total_scl + scale_loss(y_tilde, teacher, w)

        if n_pairs == 0:
            return zero, zero, zero

        total_ang = total_ang / n_pairs
        total_scl = total_scl / n_pairs
        total_cos = total_cos / n_pairs
        return total_ang, total_scl, total_cos.detach()

    # ------------------------------------------------------------------
    # Distributed helpers.
    # ------------------------------------------------------------------
    def _sync_gf_grads(self):
        if not dist.is_initialized() or self.gf_head is None:
            return
        world_size = dist.get_world_size()
        for param in self.gf_head.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    # ------------------------------------------------------------------
    # Training step.
    # ------------------------------------------------------------------
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        need_align = (
            self.gf_head is not None
            and (self.gf_lambda_angular > 0 or self.gf_lambda_scale > 0)
            and self.step >= self.gf_start_step
            and self.step % self.vggt_loss_interval == 0
        )

        transformer_input = input_dict
        if need_align:
            transformer_input = dict(input_dict)
            transformer_input['return_hidden_layers'] = True
            transformer_input['align_layer_idx'] = list(self.gf_student_layers)
            transformer_input['align_video_tokens_only'] = True

        output = self.transformer(transformer_input, train_mode=True)
        align_hidden_states = None
        if isinstance(output, dict):
            align_hidden_states = output.get('align_hidden_states')
            output = output['pred']

        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        # Optional depth/point loss (hybrid mode).
        vggt_loss_val = torch.tensor(0.0, device=self.device)
        should_run_vggt = self.step % self.vggt_loss_interval == 0
        if (
            self.gf_mode == 'hybrid'
            and self.vggt_loss_weight > 0
            and self.step >= self.vggt_loss_start_step
            and should_run_vggt
        ):
            vggt_input = latent_pred if self.vggt_grad_enabled else latent_pred.detach()
            vggt_loss_val = self.compute_vggt_loss(vggt_input, input_dict)
            loss = loss + self.vggt_loss_weight * vggt_loss_val / self.gradient_accumulation_steps

        # Geometry Forcing losses.
        ang_loss_val = torch.tensor(0.0, device=self.device)
        scl_loss_val = torch.tensor(0.0, device=self.device)
        cos_val = torch.tensor(0.0, device=self.device)
        if need_align:
            ang_loss_val, scl_loss_val, cos_val = self.compute_geometry_forcing_loss(
                align_hidden_states=align_hidden_states,
                input_dict=input_dict,
            )
            if self.gf_lambda_angular > 0:
                loss = loss + self.gf_lambda_angular * ang_loss_val / self.gradient_accumulation_steps
            if self.gf_lambda_scale > 0:
                loss = loss + self.gf_lambda_scale * scl_loss_val / self.gradient_accumulation_steps

        loss.backward()
        self._sync_gf_grads()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'vggt_loss': vggt_loss_val.detach(),
            'gf_angular_loss': ang_loss_val.detach(),
            'gf_scale_loss': scl_loss_val.detach(),
            'gf_cosine': cos_val.detach(),
        }

        if should_sync:
            max_norm = 2.0
            # Clip transformer and GF head separately (DTensor vs Tensor).
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), max_norm, foreach=False
            )
            if self.gf_head is not None:
                h_norm = torch.nn.utils.clip_grad_norm_(
                    self.gf_head.parameters(), max_norm, foreach=False
                )
                total_norm = (total_norm * total_norm + h_norm * h_norm).sqrt()
            self.optimizer.step()
            if self.optimizer_gf is not None:
                self.optimizer_gf.step()
            self.lr_scheduler.step()
            if self.lr_scheduler_gf is not None:
                self.lr_scheduler_gf.step()
            self.optimizer.zero_grad()
            if self.optimizer_gf is not None:
                self.optimizer_gf.zero_grad()
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    # ------------------------------------------------------------------
    # Checkpointing: also save the GF head alongside transformer.
    # ------------------------------------------------------------------
    def save_checkpoint(self):
        super().save_checkpoint()
        if self.config.rank == 0 and self.gf_head is not None:
            checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
            gf_head_file = checkpoint_dir / "geometry_forcing_head.safetensors"
            save_file(
                {
                    k: v.detach().cpu().to(
                        torch.bfloat16 if v.is_floating_point() else v.dtype
                    )
                    for k, v in self.gf_head.state_dict().items()
                },
                str(gf_head_file),
            )

    # ------------------------------------------------------------------
    # Main training loop.
    # ------------------------------------------------------------------
    def train(self):
        logger.info(f"Starting Geometry Forcing training for {self.config.num_steps} steps...")
        logger.info(
            f"GF mode={self.gf_mode}, pairs={list(zip(self.gf_student_layers, self.gf_teacher_layers))}, "
            f"lambda_A={self.gf_lambda_angular}, lambda_S={self.gf_lambda_scale}, "
            f"use_sigma_weight={self.gf_use_sigma_weight}"
        )
        self.transformer.train()
        if self.gf_head is not None:
            self.gf_head.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (GeometryForcing)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        if self.optimizer_gf is not None:
            self.optimizer_gf.zero_grad()

        buf = {
            'latent': [], 'action': [], 'vggt': [],
            'gf_ang': [], 'gf_scl': [], 'gf_cos': [],
        }
        step_in_accum = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accum)

            buf['latent'].append(losses['latent_loss'])
            buf['action'].append(losses['action_loss'])
            buf['vggt'].append(losses['vggt_loss'])
            buf['gf_ang'].append(losses['gf_angular_loss'])
            buf['gf_scl'].append(losses['gf_scale_loss'])
            buf['gf_cos'].append(losses['gf_cosine'])
            step_in_accum += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                def _m(k):
                    return dist_mean(torch.stack(buf[k]).sum()).detach().cpu().item()

                def _mx(k):
                    return dist_max(torch.stack(buf[k]).sum()).detach().cpu().item()

                latent_v = _m('latent'); action_v = _m('action')
                vggt_v = _m('vggt')
                gf_a = _m('gf_ang'); gf_s = _m('gf_scl'); gf_c = _m('gf_cos')
                max_l = _mx('latent'); max_a = _mx('action')

                for k in buf:
                    buf[k] = []
                step_in_accum = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        {
                            'lat': f'{latent_v:.4f}',
                            'act': f'{action_v:.4f}',
                            'vggt': f'{vggt_v:.4f}',
                            'gfA': f'{gf_a:.4f}',
                            'gfS': f'{gf_s:.4f}',
                            'cos': f'{gf_c:.3f}',
                            'step': self.step,
                            'gn': f'{total_norm.item():.2f}',
                            'lr': f'{lr:.2e}',
                        }
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', latent_v, self.step)
                        self.tb_writer.add_scalar('loss/action', action_v, self.step)
                        self.tb_writer.add_scalar('loss/vggt', vggt_v, self.step)
                        self.tb_writer.add_scalar('loss/gf_angular', gf_a, self.step)
                        self.tb_writer.add_scalar('loss/gf_scale', gf_s, self.step)
                        self.tb_writer.add_scalar('train/gf_cosine', gf_c, self.step)
                        self.tb_writer.add_scalar('loss/video_max', max_l, self.step)
                        self.tb_writer.add_scalar('loss/action_max', max_a, self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                'loss/video': latent_v,
                                'loss/action': action_v,
                                'loss/vggt': vggt_v,
                                'loss/gf_angular': gf_a,
                                'loss/gf_scale': gf_s,
                                'train/gf_cosine': gf_c,
                                'loss/video_max': max_l,
                                'loss/action_max': max_a,
                                'train/grad_norm': total_norm.item(),
                                'train/lr': lr,
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


# Entry-point config registry (same pattern as train_vggt_spatial_forcing.py).
GEOMETRY_FORCING_CONFIGS = dict(BASE_CONFIGS)
GEOMETRY_FORCING_CONFIGS['robotwin_train_vggt_geometry_forcing'] = (
    va_robotwin_train_vggt_geometry_forcing_cfg
)
GEOMETRY_FORCING_CONFIGS['robotwin_train_vggt_geometry_forcing_debug'] = (
    va_robotwin_train_vggt_geometry_forcing_debug_cfg
)


def run(args):
    config = GEOMETRY_FORCING_CONFIGS[args.config_name]

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

    trainer = GeometryForcingTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train Wan-VA model with Geometry Forcing (Angular + Scale alignment)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_vggt_geometry_forcing',
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
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
