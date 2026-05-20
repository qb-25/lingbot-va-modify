# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# Geometry Forcing + Internal head training for Wan-VA.
#
# Combines two ideas:
#   (1) Geometry Forcing (Wu et al., 2025): multi-layer Angular + Scale
#       alignment between the DiT hidden states and a frozen VGGT backbone.
#   (2) Internal Guidance (Zhou et al., 2025): a parallel internal head
#       D_i forked from a chosen depth of the DiT, trained jointly with
#       the main head D_f via deep supervision.
#
# In addition, the trainer periodically dumps cross-attention overlays
# ("which spatial patch each Wan layer attends to in the prompt") for
# inspection.  See ``modules/attention_visualizer.py``.
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from tqdm import tqdm

from . import train_vggt_geometry_forcing as gf_train
from .configs import VA_CONFIGS as BASE_CONFIGS
from .configs.va_robotwin_train_vggt_geometry_forcing_internal_cfg import (
    va_robotwin_train_vggt_geometry_forcing_internal_cfg,
)
from .configs.va_robotwin_train_vggt_geometry_forcing_internal_debug_cfg import (
    va_robotwin_train_vggt_geometry_forcing_internal_debug_cfg,
)
from .distributed.util import dist_max, dist_mean, init_distributed
from .modules.attention_visualizer import (
    cross_attn_to_video_heatmaps,
    save_layered_attention_grid,
)
from .modules.utils_geometry_forcing_internal import (
    load_transformer as load_transformer_gf_internal,
)
from .utils import init_logger, logger


# Make GeometryForcingTrainer pick up the GF+internal backbone.
gf_train.load_transformer_geometry_forcing = load_transformer_gf_internal
gf_train.base_train.load_transformer = load_transformer_gf_internal


class GeometryForcingInternalTrainer(gf_train.GeometryForcingTrainer):
    """GF trainer + internal deep-supervision head + attention visualizer.

    Overrides only ``_train_step`` and ``train`` (latter to re-export logs);
    everything else (FSDP, GF projector heads, VAE/VGGT teachers, save flow)
    is inherited from ``GeometryForcingTrainer``.
    """

    def __init__(self, config):
        # Internal-head config.
        self.enable_internal = bool(getattr(config, 'enable_internal', True))
        self.internal_depth = int(getattr(config, 'internal_depth', 24))
        self.num_internal_blocks = int(getattr(config, 'num_internal_blocks', 2))
        self.lambda_internal = float(getattr(config, 'lambda_internal', 1.0))
        self.internal_start_step = int(getattr(config, 'internal_start_step', 0))

        # Visualization config.
        self.attn_vis_enabled = bool(getattr(config, 'attn_vis_enabled', False))
        self.attn_vis_interval = int(getattr(config, 'attn_vis_interval', 5000))
        self.attn_vis_layers = list(getattr(config, 'attn_vis_layers', [0, 10, 20, 29]))
        self.attn_vis_frames = list(getattr(config, 'attn_vis_frames', [0, 2, 5]))
        self.attn_vis_token_meta = dict(
            getattr(config, 'attn_vis_token_meta', {'mode': 'content_only', 'top_k': 16})
        )
        self.attn_vis_alpha = float(getattr(config, 'attn_vis_alpha', 0.5))

        # Inject extra kwargs into the model loader (the GF parent's __init__
        # calls load_transformer with positional args; the new loader receives
        # the relevant kwargs via load_transformer_geometry_forcing being
        # a partial-style closure).  We use a small trick: monkey-patch the
        # loader to capture our internal kwargs at this scope.
        _internal_kwargs = dict(
            enable_internal=self.enable_internal,
            internal_depth=self.internal_depth,
            num_internal_blocks=self.num_internal_blocks,
        )

        def _loader(transformer_path, torch_dtype, torch_device):
            return load_transformer_gf_internal(
                transformer_path,
                torch_dtype=torch_dtype,
                torch_device=torch_device,
                **_internal_kwargs,
            )

        gf_train.base_train.load_transformer = _loader

        super().__init__(config)

    # ------------------------------------------------------------------
    # Train step: GF losses (parent) + internal deep-supervision loss.
    # ------------------------------------------------------------------
    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        # GF: alignment hooks
        need_align = (
            self.gf_head is not None
            and (self.gf_lambda_angular > 0 or self.gf_lambda_scale > 0)
            and self.step >= self.gf_start_step
            and self.step % self.vggt_loss_interval == 0
        )

        # Attention visualization (only at sparse intervals, rank 0 only).
        do_vis = (
            self.attn_vis_enabled
            and self.config.rank == 0
            and self.step > 0
            and (self.step % self.attn_vis_interval == 0)
        )

        transformer_input = dict(input_dict)
        if need_align:
            transformer_input['return_hidden_layers'] = True
            transformer_input['align_layer_idx'] = list(self.gf_student_layers)
            transformer_input['align_video_tokens_only'] = True
        if do_vis:
            transformer_input['capture_cross_attn'] = True
            transformer_input['capture_layers'] = list(self.attn_vis_layers)

        output = self.transformer(transformer_input, train_mode=True)

        align_hidden_states = None
        attn_probs = None
        attn_split_list = None
        internal_pred = None
        if isinstance(output, dict):
            align_hidden_states = output.get('align_hidden_states')
            attn_probs = output.get('attn_probs')
            attn_split_list = output.get('split_list')
            internal_pred = output.get('internal_pred')
            output_pred = output['pred']
        else:
            output_pred = output

        # Main-head loss.
        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output_pred)
        loss = latent_loss + action_loss

        # Internal-head deep supervision (Q1=A, Q2=a).
        int_latent_loss = torch.tensor(0.0, device=self.device)
        int_action_loss = torch.tensor(0.0, device=self.device)
        if (
            self.enable_internal
            and internal_pred is not None
            and internal_pred[0] is not None
            and self.step >= self.internal_start_step
        ):
            int_latent_loss, int_action_loss, _ = self.compute_loss(
                input_dict, internal_pred
            )
            loss = loss + self.lambda_internal * (int_latent_loss + int_action_loss)

        # Hybrid VGGT depth/point loss (parent class).
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

        # Geometry Forcing alignment losses.
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

        # Optional: dump per-layer cross-attn heatmaps to disk.
        if do_vis and attn_probs:
            try:
                self._dump_attention_grid(
                    attn_probs=attn_probs,
                    split_list=attn_split_list,
                    input_dict=input_dict,
                )
            except Exception as e:  # pragma: no cover — vis must never crash training
                logger.warning(f"Attention visualization failed: {e}")

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'vggt_loss': vggt_loss_val.detach(),
            'gf_angular_loss': ang_loss_val.detach(),
            'gf_scale_loss': scl_loss_val.detach(),
            'gf_cosine': cos_val.detach(),
            'int_latent_loss': int_latent_loss.detach(),
            'int_action_loss': int_action_loss.detach(),
        }

        if should_sync:
            max_norm = 2.0
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
    # Visualization helper.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _dump_attention_grid(self, attn_probs, split_list, input_dict):
        latent_dict = input_dict['latent_dict']
        # Token grid (F_token, H_token, W_token) for the noisy video segment.
        # Match how the model embeds the video: patch_size = (p1, p2, p3) on (F, H, W).
        _, _, F_lat, H_lat, W_lat = latent_dict['noisy_latents'].shape
        F_t = F_lat // self.patch_size[0]
        H_t = H_lat // self.patch_size[1]
        W_t = W_lat // self.patch_size[2]

        heatmaps = cross_attn_to_video_heatmaps(
            attn_probs=attn_probs,
            split_list=split_list,
            video_grid_thw=(F_t, H_t, W_t),
            token_meta=self.attn_vis_token_meta,
        )

        # Decode a single sample of GT latent → RGB → cam_high crop, just so
        # we have something to overlay; we use sample 0 of the batch.
        gt_clean = latent_dict['latent']
        frame_indices = list(range(F_t))
        gt_pixels = self._decode_latent_to_pixels(gt_clean, frame_indices, enable_grad=False)
        gt_pixels = self._select_vggt_supervision_pixels(gt_pixels)
        rgb = gt_pixels[0]  # (F, 3, H, W) in [0, 1]

        out_dir = Path(self.config.save_root) / 'attn_vis'
        out_path = out_dir / f'attn_step_{self.step:07d}.png'
        save_layered_attention_grid(
            heatmaps=heatmaps,
            rgb_frames=rgb,
            out_path=out_path,
            sample_layers=self.attn_vis_layers,
            sample_frames=self.attn_vis_frames,
            alpha=self.attn_vis_alpha,
        )
        logger.info(f"Saved attention grid -> {out_path}")
        if self.tb_writer is not None:
            try:
                from PIL import Image
                arr = np.asarray(Image.open(str(out_path)).convert('RGB'))
                self.tb_writer.add_image(
                    'attn/cross_attn_grid', arr.transpose(2, 0, 1), self.step
                )
            except Exception:
                pass
        if self.wandb is not None:
            try:
                self.wandb.log({
                    'attn/cross_attn_grid': self.wandb.Image(str(out_path)),
                }, step=self.step)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Save additional internal head weights.
    # ------------------------------------------------------------------
    def save_checkpoint(self):
        super().save_checkpoint()
        if self.config.rank != 0 or not self.enable_internal:
            return
        ckpt_dir = self.save_dir / f"checkpoint_step_{self.step}"
        # The internal_* weights live in the (FSDP-sharded) transformer; the
        # parent class already saves the full transformer state. Here we only
        # save a tiny manifest so users can verify which fork-depth was used.
        manifest = {
            'enable_internal': bool(self.enable_internal),
            'internal_depth': int(self.internal_depth),
            'num_internal_blocks': int(self.num_internal_blocks),
            'lambda_internal': float(self.lambda_internal),
            'gf_student_layers': list(self.gf_student_layers),
            'gf_teacher_layers': list(self.gf_teacher_layers),
        }
        try:
            (ckpt_dir / 'gf_internal_manifest.json').write_text(
                __import__('json').dumps(manifest, indent=2)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Training loop (extends parent to also log internal losses).
    # ------------------------------------------------------------------
    def train(self):
        logger.info(
            "Starting Geometry Forcing + Internal training for "
            f"{self.config.num_steps} steps... internal_depth={self.internal_depth}, "
            f"num_internal_blocks={self.num_internal_blocks}, "
            f"lambda_internal={self.lambda_internal}, "
            f"attn_vis_enabled={self.attn_vis_enabled}"
        )
        self.transformer.train()
        if self.gf_head is not None:
            self.gf_head.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (GF+Internal)",
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
            'int_lat': [], 'int_act': [],
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
            buf['int_lat'].append(losses['int_latent_loss'])
            buf['int_act'].append(losses['int_action_loss'])
            step_in_accum += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                def _m(k):
                    return dist_mean(torch.stack(buf[k]).sum()).detach().cpu().item()

                latent_v = _m('latent'); action_v = _m('action'); vggt_v = _m('vggt')
                gf_a = _m('gf_ang'); gf_s = _m('gf_scl'); gf_c = _m('gf_cos')
                int_lat_v = _m('int_lat'); int_act_v = _m('int_act')

                for k in buf:
                    buf[k] = []
                step_in_accum = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'lat': f'{latent_v:.4f}',
                        'act': f'{action_v:.4f}',
                        'iLat': f'{int_lat_v:.4f}',
                        'iAct': f'{int_act_v:.4f}',
                        'gfA': f'{gf_a:.4f}',
                        'gfS': f'{gf_s:.4f}',
                        'cos': f'{gf_c:.3f}',
                        'step': self.step,
                        'gn': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                    })
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', latent_v, self.step)
                        self.tb_writer.add_scalar('loss/action', action_v, self.step)
                        self.tb_writer.add_scalar('loss/vggt', vggt_v, self.step)
                        self.tb_writer.add_scalar('loss/gf_angular', gf_a, self.step)
                        self.tb_writer.add_scalar('loss/gf_scale', gf_s, self.step)
                        self.tb_writer.add_scalar('loss/int_latent', int_lat_v, self.step)
                        self.tb_writer.add_scalar('loss/int_action', int_act_v, self.step)
                        self.tb_writer.add_scalar('train/gf_cosine', gf_c, self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.wandb is not None:
                        self.wandb.log({
                            'loss/video': latent_v,
                            'loss/action': action_v,
                            'loss/vggt': vggt_v,
                            'loss/gf_angular': gf_a,
                            'loss/gf_scale': gf_s,
                            'loss/int_latent': int_lat_v,
                            'loss/int_action': int_act_v,
                            'train/gf_cosine': gf_c,
                            'train/grad_norm': total_norm.item(),
                            'train/lr': lr,
                        }, step=self.step)

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


GFI_CONFIGS = dict(BASE_CONFIGS)
GFI_CONFIGS['robotwin_train_vggt_geometry_forcing_internal'] = (
    va_robotwin_train_vggt_geometry_forcing_internal_cfg
)
GFI_CONFIGS['robotwin_train_vggt_geometry_forcing_internal_debug'] = (
    va_robotwin_train_vggt_geometry_forcing_internal_debug_cfg
)


def run(args):
    config = GFI_CONFIGS[args.config_name]

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

    trainer = GeometryForcingInternalTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train Wan-VA with Geometry Forcing + Internal head + attn vis"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_vggt_geometry_forcing_internal',
    )
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
