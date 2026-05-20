# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# Geometry Forcing + Cross-Stream Alignment training for Wan-VA.
#
#   * Geometry Forcing (Wu et al., 2025): multi-layer Angular + Scale
#     alignment between Wan DiT hidden states and a frozen VGGT teacher.
#   * Cross-Stream Alignment: at a chosen DiT block (default 25, NOT a
#     GF layer), pool video and action tokens to per-frame embeddings
#     and apply a symmetric InfoNCE objective with all-gather across
#     ranks and sigma-aware gating.
#
# Also: dump a Wan cross-attention overlay grid every ``attn_vis_interval``
# (default 2000) steps under ``${save_root}/attn_vis/``.
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
from .configs.va_robotwin_train_vggt_geometry_forcing_xstream_cfg import (
    va_robotwin_train_vggt_geometry_forcing_xstream_cfg,
)
from .configs.va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg import (
    va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg,
)
from .distributed.util import dist_max, dist_mean, init_distributed
from .modules.attention_visualizer import (
    cross_attn_to_video_heatmaps,
    save_layered_attention_grid,
)
from .modules.cross_stream_align import (
    CrossStreamProjector,
    compute_cross_stream_loss,
    pool_action_tokens,
    pool_video_tokens,
)
from .modules.utils_geometry_forcing_internal import (
    load_transformer as load_transformer_gf_internal,
)
from .utils import init_logger, logger, warmup_constant_lambda


# Use the GF+Internal backbone (which carries the xstream / attn capture
# hooks) but with internal head DISABLED. This way the model behaves as
# a vanilla GF backbone plus the extra hooks we need.
def _loader_gf_xstream(transformer_path, torch_dtype, torch_device):
    return load_transformer_gf_internal(
        transformer_path,
        torch_dtype=torch_dtype,
        torch_device=torch_device,
        enable_internal=False,
        internal_depth=0,
        num_internal_blocks=0,
    )


# Patch the parent trainer's load_transformer entry point.
gf_train.base_train.load_transformer = _loader_gf_xstream


class GFCrossStreamTrainer(gf_train.GeometryForcingTrainer):
    """GF trainer + cross-stream contrastive head + attention visualizer."""

    def __init__(self, config):
        # Cross-stream config.
        self.xstream_enable = bool(getattr(config, 'xstream_enable', True))
        self.xstream_layers = list(getattr(config, 'xstream_layers', [25]))
        self.xstream_proj_dim = int(getattr(config, 'xstream_proj_dim', 256))
        self.xstream_proj_hidden = int(getattr(config, 'xstream_proj_hidden', 1024))
        self.xstream_pos_mode = str(getattr(config, 'xstream_pos_mode', 'window'))
        self.xstream_window = int(getattr(config, 'xstream_window', 2))
        self.xstream_tau = float(getattr(config, 'xstream_tau', 0.15))
        self.lambda_xstream = float(getattr(config, 'lambda_xstream', 0.1))
        self.xstream_interval = int(getattr(config, 'xstream_interval', 1))
        self.xstream_start_step = int(getattr(config, 'xstream_start_step', 0))
        self.xstream_use_all_gather = bool(getattr(config, 'xstream_use_all_gather', True))
        self.xstream_sigma_soft = bool(getattr(config, 'xstream_sigma_soft', True))
        self.xstream_sigma_threshold = float(getattr(config, 'xstream_sigma_threshold', 0.5))

        # Visualization config (project requirement: every 2000 steps).
        self.attn_vis_enabled = bool(getattr(config, 'attn_vis_enabled', True))
        self.attn_vis_interval = int(getattr(config, 'attn_vis_interval', 2000))
        self.attn_vis_layers = list(getattr(config, 'attn_vis_layers', [0, 10, 20, 29]))
        self.attn_vis_frames = list(getattr(config, 'attn_vis_frames', [0, 2, 5]))
        self.attn_vis_token_meta = dict(
            getattr(config, 'attn_vis_token_meta', {'mode': 'content_only', 'top_k': 16})
        )
        self.attn_vis_alpha = float(getattr(config, 'attn_vis_alpha', 0.5))

        self.xstream_projector = None
        self.optimizer_xstream = None
        self.lr_scheduler_xstream = None

        super().__init__(config)

        # Build cross-stream projector AFTER super().__init__ so the
        # transformer config is available.
        if self.xstream_enable and self.lambda_xstream > 0:
            student_dim = int(
                self.transformer.config.num_attention_heads
                * self.transformer.config.attention_head_dim
            )
            # Both video and action tokens live in inner_dim (the trunk
            # dim) at any block; they're not projected back to their
            # input vocab until the proj_out layer.
            self.xstream_projector = CrossStreamProjector(
                d_v=student_dim,
                d_a=student_dim,
                d_proj=self.xstream_proj_dim,
                hidden=self.xstream_proj_hidden,
            ).to(self.device)

            # Same FSDP-friendly recipe as gf_head: separate non-fused
            # AdamW (the projector is a plain nn.Module, not DTensor).
            self.optimizer_xstream = torch.optim.AdamW(
                [p for p in self.xstream_projector.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=False,
                foreach=False,
            )
            self.lr_scheduler_xstream = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_xstream,
                lr_lambda=lambda step: warmup_constant_lambda(
                    step, warmup_steps=config.warmup_steps
                ),
            )
            logger.info(
                "[xstream] CrossStreamProjector built: "
                f"layers={self.xstream_layers}, d_proj={self.xstream_proj_dim}, "
                f"tau={self.xstream_tau}, pos_mode={self.xstream_pos_mode}, "
                f"all_gather={self.xstream_use_all_gather}, "
                f"lambda={self.lambda_xstream}"
            )

    # ------------------------------------------------------------------
    # Distributed grad sync for the xstream projector (mirrors _sync_gf_grads).
    # ------------------------------------------------------------------
    def _sync_xstream_grads(self):
        if not dist.is_initialized() or self.xstream_projector is None:
            return
        world_size = dist.get_world_size()
        for param in self.xstream_projector.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    # ------------------------------------------------------------------
    # Helper: derive sigma_v / sigma_a aligned to the *sampled* video frames
    # used by the cross-stream pooling.
    # ------------------------------------------------------------------
    def _xstream_sigma(self, input_dict, F_t):
        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        n_train = self.train_scheduler_latent.num_train_timesteps
        sigma_v = (latent_dict['timesteps'].float() / n_train).clamp(0.0, 1.0)
        sigma_a = (action_dict['timesteps'].float() / n_train).clamp(0.0, 1.0)
        # Both should be (B, F_lat) and (B, F_act). Down-/up-sample to F_t
        # if patch_size[0] != 1; with the project default patch_size=[1,2,2]
        # they are already at the same temporal grid.
        if sigma_v.shape[-1] != F_t:
            # Pool/repeat to length F_t (linear interpolation along F).
            sigma_v = torch.nn.functional.interpolate(
                sigma_v.unsqueeze(1), size=F_t, mode='linear', align_corners=False
            ).squeeze(1)
        if sigma_a.shape[-1] != F_t:
            sigma_a = torch.nn.functional.interpolate(
                sigma_a.unsqueeze(1), size=F_t, mode='linear', align_corners=False
            ).squeeze(1)
        return sigma_v, sigma_a

    # ------------------------------------------------------------------
    # Train step.
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

        # Cross-stream: capture both video and action hidden at xstream layers.
        need_xstream = (
            self.xstream_enable
            and self.xstream_projector is not None
            and self.lambda_xstream > 0
            and self.step >= self.xstream_start_step
            and self.step % max(1, self.xstream_interval) == 0
        )

        # Attention visualization (every 2000 steps by default).
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
        if need_xstream:
            transformer_input['xstream_layers'] = list(self.xstream_layers)
        if do_vis:
            transformer_input['capture_cross_attn'] = True
            transformer_input['capture_layers'] = list(self.attn_vis_layers)

        output = self.transformer(transformer_input, train_mode=True)

        align_hidden_states = None
        xstream_hidden = None
        attn_probs = None
        attn_split_list = None
        if isinstance(output, dict):
            align_hidden_states = output.get('align_hidden_states')
            xstream_hidden = output.get('xstream_hidden_states')
            attn_probs = output.get('attn_probs')
            attn_split_list = output.get('split_list')
            output_pred = output['pred']
        else:
            output_pred = output

        # Main loss.
        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output_pred)
        loss = latent_loss + action_loss

        # Hybrid VGGT depth/point (parent class behaviour).
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

        # GF Angular + Scale.
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

        # Cross-stream contrastive.
        xstream_loss_val = torch.tensor(0.0, device=self.device)
        xstream_metrics = {}
        if need_xstream and xstream_hidden:
            xstream_loss_val, xstream_metrics = self._compute_xstream_loss(
                xstream_hidden=xstream_hidden,
                input_dict=input_dict,
            )
            loss = loss + self.lambda_xstream * xstream_loss_val / self.gradient_accumulation_steps

        loss.backward()
        self._sync_gf_grads()
        self._sync_xstream_grads()

        # Attention vis dump.
        if do_vis and attn_probs:
            try:
                self._dump_attention_grid(
                    attn_probs=attn_probs,
                    split_list=attn_split_list,
                    input_dict=input_dict,
                )
            except Exception as e:  # pragma: no cover -- vis must never crash training
                logger.warning(f"Attention visualization failed: {e}")

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'vggt_loss': vggt_loss_val.detach(),
            'gf_angular_loss': ang_loss_val.detach(),
            'gf_scale_loss': scl_loss_val.detach(),
            'gf_cosine': cos_val.detach(),
            'xstream_loss': xstream_loss_val.detach(),
        }
        for k, v in xstream_metrics.items():
            losses[k] = torch.tensor(float(v), device=self.device)

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
            if self.xstream_projector is not None:
                x_norm = torch.nn.utils.clip_grad_norm_(
                    self.xstream_projector.parameters(), max_norm, foreach=False
                )
                total_norm = (total_norm * total_norm + x_norm * x_norm).sqrt()
            self.optimizer.step()
            if self.optimizer_gf is not None:
                self.optimizer_gf.step()
            if self.optimizer_xstream is not None:
                self.optimizer_xstream.step()
            self.lr_scheduler.step()
            if self.lr_scheduler_gf is not None:
                self.lr_scheduler_gf.step()
            if self.lr_scheduler_xstream is not None:
                self.lr_scheduler_xstream.step()
            self.optimizer.zero_grad()
            if self.optimizer_gf is not None:
                self.optimizer_gf.zero_grad()
            if self.optimizer_xstream is not None:
                self.optimizer_xstream.zero_grad()
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    # ------------------------------------------------------------------
    # Cross-stream loss helper.
    # ------------------------------------------------------------------
    def _compute_xstream_loss(self, xstream_hidden, input_dict):
        latent_dict = input_dict['latent_dict']
        _, _, F_lat, H_lat, W_lat = latent_dict['noisy_latents'].shape
        F_t = F_lat // self.patch_size[0]
        H_t = H_lat // self.patch_size[1]
        W_t = W_lat // self.patch_size[2]
        # action_per_frame controls the action-stream "spatial" dim K
        K = int(getattr(self.config, 'action_per_frame', 16))

        sigma_v, sigma_a = self._xstream_sigma(input_dict, F_t)

        # Loop over xstream layers and average their losses + metrics.
        total_loss = torch.tensor(0.0, device=self.device)
        agg_metrics = {}
        n = 0
        for layer in self.xstream_layers:
            cap = xstream_hidden.get(int(layer))
            if cap is None:
                continue
            h_v_layer = cap['video']    # (B, F_t*H_t*W_t, D)
            h_a_layer = cap['action']   # (B, F_t*K, D)
            h_v_pooled = pool_video_tokens(h_v_layer, F_t, H_t, W_t)
            h_a_pooled = pool_action_tokens(h_a_layer, F_t, K)

            loss_l, metrics = compute_cross_stream_loss(
                h_v_pooled,
                h_a_pooled,
                self.xstream_projector,
                sigma_v=sigma_v,
                sigma_a=sigma_a,
                pos_mode=self.xstream_pos_mode,
                window=self.xstream_window,
                tau=self.xstream_tau,
                use_all_gather=self.xstream_use_all_gather,
                sigma_threshold=self.xstream_sigma_threshold,
                sigma_soft=self.xstream_sigma_soft,
            )
            total_loss = total_loss + loss_l
            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + float(v)
            n += 1
        if n > 0:
            total_loss = total_loss / n
            agg_metrics = {k: v / n for k, v in agg_metrics.items()}
        return total_loss, agg_metrics

    # ------------------------------------------------------------------
    # Visualization helper.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _dump_attention_grid(self, attn_probs, split_list, input_dict):
        latent_dict = input_dict['latent_dict']
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

        gt_clean = latent_dict['latent']
        rgb = self._decode_latent_to_pixels(
            gt_clean, list(range(F_t)), enable_grad=False
        )
        rgb = self._select_vggt_supervision_pixels(rgb)[0]

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
    # Save the cross-stream projector alongside the GF head.
    # ------------------------------------------------------------------
    def save_checkpoint(self):
        super().save_checkpoint()
        if self.config.rank != 0 or self.xstream_projector is None:
            return
        ckpt_dir = self.save_dir / f"checkpoint_step_{self.step}"
        out = ckpt_dir / 'cross_stream_projector.safetensors'
        try:
            save_file(
                {
                    k: v.detach().cpu().to(
                        torch.bfloat16 if v.is_floating_point() else v.dtype
                    )
                    for k, v in self.xstream_projector.state_dict().items()
                },
                str(out),
            )
        except Exception as e:
            logger.warning(f"Failed to save cross_stream_projector: {e}")

    # ------------------------------------------------------------------
    # Training loop with extra logging.
    # ------------------------------------------------------------------
    def train(self):
        logger.info(
            f"Starting GF + Cross-Stream training for {self.config.num_steps} steps... "
            f"xstream_layers={self.xstream_layers}, "
            f"lambda_xstream={self.lambda_xstream}, "
            f"attn_vis_interval={self.attn_vis_interval}"
        )
        self.transformer.train()
        if self.gf_head is not None:
            self.gf_head.train()
        if self.xstream_projector is not None:
            self.xstream_projector.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (GF+XStream)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        if self.optimizer_gf is not None:
            self.optimizer_gf.zero_grad()
        if self.optimizer_xstream is not None:
            self.optimizer_xstream.zero_grad()

        buf = {
            'latent': [], 'action': [], 'vggt': [],
            'gf_ang': [], 'gf_scl': [], 'gf_cos': [],
            'xs': [], 'xs_diag': [], 'xs_gap': [], 'xs_topk1': [], 'xs_gate': [],
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
            buf['xs'].append(losses['xstream_loss'])
            buf['xs_diag'].append(losses.get('cross_stream_logits_diag', torch.tensor(0.0, device=self.device)))
            buf['xs_gap'].append(losses.get('cross_stream_pos_neg_gap', torch.tensor(0.0, device=self.device)))
            buf['xs_topk1'].append(losses.get('cross_stream_topk1_acc', torch.tensor(0.0, device=self.device)))
            buf['xs_gate'].append(losses.get('cross_stream_gate', torch.tensor(0.0, device=self.device)))
            step_in_accum += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                def _m(k):
                    return dist_mean(torch.stack(buf[k]).sum()).detach().cpu().item()

                vals = {k: _m(k) for k in buf}
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
                        'lat': f"{vals['latent']:.4f}",
                        'act': f"{vals['action']:.4f}",
                        'gfA': f"{vals['gf_ang']:.4f}",
                        'gfS': f"{vals['gf_scl']:.4f}",
                        'cos': f"{vals['gf_cos']:.3f}",
                        'xs':  f"{vals['xs']:.4f}",
                        'gap': f"{vals['xs_gap']:.3f}",
                        'top1': f"{vals['xs_topk1']:.3f}",
                        'step': self.step,
                        'gn': f"{total_norm.item():.2f}",
                        'lr': f"{lr:.2e}",
                    })
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', vals['latent'], self.step)
                        self.tb_writer.add_scalar('loss/action', vals['action'], self.step)
                        self.tb_writer.add_scalar('loss/vggt', vals['vggt'], self.step)
                        self.tb_writer.add_scalar('loss/gf_angular', vals['gf_ang'], self.step)
                        self.tb_writer.add_scalar('loss/gf_scale', vals['gf_scl'], self.step)
                        self.tb_writer.add_scalar('loss/cross_stream', vals['xs'], self.step)
                        self.tb_writer.add_scalar('train/gf_cosine', vals['gf_cos'], self.step)
                        self.tb_writer.add_scalar('train/cross_stream_logits_diag', vals['xs_diag'], self.step)
                        self.tb_writer.add_scalar('train/cross_stream_pos_neg_gap', vals['xs_gap'], self.step)
                        self.tb_writer.add_scalar('train/cross_stream_topk1_acc', vals['xs_topk1'], self.step)
                        self.tb_writer.add_scalar('train/cross_stream_gate', vals['xs_gate'], self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.wandb is not None:
                        self.wandb.log({
                            'loss/video': vals['latent'],
                            'loss/action': vals['action'],
                            'loss/vggt': vals['vggt'],
                            'loss/gf_angular': vals['gf_ang'],
                            'loss/gf_scale': vals['gf_scl'],
                            'loss/cross_stream': vals['xs'],
                            'train/gf_cosine': vals['gf_cos'],
                            'train/cross_stream_logits_diag': vals['xs_diag'],
                            'train/cross_stream_pos_neg_gap': vals['xs_gap'],
                            'train/cross_stream_topk1_acc': vals['xs_topk1'],
                            'train/cross_stream_gate': vals['xs_gate'],
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


GFXSTREAM_CONFIGS = dict(BASE_CONFIGS)
GFXSTREAM_CONFIGS['robotwin_train_vggt_geometry_forcing_xstream'] = (
    va_robotwin_train_vggt_geometry_forcing_xstream_cfg
)
GFXSTREAM_CONFIGS['robotwin_train_vggt_geometry_forcing_xstream_debug'] = (
    va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg
)


def run(args):
    config = GFXSTREAM_CONFIGS[args.config_name]

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

    trainer = GFCrossStreamTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Train Wan-VA with Geometry Forcing + Cross-Stream Alignment + attn vis"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_vggt_geometry_forcing_xstream',
    )
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
