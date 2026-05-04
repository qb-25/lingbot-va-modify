# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Modified from train.py: adds VGGT geometric alignment loss for world model.
import argparse
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
    load_vae,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
import gc


class VGGTTrainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin_vggt"),
                config=config,
                mode="online",
                name='train_vggt'
            )
            logger.info("WandB logging enabled")
        self.tb_writer = None
        if config.rank == 0 and SummaryWriter is not None:
            tb_log_dir = str(Path(config.save_root) / "tb_logs")
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard logging to {tb_log_dir}")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        gc.collect()
        torch.cuda.empty_cache()

        # Load frozen VAE decoder for VGGT loss
        self.vggt_loss_weight = getattr(config, 'vggt_loss_weight', 0.1)
        self.vggt_loss_start_step = getattr(config, 'vggt_loss_start_step', 0)
        self.vggt_loss_interval = getattr(config, 'vggt_loss_interval', 1)
        self.vggt_num_sample_frames = getattr(config, 'vggt_num_sample_frames', 2)

        self.vggt_grad_enabled = getattr(config, 'vggt_grad_enabled', True)

        if self.vggt_loss_weight > 0:
            logger.info("Loading frozen VAE for VGGT loss...")
            vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'vae')
            self.vae = load_vae(vae_path, torch_dtype=self.dtype, torch_device=self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)

            logger.info("Loading frozen VGGT model...")
            from vggt.models.vggt import VGGT as VGGTModel
            self.vggt = VGGTModel.from_pretrained("facebook/VGGT-1B")
            del self.vggt.camera_head
            del self.vggt.track_head
            self.vggt.camera_head = None
            self.vggt.track_head = None
            self.vggt.eval()
            self.vggt.requires_grad_(False)
            self.vggt.to(dtype=self.dtype, device=self.device)
            logger.info(f"VGGT model loaded (grad_enabled={self.vggt_grad_enabled}).")
        else:
            self.vae = None
            self.vggt = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        max_samples = getattr(config, 'max_train_samples', 0)
        if max_samples > 0 and max_samples < len(train_dataset):
            logger.info(f"Limiting dataset from {len(train_dataset)} to {max_samples} samples (debug mode)")
            train_dataset = torch.utils.data.Subset(
                train_dataset, list(range(max_samples)))
        logger.info(f"Dataset size: {len(train_dataset)}")
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
    
    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)
        return input_dict

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        action_loss = action_loss.permute(0, 2, 3, 4, 1)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)
        action_loss = action_loss.flatten(0, 1).flatten(1)
        action_mask = action_mask.flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1)
        action_mask_per_frame = action_mask.sum(dim=1)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps, latent_pred

    def _velocity_to_clean_latent(self, latent_pred, latent_dict):
        """Approximate single-step denoising: recover clean latent from predicted velocity.

        Flow matching:  x_t = (1-σ)*x_0 + σ*ε,  v = ε - x_0
        => x_0 = x_t - σ * v

        Returns (pred_clean, per_frame_sigmas) so downstream can weight by noise level.
        """
        noisy = latent_dict['noisy_latents']
        timesteps = latent_dict['timesteps']  # [B, F]
        sigmas = (timesteps / self.train_scheduler_latent.num_train_timesteps)
        sigmas_expanded = sigmas[:, None, :, None, None]  # [B, 1, F, 1, 1]
        pred_clean = noisy - sigmas_expanded.to(latent_pred.device) * latent_pred
        return pred_clean, sigmas  # sigmas shape [B, F]

    def _denormalize_latent(self, latent):
        """Reverse the (mu - mean) / std normalization before VAE decode."""
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1).to(latent.device, latent.dtype))
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, -1, 1, 1, 1).to(latent.device, latent.dtype))
        return latent * latents_std + latents_mean

    def _decode_latent_to_pixels(self, latent, frame_indices, enable_grad=False):
        """Decode selected latent frames to pixel images via frozen VAE decoder.

        Handles latent denormalization and pixel range conversion (VAE outputs
        [-1,1], VGGT expects [0,1]).
        """
        B, C, F, H, W = latent.shape
        selected = latent[:, :, frame_indices]
        selected = self._denormalize_latent(selected).to(self.dtype)
        selected = rearrange(selected, 'b c f h w -> (b f) c 1 h w')
        if enable_grad:
            pixels = self.vae.decode(selected).sample
        else:
            with torch.no_grad():
                pixels = self.vae.decode(selected).sample
        pixels = rearrange(pixels, '(b f) c 1 h w -> b f c h w', b=B)
        pixels = ((pixels + 1.0) / 2.0).clamp(0, 1)
        return pixels

    def _compute_vggt_features(self, pixel_frames, enable_grad=False):
        """Extract VGGT depth and point features from pixel frames.

        When enable_grad=True, gradients flow from features back to input pixels.
        VGGT weights are frozen but autograd traces through ops for the input.
        """
        B, num_f, C, H, W = pixel_frames.shape
        target_h, target_w = 518, 518
        frames_resized = F.interpolate(
            pixel_frames.reshape(B * num_f, C, H, W),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        ).reshape(B, num_f, C, target_h, target_w)

        if enable_grad:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(frames_resized)
                depth_map, depth_conf = self.vggt.depth_head(
                    aggregated_tokens_list, images=frames_resized, patch_start_idx=ps_idx)
                point_map, point_conf = self.vggt.point_head(
                    aggregated_tokens_list, images=frames_resized, patch_start_idx=ps_idx)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    aggregated_tokens_list, ps_idx = self.vggt.aggregator(frames_resized)
                    depth_map, depth_conf = self.vggt.depth_head(
                        aggregated_tokens_list, images=frames_resized, patch_start_idx=ps_idx)
                    point_map, point_conf = self.vggt.point_head(
                        aggregated_tokens_list, images=frames_resized, patch_start_idx=ps_idx)

        return {
            'depth': depth_map,
            'point_map': point_map,
            'depth_conf': depth_conf,
            'point_conf': point_conf,
        }

    def compute_vggt_loss(self, latent_pred, input_dict):
        """Compute VGGT geometric alignment loss.

        Decodes a subset of frames from predicted and GT latents, runs VGGT,
        and computes confidence-weighted L1 loss on depth and point maps.

        Frames with lower noise (small sigma) get higher weight because the
        single-step x_0 estimate is more accurate there.

        Gradient flow (when vggt_grad_enabled=True):
            transformer -> latent_pred -> velocity_to_clean -> VAE decode -> VGGT -> loss
        """
        if self.vggt is None or self.vae is None:
            return torch.tensor(0.0, device=self.device)

        latent_dict = input_dict['latent_dict']
        pred_clean, sigmas = self._velocity_to_clean_latent(latent_pred, latent_dict)
        gt_clean = latent_dict['latent']

        B, C, F, H, W = pred_clean.shape
        num_sample = min(self.vggt_num_sample_frames, F)
        if F <= num_sample:
            frame_indices = list(range(F))
        else:
            frame_indices = torch.linspace(0, F - 1, num_sample).long().tolist()

        # Per-frame weight: low noise → high weight (sigma=0 → w=1, sigma=1 → w=0)
        frame_sigmas = sigmas[:, frame_indices].detach()  # [B, num_sample]
        sigma_weights = (1.0 - frame_sigmas).clamp(min=0.05)  # [B, num_sample]

        grad_on = self.vggt_grad_enabled

        pred_pixels = self._decode_latent_to_pixels(pred_clean, frame_indices, enable_grad=grad_on)
        pred_feats = self._compute_vggt_features(pred_pixels, enable_grad=grad_on)

        with torch.no_grad():
            gt_pixels = self._decode_latent_to_pixels(gt_clean, frame_indices, enable_grad=False)
            gt_feats = self._compute_vggt_features(gt_pixels, enable_grad=False)

        gt_depth_conf = gt_feats['depth_conf'].float().detach()
        gt_point_conf = gt_feats['point_conf'].float().detach()
        depth_conf_mask = (gt_depth_conf > gt_depth_conf.median()).float()
        point_conf_mask = (gt_point_conf > gt_point_conf.median()).float()

        depth_diff = (pred_feats['depth'].float() - gt_feats['depth'].float().detach()).abs()
        point_diff = (pred_feats['point_map'].float() - gt_feats['point_map'].float().detach()).abs()

        # Apply sigma weight per frame: [B, num_sample] -> [B, num_sample, 1, 1]
        sw = sigma_weights[:, :, None, None]

        depth_loss = (depth_diff.squeeze(-1) * depth_conf_mask * sw).sum() / (depth_conf_mask.sum() + 1e-6)
        point_loss = (point_diff.mean(-1) * point_conf_mask * sw).sum() / (point_conf_mask.sum() + 1e-6)

        return depth_loss + point_loss

    def _select_vggt_supervision_pixels(self, pixel_frames):
        """Select the image region used for VGGT supervision.

        Wan-VA decodes a vertically-stacked multi-camera image whose height
        equals sum(cam_i.height). When ``config.vggt_supervision_cam ==
        'cam_high'`` (default), we crop the bottom ``config.height x
        config.width`` region (which corresponds to the cam_high view in the
        stacked layout). For any other value, the full pixel tensor is
        returned unchanged.

        Args:
            pixel_frames: Tensor of shape (B, F, C, full_h, full_w) in [0, 1].

        Returns:
            Tensor of shape (B, F, C, target_h, target_w).
        """
        cam = getattr(self.config, 'vggt_supervision_cam', 'cam_high')
        if cam != 'cam_high':
            return pixel_frames
        target_h = self.config.height
        target_w = self.config.width
        _, _, _, full_h, full_w = pixel_frames.shape
        if full_h < target_h or full_w < target_w:
            raise ValueError(
                f"Decoded video is smaller than cam_high crop: got "
                f"({full_h}, {full_w}), expected at least ({target_h}, {target_w})"
            )
        h_start = full_h - target_h
        return pixel_frames[:, :, :, h_start:, :target_w]

    def _tb_add_scalar(self, tag, scalar_value, global_step):
        """Safe TensorBoard scalar writer.

        Writes to ``self.tb_writer`` if available; on ``OSError`` (e.g. disk
        full / device error from cloud storage), logs a warning on rank 0,
        closes the writer, and disables further TensorBoard logging for the
        rest of training. Any other exception is re-raised.
        """
        if self.tb_writer is None:
            return
        try:
            self.tb_writer.add_scalar(tag, scalar_value, global_step)
        except OSError as e:
            if self.config.rank == 0:
                logger.warning(
                    'TensorBoard write failed (%s); disabling TensorBoard.', e
                )
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        vggt_loss_val = torch.tensor(0.0, device=self.device)
        if self.vggt_loss_weight > 0 and self.step >= self.vggt_loss_start_step:
            vggt_input = latent_pred if self.vggt_grad_enabled else latent_pred.detach()
            vggt_loss_val = self.compute_vggt_loss(vggt_input, input_dict)
            loss = loss + self.vggt_loss_weight * vggt_loss_val / self.gradient_accumulation_steps

        loss.backward()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'vggt_loss': vggt_loss_val.detach(),
        }
        
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self):
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        logger.info(f"Starting VGGT-augmented training for {self.config.num_steps} steps...")
        logger.info(f"VGGT loss weight: {self.vggt_loss_weight}, start step: {self.vggt_loss_start_step}, interval: {self.vggt_loss_interval}")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (VGGT)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_vggt_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_vggt_losses.append(losses['vggt_loss'])
            step_in_accumulation += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                vggt_loss_show = dist_mean(torch.stack(accumulated_vggt_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_vggt_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'lat': f'{latent_loss_show:.4f}',
                        'act': f'{action_loss_show:.4f}',
                        'vggt': f'{vggt_loss_show:.4f}',
                        'step': self.step,
                        'gn': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action', action_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/vggt', vggt_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/video_max', max_latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action_max', max_action_loss_show, self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_avg_vggt_loss': vggt_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
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
        logger.info("Training completed!")


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = VGGTTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train WAN model with VGGT alignment loss")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_vggt',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
