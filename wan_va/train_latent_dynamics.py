import argparse
import contextlib
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Force HF / datasets caches away from the full /mnt/data mount when the
# launcher script does not provide explicit paths.
_default_hf_home = "/mnt/nas/qb/hf_cache"
os.environ.setdefault("HF_HOME", _default_hf_home)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_default_hf_home, "datasets"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_default_hf_home, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(_default_hf_home, "assets"))
os.environ.setdefault("TMPDIR", "/tmp")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import dist_max, dist_mean, init_distributed
from modules.latent_dynamics import LatentDynamicsBranch
from train_vggt import VGGTTrainer
from utils import init_logger, logger, warmup_constant_lambda


class LatentDynamicsTrainer(VGGTTrainer):
    def __init__(self, config):
        # Reuse the copied VGGT trainer utilities, but disable VGGT itself.
        config.vggt_loss_weight = 0.0
        super().__init__(config)

        self.latent_dynamics_pred_weight = getattr(config, 'latent_dynamics_pred_weight', 0.05)
        self.latent_dynamics_reg_weight = getattr(config, 'latent_dynamics_reg_weight', 0.01)
        self.latent_dynamics_num_projections = getattr(config, 'latent_dynamics_num_projections', 256)
        self.latent_dynamics_state_dim = getattr(config, 'latent_dynamics_state_dim', 256)
        self.latent_dynamics_hidden_dim = getattr(config, 'latent_dynamics_hidden_dim', 512)
        self.latent_dynamics_start_step = getattr(config, 'latent_dynamics_start_step', 0)

        latent_channels = int(self.transformer.config.in_channels)
        branch = LatentDynamicsBranch(
            latent_channels=latent_channels,
            action_dim=self.config.action_dim,
            action_per_frame=self.config.action_per_frame,
            state_dim=self.latent_dynamics_state_dim,
            hidden_dim=self.latent_dynamics_hidden_dim,
            num_projections=self.latent_dynamics_num_projections,
        ).to(self.device)

        if getattr(config, 'resume_from', None):
            self._load_latent_dynamics_state(branch, config.resume_from)

        if self.config.world_size > 1:
            branch = DDP(branch, device_ids=[self.config.local_rank], output_device=self.config.local_rank)
        self.latent_dynamics_head = branch

        self.branch_optimizer = torch.optim.AdamW(
            [p for p in self._latent_dynamics_module().parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.branch_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.branch_optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )

    def _latent_dynamics_module(self):
        return self.latent_dynamics_head.module if isinstance(self.latent_dynamics_head, DDP) else self.latent_dynamics_head

    def _load_latent_dynamics_state(self, branch, checkpoint_root):
        model_file = Path(checkpoint_root) / 'latent_dynamics_head' / 'model.safetensors'
        if not model_file.exists():
            if self.config.rank == 0:
                logger.info("No latent dynamics state found in resume checkpoint. Initializing branch from scratch.")
            return

        state_dict = load_file(model_file)
        branch.load_state_dict(state_dict, strict=True)
        if self.config.rank == 0:
            logger.info(f"Loaded latent dynamics branch from {model_file}")

    def _clip_gradients(self, max_norm: float):
        transformer_norm = torch.nn.utils.clip_grad_norm_(
            self.transformer.parameters(),
            max_norm,
        )
        branch_norm = torch.nn.utils.clip_grad_norm_(
            self._latent_dynamics_module().parameters(),
            max_norm,
        )
        total_norm = torch.sqrt(transformer_norm.float().pow(2) + branch_norm.float().pow(2))
        return total_norm

    def compute_latent_dynamics_loss(self, latent_pred, input_dict):
        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']

        pred_clean, sigmas = self._velocity_to_clean_latent(latent_pred, latent_dict)
        gt_clean = latent_dict['latent']
        gt_actions = action_dict['latent']

        branch = self.latent_dynamics_head
        pred_states = branch.module.encode_states(pred_clean) if isinstance(branch, DDP) else branch.encode_states(pred_clean)
        gt_states = branch.module.encode_states(gt_clean) if isinstance(branch, DDP) else branch.encode_states(gt_clean)
        action_states = branch.module.encode_actions(gt_actions) if isinstance(branch, DDP) else branch.encode_actions(gt_actions)
        pred_next = branch.module.predict_next(pred_states, action_states) if isinstance(branch, DDP) else branch.predict_next(pred_states, action_states)

        if pred_next.shape[1] == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero, zero, zero

        target_next = gt_states[:, 1:]
        sigma_pairs = (1.0 - 0.5 * (sigmas[:, :-1] + sigmas[:, 1:])).clamp(min=0.05)

        pred_mse = (pred_next.float() - target_next.float()).pow(2).mean(dim=-1)
        pred_loss = (pred_mse * sigma_pairs).sum() / (sigma_pairs.sum() + 1e-6)

        reg_loss = (
            branch.module.gaussian_reg(gt_states) if isinstance(branch, DDP) else branch.gaussian_reg(gt_states)
        )
        state_std = gt_states.float().std(dim=(0, 1)).mean()
        total = self.latent_dynamics_pred_weight * pred_loss + self.latent_dynamics_reg_weight * reg_loss
        return total, pred_loss.detach(), reg_loss.detach(), state_std.detach()

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        sync_ctx = contextlib.nullcontext()
        if isinstance(self.latent_dynamics_head, DDP) and not should_sync:
            sync_ctx = self.latent_dynamics_head.no_sync()

        with sync_ctx:
            output = self.transformer(input_dict, train_mode=True)
            latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output)
            loss = latent_loss + action_loss

            latent_dynamics_loss = torch.tensor(0.0, device=self.device)
            latent_dynamics_pred = torch.tensor(0.0, device=self.device)
            latent_dynamics_reg = torch.tensor(0.0, device=self.device)
            latent_state_std = torch.tensor(0.0, device=self.device)

            if self.step >= self.latent_dynamics_start_step:
                latent_dynamics_loss, latent_dynamics_pred, latent_dynamics_reg, latent_state_std = (
                    self.compute_latent_dynamics_loss(latent_pred, input_dict)
                )
                loss = loss + latent_dynamics_loss / self.gradient_accumulation_steps

            loss.backward()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'latent_dynamics_loss': latent_dynamics_loss.detach(),
            'latent_dynamics_pred': latent_dynamics_pred,
            'latent_dynamics_reg': latent_dynamics_reg,
            'latent_state_std': latent_state_std,
        }

        if should_sync:
            total_norm = self._clip_gradients(2.0)
            self.optimizer.step()
            self.branch_optimizer.step()
            self.lr_scheduler.step()
            self.branch_lr_scheduler.step()
            self.optimizer.zero_grad()
            self.branch_optimizer.zero_grad()
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self):
        super().save_checkpoint()
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        if self.config.rank == 0:
            branch_dir = checkpoint_dir / 'latent_dynamics_head'
            branch_dir.mkdir(parents=True, exist_ok=True)
            state_dict = {
                k: v.detach().to(torch.bfloat16)
                for k, v in self._latent_dynamics_module().state_dict().items()
            }
            save_file(state_dict, branch_dir / 'model.safetensors')
            with open(branch_dir / 'config.json', 'w') as f:
                json.dump(
                    {
                        'latent_channels': self._latent_dynamics_module().latent_channels,
                        'action_dim': self._latent_dynamics_module().action_dim,
                        'action_per_frame': self._latent_dynamics_module().action_per_frame,
                        'state_dim': self._latent_dynamics_module().state_dim,
                        'hidden_dim': self._latent_dynamics_module().hidden_dim,
                        'num_projections': self.latent_dynamics_num_projections,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved latent dynamics branch to {branch_dir}")
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        logger.info(f"Starting latent-dynamics-augmented training for {self.config.num_steps} steps...")
        logger.info(
            f"Latent dynamics weights: pred={self.latent_dynamics_pred_weight}, "
            f"reg={self.latent_dynamics_reg_weight}, projections={self.latent_dynamics_num_projections}, "
            f"state_dim={self.latent_dynamics_state_dim}"
        )
        self.transformer.train()
        self._latent_dynamics_module().train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (LatentDyn)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        self.branch_optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_branch_losses = []
        accumulated_pred_losses = []
        accumulated_reg_losses = []
        accumulated_state_stds = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)

            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_branch_losses.append(losses['latent_dynamics_loss'])
            accumulated_pred_losses.append(losses['latent_dynamics_pred'])
            accumulated_reg_losses.append(losses['latent_dynamics_reg'])
            accumulated_state_stds.append(losses['latent_state_std'])
            step_in_accumulation += 1

            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                branch_loss_show = dist_mean(torch.stack(accumulated_branch_losses).sum()).detach().cpu().item()
                pred_loss_show = dist_mean(torch.stack(accumulated_pred_losses).sum()).detach().cpu().item()
                reg_loss_show = dist_mean(torch.stack(accumulated_reg_losses).sum()).detach().cpu().item()
                state_std_show = dist_mean(torch.stack(accumulated_state_stds).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_branch_losses = []
                accumulated_pred_losses = []
                accumulated_reg_losses = []
                accumulated_state_stds = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        {
                            'lat': f'{latent_loss_show:.4f}',
                            'act': f'{action_loss_show:.4f}',
                            'ld': f'{branch_loss_show:.4f}',
                            'ldp': f'{pred_loss_show:.4f}',
                            'ldr': f'{reg_loss_show:.4f}',
                            'std': f'{state_std_show:.3f}',
                            'step': self.step,
                            'gn': f'{total_norm.item():.2f}',
                            'lr': f'{lr:.2e}',
                        }
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar('loss/video', latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action', action_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/latent_dynamics_total', branch_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/latent_dynamics_pred', pred_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/latent_dynamics_reg', reg_loss_show, self.step)
                        self.tb_writer.add_scalar('metric/latent_state_std', state_std_show, self.step)
                        self.tb_writer.add_scalar('loss/video_max', max_latent_loss_show, self.step)
                        self.tb_writer.add_scalar('loss/action_max', max_action_loss_show, self.step)
                        self.tb_writer.add_scalar('train/grad_norm', total_norm.item(), self.step)
                        self.tb_writer.add_scalar('train/lr', lr, self.step)

                    if self.config.enable_wandb:
                        self.wandb.log(
                            {
                                'loss_metrics/global_avg_video_loss': latent_loss_show,
                                'loss_metrics/global_avg_action_loss': action_loss_show,
                                'loss_metrics/global_avg_latent_dynamics_loss': branch_loss_show,
                                'loss_metrics/global_avg_latent_dynamics_pred_loss': pred_loss_show,
                                'loss_metrics/global_avg_latent_dynamics_reg_loss': reg_loss_show,
                                'loss_metrics/global_avg_latent_state_std': state_std_show,
                                'loss_metrics/global_max_video_loss': max_latent_loss_show,
                                'loss_metrics/global_max_action_loss': max_action_loss_show,
                                'grad_norm': total_norm.item(),
                                'lr': lr,
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

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = LatentDynamicsTrainer(config)
    trainer.train()


@record
def main():
    parser = argparse.ArgumentParser(description="Train WAN model with latent dynamics branch")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_latent_dynamics',
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
