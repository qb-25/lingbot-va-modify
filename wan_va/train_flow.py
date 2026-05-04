import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import dist_max, dist_mean, init_distributed
from modules.utils import load_vae
from train_vggt import VGGTTrainer
from utils import init_logger, logger


class FlowTrainer(VGGTTrainer):
    def __init__(self, config):
        # Keep the original post-train pipeline untouched. This experiment only
        # adds an optical-flow loss on top of the copied trainer entry.
        config.vggt_loss_weight = 0.0
        super().__init__(config)

        self.flow_loss_weight = getattr(config, "flow_loss_weight", 0.0)
        self.flow_loss_start_step = getattr(config, "flow_loss_start_step", 0)
        self.flow_num_pairs = getattr(config, "flow_num_pairs", 1)
        self.flow_grad_enabled = getattr(config, "flow_grad_enabled", True)
        self.flow_model_name = getattr(config, "flow_model_name", "dpflow")
        self.flow_model_ckpt = getattr(config, "flow_model_ckpt", "things")
        self.flow_input_height = getattr(config, "flow_input_height", self.config.height)
        self.flow_input_width = getattr(config, "flow_input_width", self.config.width)
        self.flow_supervision_cam = getattr(config, "flow_supervision_cam", "cam_high")
        self.vggt_supervision_cam = self.flow_supervision_cam
        self.flow_model = None

        if self.flow_loss_weight > 0 and self.vae is None:
            logger.info("Loading frozen VAE for flow loss...")
            vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, "vae")
            self.vae = load_vae(vae_path, torch_dtype=self.dtype, torch_device=self.device)
            self.vae.eval()
            self.vae.requires_grad_(False)

        if self.flow_loss_weight > 0:
            self._load_flow_model()

    def _load_flow_model(self):
        try:
            import ptlflow
        except ImportError as exc:
            raise ImportError(
                "Optical-flow post-training requires PTLFlow / DPFlow. "
                "Install it first, e.g. `pip install \"ptlflow>=0.4.1\" \"lightning<2.7\" jsonargparse loguru`."
            ) from exc

        logger.info(
            f"Loading optical flow model: {self.flow_model_name} (ckpt={self.flow_model_ckpt}, grad_enabled={self.flow_grad_enabled})"
        )
        self.flow_model = ptlflow.get_model(self.flow_model_name, ckpt_path=self.flow_model_ckpt)
        self.flow_model.eval()
        self.flow_model.requires_grad_(False)
        self.flow_model.to(self.device)

    def _resize_flow_pixels(self, pixel_frames):
        bsz, num_f, channels, height, width = pixel_frames.shape
        if height == self.flow_input_height and width == self.flow_input_width:
            return pixel_frames
        resized = F.interpolate(
            pixel_frames.reshape(bsz * num_f, channels, height, width),
            size=(self.flow_input_height, self.flow_input_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(bsz, num_f, channels, self.flow_input_height, self.flow_input_width)

    def _compute_optical_flow(self, image_pairs, enable_grad=False):
        """Estimate optical flow on image pairs shaped [N, 2, 3, H, W]."""
        if self.flow_model is None:
            raise RuntimeError("Flow model has not been initialized.")

        model_inputs = {"images": image_pairs.contiguous().float()}
        if enable_grad:
            predictions = self.flow_model(model_inputs)
        else:
            with torch.no_grad():
                predictions = self.flow_model(model_inputs)

        flows = predictions["flows"]
        if flows.ndim == 5:
            if flows.shape[1] == 1:
                flows = flows[:, 0]
            else:
                flows = flows[:, -1]
        return flows.float()

    def compute_flow_loss(self, latent_pred, input_dict):
        if self.flow_model is None or self.vae is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        latent_dict = input_dict["latent_dict"]
        pred_clean, sigmas = self._velocity_to_clean_latent(latent_pred, latent_dict)
        gt_clean = latent_dict["latent"]

        _, _, num_frames, _, _ = pred_clean.shape
        num_pairs = min(self.flow_num_pairs, max(0, num_frames - 1))
        if num_pairs < 1:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        if num_frames - 1 <= num_pairs:
            pair_starts = list(range(num_frames - 1))
        else:
            pair_starts = torch.linspace(0, num_frames - 2, num_pairs).long().tolist()
            pair_starts = list(dict.fromkeys(pair_starts))

        frame_indices = []
        for start_idx in pair_starts:
            frame_indices.extend([start_idx, start_idx + 1])
        frame_indices = list(dict.fromkeys(frame_indices))

        grad_on = self.flow_grad_enabled

        pred_pixels = self._decode_latent_to_pixels(pred_clean, frame_indices, enable_grad=grad_on)
        pred_pixels = self._select_vggt_supervision_pixels(pred_pixels)
        pred_pixels = self._resize_flow_pixels(pred_pixels)

        with torch.no_grad():
            gt_pixels = self._decode_latent_to_pixels(gt_clean, frame_indices, enable_grad=False)
            gt_pixels = self._select_vggt_supervision_pixels(gt_pixels)
            gt_pixels = self._resize_flow_pixels(gt_pixels)

        frame_to_local = {frame_idx: local_idx for local_idx, frame_idx in enumerate(frame_indices)}
        pred_pairs = []
        gt_pairs = []
        pair_weights = []
        for start_idx in pair_starts:
            pred_pairs.append(
                torch.stack(
                    [
                        pred_pixels[:, frame_to_local[start_idx]],
                        pred_pixels[:, frame_to_local[start_idx + 1]],
                    ],
                    dim=1,
                )
            )
            gt_pairs.append(
                torch.stack(
                    [
                        gt_pixels[:, frame_to_local[start_idx]],
                        gt_pixels[:, frame_to_local[start_idx + 1]],
                    ],
                    dim=1,
                )
            )
            pair_weights.append((1.0 - 0.5 * (sigmas[:, start_idx] + sigmas[:, start_idx + 1])).clamp(min=0.05))

        pred_pairs = torch.cat(pred_pairs, dim=0)
        gt_pairs = torch.cat(gt_pairs, dim=0)
        pair_weights = torch.stack(pair_weights, dim=1).reshape(-1).float().detach()

        pred_flow = self._compute_optical_flow(pred_pairs, enable_grad=grad_on)
        gt_flow = self._compute_optical_flow(gt_pairs, enable_grad=False)

        pair_l1 = (pred_flow - gt_flow.detach()).abs().mean(dim=(1, 2, 3))
        pair_epe = torch.norm(pred_flow - gt_flow.detach(), dim=1).mean(dim=(1, 2))
        flow_loss = (pair_l1 * pair_weights).sum() / (pair_weights.sum() + 1e-6)
        flow_epe = (pair_epe * pair_weights).sum() / (pair_weights.sum() + 1e-6)
        return flow_loss, flow_epe.detach()

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        self.transformer.set_requires_gradient_sync(should_sync)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss, latent_pred = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        flow_loss_val = torch.tensor(0.0, device=self.device)
        flow_epe_val = torch.tensor(0.0, device=self.device)
        if self.flow_loss_weight > 0 and self.step >= self.flow_loss_start_step:
            flow_input = latent_pred if self.flow_grad_enabled else latent_pred.detach()
            flow_loss_val, flow_epe_val = self.compute_flow_loss(flow_input, input_dict)
            loss = loss + self.flow_loss_weight * flow_loss_val / self.gradient_accumulation_steps

        loss.backward()

        losses = {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "flow_loss": flow_loss_val.detach(),
            "flow_epe": flow_epe_val,
        }

        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            losses["total_norm"] = total_norm
            losses["should_log"] = True
        else:
            losses["should_log"] = False

        return losses

    def train(self):
        logger.info(f"Starting flow-augmented training for {self.config.num_steps} steps...")
        logger.info(
            f"Flow loss weight: {self.flow_loss_weight}, start step: {self.flow_loss_start_step}, "
            f"pairs: {self.flow_num_pairs}, supervision_cam: {self.flow_supervision_cam}, "
            f"model: {self.flow_model_name}/{self.flow_model_ckpt}"
        )
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training (Flow)",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_flow_losses = []
        accumulated_flow_epes = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)

            accumulated_latent_losses.append(losses["latent_loss"])
            accumulated_action_losses.append(losses["action_loss"])
            accumulated_flow_losses.append(losses["flow_loss"])
            accumulated_flow_epes.append(losses["flow_epe"])
            step_in_accumulation += 1

            if losses["should_log"]:
                lr = self.lr_scheduler.get_last_lr()[0]
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                flow_loss_show = dist_mean(torch.stack(accumulated_flow_losses).sum()).detach().cpu().item()
                flow_epe_show = dist_mean(torch.stack(accumulated_flow_epes).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_flow_losses = []
                accumulated_flow_epes = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses["total_norm"]
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix(
                        {
                            "lat": f"{latent_loss_show:.4f}",
                            "act": f"{action_loss_show:.4f}",
                            "flow": f"{flow_loss_show:.4f}",
                            "epe": f"{flow_epe_show:.4f}",
                            "step": self.step,
                            "gn": f"{total_norm.item():.2f}",
                            "lr": f"{lr:.2e}",
                        }
                    )
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar("loss/video", latent_loss_show, self.step)
                        self.tb_writer.add_scalar("loss/action", action_loss_show, self.step)
                        self.tb_writer.add_scalar("loss/flow", flow_loss_show, self.step)
                        self.tb_writer.add_scalar("metric/flow_epe", flow_epe_show, self.step)
                        self.tb_writer.add_scalar("loss/video_max", max_latent_loss_show, self.step)
                        self.tb_writer.add_scalar("loss/action_max", max_action_loss_show, self.step)
                        self.tb_writer.add_scalar("train/grad_norm", total_norm.item(), self.step)
                        self.tb_writer.add_scalar("train/lr", lr, self.step)

                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "loss/video": latent_loss_show,
                                "loss/action": action_loss_show,
                                "loss/flow": flow_loss_show,
                                "metric/flow_epe": flow_epe_show,
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


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    debug_seed = getattr(config, "debug_seed", None)
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

    trainer = FlowTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train WAN model with optical flow alignment loss")
    parser.add_argument(
        "--config-name",
        type=str,
        default="robotwin_train_flow",
        help="Config name",
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
