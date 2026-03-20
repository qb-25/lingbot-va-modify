# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Joint Parallel Denoising server: video and action are denoised
# simultaneously in a single forward pass, closing the train-inference gap.

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan_va_server import VA_Server
from configs import VA_CONFIGS
from distributed.util import _configure_model, init_distributed
from modules.model_joint import patch_model_with_joint_forward
from utils import (
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server_Joint(VA_Server):
    """
    Joint Parallel Denoising variant of VA_Server.

    Key difference from the base server:
      Original: video is fully denoised (25 steps), then action is fully
                denoised (50 steps) conditioned on the *clean* video KV.
      Joint:    video and action are denoised together in a single loop
                (50 steps), so action sees the *noisy* video at matching
                noise level — exactly as during training.

    Benefits:
      1. Eliminates train-inference gap (noise→noise interaction preserved)
      2. Fewer total forward passes  (50 joint vs 25+50=75 serial)
      3. Better video-action coherence (real-time mutual conditioning)
    """

    def __init__(self, job_config):
        super().__init__(job_config)
        patch_model_with_joint_forward(self.transformer)
        logger.info("Patched transformer with forward_joint for joint denoising")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _prepare_joint_inputs_for_cfg(
        self, latents, actions, t_video, t_action,
        latent_cond, action_cond, frame_st_id,
    ):
        """Build video/action input dicts and apply CFG batch duplication."""
        ps = self.job_config.patch_size

        video_input = {
            'noisy_latents': latents.clone(),
            'timesteps': (
                torch.ones([latents.shape[2]], dtype=torch.float32,
                            device=self.device) * t_video),
            'grid_id': get_mesh_id(
                latents.shape[-3] // ps[0],
                latents.shape[-2] // ps[1],
                latents.shape[-1] // ps[2],
                0, 1, frame_st_id).to(self.device),
            'text_emb': self.prompt_embeds.to(self.dtype).clone(),
        }
        if latent_cond is not None:
            video_input['noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
            video_input['timesteps'][0:1] *= 0

        action_input = {
            'noisy_latents': actions.clone(),
            'timesteps': (
                torch.ones([actions.shape[2]], dtype=torch.float32,
                            device=self.device) * t_action),
            'grid_id': get_mesh_id(
                actions.shape[-3], actions.shape[-2], actions.shape[-1],
                1, 1, frame_st_id, action=True).to(self.device),
            'text_emb': self.prompt_embeds.to(self.dtype).clone(),
        }
        if action_cond is not None:
            action_input['noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
            action_input['timesteps'][0:1] *= 0
        action_input['noisy_latents'][:, ~self.action_mask] *= 0

        if self.use_cfg:
            for d in (video_input, action_input):
                d['noisy_latents'] = d['noisy_latents'].repeat(2, 1, 1, 1, 1)
                d['text_emb'] = torch.cat([
                    self.prompt_embeds.to(self.dtype).clone(),
                    self.negative_prompt_embeds.to(self.dtype).clone(),
                ], dim=0)
                d['grid_id'] = d['grid_id'][None].repeat(2, 1, 1)
                d['timesteps'] = d['timesteps'][None].repeat(2, 1)
        else:
            for d in (video_input, action_input):
                d['grid_id'] = d['grid_id'][None]
                d['timesteps'] = d['timesteps'][None]

        return video_input, action_input

    # ------------------------------------------------------------------
    # core: joint denoising loop
    # ------------------------------------------------------------------
    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size

        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1, 48, frame_chunk_size,
            self.latent_height, self.latent_width,
            device=self.device, dtype=self.dtype)
        actions = torch.randn(
            1, self.job_config.action_dim, frame_chunk_size,
            self.action_per_frame, 1,
            device=self.device, dtype=self.dtype)

        # Unified step count: use the larger of the two original schedules
        joint_steps = getattr(
            self.job_config, 'joint_num_inference_steps',
            self.job_config.action_num_inference_steps)  # default 50

        self.scheduler.set_timesteps(joint_steps)
        self.action_scheduler.set_timesteps(joint_steps)

        video_ts = F.pad(self.scheduler.timesteps, (0, 1),
                         mode='constant', value=0)
        action_ts = F.pad(self.action_scheduler.timesteps, (0, 1),
                          mode='constant', value=0)

        cfg_batch = 2 if self.use_cfg else 1

        with torch.no_grad():
            for i in tqdm(range(len(video_ts)), desc='Joint denoise'):
                last_step = (i == len(video_ts) - 1)
                t_v = video_ts[i]
                t_a = action_ts[i]

                latent_cond = (
                    self.init_latent[:, :, 0:1].to(self.dtype)
                    if frame_st_id == 0 else None)
                action_cond = (
                    torch.zeros(
                        [1, self.job_config.action_dim, 1,
                         self.action_per_frame, 1],
                        device=self.device, dtype=self.dtype)
                    if frame_st_id == 0 else None)

                v_in, a_in = self._prepare_joint_inputs_for_cfg(
                    latents, actions, t_v, t_a,
                    latent_cond, action_cond, frame_st_id)

                video_pred, action_pred = self.transformer(
                    {'__joint_mode__': True,
                     'video_input': v_in,
                     'action_input': a_in},
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name)

                if not last_step:
                    # --- video scheduler step ---
                    video_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=cfg_batch)
                    if self.job_config.guidance_scale > 1:
                        video_pred = (
                            video_pred[1:]
                            + self.job_config.guidance_scale
                            * (video_pred[:1] - video_pred[1:]))
                    else:
                        video_pred = video_pred[:1]
                    latents = self.scheduler.step(
                        video_pred, t_v, latents, return_dict=False)

                    # --- action scheduler step ---
                    action_pred = rearrange(
                        action_pred, 'b (f n) c -> b c f n 1',
                        f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_pred = (
                            action_pred[1:]
                            + self.job_config.action_guidance_scale
                            * (action_pred[:1] - action_pred[1:]))
                    else:
                        action_pred = action_pred[:1]
                    actions = self.action_scheduler.step(
                        action_pred, t_a, actions, return_dict=False)

                # Re-apply conditions on the first frame
                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0

        save_async(latents,
                   os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions,
                   os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------
def run(args):
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    model = VA_Server_Joint(config)

    infer_mode = getattr(config, 'infer_mode', 'server')
    if infer_mode == 'i2va':
        logger.info("USE I2AV mode (joint parallel denoising)")
        model.generate()
    elif infer_mode == 'server':
        logger.info("USE Server mode (joint parallel denoising)")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {infer_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default='robotwin')
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!")


if __name__ == "__main__":
    init_logger()
    main()
