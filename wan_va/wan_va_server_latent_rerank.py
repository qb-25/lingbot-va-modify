import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from modules.latent_dynamics import LatentDynamicsBranch
from utils import init_logger, logger, run_async_server_mode, save_async
from wan_va_server import VA_Server


class VA_Server_LatentRerank(VA_Server):
    """Copied server variant with lightweight latent-dynamics action reranking."""

    def __init__(self, job_config):
        super().__init__(job_config)
        self.rerank_num_candidates = max(
            1, int(getattr(job_config, "action_rerank_num_candidates", 4))
        )
        self.rerank_topk = max(
            1,
            min(
                int(getattr(job_config, "action_rerank_topk", 2)),
                self.rerank_num_candidates,
            ),
        )
        self.latent_dynamics_ckpt_path = getattr(
            job_config, "latent_dynamics_ckpt_path", None
        ) or getattr(job_config, "ckpt_path", None)
        self.latent_dynamics_head = self._load_latent_dynamics_head(
            self.latent_dynamics_ckpt_path
        )
        self.latent_dynamics_head.eval().requires_grad_(False)
        logger.info(
            "Latent rerank enabled: "
            f"candidates={self.rerank_num_candidates}, topk={self.rerank_topk}, "
            f"branch_ckpt={self.latent_dynamics_ckpt_path}"
        )

    def _load_latent_dynamics_head(self, checkpoint_root):
        if checkpoint_root is None:
            raise ValueError(
                "latent_dynamics_ckpt_path is required for rerank server."
            )

        branch_dir = Path(checkpoint_root)
        if branch_dir.name != "latent_dynamics_head":
            branch_dir = branch_dir / "latent_dynamics_head"

        model_file = branch_dir / "model.safetensors"
        config_file = branch_dir / "config.json"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Latent dynamics checkpoint not found: {model_file}"
            )

        branch_cfg = {}
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                branch_cfg = json.load(f)

        branch = LatentDynamicsBranch(
            latent_channels=int(
                branch_cfg.get("latent_channels", self.transformer.config.in_channels)
            ),
            action_dim=int(branch_cfg.get("action_dim", self.job_config.action_dim)),
            action_per_frame=int(
                branch_cfg.get("action_per_frame", self.job_config.action_per_frame)
            ),
            state_dim=int(branch_cfg.get("state_dim", 256)),
            hidden_dim=int(branch_cfg.get("hidden_dim", 512)),
            num_projections=int(branch_cfg.get("num_projections", 256)),
        ).to(self.device)
        branch.load_state_dict(load_file(model_file), strict=True)
        return branch

    def _iter_cache_modules(self):
        for block in self.transformer.blocks:
            attn = getattr(block, "attn1", None)
            if attn is not None and getattr(attn, "attn_caches", None) is not None:
                yield attn

    def _clone_cache(self, source_cache_name, target_cache_name):
        for attn in self._iter_cache_modules():
            source_cache = attn.attn_caches.get(source_cache_name)
            if source_cache is None:
                attn.attn_caches[target_cache_name] = None
                continue
            attn.attn_caches[target_cache_name] = {
                key: value.clone() for key, value in source_cache.items()
            }

    def _drop_cache(self, cache_name):
        for attn in self._iter_cache_modules():
            attn.attn_caches.pop(cache_name, None)

    def _prepare_inference_schedules(self):
        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        action_timesteps = F.pad(
            self.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        if video_step != -1:
            timesteps = timesteps[:video_step]
        return timesteps, action_timesteps

    def _infer_video_latents(self, obs, frame_st_id):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
        else:
            init_latent = None

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        timesteps, _ = self._prepare_inference_schedules()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = (
                    init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                )
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )

                if not last_step or self.job_config.video_exec_step != -1:
                    video_noise_pred = self._reshape_video_noise_pred(
                        video_noise_pred, frame_chunk_size
                    )
                    latents = self.scheduler.step(
                        video_noise_pred,
                        t,
                        latents,
                        return_dict=False,
                    )

                latents[:, :, 0:1] = (
                    latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                )
        return latents

    def _reshape_video_noise_pred(self, video_noise_pred, frame_chunk_size):
        from utils import data_seq_to_patch

        video_noise_pred = data_seq_to_patch(
            self.job_config.patch_size,
            video_noise_pred,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            batch_size=2 if self.use_cfg else 1,
        )
        if self.job_config.guidance_scale > 1:
            video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (
                video_noise_pred[:1] - video_noise_pred[1:]
            )
        else:
            video_noise_pred = video_noise_pred[:1]
        return video_noise_pred

    def _sample_action_candidate(self, frame_st_id, cache_name, init_actions=None):
        frame_chunk_size = self.job_config.frame_chunk_size
        _, action_timesteps = self._prepare_inference_schedules()
        actions = init_actions
        if actions is None:
            actions = torch.randn(
                1,
                self.job_config.action_dim,
                frame_chunk_size,
                self.action_per_frame,
                1,
                device=self.device,
                dtype=self.dtype,
            )

        with torch.no_grad():
            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=cache_name,
                    action_mode=True,
                )

                if not last_step:
                    action_noise_pred = self._reshape_action_noise_pred(
                        action_noise_pred, frame_chunk_size
                    )
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        t,
                        actions,
                        return_dict=False,
                    )

                actions[:, :, 0:1] = (
                    action_cond if frame_st_id == 0 else actions[:, :, 0:1]
                )
        return actions

    def _reshape_action_noise_pred(self, action_noise_pred, frame_chunk_size):
        from einops import rearrange

        action_noise_pred = rearrange(
            action_noise_pred,
            "b (f n) c -> b c f n 1",
            f=frame_chunk_size,
        )
        if self.job_config.action_guidance_scale > 1:
            action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (
                action_noise_pred[:1] - action_noise_pred[1:]
            )
        else:
            action_noise_pred = action_noise_pred[:1]
        return action_noise_pred

    def _score_action_candidate(self, pred_latents, candidate_actions):
        branch = self.latent_dynamics_head
        video_states = branch.encode_states(pred_latents)
        if video_states.shape[1] < 2:
            return {"score": 0.0, "rollout_mse": 0.0, "action_smooth": 0.0}

        action_states = branch.encode_actions(candidate_actions)
        rollout_states = branch.rollout(video_states[:, 0], action_states[:, :-1])
        target_states = video_states[:, 1:]

        rollout_mse = F.mse_loss(
            rollout_states.float(),
            target_states.float(),
            reduction="mean",
        )
        action_smooth = (
            candidate_actions[:, :, 1:] - candidate_actions[:, :, :-1]
        ).float().pow(2).mean()
        score = rollout_mse + 0.01 * action_smooth
        return {
            "score": float(score.detach().cpu().item()),
            "rollout_mse": float(rollout_mse.detach().cpu().item()),
            "action_smooth": float(action_smooth.detach().cpu().item()),
        }

    def _rerank_actions(self, pred_latents, frame_st_id):
        candidates = []
        for candidate_id in range(self.rerank_num_candidates):
            candidate_cache_name = f"{self.cache_name}_cand_{frame_st_id}_{candidate_id}"
            self._clone_cache(self.cache_name, candidate_cache_name)
            candidate_actions = self._sample_action_candidate(
                frame_st_id,
                candidate_cache_name,
            )
            score_dict = self._score_action_candidate(pred_latents, candidate_actions)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "cache_name": candidate_cache_name,
                    "actions": candidate_actions.detach().clone(),
                    **score_dict,
                }
            )

        ranked = sorted(candidates, key=lambda item: item["score"])
        best = ranked[0]
        self._clone_cache(best["cache_name"], self.cache_name)

        topk_log = []
        for item in ranked[: self.rerank_topk]:
            topk_log.append(
                {
                    "candidate_id": item["candidate_id"],
                    "score": item["score"],
                    "rollout_mse": item["rollout_mse"],
                    "action_smooth": item["action_smooth"],
                }
            )
        rerank_file = os.path.join(
            self.exp_save_root, f"action_rerank_{frame_st_id}.json"
        )
        with open(rerank_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_candidate_id": best["candidate_id"],
                    "topk": topk_log,
                },
                f,
                indent=2,
            )

        for item in candidates:
            self._drop_cache(item["cache_name"])

        logger.info(
            "Latent rerank scores: "
            + ", ".join(
                [
                    f"id={item['candidate_id']} score={item['score']:.4f}"
                    for item in ranked[: self.rerank_topk]
                ]
            )
        )
        return best["actions"]

    def _infer(self, obs, frame_st_id=0):
        torch.cuda.synchronize()
        infer_start = time.perf_counter()
        pred_latents = self._infer_video_latents(obs, frame_st_id)
        torch.cuda.synchronize()
        video_end = time.perf_counter()

        best_actions = self._rerank_actions(pred_latents, frame_st_id)
        torch.cuda.synchronize()
        action_end = time.perf_counter()

        best_actions[:, ~self.action_mask] *= 0
        save_async(
            pred_latents, os.path.join(self.exp_save_root, f"latents_{frame_st_id}.pt")
        )
        save_async(
            best_actions,
            os.path.join(self.exp_save_root, f"actions_{frame_st_id}.pt"),
        )

        logger.info(
            f"[LatentRerank Timing] chunk={frame_st_id} | "
            f"total={action_end - infer_start:.3f}s | "
            f"video={video_end - infer_start:.3f}s | "
            f"rerank_action={action_end - video_end:.3f}s | "
            f"num_candidates={self.rerank_num_candidates}"
        )

        actions = self.postprocess_action(best_actions)
        torch.cuda.empty_cache()
        return actions, pred_latents


def run(args):
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.ckpt_path is not None:
        config.ckpt_path = args.ckpt_path
    if args.latent_dynamics_ckpt_path is not None:
        config.latent_dynamics_ckpt_path = args.latent_dynamics_ckpt_path
    config.action_rerank_num_candidates = args.action_rerank_num_candidates
    config.action_rerank_topk = args.action_rerank_topk

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    model = VA_Server_LatentRerank(config)
    if config.infer_mode == "i2va":
        logger.info("******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == "server":
        logger.info("***************************USE Latent Rerank Server***************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--latent-dynamics-ckpt-path", type=str, default=None)
    parser.add_argument("--action-rerank-num-candidates", type=int, default=4)
    parser.add_argument("--action-rerank-topk", type=int, default=2)
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
