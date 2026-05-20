# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Inference server for the GF + Internal training variant.

This file does the minimum needed to put the GF+Internal model on
serve duty — no protocol, no batching changes, just:

* swap ``load_transformer`` for the GF+Internal loader so the
  ``internal_blocks / internal_*_proj_out`` weights are restored;
* offer two new sampling modes that use the internal head:
    - ``'short_path'`` — every step except the first / every Kth / the
      last one calls ``forward_internal`` (qb_xx style step-skipping
      that trades a tiny FID hit for ~25 % wall-time per step);
    - ``'ig_extrapolate'`` — within an optional ``[t_lo, t_hi]`` interval
      compute both ``D_f`` and ``D_i`` and apply the IG extrapolation
      ``D_w = D_i + w (D_f - D_i)`` (faithful to Zhou et al., 2025).

Everything else (text / VAE encode / KV-cache lifecycle / CFG / action
post-processing) is inherited verbatim from ``VA_Server`` so behaviour
is identical when ``internal_infer_mode == 'main_only'``.

Usage::

    # Multi-GPU launcher mirrors evaluation/robotwin/launch_server_*.sh
    # (see script/run_server_gf_internal.sh below).
    python wan_va/wan_va_server_gf_internal.py \
        --port 29556 \
        --config-name robotwin_train_vggt_geometry_forcing_internal \
        --save_root /path/to/save
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS  # noqa: E402
from distributed.fsdp import shard_model  # noqa: E402
from distributed.util import _configure_model, init_distributed  # noqa: E402
from utils import init_logger, logger, run_async_server_mode  # noqa: E402

# Reuse all the heavy lifting from the standard server.
from wan_va_server import VA_Server  # noqa: E402

# GF+Internal loader (knows how to materialise internal_* heads).
from modules.utils_geometry_forcing_internal import (  # noqa: E402
    load_transformer as load_transformer_gf_internal,
)


# ----------------------------------------------------------------------
# Wrapper that decides at each forward whether to run the main path,
# the internal short path, or both for IG extrapolation.
# ----------------------------------------------------------------------
class _GFInternalTransformerWrapper(torch.nn.Module):
    """Drop-in replacement for ``self.transformer`` in ``VA_Server._infer``.

    The base ``VA_Server`` calls ``self.transformer(input_dict, update_cache,
    cache_name, action_mode)`` — exactly the same signature as
    ``WanTransformer3DModel.forward``. We expose that interface but let
    the GF+Internal model dispatch to ``forward`` (D_f) / ``forward_internal``
    (D_i) per the chosen mode and per-step counters.
    """

    def __init__(self, model, cfg):
        super().__init__()
        # The actual GF+Internal transformer.
        self.model = model

        self.mode = str(getattr(cfg, 'internal_infer_mode', 'main_only')).lower()
        # Step-skipping schedule (only used when mode == 'short_path').
        self.video_gap = int(getattr(cfg, 'video_internal_gap', 5))
        self.action_gap = int(getattr(cfg, 'action_internal_gap', 10))

        # IG extrapolation hyperparams (only used when mode == 'ig_extrapolate').
        self.ig_scale = float(getattr(cfg, 'internal_guidance_scale', 1.5))
        # Guidance interval expressed as a [lo, hi) fraction of the
        # denoising trajectory; matches the paper's convention. Default
        # disables the interval (always on).
        self.ig_interval = tuple(getattr(cfg, 'internal_guidance_interval', (0.0, 1.0)))

        # Per-stream step counters; the server is expected to call
        # ``begin_chunk()`` whenever a new chunk starts so we know to
        # reset the counters / reschedule.
        self._step_video = 0
        self._step_action = 0
        self._n_video = int(getattr(cfg, 'num_inference_steps', 25))
        self._n_action = int(getattr(cfg, 'action_num_inference_steps', 50))

        # Forwarded straightforward attributes the rest of VA_Server
        # touches on ``self.transformer``.
        for attr in (
            'clear_cache', 'clear_pred_cache', 'create_empty_cache',
            'patch_size', 'config',
        ):
            if hasattr(model, attr):
                setattr(self, attr, getattr(model, attr))

    def begin_chunk(self):
        """Call this at the start of every denoising chunk."""
        self._step_video = 0
        self._step_action = 0

    # ------------------------------------------------------------------
    def _is_short_path_step(self, action_mode: bool) -> bool:
        if self.mode != 'short_path':
            return False
        i = self._step_action if action_mode else self._step_video
        n = self._n_action if action_mode else self._n_video
        gap = self.action_gap if action_mode else self.video_gap
        last_step = (i == n - 1)
        return not (i == 0 or (i + 1) % max(1, gap) == 0 or last_step)

    def _is_ig_step(self, action_mode: bool) -> bool:
        if self.mode != 'ig_extrapolate':
            return False
        i = self._step_action if action_mode else self._step_video
        n = max(1, self._n_action if action_mode else self._n_video)
        frac = i / n
        lo, hi = self.ig_interval
        return lo <= frac < hi

    # ------------------------------------------------------------------
    def forward(self, input_dict, update_cache=0, cache_name='pos', action_mode=False):
        if self._is_short_path_step(action_mode):
            out = self.model.forward_internal_lora(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            ) if hasattr(self.model, 'forward_internal_lora') else self.model.forward_internal(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
        elif self._is_ig_step(action_mode):
            # Two passes: D_f writes / consults KV-cache as usual; the
            # internal pass uses ``update_cache=0`` so it doesn't touch
            # the cache pool (avoid double-counting noisy slots).
            d_f = self.model(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
            d_i = self._call_internal(
                input_dict,
                update_cache=0,
                cache_name=cache_name,
                action_mode=action_mode,
            )
            # IG extrapolation:  D_w = D_i + w (D_f - D_i)
            # Cast both to a common dtype to avoid bf16/float32 mismatch.
            common = d_f.dtype
            out = d_i.to(common) + self.ig_scale * (d_f - d_i.to(common))
        else:
            out = self.model(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )

        # Bump per-stream counter AFTER the forward so guidance interval
        # counts the just-finished step.
        if action_mode:
            self._step_action += 1
        else:
            self._step_video += 1
        return out

    def _call_internal(self, input_dict, update_cache, cache_name, action_mode):
        """Dispatch to the internal forward (different model variants name
        it slightly differently)."""
        if hasattr(self.model, 'forward_internal'):
            return self.model.forward_internal(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
        if hasattr(self.model, 'forward_internal_lora'):
            return self.model.forward_internal_lora(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
            )
        raise AttributeError(
            "GF+Internal transformer has no `forward_internal` or "
            "`forward_internal_lora`; cannot run IG / short-path inference."
        )

    # ------------------------------------------------------------------
    # Unwrapped __getattr__: anything we did not pre-expose still tunnels
    # through to the inner model (keeps `self.transformer.config`,
    # `self.transformer.parameters()`, etc. all working).
    def __getattr__(self, name):  # pragma: no cover — straightforward delegation
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__['_modules']['model'], name)


# ----------------------------------------------------------------------
# The actual server class.
# ----------------------------------------------------------------------
class VA_Server_GF_Internal(VA_Server):
    """``VA_Server`` with the transformer swapped for the GF+Internal model.

    The constructor sequence is identical to the parent except for the
    ``self.transformer = load_transformer(...)`` block, so we override
    only what is necessary by re-running the relevant slice of code with
    a different loader.
    """

    def __init__(self, job_config):
        # We do NOT call super().__init__(job_config) because that would
        # build the transformer with the wrong loader; instead we replicate
        # the parent's setup verbatim while replacing one block.
        from utils import FlowMatchScheduler  # delayed to avoid cycle
        from modules.utils import (
            WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae,
        )

        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)

        self.scheduler = FlowMatchScheduler(
            shift=job_config.snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=job_config.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path, 'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path, 'tokenizer'),
        )
        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path, 'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        # ---- The only material change vs. base VA_Server ------------
        transformer_path = os.path.join(
            job_config.wan22_pretrained_model_name_or_path, 'transformer'
        )
        if getattr(job_config, 'ckpt_path', None):
            transformer_path = os.path.join(job_config.ckpt_path, 'transformer')
            logger.info(
                f"[GF+Internal] Loading transformer from checkpoint: {transformer_path}"
            )
        base_transformer = load_transformer_gf_internal(
            transformer_path,
            torch_dtype=self.dtype,
            torch_device=self.device,
            enable_internal=bool(getattr(job_config, 'enable_internal', True)),
            internal_depth=int(getattr(job_config, 'internal_depth', 24)),
            num_internal_blocks=int(getattr(job_config, 'num_internal_blocks', 2)),
        )
        # FSDP-shard the GF+Internal transformer (same as base server).
        base_transformer = _configure_model(
            model=base_transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        # Wrap so VA_Server._infer’s ``self.transformer(...)`` calls go
        # through our dispatcher (main / short_path / IG).
        self.transformer = _GFInternalTransformerWrapper(base_transformer, job_config)
        # -------------------------------------------------------------

        # Same env-specific helpers as base server.
        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path, 'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    # ------------------------------------------------------------------
    # Reset per-chunk counters in addition to the parent's bookkeeping.
    # ------------------------------------------------------------------
    def _infer(self, obs, frame_st_id=0):
        if hasattr(self.transformer, 'begin_chunk'):
            self.transformer.begin_chunk()
        return super()._infer(obs, frame_st_id=frame_st_id)


# ----------------------------------------------------------------------
# CLI entry point — mirrors wan_va_server.py.
# ----------------------------------------------------------------------
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

    model = VA_Server_GF_Internal(config)

    if config.infer_mode == 'i2va':
        logger.info("****************************** USE I2AV mode ******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info("****************************** USE Server mode (GF+Internal) ******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def main():
    parser = argparse.ArgumentParser(
        description="GF+Internal inference server (drop-in replacement for wan_va_server.py)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train_vggt_geometry_forcing_internal',
        help="Key in VA_CONFIGS. Defaults to the GF+Internal training cfg "
             "(use the matching cfg so internal_depth / num_internal_blocks "
             "are correctly read at load time).",
    )
    parser.add_argument("--port", type=int, default=None, help='(start) port')
    parser.add_argument("--save_root", type=str, default=None, help='save root')
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
