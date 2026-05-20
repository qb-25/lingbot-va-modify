# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Geometry Forcing + Internal Guidance training config (RoboTwin)."""
import os

from easydict import EasyDict

from .va_robotwin_train_vggt_geometry_forcing_cfg import (
    va_robotwin_train_vggt_geometry_forcing_cfg,
)


va_robotwin_train_vggt_geometry_forcing_internal_cfg = EasyDict(
    __name__='Config: VA robotwin train with Geometry Forcing + Internal head'
)
va_robotwin_train_vggt_geometry_forcing_internal_cfg.update(
    va_robotwin_train_vggt_geometry_forcing_cfg
)

# ---- IO --------------------------------------------------------------
va_robotwin_train_vggt_geometry_forcing_internal_cfg.save_root = (
    './train_out/vggt_geometry_forcing_internal'
)
va_robotwin_train_vggt_geometry_forcing_internal_cfg.wandb_run_name = (
    'train_vggt_geometry_forcing_internal'
)

# ---- Internal Guidance (Q1=A) ---------------------------------------
va_robotwin_train_vggt_geometry_forcing_internal_cfg.enable_internal = True
# Wan DiT has 30 blocks; fork after the 24th (index 24).
va_robotwin_train_vggt_geometry_forcing_internal_cfg.internal_depth = 24
# Internal branch carries this many extra blocks (deepcopy of main tail).
va_robotwin_train_vggt_geometry_forcing_internal_cfg.num_internal_blocks = 2
# (Q2=a) deep-supervision weight for the internal head.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.lambda_internal = 1.0
# Train only after this step (warmup).
va_robotwin_train_vggt_geometry_forcing_internal_cfg.internal_start_step = 0

# ---- Attention visualization (Q3=V1, Q4=c, Q5=c) ---------------------
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_enabled = True
# How often to dump attention overlays during training.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_interval = 5000
# Wan DiT layers to visualize.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_layers = [0, 10, 20, 29]
# Frames to sample from the (decoded) video for the overlay grid.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_frames = [0, 2, 5]
# Task-target token selection strategy for prompt tokens
# ('content_only' just keeps top-K most-attended-to text tokens, robust
# fallback when explicit indices are not available).
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_token_meta = {
    'mode': 'content_only',
    'top_k': 16,
}
# Heatmap blend.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.attn_vis_alpha = 0.5

# ---- Inference-time internal guidance --------------------------------
# How the GF+Internal inference server uses the internal branch.
#   'main_only'      : ignore internal head, behave like vanilla VA_Server
#   'short_path'     : per-step skipping, only run blocks[:internal_depth]
#                      + internal_blocks on most steps (qb_xx style accel).
#   'ig_extrapolate' : Internal Guidance from Zhou et al., 2025, i.e.
#                      D_w = D_i + w * (D_f - D_i).
va_robotwin_train_vggt_geometry_forcing_internal_cfg.internal_infer_mode = 'main_only'

# Step skipping cadence used when ``internal_infer_mode == 'short_path'``.
# At i==0 / (i+1)%gap==0 / last_step we always run the main path.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.video_internal_gap = 5
va_robotwin_train_vggt_geometry_forcing_internal_cfg.action_internal_gap = 10

# IG extrapolation hyperparams used when ``internal_infer_mode == 'ig_extrapolate'``.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.internal_guidance_scale = 1.5
# Apply IG only when (i / num_steps) is in [lo, hi). (0.0, 1.0) = always on.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.internal_guidance_interval = (0.0, 1.0)

# Resume entry — same env-var convention as the GF cfg.
va_robotwin_train_vggt_geometry_forcing_internal_cfg.resume_from = None
_resume_env = os.getenv("LINGBOT_RESUME_FROM", "").strip()
if _resume_env:
    va_robotwin_train_vggt_geometry_forcing_internal_cfg.resume_from = _resume_env
