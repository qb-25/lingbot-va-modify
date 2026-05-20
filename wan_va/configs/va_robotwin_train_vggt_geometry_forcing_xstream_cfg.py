# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Geometry Forcing + Cross-Stream Alignment training config (RoboTwin).

Inherits the GF config and adds the cross-stream contrastive objective.
By default the cross-stream layer is **block 25** (NOT one of the GF
layers ``[10, 20, 29]``) so the two objectives do not fight for the same
representation.
"""
import os

from easydict import EasyDict

from .va_robotwin_train_vggt_geometry_forcing_cfg import (
    va_robotwin_train_vggt_geometry_forcing_cfg,
)


va_robotwin_train_vggt_geometry_forcing_xstream_cfg = EasyDict(
    __name__='Config: VA robotwin train with GF + Cross-Stream Alignment'
)
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.update(
    va_robotwin_train_vggt_geometry_forcing_cfg
)

# ---- IO --------------------------------------------------------------
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.save_root = (
    './train_out/vggt_geometry_forcing_xstream'
)
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.wandb_run_name = (
    'train_vggt_geometry_forcing_xstream'
)

# ---- Cross-Stream Alignment ------------------------------------------
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_enable = True

# DiT block(s) at which to capture (video, action) hidden for InfoNCE.
# **Stay away from gf_student_layers=[10,20,29]** so the two objectives
# do not compete for the same subspace. block 25 is a good default
# (mid-late, after most representation is formed but before the final
# specialization to the velocity head).
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_layers = [25]

# Projector geometry.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_proj_dim = 256
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_proj_hidden = 1024

# Positive convention: 'frame' / 'window' / 'trajectory'.
# On RoboTwin (slow visual change between frames), 'window' is a robust
# default; switch to 'trajectory' if `cross_stream_pos_neg_gap` stalls.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_pos_mode = 'window'
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_window = 2

# InfoNCE temperature. Default 0.15 (NOT CLIP's 0.07) because batch is
# small even after all-gather; a larger tau keeps the softmax less peaky
# and the gradient less degenerate.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_tau = 0.15

# Loss weight in the total objective.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.lambda_xstream = 0.1

# Compute interval (in steps). Set > 1 to skip cross-stream loss on most
# steps if it costs too much wall-time; 1 = every step.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_interval = 1

# Warmup: only enable cross-stream loss after this step.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_start_step = 0

# Whether to gather z across DDP ranks before InfoNCE.
# **MUST be True** on Wan-VA defaults (batch_size=1, world_size=8) for
# the contrastive objective to have non-trivial negatives.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_use_all_gather = True

# Sigma-aware gating: scale the loss by mean((1-sigma_v)*(1-sigma_a)).
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_sigma_soft = True
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.xstream_sigma_threshold = 0.5

# ---- Attention visualization ----------------------------------------
# Per the project requirement, dump the cross-attention heatmaps every
# 2000 steps (instead of the GF+Internal default 5000).
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_enabled = True
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_interval = 2000
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_layers = [0, 10, 20, 29]
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_frames = [0, 2, 5]
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_token_meta = {
    'mode': 'content_only',
    'top_k': 16,
}
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.attn_vis_alpha = 0.5

# Resume entry — same env-var convention as the GF cfg.
va_robotwin_train_vggt_geometry_forcing_xstream_cfg.resume_from = None
_resume_env = os.getenv("LINGBOT_RESUME_FROM", "").strip()
if _resume_env:
    va_robotwin_train_vggt_geometry_forcing_xstream_cfg.resume_from = _resume_env
