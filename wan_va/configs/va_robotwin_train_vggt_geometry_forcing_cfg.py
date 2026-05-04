# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Geometry Forcing training config for VA / RoboTwin.

Follows the Geometry Forcing paper (Wu et al., 2025):

    L = L_FM + L_action + lambda_angular * L_angular + lambda_scale * L_scale
        (+ optional lambda_depth_point * L_depth_point  in `hybrid` mode)

Key knobs
---------
* `gf_student_layers`:  list[int]  student DiT block indices whose hidden
                        states are aligned with VGGT features.
* `gf_teacher_layers`:  list[int]  VGGT aggregator layer indices (negatives
                        allowed, -1 = last).  Must have the same length as
                        `gf_student_layers`.
* `gf_lambda_angular`:  float      paper default 0.5.
* `gf_lambda_scale`:    float      paper default 0.05.
* `gf_mode`:
    - `'pure'`    : only Angular + Scale (paper-faithful, default).
    - `'hybrid'`  : also enable the VGGT depth/point pixel-level loss
                    (as in train_vggt_spatial_forcing.py).
"""
import os

from easydict import EasyDict

from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg


va_robotwin_train_vggt_geometry_forcing_cfg = EasyDict(
    __name__='Config: VA robotwin train with Geometry Forcing'
)
va_robotwin_train_vggt_geometry_forcing_cfg.update(va_robotwin_train_vggt_cfg)

# ---- IO / logging --------------------------------------------------------
va_robotwin_train_vggt_geometry_forcing_cfg.save_root = './train_out/vggt_geometry_forcing'
va_robotwin_train_vggt_geometry_forcing_cfg.wandb_run_name = 'train_vggt_geometry_forcing'
va_robotwin_train_vggt_geometry_forcing_cfg.save_interval = 10000

# ---- Geometry Forcing settings ------------------------------------------
# 'pure'   : paper-faithful, only Angular + Scale losses.
# 'hybrid' : additionally enable depth/point loss from train_vggt.VGGTTrainer.
va_robotwin_train_vggt_geometry_forcing_cfg.gf_mode = 'pure'

# Multi-layer alignment: each entry pairs one student layer with one teacher
# layer (lengths must match). VGGT-1B aggregator has 24 layers (negatives ok).
va_robotwin_train_vggt_geometry_forcing_cfg.gf_student_layers = [10, 20, 29]
va_robotwin_train_vggt_geometry_forcing_cfg.gf_teacher_layers = [-9, -5, -1]

# Loss weights (paper defaults: 0.5 / 0.05).
va_robotwin_train_vggt_geometry_forcing_cfg.gf_lambda_angular = 0.5
va_robotwin_train_vggt_geometry_forcing_cfg.gf_lambda_scale = 0.05

# Projector / predictor architecture.
va_robotwin_train_vggt_geometry_forcing_cfg.gf_proj_hidden_dim = 2048
va_robotwin_train_vggt_geometry_forcing_cfg.gf_use_bn = True

# Start step for GF loss (warmup). 0 = from the beginning.
va_robotwin_train_vggt_geometry_forcing_cfg.gf_start_step = 0

# Use sigma-based per-frame weighting (clean frames weigh more).
# Paper does uniform averaging; we keep Wan's flow-matching weighting by
# default because Wan trains with per-frame noise levels.
va_robotwin_train_vggt_geometry_forcing_cfg.gf_use_sigma_weight = True

# When gf_mode == 'hybrid', enable the pixel-level VGGT depth/point loss.
# Otherwise it is forcibly disabled regardless of vggt_loss_weight in the
# base config.
# (weight read from the inherited `vggt_loss_weight`, default 0.01).

# ---- Resume / hot-start --------------------------------------------------
# Path to a checkpoint directory containing `transformer/`. Optional.
va_robotwin_train_vggt_geometry_forcing_cfg.resume_from = None
_resume_env = os.getenv("LINGBOT_RESUME_FROM", "").strip()
if _resume_env:
    va_robotwin_train_vggt_geometry_forcing_cfg.resume_from = _resume_env
