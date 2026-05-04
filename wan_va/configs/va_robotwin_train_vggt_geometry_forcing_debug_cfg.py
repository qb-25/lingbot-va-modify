# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Small-data debug config for Geometry Forcing training."""
from easydict import EasyDict

from .va_robotwin_train_vggt_geometry_forcing_cfg import (
    va_robotwin_train_vggt_geometry_forcing_cfg,
)


va_robotwin_train_vggt_geometry_forcing_debug_cfg = EasyDict(
    __name__='Config: VA robotwin Geometry Forcing debug'
)
va_robotwin_train_vggt_geometry_forcing_debug_cfg.update(
    va_robotwin_train_vggt_geometry_forcing_cfg
)

va_robotwin_train_vggt_geometry_forcing_debug_cfg.save_root = (
    './train_out/vggt_geometry_forcing_debug'
)
va_robotwin_train_vggt_geometry_forcing_debug_cfg.max_train_samples = 8
va_robotwin_train_vggt_geometry_forcing_debug_cfg.load_worker = 0
va_robotwin_train_vggt_geometry_forcing_debug_cfg.disable_train_shuffle = True
va_robotwin_train_vggt_geometry_forcing_debug_cfg.debug_seed = 42
va_robotwin_train_vggt_geometry_forcing_debug_cfg.cfg_prob = 0.0
va_robotwin_train_vggt_geometry_forcing_debug_cfg.latent_noisy_cond_prob = 0.0
va_robotwin_train_vggt_geometry_forcing_debug_cfg.fixed_chunk_size = 1
va_robotwin_train_vggt_geometry_forcing_debug_cfg.fixed_window_size = 16
va_robotwin_train_vggt_geometry_forcing_debug_cfg.fixed_latent_timestep_id = 100
va_robotwin_train_vggt_geometry_forcing_debug_cfg.fixed_action_timestep_id = 100

va_robotwin_train_vggt_geometry_forcing_debug_cfg.num_steps = 300
va_robotwin_train_vggt_geometry_forcing_debug_cfg.save_interval = 100
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gc_interval = 20

# Smaller weights / single-layer for fast sanity check
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gf_student_layers = [29]
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gf_teacher_layers = [-1]
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gf_lambda_angular = 0.5
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gf_lambda_scale = 0.05
va_robotwin_train_vggt_geometry_forcing_debug_cfg.gf_start_step = 0
