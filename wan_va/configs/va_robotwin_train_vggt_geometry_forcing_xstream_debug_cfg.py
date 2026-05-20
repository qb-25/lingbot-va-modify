# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Single-GPU debug config for GF + Cross-Stream training."""
from easydict import EasyDict

from .va_robotwin_train_vggt_geometry_forcing_xstream_cfg import (
    va_robotwin_train_vggt_geometry_forcing_xstream_cfg,
)


va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg = EasyDict(
    __name__='Config: VA robotwin GF + XStream debug'
)
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.update(
    va_robotwin_train_vggt_geometry_forcing_xstream_cfg
)

va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.save_root = (
    './train_out/vggt_geometry_forcing_xstream_debug'
)
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.max_train_samples = 8
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.load_worker = 0
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.disable_train_shuffle = True
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.debug_seed = 42
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.cfg_prob = 0.0
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.latent_noisy_cond_prob = 0.0
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.fixed_chunk_size = 1
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.fixed_window_size = 16
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.fixed_latent_timestep_id = 100
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.fixed_action_timestep_id = 100

va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.num_steps = 300
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.save_interval = 100
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.gc_interval = 20

# Smaller GF alignment for fast sanity check.
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.gf_student_layers = [29]
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.gf_teacher_layers = [-1]

# Cross-stream and attn vis fire often during debug.
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.xstream_layers = [25]
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.xstream_interval = 1
va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.attn_vis_interval = 100
