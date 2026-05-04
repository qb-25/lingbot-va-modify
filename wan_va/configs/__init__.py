# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_train_latent_dynamics_cfg import va_robotwin_train_latent_dynamics_cfg
from .va_robotwin_train_flow_cfg import va_robotwin_train_flow_cfg
from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg
from .va_robotwin_train_vggt_debug_cfg import va_robotwin_train_vggt_debug_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'robotwin_train_latent_dynamics': va_robotwin_train_latent_dynamics_cfg,
    'robotwin_train_flow': va_robotwin_train_flow_cfg,
    'robotwin_train_vggt': va_robotwin_train_vggt_cfg,
    'robotwin_train_vggt_debug': va_robotwin_train_vggt_debug_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
}