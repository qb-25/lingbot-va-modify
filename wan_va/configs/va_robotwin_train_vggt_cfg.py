# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Training config with VGGT geometric alignment loss.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_vggt_cfg = EasyDict(__name__='Config: VA robotwin train with VGGT loss')
va_robotwin_train_vggt_cfg.update(va_robotwin_cfg)

va_robotwin_train_vggt_cfg.save_root = './train_out/vggt'

va_robotwin_train_vggt_cfg.dataset_path = '/mnt/data/datasets/robotwin'
va_robotwin_train_vggt_cfg.empty_emb_path = os.path.join(va_robotwin_train_vggt_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_vggt_cfg.enable_wandb = False
va_robotwin_train_vggt_cfg.load_worker = 16
va_robotwin_train_vggt_cfg.save_interval = 1000
va_robotwin_train_vggt_cfg.gc_interval = 50
va_robotwin_train_vggt_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_vggt_cfg.learning_rate = 1e-5
va_robotwin_train_vggt_cfg.beta1 = 0.9
va_robotwin_train_vggt_cfg.beta2 = 0.95
va_robotwin_train_vggt_cfg.weight_decay = 0.1
va_robotwin_train_vggt_cfg.warmup_steps = 10
va_robotwin_train_vggt_cfg.batch_size = 1
va_robotwin_train_vggt_cfg.gradient_accumulation_steps = 1
va_robotwin_train_vggt_cfg.num_steps = 50000

# VGGT alignment loss parameters
va_robotwin_train_vggt_cfg.vggt_loss_weight = 0.01
va_robotwin_train_vggt_cfg.vggt_loss_start_step = 0
va_robotwin_train_vggt_cfg.vggt_loss_interval = 1
va_robotwin_train_vggt_cfg.vggt_num_sample_frames = 2
va_robotwin_train_vggt_cfg.vggt_grad_enabled = True
