import os

from easydict import EasyDict

from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg


va_robotwin_train_vggt_repa_vae_cfg = EasyDict(
    __name__='Config: VA robotwin train with VGGT + VAE-REPA alignment'
)
va_robotwin_train_vggt_repa_vae_cfg.update(va_robotwin_train_vggt_cfg)

# Default output root; override with CLI --save-root or env-driven configs on DLC.
va_robotwin_train_vggt_repa_vae_cfg.save_root = '/mnt/nas/qianbin/train_repa'
va_robotwin_train_vggt_repa_vae_cfg.wandb_run_name = 'train_vggt_repa_vae'
va_robotwin_train_vggt_repa_vae_cfg.save_interval = 1000

# Keep the existing VGGT geometric branch available; set mode='feature' and vggt_loss_weight=0
# if you want behavior closest to the paper's pure denoising + REPA setup.
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_mode = 'feature'
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_weight = 1.0
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_use_bn = True
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_proj_dim = 2048
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_num_layers = 5
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_start_step = 0
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_beta = 0.05

# VAE-REPA prefers earlier layers. For Wan's deeper video transformer, use block 4 as a
# conservative early-layer default; try {2, 4, 6} first if you want to sweep this.
va_robotwin_train_vggt_repa_vae_cfg.vggt_align_layer_idx = 4

resume_from = os.environ.get('LINGBOT_RESUME_FROM')
if resume_from:
    va_robotwin_train_vggt_repa_vae_cfg.resume_from = resume_from
