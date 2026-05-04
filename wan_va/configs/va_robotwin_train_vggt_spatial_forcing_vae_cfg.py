from easydict import EasyDict

from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg


va_robotwin_train_vggt_spatial_forcing_vae_cfg = EasyDict(
    __name__='Config: VA robotwin train with VGGT VAE-encoder spatial alignment'
)
va_robotwin_train_vggt_spatial_forcing_vae_cfg.update(va_robotwin_train_vggt_cfg)

va_robotwin_train_vggt_spatial_forcing_vae_cfg.save_root = './train_out/vggt_spatial_forcing_vae'
va_robotwin_train_vggt_spatial_forcing_vae_cfg.wandb_run_name = 'train_vggt_spatial_forcing_vae'
va_robotwin_train_vggt_spatial_forcing_vae_cfg.save_interval = 10000

# Same high-level knobs as transformer spatial forcing; "feature" = VAE encoder pre-quant align.
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_align_mode = 'hybrid'
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_align_weight = 0.1
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_align_use_bn = True
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_align_proj_dim = 2048
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_align_start_step = 0
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vggt_teacher_layer_idx = -1

# Student branch: pred = from predicted velocity->x0 decode; gt = from dataset latent decode (no grad to transformer)
va_robotwin_train_vggt_spatial_forcing_vae_cfg.vae_align_student_source = 'pred'
