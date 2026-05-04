import os

from easydict import EasyDict

from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg


va_robotwin_train_vggt_spatial_forcing_cfg = EasyDict(
    __name__='Config: VA robotwin train with VGGT spatial forcing'
)
va_robotwin_train_vggt_spatial_forcing_cfg.update(va_robotwin_train_vggt_cfg)

va_robotwin_train_vggt_spatial_forcing_cfg.save_root = './train_out/vggt_spatial_forcing_29'
va_robotwin_train_vggt_spatial_forcing_cfg.wandb_run_name = 'train_vggt_spatial_forcing'
va_robotwin_train_vggt_spatial_forcing_cfg.save_interval = 10000
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_mode = 'hybrid'
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_layer_idx = 29
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_weight = 0.1
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_use_bn = True
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_proj_dim = 2048
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_align_start_step = 0
va_robotwin_train_vggt_spatial_forcing_cfg.vggt_teacher_layer_idx = -1

# 继续训练：指向「含 transformer/ 子目录」的 checkpoint 根目录（不要只写到 transformer 这一层）。
# 也可在启动前设置环境变量 LINGBOT_RESUME_FROM=/path/to/checkpoint_step_xxx覆盖下面默认值。
va_robotwin_train_vggt_spatial_forcing_cfg.resume_from = None
_resume_env = os.getenv("LINGBOT_RESUME_FROM", "").strip()
if _resume_env:
    va_robotwin_train_vggt_spatial_forcing_cfg.resume_from = _resume_env
