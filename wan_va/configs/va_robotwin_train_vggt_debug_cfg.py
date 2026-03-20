# Debug config for quickly validating VGGT loss convergence on a small data subset.
# Usage: CONFIG_NAME=robotwin_train_vggt_debug bash script/run_va_posttrain_vggt.sh
from easydict import EasyDict
from .va_robotwin_train_vggt_cfg import va_robotwin_train_vggt_cfg

va_robotwin_train_vggt_debug_cfg = EasyDict(
    __name__='Config: VA robotwin VGGT debug')
va_robotwin_train_vggt_debug_cfg.update(va_robotwin_train_vggt_cfg)

va_robotwin_train_vggt_debug_cfg.save_root = './train_out/vggt_debug'

# Small dataset: only use first 50 samples, repeated
va_robotwin_train_vggt_debug_cfg.max_train_samples = 50

# Short run
va_robotwin_train_vggt_debug_cfg.num_steps = 300
va_robotwin_train_vggt_debug_cfg.save_interval = 100
va_robotwin_train_vggt_debug_cfg.gc_interval = 20

# VGGT from step 0 with weight 0.01
va_robotwin_train_vggt_debug_cfg.vggt_loss_weight = 0.01
va_robotwin_train_vggt_debug_cfg.vggt_loss_start_step = 0
