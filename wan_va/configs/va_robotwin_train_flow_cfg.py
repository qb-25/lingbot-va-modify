from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_flow_cfg = EasyDict(__name__='Config: VA robotwin train with flow loss')
va_robotwin_train_flow_cfg.update(va_robotwin_cfg)

va_robotwin_train_flow_cfg.save_root = './train_out/flow'

va_robotwin_train_flow_cfg.dataset_path = '/mnt/data/datasets/robotwin'
va_robotwin_train_flow_cfg.empty_emb_path = os.path.join(va_robotwin_train_flow_cfg.dataset_path, 'empty_emb.pt')
# 开启方式：设为 True，或启动命令加 --enable-wandb（与 train_vggt_spatial_forcing 一致）
va_robotwin_train_flow_cfg.enable_wandb = True
va_robotwin_train_flow_cfg.wandb_run_name = "train_flow"
va_robotwin_train_flow_cfg.load_worker = 16
va_robotwin_train_flow_cfg.save_interval = 1000
va_robotwin_train_flow_cfg.gc_interval = 50
va_robotwin_train_flow_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_flow_cfg.learning_rate = 1e-5
va_robotwin_train_flow_cfg.beta1 = 0.9
va_robotwin_train_flow_cfg.beta2 = 0.95
va_robotwin_train_flow_cfg.weight_decay = 0.1
va_robotwin_train_flow_cfg.warmup_steps = 10
va_robotwin_train_flow_cfg.batch_size = 1
va_robotwin_train_flow_cfg.gradient_accumulation_steps = 1
va_robotwin_train_flow_cfg.num_steps = 50000
# va_robotwin_train_flow_cfg.resume_from = va_robotwin_train_flow_cfg.ckpt_path

# Disable VGGT in the flow-only experiment.
va_robotwin_train_flow_cfg.vggt_loss_weight = 0.0

# Optical-flow alignment loss parameters
va_robotwin_train_flow_cfg.flow_loss_weight = 0.02
va_robotwin_train_flow_cfg.flow_loss_start_step = 0
va_robotwin_train_flow_cfg.flow_num_pairs = 1
va_robotwin_train_flow_cfg.flow_grad_enabled = True
va_robotwin_train_flow_cfg.flow_model_name = 'dpflow'
va_robotwin_train_flow_cfg.flow_model_ckpt = '/mnt/data/qianbin/models/dpflow-things.ckpt'
va_robotwin_train_flow_cfg.flow_input_height = 256
va_robotwin_train_flow_cfg.flow_input_width = 320
va_robotwin_train_flow_cfg.flow_supervision_cam = 'cam_high'
