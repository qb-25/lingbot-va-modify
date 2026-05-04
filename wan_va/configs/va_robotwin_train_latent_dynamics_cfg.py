from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_latent_dynamics_cfg = EasyDict(
    __name__='Config: VA robotwin train with latent dynamics branch')
va_robotwin_train_latent_dynamics_cfg.update(va_robotwin_cfg)

va_robotwin_train_latent_dynamics_cfg.save_root = './train_out/latent_dynamics'

va_robotwin_train_latent_dynamics_cfg.dataset_path = '/mnt/data/datasets/robotwin'
va_robotwin_train_latent_dynamics_cfg.empty_emb_path = os.path.join(
    va_robotwin_train_latent_dynamics_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_latent_dynamics_cfg.enable_wandb = False
va_robotwin_train_latent_dynamics_cfg.load_worker = 4
va_robotwin_train_latent_dynamics_cfg.dataset_init_worker = 1
va_robotwin_train_latent_dynamics_cfg.save_interval = 1000
va_robotwin_train_latent_dynamics_cfg.gc_interval = 50
va_robotwin_train_latent_dynamics_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_latent_dynamics_cfg.learning_rate = 1e-5
va_robotwin_train_latent_dynamics_cfg.beta1 = 0.9
va_robotwin_train_latent_dynamics_cfg.beta2 = 0.95
va_robotwin_train_latent_dynamics_cfg.weight_decay = 0.1
va_robotwin_train_latent_dynamics_cfg.warmup_steps = 10
va_robotwin_train_latent_dynamics_cfg.batch_size = 1
va_robotwin_train_latent_dynamics_cfg.gradient_accumulation_steps = 1
va_robotwin_train_latent_dynamics_cfg.num_steps = 50000
# va_robotwin_train_latent_dynamics_cfg.resume_from = va_robotwin_train_latent_dynamics_cfg.ckpt_path

# Latent dynamics branch
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_state_dim = 256
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_hidden_dim = 512
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_pred_weight = 0.05
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_reg_weight = 0.01
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_num_projections = 256
va_robotwin_train_latent_dynamics_cfg.latent_dynamics_start_step = 0
