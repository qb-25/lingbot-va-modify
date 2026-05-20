# Geometry Forcing + Internal 训练（Wan-VA）说明

## 总体思路

本分支在 `train_vggt_geometry_forcing.py` 基础上**叠加 Internal Guidance**
（Zhou et al., 2025）的"训练侧"做法，并加上**Wan 每层对任务目标的
cross-attention 可视化**。

```
       noisy/clean (video + action) tokens
                        │
                        ▼
              blocks[0 : internal_depth]      ← 共享 trunk（GF hooks 在这一段也生效）
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   blocks[internal_depth :]    internal_blocks  (deepcopy of last N main blocks)
       (main path D_f)             (D_i)
            │                       │
       norm_out + proj_out    internal_norm_out + internal_proj_out
       action_proj_out            internal_action_proj_out
            │                       │
            ▼                       ▼
       (main_lat, main_act)   (int_lat, int_act)

         L_main = MSE(main_*, target)
         L_int  = MSE(int_*,  target)
         L_total = L_main + λ_int · L_int + λ_A · L_Angular + λ_S · L_Scale
                  ( + 可选 hybrid 模式下的 VGGT depth/point 几何 loss )
```

> Geometry Forcing 的 Angular + Scale alignment 仍然在主干（D_f）的
> `[10, 20, 29]` 三层做对齐；internal head 用主干末尾 N 个 block 的
> deepcopy 初始化，等价于"主干末端的浅副本"，开训前一刻 D_i ≈ D_f，
> 训练过程中再让 D_i 在更短路径上独立收敛。

## 新增文件

```
wan_va/
  train_vggt_geometry_forcing_internal.py        # 训练入口（继承 GeometryForcingTrainer）
  vis_attention.py                                # 离线注意力可视化脚本
  modules/
    model_geometry_forcing_internal.py           # Wan + GF hooks + internal 分支 + cross-attn 捕获
    utils_geometry_forcing_internal.py           # load_transformer：从 base 末尾 deepcopy 初始化
    attention_visualizer.py                      # cross-attn 概率 → 任务目标热力图 → RGB 叠图
  configs/
    va_robotwin_train_vggt_geometry_forcing_internal_cfg.py
    va_robotwin_train_vggt_geometry_forcing_internal_debug_cfg.py
script/
  run_va_posttrain_vggt_geometry_forcing_internal.sh
  run_va_posttrain_vggt_geometry_forcing_internal_debug.sh
```

未修改任何已有文件。

## 关键 Config

`va_robotwin_train_vggt_geometry_forcing_internal_cfg.py`：

| 配置 | 默认 | 含义 |
|---|---|---|
| `enable_internal` | `True` | 是否启用 internal 分支 |
| `internal_depth` | `24` | 在主干第几个 block 后分叉（Wan 30 层，留末段做 D_f） |
| `num_internal_blocks` | `2` | internal 分支自带几层（用 base 末尾 N 个 block deepcopy 初始化） |
| `lambda_internal` | `1.0` | 内部 head 的 deep-supervision 权重 |
| `internal_start_step` | `0` | 第几步开始算 internal loss |
| `gf_student_layers` | `[10, 20, 29]` | GF Angular/Scale 的学生层（继承自 GF cfg） |
| `gf_teacher_layers` | `[-9, -5, -1]` | VGGT 老师层 |
| `attn_vis_enabled` | `True` | 训练中是否定期 dump 注意力 PNG |
| `attn_vis_interval` | `5000` | 间隔 |
| `attn_vis_layers` | `[0, 10, 20, 29]` | 可视化层（默认 4 层） |
| `attn_vis_frames` | `[0, 2, 5]` | 取哪几帧 |
| `attn_vis_token_meta` | `{mode: 'content_only', top_k: 16}` | 任务目标 token 选择策略 |
| `attn_vis_alpha` | `0.5` | heatmap 与 RGB 叠图透明度 |

## 损失日志（wandb / TensorBoard）

新增 key：

* `loss/int_latent` —— 内部 head 视频 deep-supervision loss
* `loss/int_action` —— 内部 head 动作 deep-supervision loss
* `attn/cross_attn_grid` —— 跨层注意力叠图（W&B Image / TB Image）

GF 原有 key 全部保留（`loss/gf_angular`, `loss/gf_scale`, `train/gf_cosine`
等）。

## Cross-Attention 可视化（V1）

每层 `attn2`（video↔text cross-attn）的概率矩阵 `softmax(QK^T/√d)`
在训练时不会被 flex_attention / FlashAttn 直接返回，所以采用 **slow path
重算**（仅可视化时启用，不影响主路径速度）：

1. 训练 step 命中 `attn_vis_interval` 时，把 `capture_cross_attn=True`
   传进 transformer。
2. 模型在每个目标 block 上额外用手写 SDPA 算一次 cross-attn 概率（不复用
   主路径的 attn1/attn2 输出）。
3. 取 video 段 query → 选定的 prompt token（默认 top-K 内容 token）的
   注意力质量，reshape 成 `(F, H, W)`，叠到 GT-decoded RGB 上。
4. 多层 / 多帧拼成网格 PNG，写入 `train_out/.../attn_vis/`，同步推 W&B。

支持三种"任务目标"选择：
* `content_only`（默认）：找出注意力质量最大的 K 个 prompt token（鲁棒）。
* `token_indices`：传入显式的 K/V 下标。
* `span`：取 `[start, end)` 区间。

离线脚本 `wan_va/vis_attention.py` 支持任意 checkpoint + 任意层 + 任意帧
按需出图。

## 启动

```bash
# Debug (单卡)
NGPU=1 bash script/run_va_posttrain_vggt_geometry_forcing_internal_debug.sh

# 多卡训练
NGPU=8 bash script/run_va_posttrain_vggt_geometry_forcing_internal.sh

# W&B
ENABLE_WANDB=1 WANDB_API_KEY=... WANDB_PROJECT=lingbotva \
WANDB_RUN_NAME=gfi_d24_n2 NGPU=8 \
bash script/run_va_posttrain_vggt_geometry_forcing_internal.sh

# 离线注意力可视化（拿任意 checkpoint）
python -m wan_va.vis_attention \
    --ckpt /path/to/checkpoint_step_NNN \
    --out  ./attn_vis_offline \
    --layers 0,5,10,15,20,25,29 \
    --frames 0,2,5
```

## 排雷点

* **internal 分支用主干末尾 N 个 block deepcopy 初始化**：开训前一刻
  `D_i` 几乎等于 `D_f`（只是少了 `internal_depth → 30` 那段计算），
  loss 一开始可能出现 `int_loss ≈ main_loss`。这正常，等几百步后两者
  会逐渐分离。
* **可视化对训练速度的影响**：`attn_vis_interval` 太小会拖慢；默认
  5000 步一次，2 万步训练只触发 4 次。
* **如何关掉可视化**：设 `attn_vis_enabled = False` 或把 interval 调大。
