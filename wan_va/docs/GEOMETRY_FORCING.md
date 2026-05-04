# Geometry Forcing 训练（Wan-VA）说明

## 背景

参考论文：Wu et al., *Geometry Forcing: Marrying Video Diffusion and
3D Representation for Consistent World Modeling*, 2025。

核心思想：把冻结的 **VGGT（Visual Geometry Grounded Transformer）** 中间层
patch token 当作"几何老师"，通过**特征对齐**把 3D 结构先验蒸馏进视频扩散模型。
论文给出两种互补目标：

| 目标 | 公式 | 作用 |
|------|------|------|
| **Angular Alignment** | `L_A = 1/LNP Σ (1 - cos(y, f_φ(h)))` | 保证学生/老师特征方向一致 |
| **Scale Alignment**   | `L_S = 1/LNP Σ ‖g_φ(normalize(f_φ(h))) - y‖²` | 保持未归一化特征的幅度，含更丰富几何信号 |

总 loss：`L = L_FM + λ_A · L_A + λ_S · L_S`（论文默认 λ_A=0.5, λ_S=0.05）。

## 与已有 `train_vggt_spatial_forcing.py` 的区别

| 维度 | `train_vggt_spatial_forcing.py` | 本版 `train_vggt_geometry_forcing.py` |
|---|---|---|
| 对齐目标 | 只有 Angular（cosine） | Angular + **Scale**（新增 g_φ 回归头） |
| 学生层 | 单层（`vggt_align_layer_idx=29`） | **多层**（`gf_student_layers=[10,20,29]`） |
| 老师层 | 单层（`vggt_teacher_layer_idx=-1`） | **多层**（`gf_teacher_layers=[-9,-5,-1]`） |
| 每层投影头 | 单个 `TokenAlignProjectionHead` | `ModuleList[f_φ]` + `ModuleList[g_φ]`（各层独立） |
| 像素级 VGGT 几何 loss | 默认叠加（`hybrid`） | 默认**关闭**（`gf_mode='pure'`），可通过 `gf_mode='hybrid'` 打开 |
| loss 权重 | `vggt_loss_weight=0.01`, `vggt_align_weight=0.1` | `gf_lambda_angular=0.5`, `gf_lambda_scale=0.05`（贴近论文） |

## 新增文件一览

```
wan_va/
  train_vggt_geometry_forcing.py           # 训练入口（GeometryForcingTrainer）
  modules/
    geometry_forcing_head.py               # AngularProjectors + ScalePredictors + loss fn
    utils_geometry_forcing.py              # load_transformer 绑定 model_spatial_forcing
  configs/
    va_robotwin_train_vggt_geometry_forcing_cfg.py
    va_robotwin_train_vggt_geometry_forcing_debug_cfg.py
script/
  run_va_posttrain_vggt_geometry_forcing.sh
  run_va_posttrain_vggt_geometry_forcing_debug.sh
wan_va/docs/
  GEOMETRY_FORCING.md                      # 本文档
```

**未修改**任何现有文件：`train_vggt.py` / `train_vggt_spatial_forcing.py` /
`model_spatial_forcing.py` 保持原状；本训练复用 `model_spatial_forcing.py`
已有的 `return_hidden_layers / align_layer_idx=list(...)` 多层 hook。

## 数据流（Angular + Scale）

对每个被选中的 (student_layer, teacher_layer) 配对：

1. Wan DiT 一次 forward 同时返回最终 `pred` 和选中层的 **video token 隐状态** `h`。
2. 冻结 VAE 解码 GT latent → 多摄像头拼接图像 → `_select_vggt_supervision_pixels`
   裁出 `cam_high` 区域 → 冻结 VGGT aggregator 取老师 patch token `y`。
3. 老师 token 双线性插值到学生 token 网格 `(Hs, Ws)`。
4. **Angular**：
   - `proj = f_φ(h)`（每层独立投影头）
   - `L_A += (1 - cos(normalize(proj), normalize(y)))`，带 `(1-σ)` 的帧级权重
5. **Scale**（仅当 `gf_lambda_scale > 0` 且 `use_scale=True`）：
   - `ĥ = normalize(proj)`，`ỹ = g_φ(ĥ)`
   - `L_S += MSE(ỹ, y)`，同样的 per-token 权重

最终：`L = L_video + L_action + λ_A·L_A + λ_S·L_S (+ optional λ_DP·L_depth_point)`。

## 关键配置项

`va_robotwin_train_vggt_geometry_forcing_cfg.py`：

| 配置 | 默认值 | 含义 |
|------|--------|------|
| `gf_mode` | `'pure'` | `pure`：仅 Angular+Scale；`hybrid`：再叠加像素级 depth/point loss |
| `gf_student_layers` | `[10, 20, 29]` | Wan DiT 的 block 下标 |
| `gf_teacher_layers` | `[-9, -5, -1]` | VGGT aggregator 层下标（支持负数） |
| `gf_lambda_angular` | `0.5` | Angular 权重 |
| `gf_lambda_scale` | `0.05` | Scale 权重（设为 0 即可退化为纯 Angular） |
| `gf_proj_hidden_dim` | `2048` | MLP bottleneck 宽度 |
| `gf_use_bn` | `True` | `True` 用 BatchNorm1d，`False` 用 LayerNorm |
| `gf_start_step` | `0` | 从第几步开始启用 GF loss（warmup） |
| `gf_use_sigma_weight` | `True` | 是否按 `(1-σ)` 做帧级权重（论文无此项；机器人任务更稳） |
| `resume_from` | `None` | checkpoint 根目录（含 `transformer/` 与可选 `geometry_forcing_head.safetensors`） |

调节建议：
- 单层先验证：把 `gf_student_layers=[29]`，`gf_teacher_layers=[-1]`。
- 若 Scale loss 发散，优先把 `gf_lambda_scale` 降到 0.01 或临时设 0。

## 优化器 & 分布式

- Transformer：FSDP + `AdamW(fused=True)`，DTensor 参数，沿用父类 `VGGTTrainer`。
- `geometry_forcing_head`：独立 `AdamW(fused=False)`，避免 DTensor 与普通 Tensor 混合 fused 更新。
- 梯度：反向后对 head 做 `all_reduce` 平均；transformer 与 head 的 `clip_grad_norm_`
  分别调用再合成总 norm，避免 foreach 路径类型不匹配报错。

## 启动示例

```bash
# 单卡 debug（小样本 + 固定种子）
NGPU=1 bash script/run_va_posttrain_vggt_geometry_forcing_debug.sh

# 多卡正式训练
NGPU=8 bash script/run_va_posttrain_vggt_geometry_forcing.sh

# 启用 W&B
ENABLE_WANDB=1 WANDB_API_KEY=... WANDB_PROJECT=lingbotva \
WANDB_RUN_NAME=gf_pure_3layers NGPU=8 \
bash script/run_va_posttrain_vggt_geometry_forcing.sh

# 从 checkpoint 续训（包含可选的 head safetensors）
LINGBOT_RESUME_FROM=/path/to/checkpoint_step_10000 NGPU=8 \
bash script/run_va_posttrain_vggt_geometry_forcing.sh

# 切到 hybrid 模式（再叠加 depth/point loss）
CONFIG_NAME=robotwin_train_vggt_geometry_forcing NGPU=8 \
bash script/run_va_posttrain_vggt_geometry_forcing.sh \
  gf_mode=hybrid  # 如需 CLI 覆写需再做参数透传，常规做法是直接改 cfg
```

## 日志曲线

TensorBoard / W&B 会写入：

- `loss/video`, `loss/action`
- `loss/vggt`（depth/point, 仅 hybrid 模式非零）
- `loss/gf_angular`
- `loss/gf_scale`
- `train/gf_cosine`（学生-老师 patch 级平均 cos 相似度）
- `train/grad_norm`, `train/lr`

## 推理 / 评测

推理阶段**只加载 transformer checkpoint**，`geometry_forcing_head.safetensors`
只是训练辅助，评测脚本沿用现有流程。保存目录结构：

```
checkpoint_step_XXXX/
├── transformer/
│   ├── diffusion_pytorch_model.safetensors
│   └── config.json
└── geometry_forcing_head.safetensors   (训练辅助, 可选)
```
