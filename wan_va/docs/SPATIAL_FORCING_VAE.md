# Spatial Forcing（VAE 编码器版）说明

## 背景

- **原版 `train_vggt_spatial_forcing.py`**：在 Wan **扩散 Transformer** 指定层取 **noisy 视频 token**，经投影后与冻结 **VGGT aggregator** 的 patch token 做 cosine 对齐（`model_spatial_forcing.py` 需暴露中间层 hidden）。
- **本版 `train_vggt_spatial_forcing_vae.py`**：不对扩散主干动刀；改为对齐 **Wan VAE encoder 在 `quant_conv` 之前的特征图**（与 `WanVAEStreamingWrapper.encode_chunk` 中 `encoder(...)` 输出、**进入 `quant_conv` 之前** 的张量一致）与同一套 **VGGT teacher**。

## 新增 / 拷贝的文件

| 文件 | 作用 |
|------|------|
| `wan_va/train_vggt_spatial_forcing_vae.py` | 独立训练入口（新 `VGGTSpatialForcingVaeTrainer`），**不 import** `train_vggt_spatial_forcing`，避免对 `load_transformer` 的 monkey-patch。 |
| `wan_va/modules/vae_encoder_utils.py` | 抽取 VAE encoder **pre-quant** 空间特征的工具函数。 |
| `wan_va/configs/va_robotwin_train_vggt_spatial_forcing_vae_cfg.py` | 正式训练默认配置。 |
| `wan_va/configs/va_robotwin_train_vggt_spatial_forcing_vae_debug_cfg.py` | 小数据 debug 配置。 |
| `script/run_va_posttrain_vggt_spatial_forcing_vae.sh` | 多卡启动脚本。 |
| `script/run_va_posttrain_vggt_spatial_forcing_vae_debug.sh` | 单卡 debug 脚本。 |

**未修改**现有 `train_vggt_spatial_forcing.py` / `model_spatial_forcing.py`。

## 数据流（feature 分支）

1. 从 batch 得到 `latent_pred`（与现有训练一致）。
2. 依 `vae_align_student_source` 选择学生图像来源：
   - **`pred`（默认）**：`pred_clean = x_t - σ·v` → VAE **decode** 得 RGB → 与 `train_vggt` 一致做 `cam_high` 等裁剪（`_select_vggt_supervision_pixels`）。  
     若 `vggt_grad_enabled=True`，decode 与后续encoder可反传至 **Transformer**。
   - **`gt`**：用数据集 `latent` decode，**不**对 Transformer 回传该支路梯度。
3. 学生 RGB（`[0,1]`）转为 VAE 输入 `[-1,1]`，经 `WanVAEStreamingWrapper` 走 encoder，**取 `quant_conv` 之前的特征**，在 **时间维**上 squeeze/mean 后得到 `(B, F, C, Hs, Ws)`。
4. Teacher：对 **GT decode + 同裁剪** 的 RGB 跑冻结 **VGGT aggregator**，取指定层 patch token（与 spatial forcing 相同），再 **双线性插值** 到 `(Hs, Ws)` 与学生对齐。
5. Per-location cosine：**学生**经可训 **`vae_align_head`（MLP+BN/LN）** 投到 teacher 维；损失为加权 `1 - cos`（权重与 timestep 相关，同 spatial forcing 思路）。

`depth_point` 分支仍走父类 `compute_vggt_loss`，与原版一致。

## 优化器与分布式

- **Transformer**：原有 FSDP + `AdamW(fused=True)`（仅 DTensor 参数）。
- **`vae_align_head`**：单独 **`AdamW(fused=False)`**，避免与 DTensor 混用 fused 更新。
- 对齐头梯度在 `backward` 后做 **`all_reduce`**（与 `train_vggt_spatial_forcing` 相同意图）。

## 推理 / 评测

推理只加载 **扩散 Transformer checkpoint**；**`vae_align_head.safetensors` 仅训练辅助**，评测脚本与此前一致，设置 `CKPT_PATH` 指向含 `transformer/` 的目录即可。

## 配置项说明（常用）

| 配置 | 含义 |
|------|------|
| `vggt_align_mode` | `feature`：仅 VAE 对齐；`depth_point`：仅几何；`hybrid`：两者。 |
| `vggt_align_weight` | VAE 对齐项权重。 |
| `vae_align_student_source` | `pred` 或 `gt`，见上文。 |
| `save_interval` | 保存间隔（正式配置默认 `10000`）。 |

## 启动示例

```bash
# Debug
NGPU=1 bash script/run_va_posttrain_vggt_spatial_forcing_vae_debug.sh

# 多卡训练
NGPU=8 bash script/run_va_posttrain_vggt_spatial_forcing_vae.sh
```

## W&B / TensorBoard 曲线名

- `loss/vae_align`
- `train/vae_align_cosine`

（与 transformer 版 `loss/vggt_feature_align` 区分，避免混淆。）
