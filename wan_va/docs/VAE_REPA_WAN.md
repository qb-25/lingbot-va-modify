# Wan VAE-REPA 对齐版说明

## 目标

这版不是继续对齐 `VGGT token` 或 `VAE encoder pre-quant hidden`，而是按 **VAE-REPA** 论文的主思路，改成：

- **student**：Wan diffusion transformer 的**早期中间层视频 token**
- **teacher**：数据集中已经存在的 **clean VAE latent**（也就是 diffusion 二阶段训练本来就在用的 VAE 特征）
- **loss**：`smooth_l1`
- **head**：默认 **5 层 MLP**

也就是说，这版更接近论文里的“用现成 VAE latent 特征去引导 diffusion trunk 的中间表示”，而不是“再引入一个外部视觉 teacher”。

## 论文关键信息与代码映射

根据 `VAE-REPA: Variational Autoencoder Representation Alignment for Efficient Diffusion Training`：

1. 对齐目标是 **off-the-shelf VAE features**，不是额外编码器，也不是 VAE 更深的隐藏层。
2. 对齐对象是 **diffusion transformer 的中间层特征**。
3. 默认对齐损失是 **smooth-L1**，论文里 `beta=0.05`。
4. **更早的层更有效**，论文表格里越往深层效果越差。
5. **5-layer MLP** 比 2-layer 更好。
6. timestep 范围用 **全范围 `[0, 1]`**。

本实现对应关系如下：

- `latent_dict['latent']` 直接作为 teacher 的 clean VAE feature 来源。
- `modules/model_repa_vae.py` 复制了一份带中间层暴露能力的 Wan transformer。
- `train_vggt_repa_vae.py` 在指定 block 抓取 `align_hidden_states`，通过 `repa_align_head` 投到 VAE latent patch 空间。
- teacher 不再需要 decode / re-encode，直接把 clean latent 按 `patch_size` patchify 后监督 student token。

## 新增文件

- `wan_va/modules/model_repa_vae.py`
- `wan_va/modules/utils_repa_vae.py`
- `wan_va/train_vggt_repa_vae.py`
- `wan_va/configs/va_robotwin_train_vggt_repa_vae_cfg.py`
- `wan_va/configs/va_robotwin_train_vggt_repa_vae_debug_cfg.py`
- `script/run_va_posttrain_vggt_repa_vae.sh`
- `script/run_va_posttrain_vggt_repa_vae_debug.sh`

## 对齐层选择

论文结论是“**早层优于深层**”。但 Wan 是视频 transformer，层数和 SiT 不同，不能机械照搬论文里的 block id。

因此这里默认选：

- `vggt_align_layer_idx = 4`

理由：

1. 它仍属于明显的**早层**，符合论文趋势。
2. 比 `2` 稍晚一点，给视频模型留下最初几层做跨帧/条件融合的空间。
3. 对当前 Wan 规模来说，`4` 更像是一个“保守但合理”的起点。

如果你要 sweep，建议优先试：

- `2`
- `4`
- `6`

## 训练分支设计

这版仍然**复制自现有 `train_vggt` 体系**，所以保留了原有 VGGT depth/point 分支，方便与你现在的实验直接对齐：

- `vggt_align_mode='feature'`：最接近论文原始形态，只做 denoising + REPA
- `vggt_align_mode='depth_point'`：只做原来的 VGGT 几何监督
- `vggt_align_mode='hybrid'`：两者同时开

默认配置用的是 `hybrid`，这样更适合你当前项目的延续实验；如果你想做“尽量贴论文”的 ablation，建议：

```python
vggt_align_mode = 'feature'
vggt_loss_weight = 0.0
vggt_align_weight = 1.0
```

## loss 细节

`compute_repa_feature_align_loss()` 的流程：

1. 从 transformer 指定 early block 取 `align_hidden_states`
2. reshape 成 `(B, F, Ht, Wt, C)`
3. 用 `repa_align_head` 投影到 teacher 维度
4. 把 `clean latent` patchify 成 `(B, F, Ht, Wt, C_teacher)`
5. 计算逐元素 `smooth_l1`
6. 用 timestep 权重 `(1 - sigma).clamp(min=0.05)` 做加权平均

这里 teacher 直接来自 clean latent，因此比“decode 后再走 VAE encoder”更贴近论文里的 **off-the-shelf latent feature** 思路，也更省算力。

## 日志与 checkpoint

新增日志项：

- `loss/repa_align`
- `train/repa_align_mae`

新增 checkpoint 文件：

- `repa_align_head.safetensors`

推理时仍然只需要 `transformer/`；`repa_align_head.safetensors` 只是训练辅助头。

## 启动方式

```bash
# debug
NGPU=1 bash script/run_va_posttrain_vggt_repa_vae_debug.sh

# 多卡
NGPU=8 bash script/run_va_posttrain_vggt_repa_vae.sh
```

## 建议的首轮实验

如果你想先验证“论文思路在 Wan 上是否成立”，建议按下面顺序做：

1. `feature only`, `layer=4`
2. `feature only`, `layer=2`
3. `feature only`, `layer=6`
4. `hybrid`, `layer=4`

这样最容易判断：收益到底来自 **VAE-REPA 本身**，还是来自与 VGGT 几何监督的组合。
