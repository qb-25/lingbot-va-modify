# Geometry Forcing + Cross-Stream Alignment 训练（Wan-VA）说明

## 总体思路

在 GF（Geometry Forcing）的几何先验对齐基础上，叠加一个**双流对比学习**：
让 Wan DiT 的中间层 **video token** 与 **action token** 在共享 latent space
里互为正样本，作为 **action 流的额外监督**——这是 lingbot-va 上第一次让
action 流接到外部信号（之前 action 完全靠数据集 trajectory 自监督）。

```
       noisy/clean (video + action) tokens (concat + pad)
                        │
                        ▼
                 blocks[0:30]   ← 完整 Wan DiT 主干
                        │
       ┌────────────────┼─────────────────────────────────────┐
       │                │                                     │
       ▼                ▼                                     ▼
  GF 学生层抽 hidden    main head (D_f)                   block 25 抽
   [10, 20, 29]                                    (video, action) hidden
       │                                                    │
   GF Angular + Scale                              池化 → projector_v / a
   对齐 VGGT teacher                               L2-norm → InfoNCE
       │                                                    │
       ▼                                                    ▼
 L_A + L_S                                           L_xstream

 总 loss = L_video + L_action
        + λ_A · L_A + λ_S · L_S      (GF, λ_A=0.5, λ_S=0.05)
        + λ_xstream · L_xstream      (Cross-Stream, λ=0.1)
        ( + 可选 hybrid 模式下的 VGGT depth/point loss )
```

注意一个关键设计：**cross-stream 默认放在 block 25**（GF 没占的层），
两个监督信号共生不打架；如果你把 xstream_layers 改到 [10/20/29]，会出现
"同一表征同时被 VGGT 老师和 InfoNCE 拽"，效果会反而下降。

---

## 新增文件

```
wan_va/
  train_vggt_geometry_forcing_xstream.py        # 训练入口
  modules/
    cross_stream_align.py                       # CrossStreamProjector + InfoNCE + all-gather
configs/
  va_robotwin_train_vggt_geometry_forcing_xstream_cfg.py
  va_robotwin_train_vggt_geometry_forcing_xstream_debug_cfg.py
script/
  run_va_posttrain_vggt_geometry_forcing_xstream.sh
  run_va_posttrain_vggt_geometry_forcing_xstream_debug.sh
docs/
  GEOMETRY_FORCING_XSTREAM.md                   # 本文
```

并在 `modules/model_geometry_forcing_internal.py` 上扩展了**新的 hook**：
传 `xstream_layers=[...]` 进 `forward_train_gf_internal()`，会额外返回
`xstream_hidden_states[block_idx] = {'video': (B,Lv,C), 'action': (B,La,C)}`。
GF 的旧 hook（`return_hidden_layers / align_layer_idx`）保持不变，与
xstream 互不干扰。

---

## 关键 Config

`va_robotwin_train_vggt_geometry_forcing_xstream_cfg.py`：

| 配置 | 默认 | 含义 |
|---|---|---|
| `xstream_enable` | True | 是否启用 cross-stream |
| `xstream_layers` | `[25]` | 在哪几层做对齐（建议**避开** GF 学生层） |
| `xstream_proj_dim` | 256 | 投影空间维度（CLIP/SimCLR 默认） |
| `xstream_proj_hidden` | 1024 | 投影头 hidden 宽度 |
| `xstream_pos_mode` | `'window'` | 正样本约定 `frame / window / trajectory` |
| `xstream_window` | 2 | window 模式半径（≤W 帧均算正） |
| `xstream_tau` | **0.15** | InfoNCE 温度（**不是** CLIP 的 0.07，因为这里 effective batch 较小） |
| `lambda_xstream` | 0.1 | 损失权重 |
| `xstream_use_all_gather` | **True** | **关键**：跨 GPU 拼接 z 才有足够负样本 |
| `xstream_sigma_soft` | True | True=按 (1-σ_v)(1-σ_a) 软加权；False=σ<th 硬门 |
| `xstream_interval` | 1 | 每 N 步算一次（>1 可省时间） |
| `xstream_start_step` | 0 | warmup |
| `attn_vis_interval` | **2000** | cross-attn 热图保存频率（你要的每 2000 步） |
| `attn_vis_layers` | `[0, 10, 20, 29]` | 可视化层 |

GF / 其他字段全部沿用 GF cfg，这里不重复列举。

---

## 损失日志（wandb / TensorBoard）新增项

| key | 含义 |
|---|---|
| `loss/cross_stream` | 总 InfoNCE loss（已乘 sigma gate） |
| `train/cross_stream_logits_diag` | 对角元素均值（越大表示 anchor↔自己 越近） |
| `train/cross_stream_pos_neg_gap` | mean(pos logits) − mean(neg logits)，**最关键的健康指标** |
| `train/cross_stream_topk1_acc` | InfoNCE top-1 命中率（anchor 的 argmax 是否在 pos 集合里） |
| `train/cross_stream_gate` | sigma gate 标量（前期接近 0.05～0.3，后期上升） |
| `attn/cross_attn_grid` | Wan 跨层 cross-attn 热图叠图（PNG） |

GF 原有 key（`loss/gf_angular`, `loss/gf_scale`, `train/gf_cosine` 等）保留。

### 健康曲线判断

| 现象 | 解读 |
|---|---|
| 1k step 内 `pos_neg_gap` 从 0 涨到 0.05+ | 正常，对比信号在形成 |
| 5k step 后 `pos_neg_gap` 仍 ≤ 0.02 | **不健康** — 把 `pos_mode='trajectory'` 起步 + `tau` 调到 0.2 + 把 `xstream_layers` 换成更早的层（如 [5]） |
| `topk1_acc` 长期 ≈ 1/N | 同上 — 负样本数不够（检查 all_gather 是否真的开了） |
| `loss/action` 反而上升 | λ_xstream 过大，或 cross 层与 GF 学生层冲突 — 把 lambda 降到 0.05，或把 xstream_layers 移到 [5] |

---

## Cross-Stream 工程细节（必读）

### 1. 数据形状（lingbot-va 默认配置）

* video latent: `(B, 48, F=6, H=32, W=40)` → patch_size=[1,2,2] → token = `(B, 6×16×20=1920, D=3072)`
* action: `(B, 30, F=6, K=16, 1)` → token = `(B, 6×16=96, D=3072)`
* video / action **沿 F 严格 1:1** 对齐（每 video frame 对应一个 16-step action chunk）

池化后两者都变成 `(B, 6, D)`，再过独立 MLP 降到 `(B, 6, 256)`，做 InfoNCE。

### 2. 跨 GPU all-gather（必须做）

lingbot-va 默认 `batch_size=1`、`world_size=8`、`gradient_accumulation_steps=8`。
单卡单 micro-step 只能看到 1 个轨迹 × 6 帧 = 6 个 anchor、5 个负样本。
这对 InfoNCE 是灾难性的小（CLIP 至少 256 负）。

`compute_cross_stream_loss(use_all_gather=True)` 会：
1. 每卡算自己的 `z_v`, `z_a`（各 6 个）；
2. `dist.all_gather` 把 8 卡的 z 拼起来 → 48 个 anchor / 47 个负；
3. 自动调整 `(b, f)` 索引让 8 卡的 trajectory 编号不重，positive mask 正确。

为了让本地 grad 能通过 all-gather 的副本传回去，做了一个标准 trick：把
本地 rank 那一份替换为 grad-tracking 的原 tensor（详见 `_all_gather_with_grad`）。

### 3. Sigma gating

video 与 action 的 `timesteps / num_train_timesteps` 是 σ ∈ [0, 1]。
soft 模式下 loss × `mean((1-σ_v)(1-σ_a)).clamp_min(0.05)`，
噪声越大 gate 越小，让 InfoNCE 主要受**接近 clean 的样本**驱动；
hard 模式下 σ < threshold 才计 loss。

### 4. FSDP 兼容

`xstream_projector` 是普通 `nn.Module`（不被 FSDP 切），与 `gf_head` 同等
处理：
- 独立 `AdamW(fused=False)`；
- backward 后 `dist.all_reduce` 同步梯度；
- `clip_grad_norm` 单独算后并入总 norm。

### 5. Checkpoint

每个 save 周期会额外存：
```
checkpoint_step_NNN/cross_stream_projector.safetensors
```
推理时**完全不需要**它，可单独删除。

---

## 启动

```bash
# Debug (单卡)
NGPU=1 bash script/run_va_posttrain_vggt_geometry_forcing_xstream_debug.sh

# 多卡正式训练
NGPU=8 bash script/run_va_posttrain_vggt_geometry_forcing_xstream.sh

# W&B
ENABLE_WANDB=1 WANDB_API_KEY=... WANDB_PROJECT=lingbotva \
WANDB_RUN_NAME=gfxs_block25_tau15 NGPU=8 \
bash script/run_va_posttrain_vggt_geometry_forcing_xstream.sh
```

---

## 调参手册（按优先级）

如果上线第一次发现 `pos_neg_gap` 起不来：

1. **优先 fix**：检查 `xstream_use_all_gather=True` 且 W&B 上的 `cross_stream_topk1_acc > 1/N`（N≈48）
2. 把 `xstream_pos_mode` 从 `'window'` 改到 `'trajectory'`（同 batch 全算正）
3. `xstream_tau` 从 0.15 升到 0.20~0.25
4. 把 `xstream_layers` 从 `[25]` 改到 `[5]` 或 `[15]`
5. 仍不行：`lambda_xstream` 临时降到 0.05，先看 `pos_neg_gap` 是否能稳定上涨

如果 `loss/action` 因加了 xstream 而上升超过 5%：
- 多半是层选与 GF 撞；移开（如 25 撞了 → 试 5 或 15）
- 或 `lambda_xstream` 太大；从 0.1 降到 0.05

---

## 与现有训练栈的关系

| trainer | 几何先验 | 内部 head | 跨流对齐 | 适用场景 |
|---|:---:|:---:|:---:|---|
| `train_vggt.py` | depth/point | ✗ | ✗ | 仅像素级几何 |
| `train_vggt_spatial_forcing.py` | feature cosine（单层） | ✗ | ✗ | 早期 |
| `train_vggt_geometry_forcing.py` | Angular+Scale 多层 | ✗ | ✗ | GF 论文复现 |
| `train_vggt_geometry_forcing_internal.py` | 同上 | ✓ | ✗ | GF + IG 训练侧 |
| **`train_vggt_geometry_forcing_xstream.py`** | 同上 | ✗ | ✓ | **本仓库**：GF + 跨流对比 |

如果你想把 internal head 也叠上去（成为 GF + Internal + XStream 三合一），
逻辑是：以 internal trainer 为基类，再把 xstream 那一段拷过去；目前没写，
等本版本验证有效后再做。
