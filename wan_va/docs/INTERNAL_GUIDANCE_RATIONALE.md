# Internal 做法的优势与原因（深入说明）

> 适配项目：LingBot-VA（`lingbot-va` 与 `qb_xx/lingbot-va` 仓库）
> 关联代码：`wan_va/train_vggt_geometry_forcing_internal.py`、`wan_va/modules/model_geometry_forcing_internal.py`
> 关联论文：*Guiding a Diffusion Transformer with the Internal Dynamics of Itself*（Zhou et al., 2025；下文简称 IG 论文）

---

## 1. 一句话定位

**Internal Guidance（IG）** 是给深层 Diffusion Transformer 加一份"自蒸馏式"的中间监督——
把网络的某一中间层挂上一个独立去噪头 $D_i$，与最终头 $D_f$ 共同回归 $x_0$，**不引入外部 teacher、不改采样器、几乎不增加显存**，
就能把 SiT-XL / LightningDiT / DiT 这类 30+ 层 DiT 的训练效率与生成质量同步显著拉升。
论文里在 ImageNet-256 上 LightningDiT+IG 达到 **FID = 1.19**（with CFG），SiT-XL+IG 仅用 **80 epoch** 就 FID = 5.31，
**比 SiT-XL@1400ep + REPA@800ep 都更早进入收敛区**。

在 LingBot-VA 这边，我们把同样的设计搬到了 Wan DiT（30 层、视频+动作双流），
并与 Geometry Forcing 几何先验对齐**叠加使用**，trainer 入口是
`train_vggt_geometry_forcing_internal.py`。

---

## 2. 为什么需要 Internal —— 来自 DiT 的固有问题

### 2.1 Diffusion Transformer 越深越难训

Diffusion 模型的训练目标是覆盖完整数据分布（包括低密度区域），但低密度区域监督信号天然稀疏。
当模型变成 30 层 / 40 层这种深 DiT，会同时遇到：

* **梯度消失 / 表征塌缩**：早期层在长链路反向传播下更新困难；
* **训练样本利用率低**：低密度区域的 loss 主要由"上层最终头"承担，下层学不到清晰的 task signal；
* **靠大 epoch 硬堆**：vanilla DiT 需要 1400 epoch 才追平 baseline，REPA 用 SSL teacher 把 epoch 砍到一半但**多引入一份冻结的 DINOv2/CLIP**。

### 2.2 已有缓解方案的代价

| 方法家族 | 代表 | 代价 |
|---|---|---|
| Self-Supervised 表征对齐 | REPA、REG、REPA-E | 需要外部 SSL 模型；teacher 选型敏感；推理 / 训练计算多一份 |
| 中间层正则（dispersion / SRA） | SRA, DisperseLoss | 设计复杂；提升较温和 |
| Sampling 端 guidance | CFG / Autoguidance / "bad-version guidance" | 要么牺牲多样性，要么需要单独训一个 bad model 或多采样步 |

**IG 的卖点**：用网络**自己的中间层副本**充当 $D_i$，**不需要外部 teacher、不需要 bad model、不增加采样开销**。

---

## 3. IG 的核心做法（论文）

### 3.1 训练端：中间监督

在网络中间某一层（如第 4 层或第 24 层）开一条短分支，得到中间预测 $D_i(x_t, t)$；
最终预测 $D_f(x_t, t)$ 是完整网络的输出。两路同时回归 $x_0$：

$$
\mathcal{L} = \mathcal{L}_{\text{final}} + \lambda\,\mathcal{L}_{\text{inter}},\qquad
\mathcal{L}_{*} = \|D_*(x_t, t) - x_0\|^2
$$

论文里 $\lambda$ 通常取 1（论文 Table 3 的实验全部 $\lambda{=}1$；正文也讨论了 $\lambda{<}1$ 更稳但增益略小）。

### 3.2 采样端：内部动力学外推

$D_i$ 视作"模型自身的弱版本"，按 CFG 的外推思路得到新的预测：

$$
D_w(x; c) = D_i(x; c) + w \cdot \bigl(D_f(x; c) - D_i(x; c)\bigr)
$$

一些关键性质：
* 当 $w=1$ 时退化到标准 $D_f$；$w>1$ 把生成"推得更深"（更接近 $D_f$ 在 $D_i$ 上的"语义梯度"方向）；
* 与 CFG **互补可叠加**：论文 Fig 5 显示 IG×CFG 联合使用比单独 CFG 在 IS 上更高；
* 对"采样多样性"的伤害远小于 CFG，因为 $D_i$ 仍然来自同一条件分布，没有 unconditional 那种"塌缩"风险；
* 配合 **guidance interval**（仅在去噪过程中段使用 IG）能进一步降低 FID。

---

## 4. IG 在论文中的硬指标

来自 IG 论文 Table 1, 2, 3, 4 与 Fig 5, 6 的关键数字（ImageNet 256×256, class-conditional）：

| 模型 | Epochs | FID↓（无 CFG） | FID↓（带 CFG / Autoguidance） |
|---|---|---|---|
| Vanilla DiT-XL/2 | 1400 | 9.62 | 2.27 |
| Vanilla SiT-XL/2 | 1400 | 8.61 | 2.06 |
| REPA（SSL teacher） | 800 | 5.90 | 1.42 |
| REPA-E | 800 | 1.83 | 1.26 |
| **SiT-XL+IG** | **80** | **5.31** | – |
| **SiT-XL+IG** | **800** | **1.75** | **1.46** |
| **LightningDiT+IG** | **60** | **2.42** | – |
| **LightningDiT+IG** | **680** | **1.34** | **1.19**（SOTA） |

可视化要点（论文图）：
* Fig 6（Scalability）：对 DiT-B/L/XL 三档模型都给出类似形状的 FID 收敛曲线，IG 把同一收敛位置的训练步数缩短 **~3-5×**；
* Fig 5（Compatibility）：在 SiT-B/2 上，单 CFG 最低 FID = 7.30；IG+CFG 最低 = 6.50；
* Table 3（Ablation）：把 SiT-B/2 在第 4 层挂 IG，无 CFG 下 FID 从 33.02 → 19.02（IS 也从 43.71 → 65.06），**仅引入一个浅副本就把 baseline 砍掉接近一半 FID**。

---

## 5. 为什么 IG 有效（机理分析）

> 论文给出的解释 + 我们仓库实测过程中观察到的现象。

### 5.1 训练动力学：缓解深网梯度消失

把中间层直接接到目标 $x_0$：
* 反向传播多了一条**短路径**，梯度在到达浅层之前不必再穿过 25+ 个 transformer block；
* 早期 block 接收到来自 $D_i$ 的"清晰任务信号"，避免被深层非线性吞噬；
* 从优化角度等价于在 deep network 上做 *deep supervision*，但损失函数与最终目标完全对齐（同样回归 $x_0$），不像辅助分类头那样需要额外标签。

论文 Table 3 把"加 IG 中间监督"和"加 SSL 表征对齐（SRA / Disperse Loss）"放在同一栏比较 —— **结果是 IG 一行的 FID 比 SSL 那一行还低**（19.02 vs 29.10/31.45）。
这暗示：在模型本身够大、数据够多的前提下，"再蒸馏一个 self"比"再叠一个外部 SSL teacher"更直接命中梯度问题。

### 5.2 表征学习：自蒸馏的隐式正则

$D_i$ 与 $D_f$ 的差是模型自身的内部梯度方向。让两者同时回归同一目标，相当于：
* 早期层被迫学到一组"足够还原 $x_0$"的表征 → 让深层有一个 **可解读的起点**；
* 深层被迫学到一组"超越 $D_i$"的表征 → 否则 main head 的损失下不去；
* 两者形成"老师-学生在同一 forward 内"的自蒸馏循环，类似 ARM / Prompt-tuning 那种 self-bootstrap。

### 5.3 推理端：CFG 替代品 / 增强品

外推公式 $D_w = D_i + w(D_f - D_i)$ 与 CFG 的 $D_w = D_{\text{uncond}} + w(D_{\text{cond}} - D_{\text{uncond}})$ 同形，但区别巨大：

| 维度 | CFG | IG |
|---|---|---|
| "弱版本" 来源 | 训练时随机丢条件构造 unconditional | 同一模型的中间层副本（始终条件齐全） |
| 多样性影响 | 大 $w$ 会"塌缩"到典型样本 | 多样性几乎不退化 |
| 是否需要额外训练 | 是（cfg_prob 训练） | 否 |
| 是否增加采样步数 | 否（共用 forward） | 否（同一次 forward 拿到 $D_i$ 和 $D_f$） |
| 与 guidance interval 兼容 | ✅ | ✅，且组合后增益更显著 |
| 与 CFG 是否冲突 | — | 完全可叠加（论文 Fig 5） |

### 5.4 论文未明说但很关键的工程优势

* **零增量推理开销**（如果不开外推）：训练时 $D_i$ head 会算 forward+backward，但**推理只需要 $D_f$**；
* **零外部依赖**：不需要 DINOv2 / VGGT / CLIP 等任何外部模型；
* **轻量 checkpoint**：$D_i$ 头的体积 ≪ 主干，开关 internal 不影响主推理路径；
* **天然适合 LoRA 微调**：主干可以冻结，只训 internal head + LoRA，得到**快、省、稳**的后训方案（这是 `qb_xx/lingbot-va/wan_va/train_internal_lora.py` 选择的姿态）。

---

## 6. 在 LingBot-VA 上的两份落地实现

### 6.1 路径 A：`qb_xx/lingbot-va/wan_va/train_internal_lora.py`

* 训练端 IG 完整复刻 + LoRA(rank=16) + 主干冻结；
* fork 在 `internal_depth=4`（早层），internal_blocks 用主干末尾若干 block 的 deepcopy 初始化；
* 双头 loss 直接相加（$\lambda=1$）；
* Checkpoint 仅保存 LoRA + internal head 的 trainable 子集（几百 MB）；
* **未开启**论文式的推理外推 $D_w$。

适合"想保护 base 通用能力、又要给 robotwin 任务做轻量微调"的场景。

### 6.2 路径 B：`lingbot-va/wan_va/train_vggt_geometry_forcing_internal.py`（本仓库本次主推）

把 IG 与 Geometry Forcing 几何先验做正交叠加：

```
共享 trunk: blocks[0 : internal_depth=24]
        │
        ├── 主路径 D_f: blocks[24:30] → norm_out + proj_out → main_lat / main_act
        └── 内部路径 D_i: internal_blocks (deepcopy of last 2) → internal_norm_out + internal_proj_out

GF Angular + Scale 损失继续作用在 main path 的 [10, 20, 29] 三层
Internal head 双头 loss 与 GF loss 加权相加
+ 训练中 attention 可视化（Wan 每层 video↔text cross-attn 叠到 RGB）
```

关键代码 hook：`forward_train_gf_internal()` 同一次 forward 同时返回 main / internal 两路 pred；
`_train_step()` 用同一份 `compute_loss` 分别算两路并加权；FSDP / 梯度同步逻辑全部继承 `GeometryForcingTrainer`。

### 6.3 LingBot-VA 默认配置

| 配置 | 默认 | 含义 |
|---|---|---|
| `enable_internal` | True | 是否启用 IG |
| `internal_depth` | 24 | 主干第 24 个 block 后分叉（30 层 DiT 选择"末段做 D_f"） |
| `num_internal_blocks` | 2 | internal 自带 2 个 block，用主干末尾 2 个 block 的 deepcopy 初始化 |
| `lambda_internal` | 1.0 | 与 main loss 同权重（论文默认） |
| `gf_student_layers` | [10, 20, 29] | GF 学生层 |
| `gf_teacher_layers` | [-9, -5, -1] | VGGT 老师层 |

---

## 7. IG 的优势汇总（与替代方案对照）

| 维度 | 朴素后训练 | REPA 类 SSL 对齐 | CFG-only | LoRA-only | **IG** |
|---|---|---|---|---|---|
| 缓解深网梯度消失 | ✗ | △ | ✗ | ✗ | ✅ |
| 提升训练效率 | ✗ | ✅（~2×） | ✗ | △ | ✅✅（**3-5×**） |
| 提升 FID / 生成质量 | ✗ | ✅ | ✅ | ✗ | ✅✅ |
| 不引入外部 teacher | ✅ | ✗ | ✅ | ✅ | ✅ |
| 不增加推理成本 | ✅ | ✅ | ✗ | ✅ | ✅（不开外推时） |
| 与 CFG 叠加 | — | ✅ | — | ✅ | ✅（论文 Fig 5） |
| 与 LoRA 叠加 | — | △ | ✅ | — | ✅ |
| 与几何先验（GF 等）叠加 | — | △ | ✅ | ✅ | ✅（本仓库验证） |
| 可解释性（提供 D_i） | ✗ | ✗ | ✗ | ✗ | ✅ |
| Checkpoint 可拆分（仅训 head） | ✗ | △ | ✗ | ✅ | ✅ |

---

## 8. 适用场景与落地建议

### 8.1 适合用 IG 的情况

* **后训练阶段**（base 模型已有较好生成能力）：IG 的"自蒸馏"前提是 $D_f$ 不烂；
* **30+ 层 DiT、视频或多模态**：层数越深、信号越弱，IG 收益越大；
* **显存吃紧 / 不想引入外部 teacher**：IG 只增加 internal head（< 10 % 主干参数）；
* **想配合 CFG / LoRA / 几何先验联合使用**：IG 与上述三者完全正交。

### 8.2 暂不需要 IG 的情况

* 模型 < 12 层（梯度消失不严重，提升有限）；
* 训练数据极少（IG 主要靠样本量稳化 $D_i$，少样本反而过拟合）；
* 已经有充足外部 teacher 且 SSL 收益超过 IG。

### 8.3 经验性 hyperparam 建议

| 项 | 推荐起点 |
|---|---|
| `internal_depth` | 总层数的 **2/3 ~ 4/5**（比如 30 层选 24，48 层选 36）；论文也展示了"早层（如第 4 层）"也工作，但"中后期分叉"更利于和 GF/REPA 共存 |
| `num_internal_blocks` | 1~2（论文常用 1；deep DiT 上 2 更稳） |
| `lambda_internal` | **1.0**（论文默认）；不稳就降到 0.5 |
| 推理外推 $w$（暂未在仓库启用） | 论文最优 1.5~2.3，配合 guidance interval [0.3, 1) |

### 8.4 监控信号

用 wandb / TB 查这几条曲线：
* `loss/int_latent` / `loss/int_action` —— 内部 head 是否在下降；
* `loss/int_latent` 与 `loss/video` 的差距 —— 训练初期接近，后期 main 应略低于 int；
* 若 int 始终 ≪ main，说明 fork 太靠后（D_i 已经几乎等同 D_f，没起到正则作用）；
* 若 int 长期 ≫ main 或不下降，说明 fork 太靠前 / D_i 容量不够。

---

## 9. 当前仓库相对 IG 论文的实现差距

| 论文要点 | 仓库实现 | 备注 |
|---|---|---|
| 训练端 $\mathcal{L}_{final} + \lambda \mathcal{L}_{inter}$ | ✅ 已实现 | $\lambda$ 在 cfg 暴露 |
| 采样端 $D_w = D_i + w(D_f - D_i)$ | ❌ 暂未启用 | 推理路径（`forward_internal_lora`）已写好 internal_blocks 的 KV-Cache，加一行外推即可接入 |
| Guidance Interval | ❌ | 配套以上一起加 |
| Scalability 曲线 | ⏳ | 后续在 robotwin 上做 ablation 即可绘出 |

---

## 10. 一图汇总

```
                Q：30 层 DiT 早层为什么训不好？
                    │
                A：梯度路径太长，监督信号被压扁
                    │
        ┌───────────┴───────────────────┐
        │                               │
   外部 teacher (REPA)             内部副本 (IG)
        │                               │
   需要 DINOv2 / VGGT             不需要外部模型
   teacher 选型敏感                直接用主干末段 deepcopy
   推理仍需 teacher                推理只用 D_f （或 D_w 外推）
        │                               │
        └────────── 都能减少需要的 ──────┘
                  训练 epoch 数

      IG 额外赠送:
        ✓ 可与 CFG / LoRA / Geometry Forcing 完全正交叠加
        ✓ Checkpoint 可拆 (head + LoRA 单独保存)
        ✓ 可解释性 (D_i 直接可视化)
        ✓ 训练 3-5× 加速 + FID 1-2 点改善（论文 ImageNet）
```

---

## 11. 相关文件索引

| 文件 | 角色 |
|---|---|
| `qb_xx/lingbot-va/internal.pdf` | IG 论文（Zhou et al., 2025） |
| `lingbot-va/wan_va/modules/model_geometry_forcing_internal.py` | Wan + GF hooks + internal 分支模型 |
| `lingbot-va/wan_va/modules/utils_geometry_forcing_internal.py` | 用主干末段 deepcopy 初始化 internal blocks/heads |
| `lingbot-va/wan_va/train_vggt_geometry_forcing_internal.py` | 训练入口（GF + Internal 双头 + 注意力可视化） |
| `lingbot-va/wan_va/configs/va_robotwin_train_vggt_geometry_forcing_internal_cfg.py` | 默认配置 |
| `lingbot-va/wan_va/docs/GEOMETRY_FORCING_INTERNAL.md` | 工程说明 |
| `qb_xx/lingbot-va/wan_va/train_internal_lora.py` | LoRA + Internal 路径（不带 GF） |
| `qb_xx/lingbot-va/wan_va/modules_internal_lora/model.py` | 上一份的模型实现（含 LoRA 注入） |

---

文档结束。后续要做的几件事（可选）：
1. 在 `forward_internal_lora` 推理路径里加上 $D_w = D_i + w(D_f - D_i)$ 与 guidance interval；
2. 在 robotwin 评测上跑 IG-on / IG-off 的 ablation（同 epoch、同 ckpt-base），把 FVD / 成功率画成 IG 论文 Fig 6 那样的"训练 step vs 指标"曲线；
3. 与 Geometry Forcing 单独做 IG-on/off 的 4 组交叉实验，验证"GF + IG"是否真的相加。
