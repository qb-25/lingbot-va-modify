# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Wan-VA 视频-动作扩散 Transformer 主模型实现。

本文件提供：

* ``FlexAttnFunc``         —— 训练时使用 PyTorch 2.5+ 的 ``flex_attention``，配合
                              自定义 BlockMask（causal / window / clean-noise 关系等）。
* ``custom_sdpa``           —— 推理时使用的标准 SDPA 包装。
* ``WanTimeTextImageEmbedding`` —— 时间步 + 文本嵌入（沿用 Wan 原始组件）。
* ``WanRotaryPosEmbed``     —— 3D 旋转位置编码（按帧/高/宽三轴拆分维度）。
* ``WanAttention``          —— 单层注意力，含 KV-Cache 槽位管理（推理用）。
* ``WanTransformerBlock``   —— Self-Attn + Cross-Attn(text) + FFN 三段式 DiT block。
* ``WanTransformer3DModel`` —— 顶层 DiT，同时建模视频 latent 与动作两路 token。

设计要点：
* 视频 / 动作 / 文本 走**同一个 Transformer 主干**，但在输入侧用不同 embedder
  （``patch_embedding_mlp`` vs ``action_embedder`` vs ``text_embedder``），
  在 timestep AdaLN 上也分别使用 ``condition_embedder`` / ``condition_embedder_action``，
  形成"双流 MoT"的雏形。
* 训练（``forward_train``）：把 4 段 token —— [noisy video, clean video cond,
  noisy action, clean action cond] —— 一次性 concat 进去，通过 FlexAttention 的
  block mask 实现"自回归 + 因果窗口 + clean↔noisy 关系"等多种约束。
* 推理（``forward``）：分别按 video / action 路调用，借助 ``KV-Cache`` 与
  ``cache_name`` 区分 CFG 的 conditional / unconditional 两套缓存。
"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

# 优先用 FlashAttention v2/v3 的官方接口；老环境回退到旧版包名。
try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    """PyTorch 原生 SDPA 包装。

    本工程内部约定 q/k/v 形状为 ``(B, S, H, D)``（seq 在 head 之前），
    而 ``F.scaled_dot_product_attention`` 期望 ``(B, H, S, D)``，
    因此前后各做一次 ``transpose(1, 2)``。
    """
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

class FlexAttnFunc(nn.Module):
    """训练阶段的 attention 实现（基于 ``torch.nn.attention.flex_attention``）。

    特点：
    * 用 ``BlockMask`` 提前编排"哪些 token 之间允许互相 attend"。
      训练时一次 forward 同时塞入 4 段 token（noisy/clean × video/action），
      因此需要复杂的组合 mask，见下方 ``init_mask`` 与 ``_get_mask_mod``。
    * Class-level 缓存 ``attention_mask`` / ``cross_attention_mask``：
      每个 train step 只调一次 ``init_mask`` 编译 BlockMask，所有 block 复用，
      避免每层都重新 compile。
    """
    # ---- 类级别共享对象 ----
    # `flex_attention` 自带较高编译开销，使用 torch.compile 包一层缓存。
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True,
    )
    # `create_block_mask` 同样是热点，dynamic=True 让不同 seq_len 复用图。
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    # 当前 step 的 self-attention BlockMask，由 init_mask 写入。
    attention_mask: ClassVar[BlockMask] = None
    # 当前 step 的 cross-attention（video↔text）BlockMask。
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        # is_cross=True 表示这层是 video↔text 的 cross-attn，使用 cross_attention_mask。
        self.is_cross = is_cross

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        # 进入时形状: (1, S, H, D) —— 即 batch 已经被外层 flatten 到 1。
        # flex_attention 期望 (B, H, S, D)，因此把 (S, H) 维度互换。
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        # 强制 q/k/v 在 fp16 / bf16 下做 attention，否则 triton kernel 不支持。
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        # 三者必须 dtype 完全一致（避免 fp16 vs bf16 混用导致重新编译）。
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        # 选择当前 step 已编排好的 BlockMask（self / cross）。
        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        # 调用 triton flex_attention，kernel_options 控制 block tile 大小，
        # 这套数值是针对本工程典型 seq_len 调过的较好默认值。
        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        # 还原 (B, S, H, D) 形状供后续 to_out 使用。
        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        """每个 train step 由 ``forward_train`` 调用一次，构造 self/cross BlockMask。

        token 的总排列（与 ``forward_train`` 中 ``torch.cat`` 顺序对应）：

            [ noisy_video | clean_video_cond | noisy_action | clean_action_cond | PAD ]

        通过给每个 token 打三类标签：
            * ``seq_ids``   —— 属于哪个 batch 样本（同样本内才能 attend）
            * ``frame_ids`` —— 时间索引（用于因果 / 窗口）
            * ``noise_ids`` —— 0=noisy，1=clean（控制 clean↔clean / noise↔clean / noise↔noise）

        由这三类标签组合出最终 BlockMask（见 ``_get_mask_mod``）。
        """
        # 编译期 trick：放宽 inductor 的 "realize op count" 阈值，避免 mask 函数被反复 realize。
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # ---- 1) 每个 token 的 seq_id（=batch index），用于"只能 attend 同 batch 样本"。
        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        # 4 段 concat：noisy_video, clean_video, noisy_action, clean_action
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        # ---- 2) 每个 token 的 frame_id（用于因果 / window 关系）。
        # 视频 patch 经过 patch_size[0] 时间下采样后在 chunk 内可能跨多帧，但本工程
        # 把同 chunk 视为同一"时间点"，所以 //chunk_size 后 *2 留出空位给 action 帧 +1。
        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        # video 帧编号偶数，action 帧编号奇数 → 自然形成 V0,A0,V1,A1,... 的因果链。
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        # ---- 3) 每个 token 的 noise_id：0=noisy（要去噪的），1=clean（条件上下文）。
        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),  # noisy video
                torch.ones_like(latent_frame_id),   # clean video cond
                torch.zeros_like(action_frame_id),  # noisy action
                torch.ones_like(action_frame_id),   # clean action cond
            ]
        )

        # 末尾补 padded_length 个无效槽位（标签全 -1，被 seq_mask 排除掉）。
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        # 组合出 self-attention BlockMask 并写入 class-level 缓存。
        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        # cross-attn：视频/动作 token ↔ 文本 token；文本上限固定 512 长度。
        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        """Cross-attention 的 mask：query token 只能看到**同一 batch** 的文本。"""
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        """Self-attention 的 mask：组合多种规则得到最终的 BlockMask。

        ``mask_mod`` 是一个返回 bool 的函数，签名固定为 ``(b, h, q_idx, kv_idx)``。
        flex_attention 会对每对 (q, kv) 调用一次它，返回 True 表示允许 attend。
        本函数最终的 mask 形如：

            ((clean→clean ∧ block_causal)
              ∨ (noise→clean ∧ block_causal_exclude_self)
              ∨ (noise→noise ∧ block_self))
            ∧ same_batch
            ∧ |Δframe| <= window_size

        语义：
        * clean 上下文之间允许"因果可见"（>= 当前帧的不能看）。
        * noisy token 可以看历史的 clean（不能看自己当前帧的 clean，避免泄漏 GT）。
        * noisy token 之间只在**同一帧/同一 chunk** 内互相可见。
        * 所有规则再叠加"同 batch + 帧间距 <= window"两个全局约束。
        """
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # 同一个样本内 + 排除 PAD（id=-1）。
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # 标准因果：kv 的帧号 <= q 的帧号。
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # 严格因果：连"当前帧"也排除（noisy→clean 时防止泄漏自身 GT）。
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # 同帧/同 chunk 内的所有 token 互相可见。
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            # 限制 q 与 kv 在时间维上的相对距离不超过 window_size，控制感受野。
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        # —— 三组核心规则 OR 起来 ——
        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))               # clean ↔ clean，因果
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))  # noisy → clean，严格因果
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))                 # noisy ↔ noisy，仅同 chunk
        mask = or_masks(*mask_list)
        # 再叠两个全局约束：同 batch、|Δframe| <= window。
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):
    """时间步 + 文本 token 的统一嵌入器（沿用 Wan 原始组件）。

    Args:
        dim:               主干 token 维度（inner_dim）
        time_freq_dim:     timestep 正弦频率维度（输入 ``Timesteps``）
        time_proj_dim:     timestep 在 AdaLN 调制系数侧的展开维度（=6*dim，对应 6 个 shift/scale/gate）
        text_embed_dim:    输入文本 embedding 维度（来自外部 text encoder）
        pos_embed_seq_len: 兼容字段，本工程未启用
    """

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        # timestep → 正弦特征（Diffusers 内置）。
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        # 正弦特征 → MLP 嵌入到主干维度。
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        # AdaLN 调制系数生成头（1 路 → 6 路 shift/scale/gate）。
        self.time_proj = nn.Linear(dim, time_proj_dim)
        # 文本：text_dim → 主干 dim 的非线性投影。
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        # timestep 形状 (B, L)，L 为该样本所有 token 共享的 timestep 重复次数。
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # 把 dtype 与第一层 Linear 对齐（避免 fp16/bf16/int8 混合导致 cuBLAS 报错）。
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        # temb：用于残差侧调制；timestep_proj：用于 AdaLN 6 路系数。
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    """3D 旋转位置编码（RoPE-3D）。

    将 head_dim 维拆成 3 段，分别承担 (frame, height, width) 三轴的角频率，
    然后在最后一维拼接成复数表示，便于 ``apply_rotary_emb`` 直接乘到 q/k 上。
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 把 head_dim 切成三段：f_dim 略大（吸收余数），h_dim、w_dim 各占 1/3。
        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # 预先算好 1/(theta^(2k/dim)) 三组基频，forward 时与 grid_id 相乘即可。
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))；只取每段维度的前一半（复数对的实/虚配对）。
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        # grid_ids: (1, 3, L)，3 行分别是 (frame_id, h_id, w_id)。
        with torch.no_grad():
            # 每轴 token 的位置 × 该轴的基频 → 该轴的 RoPE 角度。
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            # 转成复数 e^{iθ}：模长固定为 1，相位取 freqs。
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):
    """单层 attention，带 KV-Cache 槽位池（仅 self-attn 使用）。

    设计要点
    --------
    * **KV-Cache 是按 cache_name 索引的 dict**：
      推理阶段 CFG 同时需要 conditional / unconditional 两路缓存，
      `cache_name='pos'` / `'neg'` 分别对应；训练时 `attn_caches=None` 跳过。
    * **slot 复用**：`init_kv_cache` 一次性分配一个固定大小池子
      ``[B, total_tolen, H, D]``，每次写入找空位 (`mask=False`) 写入并标 True；
      满了就把最早的 (`id` 最小) 槽位踢掉。
    * **is_pred 区分 noisy/clean 段**：推理一帧 chunk 内有"中间步 noisy KV"和
      "去噪完成的 clean KV"两类；前者只在本步内有效，所以单独标记，
      `clear_pred_cache` 可以一键清掉它们而保留 clean。
    * cross-attn 不需要 cache（每步 text 都重新喂），构造时 `attn_caches=None`。
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        # 选择实际跑 attention 的 kernel：
        #  * torch     - 推理走 SDPA（兼容性最好）
        #  * flashattn - 推理用官方 FlashAttn 接口（最快）
        #  * flex      - 训练用，配合 BlockMask
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        # cross-attn 时 K/V 的输入维度（来自外部 encoder）可能不同。
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        # Q / K / V 投影（自带 bias）。
        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        # out_proj + dropout（diffusers 风格的 ModuleList）。
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        # Q/K 上加 RMSNorm（PixArt-α 提出的稳定 attention 训练的 trick）。
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        # 仅 self-attn 启用 KV-Cache。
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        """清掉本帧 chunk 内"中间步 noisy KV"，保留 clean 上下文。"""
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        """整片清空：把整个 cache_name 的 KV 池设回 None（重新一轮交互前调用）。"""
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        """分配 KV-Cache 池子。

        ``total_tolen`` = (attn_window/2)*L_per_chunk + (attn_window/2)*A_per_chunk
        够装下 attention 窗口内的 video+action token。
        """
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            # K / V 缓冲区（按 [B, T, H, D] 形状）。
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            # 每个 slot 的写入序号（用于 LRU 淘汰）。
            'id':
            torch.full((total_tolen, ), -1, device=device),
            # slot 是否当前有效。
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            # slot 是否属于"本步预测 noisy"段（可被 clear_pred_cache 单独清除）。
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        """为新的 ``key_size`` 个 token 在缓存里找连续可用槽位。

        若空位不足，按 ``id`` 从小到大踢掉最早写入的若干 slot（FIFO 淘汰）。
        """
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        """生成单调递增的写入序号，用作淘汰策略的"年龄"。"""
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        """把当前 batch 的 K/V 写入缓存池。"""
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        """把刚刚临时写入的槽位置失效（用于 ``update_cache=0`` 的"探查式" forward）。"""
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        """注意力层 forward。

        Args:
            q, k, v:        三路输入 token (B, S, dim)。在 self-attn 中三者相同。
            rotary_emb:     RoPE 复数频率，形状 (1, S, 1, D/2)；cross-attn 时为 None。
            update_cache:   0=不写缓存（只读取已有），1=写入 noisy 段，2=写入 clean 段。
            cache_name:     'pos' / 'neg' 等，区分 CFG 的两路缓存。
        """
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        # ---- Q/K/V 投影 + RMSNorm + 拆 head ----
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                # 把最后两维捏成"复数对"，乘以频率（复数旋转），再展平回去。
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        # ---- 推理：写入 / 读取 KV-Cache ----
        if kv_cache is not None and kv_cache['k'] is not None:
            # 把当前 batch 的 K/V 写入空闲 slots，并标记其属性（is_pred）。
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            # 取出整池中所有"当前有效"的 K/V，作为本次 attention 的 KV。
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        # update_cache==0：仅"借用一下槽位完成本次计算"，立刻把刚写入的位置撤销。
        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        # 还原 (B, S, inner_dim) 并经 to_out 投回 token 维度。
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):
    """单个 DiT block：AdaLN + Self-Attn → Cross-Attn(text) → FFN。

    AdaLN 调制：把 timestep embedding 投到 6 路 (shift_msa, scale_msa, gate_msa,
    c_shift_msa, c_scale_msa, c_gate_msa)，分别给 self-attn 和 FFN 用。
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention：附带 KV-Cache（推理用）。
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention：query=video/action token，K/V=text token；不缓存。
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        # cross_attn_norm: 是否在 cross-attn 前再做一次 LN（True 更稳）。
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward (gelu-approximate)：标准 PreNorm + FFN。
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # 每个 block 自带一组可学习的 AdaLN scale/shift 基准（与 timestep 系数相加）。
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        # AdaLN：scale_shift_table + temb → 6 路调制系数。
        # temb 形状 (B, L, 6, C)；scale_shift_table 是 (1, 6, C)。
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        # 把第 1 维 (n=6) squeeze 掉 → (B, L, C)。
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention：先 AdaLN(scale,shift) → attn → gate 残差回写。
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention：query=video/action 主干，K/V=text 编码；
        #    不写 KV-Cache（每步 text 都重投影），update_cache 强制 0。
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward：再做一次 AdaLN(c_scale, c_shift) → FFN → c_gate 残差。
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""LingBot-VA 主干：3D 视频 + 动作的扩散 Transformer。

    与原 Wan 视频 DiT 的区别：
    * 输入端：除 video latent 外，新增 ``action_embedder`` 处理 ``action_dim`` 维动作。
    * timestep AdaLN：拥有两套 condition_embedder（``condition_embedder`` /
      ``condition_embedder_action``），video / action 用各自的时间步条件。
    * 输出端：``proj_out`` 预测 video 速度，``action_proj_out`` 预测动作速度（flow matching）。

    使用建议（详见 README "attn_mode 配置"）：
    * 训练   → ``attn_mode='flex'``
    * 推理   → ``attn_mode='torch'`` 或 ``'flashattn'``
    """
    _supports_gradient_checkpointing = True
    # 这些子模块在 layerwise dtype cast（如 bf16/fp16）时**不参与转换**，保持原 dtype。
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding",
                                        "patch_embedding_mlp",
                                        "condition_embedder",
                                        'condition_embedder_action',
                                        "norm"]
    # FSDP 切分时不在 block 内部再切（保证整 block 同时分到一个 rank）。
    _no_split_modules = ["WanTransformerBlock"]
    # 这些层强制保持 fp32，避免 LayerNorm/timestep MLP 的精度损失。
    _keep_in_fp32_modules = ["time_embedder",
                             "scale_shift_table",
                             "scale_shift_table_action",
                             "norm1",
                             'action_norm1',
                             'text_norm1',
                             "norm2",
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    # 加载预训练时忽略这些键（来源于历史版本）。
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    # FSDP 在 enable_apply_recompute 时使用的"重复 block"标识。
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],          # (T,H,W) latent → token 的 patch 大小
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,                # Wan VAE 的 latent 通道数
                 out_channels=48,               # 输出预测速度也是 48 通道
                 action_dim=30,                 # 双臂 + 多 cam 拼接维度
                 text_dim=4096,                 # 文本 encoder 输出维度（如 T5）
                 freq_dim=256,                  # timestep 正弦特征维度
                 ffn_dim=14336,                 # FFN 中间层维度
                 num_layers=30,                 # DiT block 数
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        # 主干 token 维度 = num_heads * head_dim（通常 24*128=3072）。
        inner_dim = num_attention_heads * attention_head_dim
        # 3D RoPE：拆 head_dim 给 (frame, h, w) 三轴。
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        # video latent → token：先按 patch_size 切块再 Linear 投到 inner_dim。
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        # action 直接 Linear 投到 inner_dim。
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        # video 的 timestep + text 嵌入。
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,        # AdaLN 6 路调制系数
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        # action 的 timestep 嵌入：与 video 不共享参数（deepcopy 后两边独立训练）。
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        # 主干 block 堆叠。
        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        # 最后一层 LN（fp32），不带 affine（由 scale_shift_table 给）。
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        # video 输出头：把 token 维投到 (out_channels * 时空 patch) 后还原。
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        # action 输出头：直接投回 action_dim。
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        # 顶层最终的 (shift, scale) 参数表（与 timestep_proj 相加）。
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def clear_cache(self, cache_name):
        """清掉所有 block 中 ``cache_name`` 这套 KV-Cache（reset / 切换样本前调用）。"""
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        """清掉所有 block 中本步的 noisy 预测 KV，保留 clean 上下文。"""
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        """为推理预分配 KV-Cache 池子。

        ``total_tolen`` = 半窗 video token + 半窗 action token，
        覆盖一个 attention 窗口内的所有可见 token 数。
        """
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)

    def _input_embed(self, latents, input_type='latent'):
        """三种输入类型的统一 embed 入口。

        * ``latent`` —— video VAE latent，按 patch_size 切 patch 后过 MLP。
        * ``action`` —— per-frame 动作向量，直接 Linear 投影。
        * ``text``   —— 文本 encoder 输出，过 PixArt 风格的 GeLU 投影。
        """
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        """把 timestep 重复到每个 token，生成 ``temb`` 与 ``timestep_proj``。

        * 如果是 video（action_mode=False），timestep 复制到 (H/p, W/p) 个空间 token；
          action 的"空间"维度恒为 1×1，所以 patch_scale 取 1。
        * 通过 ``condition_embedder`` 或 ``condition_embedder_action`` 走不同参数。
        """
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        # 把 timestep_proj 拆成 6 路（与 block 内 6 个 AdaLN 系数对齐）。
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        """训练 forward：一次性预测视频与动作两路速度。

        与 ``forward`` 的区别在于：训练把 4 段 token 同时拼起来过 transformer，
        通过 FlexAttention 的 BlockMask 实现 noisy↔clean / 因果 / window 等约束；
        推理则按 video / action 顺序两次调用 ``forward``，并使用 KV-Cache。

        ``input_dict`` 期望结构：
            latent_dict = {
              noisy_latents:   (B, C, F, H, W),     # 加噪 video latent
              latent:          (B, C, F, H, W),     # clean video（条件分支）
              text_emb:        (B, L_text, text_dim),
              grid_id:         (B, 3, F*H*W//patches),  # 每 token 的 (frame, h, w) id
              timesteps:       (B, F),                  # 每帧的 σ → diffusion 时间步
              cond_timesteps:  (B, F),                  # 条件分支的时间步（>0 表示部分加噪）
            }
            action_dict 同理（其 latent 形状不切 patch）。
            chunk_size, window_size:  控制 BlockMask 中的 frame_id / window 行为。
        """
        # 视频 / 动作的 noisy/clean 全部转 bf16，便于 FlexAttention 的 triton kernel。
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        # ---- 1) 4 段 token 各自 embed，并 flatten 成单 sequence (B*L → 1 (B L) C) ----
        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        # 顺序：noisy_video | clean_video_cond | noisy_action | clean_action_cond
        # ↑ 这个顺序与 FlexAttnFunc.init_mask 内 noise_ids/frame_ids 的拼接顺序严格对应。
        hidden_states = torch.cat([latent_hidden_states,
                                   condition_latent_hidden_states,
                                   action_hidden_states,
                                   condition_action_hidden_states], dim=1)


        # ---- 2) 拼 grid_id 给 RoPE：4 段共享同一份位置编码（[2,2] 重复） ----
        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        # RoPE 输出 (1, S, 1, D/2) 复数频率（[:, :, None] 是为 head 维留空，attn 内会广播）。
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        # ---- 3) timestep：noisy 段用真 σ，clean 段用 cond_timesteps（通常为 0）。
        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps,
                        latent_dict['noisy_latents'].shape[-2],
                        latent_dict['noisy_latents'].shape[-1],
                        dtype=hidden_states.dtype,
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2],
                        action_dict['noisy_latents'].shape[-1],
                        dtype=hidden_states.dtype,
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        # ---- 4) pad 到 128 的整数倍（FlexAttention 的 block tile 对齐要求） ----
        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        # 记录 4+1 段的长度，最后一段是 PAD，方便 forward 完后 split 还原。
        split_list = [latent_hidden_states.shape[1],
                      condition_latent_hidden_states.shape[1],
                      action_hidden_states.shape[1],
                      condition_action_hidden_states.shape[1],
                      padded_length]

        # ---- 5) 编排本步的 BlockMask（self-attn + cross-attn 各一份） ----
        FlexAttnFunc.init_mask(latent_dict['noisy_latents'].shape,
                               action_dict['noisy_latents'].shape,
                               padded_length,
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               device=hidden_states.device
                               )

        # ---- 6) 主干 30 个 block 串行 forward；训练不写 KV-Cache。 ----
        for block in self.blocks:
            hidden_states = block(hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=False)
        # ---- 7) 顶层 AdaLN(2 路：scale,shift) + LN ----
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        # 按 split_list 把 4+1 段拆回；只有 noisy_video 与 noisy_action 走输出头。
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        # video 输出：proj_out 把 token 维投到 (out_channels * 时空 patch 数)，
        # 再 rearrange 回 (B, F*H*W*patches, out_channels) 形状供 trainer 还原 latent。
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             '1 (b l) (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size), b=batch_size)  #
        # action 输出：每个 token 直接回到 action_dim。
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""推理 forward（也是模型的默认入口）。

        与 ``forward_train`` 的关键差异：
        * **只处理一路 token**（video 或 action，由 ``action_mode`` 选择），
          而不是 4 段一起跑。这样可以利用 KV-Cache 在去噪迭代之间复用上下文。
        * **使用 KV-Cache**（``cache_name``）：CFG 同时维护 ``'pos'`` / ``'neg'``
          两套缓存，外层 server 负责按需切换。
        * **不使用 FlexAttention block mask**：直接走 SDPA / FlashAttn，由 cache
          的"哪些 slot 当前有效"来天然实现因果可见性。

        Args:
            input_dict: 见下注释中的字段。
            update_cache:  0=只读不写、1=写入 noisy 段（pred）、2=写入 clean 段（context）。
            cache_name:    KV-Cache 的命名空间（CFG 区分 'pos'/'neg'）。
            action_mode:   True 走 action 路径（embedder/proj_out 都换一套）。
            train_mode:    若为 True，转发到 forward_train（让外层只暴露一个 forward）。
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb：直接 Linear 投影到主干维度。
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb：先按 patch_size 切 patch 再 MLP。
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        # 文本 token 投影（每一步都重新算，不缓存）。
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        # RoPE：按本路 token 的 grid_id 算频率。
        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        # timestep 复制到每个 token，并选用 video / action 各自的 condition_embedder。
        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        # 主干：每个 block 的 attn1 自带 KV-Cache 控制（按 update_cache / cache_name）。
        for block in self.blocks:
            latent_hidden_states = block(latent_hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=update_cache,
                                         cache_name=cache_name)
        # 顶层 AdaLN(2 路：scale,shift) + LN。
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        # 输出头：action 路 → action_dim；video 路 → 还原回 latent 体积。
        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    # 简单的自检入口：用默认配置实例化一个模型并打印结构，便于人工核对参数量 / 子模块。
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
