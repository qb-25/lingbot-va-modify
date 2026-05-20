# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Cross-stream alignment for the Wan-VA double-stream DiT.

At a chosen DiT block, pool the *video* tokens to a per-frame embedding
``(B, F, D_v)`` and the *action* tokens to a per-frame embedding
``(B, F, D_a)`` (both 1:1 along F under the RoboTwin / `action_per_frame`
data layout). Project both into a shared space with two independent
2-layer MLPs (SimCLR / CLIP recipe), L2-normalize, and compute a
symmetric InfoNCE objective.

Two crucial details for it to actually work on Wan-VA defaults
(``batch_size=1``, ``world_size=8``, ``gradient_accumulation_steps=8``):

1. **All-gather across ranks** before forming the contrastive matrix —
   without this the effective batch is 1×F and the InfoNCE collapses
   to a trivial solution.
2. **Sigma-aware gate** — only count the loss when both streams are
   close enough to clean (``(1 - σ_v)·(1 - σ_a) > threshold``);
   defaults to a soft weight rather than a hard mask.

The module exposes:

* ``CrossStreamProjector``      — two-MLP head, save/load via state_dict.
* ``compute_cross_stream_loss`` — standalone function (no FSDP), returns
  ``(loss, metrics_dict)`` ready for direct backward / wandb logging.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """all-gather on dim 0 while keeping local gradients.

    The standard ``dist.all_gather`` does NOT propagate gradients to the
    other ranks' tensors (which is correct — they live on different GPUs)
    but we DO need the local rank's gradient to flow through the InfoNCE
    matrix. The trick: place a no-grad copy of every other rank into the
    output, but write our own *grad-tracking* tensor at our slot.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world_size = dist.get_world_size()
    if world_size == 1:
        return x
    rank = dist.get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.contiguous())
    # Restore local-rank slot to the grad-tracking original.
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


# ----------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------
def pool_video_tokens(
    h_v: torch.Tensor,
    F_t: int,
    H_t: int,
    W_t: int,
) -> torch.Tensor:
    """``(B, F_t*H_t*W_t, D)`` -> ``(B, F_t, D)`` via spatial mean + LN."""
    B, L, D = h_v.shape
    assert L == F_t * H_t * W_t, (
        f"Video token count mismatch: {L} vs {F_t}*{H_t}*{W_t}"
    )
    x = h_v.reshape(B, F_t, H_t, W_t, D).mean(dim=(2, 3))  # (B, F_t, D)
    return _ln(x)


def pool_action_tokens(
    h_a: torch.Tensor,
    F_t: int,
    K: int,
) -> torch.Tensor:
    """``(B, F_t*K, D)`` -> ``(B, F_t, D)`` via per-step mean + LN."""
    B, L, D = h_a.shape
    assert L == F_t * K, f"Action token count mismatch: {L} vs {F_t}*{K}"
    x = h_a.reshape(B, F_t, K, D).mean(dim=2)  # (B, F_t, D)
    return _ln(x)


# ----------------------------------------------------------------------
# Projector
# ----------------------------------------------------------------------
class CrossStreamProjector(nn.Module):
    """Two-MLP head: ``video / action`` -> shared L2-normalized space.

    Following SimCLR / CLIP, the contrastive objective is computed on the
    L2-normalized projector outputs (NOT the raw hidden states), which
    consistently performs better.
    """

    def __init__(
        self,
        d_v: int,
        d_a: int,
        d_proj: int = 256,
        hidden: int = 1024,
    ):
        super().__init__()
        self.proj_v = nn.Sequential(
            nn.Linear(d_v, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_proj),
        )
        self.proj_a = nn.Sequential(
            nn.Linear(d_a, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_proj),
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_v = F.normalize(self.proj_v(h_v.float()), dim=-1)
        z_a = F.normalize(self.proj_a(h_a.float()), dim=-1)
        return z_v, z_a


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def compute_cross_stream_loss(
    h_v: torch.Tensor,         # (B, F_t, D_v)  pooled video features
    h_a: torch.Tensor,         # (B, F_t, D_a)  pooled action features
    projector: CrossStreamProjector,
    sigma_v: torch.Tensor,     # (B, F_t)  in [0, 1]
    sigma_a: torch.Tensor,     # (B, F_t)  in [0, 1]
    *,
    pos_mode: str = 'window',
    window: int = 2,
    tau: float = 0.15,
    use_all_gather: bool = True,
    sigma_threshold: float = 0.5,
    sigma_soft: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Symmetric InfoNCE between (pooled) video and action embeddings.

    Args:
        pos_mode: positive-pair convention.
            * ``'frame'``      — only ``(b, f)`` itself counts as positive
                                 (strictest, CLIP-like).
            * ``'window'``     — ``(b, f')`` with ``|f' - f| <= window``
                                 counts as positive (default; balances
                                 strictness with smoothness).
            * ``'trajectory'`` — all frames within the same ``b`` count
                                 as positive (loosest; recommended when
                                 video frames change slowly).
        window: only used by ``pos_mode='window'``.
        tau: temperature. Default 0.15 (NOT 0.07) because effective batch
             on Wan-VA defaults is small even after all-gather.
        use_all_gather: gather z_v / z_a across DDP ranks before InfoNCE.
        sigma_threshold: only used when ``sigma_soft=False``.
        sigma_soft: if True, the loss is multiplied by
            ``mean((1-σ_v)(1-σ_a))`` (soft attenuation); if False, by
            ``mean(σ_v < th) * mean(σ_a < th)`` (hard gate).

    Returns:
        (loss, metrics) where ``metrics`` is a dict of scalar floats:
            * ``cross_stream_logits_diag`` — diag mean of the sim matrix
            * ``cross_stream_pos_neg_gap`` — mean(pos) - mean(neg)
            * ``cross_stream_topk1_acc``   — fraction of anchors whose
                                             argmax is a positive
            * ``cross_stream_gate``        — gate scalar applied to loss
    """
    z_v, z_a = projector(h_v, h_a)          # (B, F_t, d)

    B, F_t, d = z_v.shape

    # Flatten (B, F_t) -> (N,)
    z_v_flat = z_v.reshape(B * F_t, d)
    z_a_flat = z_a.reshape(B * F_t, d)

    # Per-anchor (b, f) ids on the local rank.
    idx_b = torch.arange(B, device=z_v.device).unsqueeze(1).expand(B, F_t).reshape(-1)
    idx_f = torch.arange(F_t, device=z_v.device).unsqueeze(0).expand(B, F_t).reshape(-1)

    if use_all_gather and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        z_v_gathered = _all_gather_with_grad(z_v_flat)  # (W*N_local, d)
        z_a_gathered = _all_gather_with_grad(z_a_flat)
        # Reconstruct (b, f) ids in the gathered layout. Each rank
        # contributes B*F_t consecutive rows; we offset b by rank*B so
        # different ranks get distinct trajectory ids.
        idx_b_kv = torch.arange(world_size * B, device=z_v.device).unsqueeze(1).expand(-1, F_t).reshape(-1)
        idx_f_kv = torch.arange(F_t, device=z_v.device).unsqueeze(0).expand(world_size * B, F_t).reshape(-1)
        # Anchor side stays local; offset its b by my rank.
        idx_b_q = idx_b + rank * B
        idx_f_q = idx_f
    else:
        z_v_gathered = z_v_flat
        z_a_gathered = z_a_flat
        idx_b_q = idx_b
        idx_f_q = idx_f
        idx_b_kv = idx_b
        idx_f_kv = idx_f

    Nq = z_a_flat.shape[0]
    Nkv = z_v_gathered.shape[0]

    # Positive mask of shape (Nq, Nkv).
    same_b = idx_b_q.unsqueeze(1) == idx_b_kv.unsqueeze(0)
    if pos_mode == 'frame':
        pos_mask = same_b & (idx_f_q.unsqueeze(1) == idx_f_kv.unsqueeze(0))
    elif pos_mode == 'window':
        pos_mask = same_b & (
            (idx_f_q.unsqueeze(1) - idx_f_kv.unsqueeze(0)).abs() <= int(window)
        )
    elif pos_mode == 'trajectory':
        pos_mask = same_b
    else:
        raise ValueError(f"Unknown pos_mode={pos_mode!r}")

    # InfoNCE both ways.
    logits_a2v = (z_a_flat @ z_v_gathered.t()) / tau   # (Nq, Nkv)
    logits_v2a = (z_v_flat @ z_a_gathered.t()) / tau   # (Nq, Nkv)

    log_prob_a2v = F.log_softmax(logits_a2v, dim=1)
    log_prob_v2a = F.log_softmax(logits_v2a, dim=1)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)
    loss_a2v = -(log_prob_a2v.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count).mean()
    loss_v2a = -(log_prob_v2a.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count).mean()
    loss = 0.5 * (loss_a2v + loss_v2a)

    # Sigma-aware gating.
    sv = sigma_v.float().reshape(-1)
    sa = sigma_a.float().reshape(-1)
    if sigma_soft:
        gate = ((1.0 - sv).clamp_min(0.0) * (1.0 - sa).clamp_min(0.0)).mean()
        gate = gate.clamp_min(0.05)
    else:
        gate = (sv < sigma_threshold).float().mean() * (sa < sigma_threshold).float().mean()
        gate = gate.clamp_min(0.05)

    loss = loss * gate

    # Cheap monitoring scalars (do NOT participate in backward).
    with torch.no_grad():
        diag_logits = logits_a2v.diagonal()  # length min(Nq, Nkv)
        pos_mean = logits_a2v[pos_mask].mean() if pos_mask.any() else logits_a2v.new_zeros(())
        neg_mean = logits_a2v[~pos_mask].mean() if (~pos_mask).any() else logits_a2v.new_zeros(())
        topk1 = logits_a2v.argmax(dim=1)
        topk1_correct = pos_mask.gather(1, topk1.unsqueeze(1)).float().mean()

    metrics = {
        'cross_stream_logits_diag': float(diag_logits.float().mean().item()),
        'cross_stream_pos_neg_gap': float((pos_mean - neg_mean).item()),
        'cross_stream_topk1_acc': float(topk1_correct.item()),
        'cross_stream_gate': float(gate.detach().item()),
    }
    return loss, metrics
