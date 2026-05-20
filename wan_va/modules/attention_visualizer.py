# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Attention visualizer for Wan-VA cross-attn (V1).

Given the captured cross-attention probabilities from
``model_geometry_forcing_internal.WanTransformer3DModel`` (with
``capture_cross_attn=True``), produce per-layer heatmaps showing
"how each spatial patch attends to the *task-target* tokens in the
prompt".

How the task target is selected
-------------------------------
Three modes for resolving which text tokens count as the "task target":

* ``token_indices``: explicit list[int] of K/V positions (computed offline).
* ``content_only``: keep tokens that are not pad. This averages over the
  whole prompt — coarse but always works.
* ``span``: take a [start, end) range of token indices.

Inputs
------
* ``attn_probs``: dict{block_idx -> Tensor(H, S_q, S_kv)} as returned by
  the model with ``capture_cross_attn=True``.
* ``split_list``: list returned by the model so we can locate the
  ``noisy_video`` slice within S_q.
* ``rgb_frames``: Tensor(B, F, 3, H_pix, W_pix) in [0, 1].
* ``token_meta``: how to resolve task-target indices, see above.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def _heatmap_jet(x_norm: np.ndarray) -> np.ndarray:
    """Apply a JET-like colormap to a 2D array in [0, 1] -> RGB uint8."""
    # Cheap matplotlib-free JET via piecewise linear interpolation.
    x = np.clip(x_norm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _select_task_target_indices(
    attn_probs_layer: torch.Tensor,
    token_meta: Dict,
) -> torch.Tensor:
    """Return a 1D index tensor over S_kv for the chosen task-target tokens."""
    S_kv = attn_probs_layer.shape[-1]
    mode = token_meta.get('mode', 'content_only')
    if mode == 'token_indices':
        idx = torch.as_tensor(token_meta['token_indices'], dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < S_kv)]
    elif mode == 'span':
        s, e = int(token_meta['start']), int(token_meta['end'])
        idx = torch.arange(max(0, s), min(S_kv, e), dtype=torch.long)
    elif mode == 'content_only':
        # Heuristic: keep top-k tokens by *total* attention mass across the
        # video query; this peels off pad and brings actual content tokens.
        total = attn_probs_layer.sum(dim=(0, 1)).cpu()
        k = int(token_meta.get('top_k', 16))
        idx = torch.topk(total, k=min(k, total.numel())).indices.sort().values
    else:
        raise ValueError(f"Unknown token_meta mode: {mode}")
    return idx


def cross_attn_to_video_heatmaps(
    attn_probs: Dict[int, torch.Tensor],
    split_list: Sequence[int],
    video_grid_thw: Sequence[int],   # (F_token, H_token, W_token) for the video portion
    token_meta: Dict,
    head_reduce: str = 'mean',
) -> Dict[int, torch.Tensor]:
    """Convert per-layer cross-attn probs into per-layer 2D heatmaps.

    Args:
        attn_probs:    dict{layer -> Tensor(H, S_q, S_kv)}.
        split_list:    [n_noisy_video, n_clean_video, n_noisy_action,
                       n_clean_action, n_pad].
        video_grid_thw: spatial-temporal token grid of the noisy video segment.
        token_meta:    see _select_task_target_indices.
        head_reduce:   'mean' or 'max' over heads.

    Returns:
        {layer -> Tensor(F_token, H_token, W_token)} heatmaps in [0, 1].
    """
    n_video = int(split_list[0])
    F_t, H_t, W_t = [int(x) for x in video_grid_thw]
    assert n_video == F_t * H_t * W_t, (
        f"Video token count mismatch: {n_video} vs {F_t}*{H_t}*{W_t}"
    )

    out = {}
    for layer, probs in attn_probs.items():
        # probs: (H, S_q, S_kv)
        idx = _select_task_target_indices(probs, token_meta).to(probs.device)
        # take video portion of queries:
        v_probs = probs[:, :n_video, :]
        # accumulate attention mass to chosen text tokens:
        mass = v_probs.index_select(dim=-1, index=idx).sum(dim=-1)  # (H, n_video)
        if head_reduce == 'max':
            agg = mass.amax(dim=0)
        else:
            agg = mass.mean(dim=0)
        # Reshape into (F, H, W) and normalize per layer.
        heat = agg.reshape(F_t, H_t, W_t)
        heat = _normalize(heat)
        out[layer] = heat.detach().float().cpu()
    return out


def overlay_heatmap_on_rgb(
    rgb_uint8: np.ndarray,
    heat: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """rgb_uint8: (H, W, 3) uint8 in [0, 255]; heat: (h, w) float in [0, 1]."""
    H, W, _ = rgb_uint8.shape
    heat_t = torch.from_numpy(heat)[None, None].float()
    heat_t = F.interpolate(heat_t, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
    cmap = _heatmap_jet(heat_t.numpy())
    out = (1.0 - alpha) * rgb_uint8.astype(np.float32) + alpha * cmap.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_layered_attention_grid(
    heatmaps: Dict[int, torch.Tensor],
    rgb_frames: torch.Tensor,
    out_path: Union[str, os.PathLike],
    sample_layers: Optional[Sequence[int]] = None,
    sample_frames: Optional[Sequence[int]] = None,
    alpha: float = 0.5,
):
    """Save a grid PNG: rows = sampled layers, cols = sampled frames.

    Args:
        heatmaps:      {layer -> (F_token, H_token, W_token)} from
                       cross_attn_to_video_heatmaps.
        rgb_frames:    Tensor(F, 3, H_pix, W_pix) in [0, 1] (single sample).
        out_path:      destination .png.
        sample_layers / sample_frames: optional subsampling.
    """
    layers = sorted(heatmaps.keys())
    if sample_layers is not None:
        layers = [l for l in layers if l in set(int(s) for s in sample_layers)]
    F_token = next(iter(heatmaps.values())).shape[0]
    if sample_frames is None:
        sample_frames = list(range(F_token))
    sample_frames = [int(f) for f in sample_frames if 0 <= f < F_token]

    rgb = (rgb_frames.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    F_pix = rgb.shape[0]
    # Map token-frame index -> pixel-frame index (assume 1:1 along F_token if equal,
    # else nearest-neighbor).
    if F_pix == F_token:
        frame_map = lambda fi: fi
    else:
        frame_map = lambda fi: int(round(fi * (F_pix - 1) / max(1, F_token - 1)))

    rows = []
    for layer in layers:
        heat_thw = heatmaps[layer]  # (F_token, H_token, W_token)
        cells = []
        for fi in sample_frames:
            rgb_frame = rgb[frame_map(fi)]
            heat = heat_thw[fi].numpy()
            cell = overlay_heatmap_on_rgb(rgb_frame, heat, alpha=alpha)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate(rows, axis=0)

    # Lazy-import PIL so non-vis deps still import cleanly.
    from PIL import Image, ImageDraw

    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)
    cell_h = grid.shape[0] // max(1, len(layers))
    cell_w = grid.shape[1] // max(1, len(sample_frames))
    for ri, layer in enumerate(layers):
        draw.text((4, ri * cell_h + 4), f"L{layer}", fill=(255, 255, 255))
    for ci, fi in enumerate(sample_frames):
        draw.text((ci * cell_w + 4, grid.shape[0] - 14), f"f{fi}", fill=(255, 255, 255))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return str(out_path)
