# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Utilities to read Wan VAE *encoder* features (pre-quant_conv), for alignment losses."""

import torch

from .utils import WanVAEStreamingWrapper, patchify


def probe_encoder_out_channels(vae, stream: WanVAEStreamingWrapper, device, dtype, height: int, width: int) -> int:
    """Run a tiny forward to infer encoder output channel dim."""
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 1, height, width, device=device, dtype=dtype)
        h = encode_encoder_hidden_pre_quant(vae, stream, dummy, enable_grad=False)
    return int(h.shape[1])


def encode_encoder_hidden_pre_quant(
    vae,
    stream: WanVAEStreamingWrapper,
    pixel_frames_bf1hw: torch.Tensor,
    enable_grad: bool = False,
) -> torch.Tensor:
    """Encoder feature map before `quant_conv` (not the sampled latent).

    Args:
        vae: AutoencoderKLWan
        stream: WanVAEStreamingWrapper bound to the same vae instance
        pixel_frames_bf1hw: (N, 3, 1, H, W) in [0, 1]

    Returns:
        Tensor of shape (N, C, Hs, Ws) with spatial dims matching encoder output.
    """
    x = pixel_frames_bf1hw * 2.0 - 1.0
    if hasattr(vae.config, "patch_size") and vae.config.patch_size is not None:
        x = patchify(x, vae.config.patch_size)

    stream.clear_cache()
    ctx = torch.enable_grad if enable_grad else torch.no_grad
    with ctx():
        out = stream.encoder(x, feat_cache=stream.feat_cache, feat_idx=[0])
        if out.dim() != 5:
            raise ValueError(f"Unexpected encoder output dim={out.dim()}, shape={tuple(out.shape)}")
        tdim = out.shape[2]
        if tdim == 1:
            out = out.squeeze(2)
        else:
            out = out.mean(dim=2)
    return out
