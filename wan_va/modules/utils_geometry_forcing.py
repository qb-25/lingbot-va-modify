# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Loader for the Geometry Forcing trainer.

Geometry Forcing reuses the same `model_spatial_forcing.WanTransformer3DModel`
backbone as spatial forcing, because that model already exposes intermediate
video-token hidden states via `return_hidden_layers=True / align_layer_idx=[...]`,
which is exactly what the per-layer Angular / Scale alignment objectives need.
"""
from .model_spatial_forcing import WanTransformer3DModel


def load_transformer(transformer_path, torch_dtype, torch_device):
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )
    return model.to(torch_device)
