# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Loader for the GF + internal trainer.

* Builds ``WanTransformer3DModel`` (the GF+internal variant) from a plain
  Wan checkpoint via ``from_pretrained``.
* When ``enable_internal=True`` and the checkpoint does not contain the new
  ``internal_*`` modules, initialises them from a deepcopy of the main
  branch's last ``num_internal_blocks`` blocks + main heads. This mirrors
  ``qb_xx/lingbot-va/wan_va/modules_internal_lora/utils.py``.
"""
import copy
from typing import Optional

import torch
import torch.nn as nn

from .model_geometry_forcing_internal import WanTransformer3DModel


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    *,
    enable_internal: bool = True,
    internal_depth: Optional[int] = None,
    num_internal_blocks: Optional[int] = None,
):
    kwargs = {"enable_internal": bool(enable_internal)}
    if enable_internal and internal_depth is not None:
        kwargs["internal_depth"] = int(internal_depth)
    if enable_internal and num_internal_blocks is not None:
        kwargs["num_internal_blocks"] = int(num_internal_blocks)

    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )

    # Initialise internal_* from the main branch's tail (only if those
    # tensors are still meta / freshly initialized — which is the typical
    # case when starting from the upstream Wan checkpoint).
    if enable_internal and num_internal_blocks is not None and num_internal_blocks > 0:
        # If the checkpoint had internal_* keys, ``from_pretrained`` already
        # filled them in; the deepcopy-overwrite below is harmless because
        # `_load_pretrained_model` only registers params it has seen.
        # Heuristic: if internal_proj_out.weight is meta or all-zero (un-trained
        # init from torch.empty + normal_), we treat this as fresh and copy.
        try:
            ip = model.internal_proj_out.weight
            need_init = ip.is_meta or torch.allclose(
                ip.detach().float().abs().mean(), torch.tensor(0.0), atol=1.0
            )
        except Exception:
            need_init = True

        if need_init:
            model.internal_proj_out = copy.deepcopy(model.proj_out)
            model.internal_action_proj_out = copy.deepcopy(model.action_proj_out)
            model.internal_scale_shift_table = nn.Parameter(
                model.scale_shift_table.detach().clone()
            )
            model.internal_norm_out = copy.deepcopy(model.norm_out)
            model.internal_blocks = copy.deepcopy(model.blocks[-int(num_internal_blocks):])

    # Materialize any remaining meta tensors (new modules absent from ckpt).
    for module in model.modules():
        for param_name, param in list(module._parameters.items()):
            if param is not None and param.is_meta:
                new_data = torch.empty(param.shape, dtype=torch_dtype, device="cpu")
                nn.init.normal_(new_data, std=0.02)
                module._parameters[param_name] = nn.Parameter(new_data)
        for buf_name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                module._buffers[buf_name] = torch.zeros(
                    buf.shape, dtype=buf.dtype, device="cpu"
                )

    return model.to(torch_device)
