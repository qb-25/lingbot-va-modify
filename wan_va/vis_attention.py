#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline attention visualization for a Wan-VA checkpoint.

Usage::

    python -m wan_va.vis_attention \
        --ckpt /path/to/checkpoint_step_NNN \
        --out  ./attn_vis_offline \
        --layers 0,5,10,15,20,25,29 \
        --frames 0,2,5

This script:

1. Loads the checkpoint with ``modules.utils_geometry_forcing_internal``
   (so the same hidden-layer + cross-attn capture hooks are available).
2. Pulls one mini-batch from the trainer's dataset (or a user-provided
   sample), runs ``forward_train`` once with ``capture_cross_attn=True``,
   and feeds the captured probabilities to
   ``modules.attention_visualizer``.
3. Writes a per-step PNG grid + the raw ``attn_probs`` tensor to disk.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS  # noqa: E402
from distributed.util import init_distributed  # noqa: E402
from modules.attention_visualizer import (  # noqa: E402
    cross_attn_to_video_heatmaps,
    save_layered_attention_grid,
)
from modules.utils_geometry_forcing_internal import (  # noqa: E402
    load_transformer as load_transformer_gf_internal,
)
from modules.utils import load_vae  # noqa: E402

# Trainer used for dataset construction + input prep:
import train_vggt_geometry_forcing_internal as gfi  # noqa: E402


def parse_layer_list(s):
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help="Path to checkpoint root containing ``transformer/``.")
    p.add_argument('--config-name', type=str,
                   default='robotwin_train_vggt_geometry_forcing_internal')
    p.add_argument('--out', type=str, default='./attn_vis_offline')
    p.add_argument('--layers', type=parse_layer_list, default='0,5,10,15,20,25,29')
    p.add_argument('--frames', type=parse_layer_list, default='0,2,5')
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--top-k-tokens', type=int, default=16)
    p.add_argument('--num-samples', type=int, default=1)
    args = p.parse_args()

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    # Force ckpt; vis script never updates weights.
    config.resume_from = args.ckpt

    # Reuse the trainer to build dataset / dtype / vae / utilities. We won't
    # call .train(); we only call its helpers.
    trainer = gfi.GeometryForcingInternalTrainer(config)
    trainer.transformer.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(args.num_samples):
        batch = trainer._get_next_batch()
        batch = trainer.convert_input_format(batch)
        input_dict = trainer._prepare_input_dict(batch)

        # Capture cross-attn at requested layers; do not run alignment heads.
        input_dict_vis = dict(input_dict)
        input_dict_vis['capture_cross_attn'] = True
        input_dict_vis['capture_layers'] = list(args.layers)

        with torch.no_grad():
            output = trainer.transformer(input_dict_vis, train_mode=True)
        attn_probs = output.get('attn_probs', None)
        split_list = output.get('split_list', None)
        if attn_probs is None:
            print(f"[sample {sample_idx}] No attn_probs returned; skipping.")
            continue

        latent_dict = input_dict['latent_dict']
        _, _, F_lat, H_lat, W_lat = latent_dict['noisy_latents'].shape
        F_t = F_lat // trainer.patch_size[0]
        H_t = H_lat // trainer.patch_size[1]
        W_t = W_lat // trainer.patch_size[2]

        token_meta = {'mode': 'content_only', 'top_k': args.top_k_tokens}
        heatmaps = cross_attn_to_video_heatmaps(
            attn_probs=attn_probs,
            split_list=split_list,
            video_grid_thw=(F_t, H_t, W_t),
            token_meta=token_meta,
        )

        gt_clean = latent_dict['latent']
        rgb = trainer._decode_latent_to_pixels(
            gt_clean, list(range(F_t)), enable_grad=False
        )
        rgb = trainer._select_vggt_supervision_pixels(rgb)[0]

        out_png = out_dir / f'sample_{sample_idx:03d}.png'
        save_layered_attention_grid(
            heatmaps=heatmaps,
            rgb_frames=rgb,
            out_path=out_png,
            sample_layers=args.layers,
            sample_frames=args.frames,
            alpha=args.alpha,
        )
        # Also dump raw heatmaps for downstream analysis.
        torch.save(
            {
                'attn_probs': {int(k): v.cpu() for k, v in attn_probs.items()},
                'heatmaps': {int(k): v.cpu() for k, v in heatmaps.items()},
                'split_list': list(split_list),
                'video_grid_thw': (F_t, H_t, W_t),
            },
            out_dir / f'sample_{sample_idx:03d}.pt',
        )
        print(f"[sample {sample_idx}] Wrote {out_png}")


if __name__ == '__main__':
    from utils import init_logger
    init_logger()
    main()
