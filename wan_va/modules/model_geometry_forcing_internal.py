# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Wan transformer with **Geometry Forcing alignment hooks** (multi-layer
intermediate hidden capture) PLUS an **Internal head** (Internal Guidance,
Zhou et al., 2025) that forks from a specified depth in the backbone.

Forking diagram (forward_train_gf_internal)::

                  noisy/clean (video+action) tokens (concat + pad)
                                       │
                                       ▼
                        ┌── blocks[0 : internal_depth] ──┐   ← shared trunk
                        │  (GF hidden capture happens    │
                        │   for any align_layer_idx in   │
                        │   this range)                  │
                        └────────────────┬───────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                              ▼                     ▼
                  blocks[internal_depth :]   internal_blocks
                       (main path D_f)        (independent D_i,
                              │                deepcopy of last
                              │                ``num_internal_blocks``
                              │                main blocks)
                              ▼                     ▼
                     norm_out + proj_out    internal_norm_out
                     action_proj_out        + internal_proj_out
                                            + internal_action_proj_out
                              │                     │
                              ▼                     ▼
                       (D_f outputs)          (D_i outputs)

The model also exposes ``return_hidden_layers / align_layer_idx`` (same
contract as ``model_spatial_forcing``) so the GF trainer can capture
intermediate tokens for Angular + Scale alignment.

For attention visualization (V1: video↔text cross-attn), set
``capture_cross_attn=True`` and ``capture_layers=[...]`` in input_dict;
the forward returns the captured probability tensors per layer. This path
is only enabled in eval mode and runs a slower manual SDPA-with-weights
to recover attention probabilities (the default ``flex_attention`` /
``flash_attn`` kernels do not return them).
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .model import FlexAttnFunc
from .model import WanTransformer3DModel as BaseWanTransformer3DModel
from .model import WanTransformerBlock


class WanTransformer3DModel(BaseWanTransformer3DModel):
    """Wan DiT extended with GF-alignment hooks + internal head.

    Extra constructor kwargs (forwarded via from_pretrained / __init__):
        enable_internal       (bool) : create internal branch (default False)
        internal_depth        (int)  : fork the trunk after this many blocks
        num_internal_blocks   (int)  : how many extra blocks inside the
                                       internal branch (parallel to the main
                                       remainder). Their weights should be
                                       initialised by the loader from a
                                       deepcopy of ``self.blocks[-N:]``.
    """

    def __init__(
        self,
        *args,
        enable_internal: bool = False,
        internal_depth: int = 24,
        num_internal_blocks: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_internal = bool(enable_internal)
        self.internal_depth = int(internal_depth)

        if self.enable_internal:
            inner_dim = self.num_attention_heads * self.attention_head_dim
            # New parameters for the internal branch (initialised here; the
            # loader (utils_geometry_forcing_internal.load_transformer)
            # overwrites them with deepcopies from the main branch).
            self.internal_blocks = nn.ModuleList([
                WanTransformerBlock(
                    inner_dim,
                    self.config.ffn_dim,
                    self.num_attention_heads,
                    self.config.cross_attn_norm,
                    self.config.eps,
                    attn_mode=self.config.attn_mode,
                ) for _ in range(int(num_internal_blocks))
            ])
            self.internal_norm_out = FP32LayerNorm(
                inner_dim, self.config.eps, elementwise_affine=False
            )
            self.internal_proj_out = nn.Linear(
                inner_dim,
                self.config.out_channels
                * self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            )
            self.internal_action_proj_out = nn.Linear(inner_dim, self.config.action_dim)
            self.internal_scale_shift_table = nn.Parameter(
                torch.randn(1, 2, inner_dim) / inner_dim**0.5
            )

    # ------------------------------------------------------------------
    # Cross-attention probability capture (slow path, only when requested).
    # ------------------------------------------------------------------
    def _manual_cross_attn_probs(
        self,
        block,
        hidden_states_normed,
        encoder_hidden_states,
        text_seq_lengths=None,
    ):
        """Compute cross-attn (video↔text) probabilities for visualization.

        Reproduces ``block.attn2`` up to softmax(QK^T/sqrt(d)). Does NOT
        affect the rest of the forward (we only consume the returned probs).

        Args:
            block: a ``WanTransformerBlock``.
            hidden_states_normed: post-``norm2`` query stream of the block,
                                  shape (1, S_q, C).
            encoder_hidden_states: text token stream, shape (1, S_kv, C).
            text_seq_lengths: optional list[int] of valid text length per
                              batch sample, used to mask padding.

        Returns:
            probs: float32 tensor (heads, S_q, S_kv) — averaged over batches
                   already (since training flattens batch into seq).
        """
        attn = block.attn2
        q = attn.to_q(hidden_states_normed)
        k = attn.to_k(encoder_hidden_states)
        q = attn.norm_q(q).unflatten(2, (attn.heads, -1))  # (1, S_q, H, D)
        k = attn.norm_k(k).unflatten(2, (attn.heads, -1))  # (1, S_kv, H, D)
        # (1, H, S_q, D)
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        scale = 1.0 / (q.shape[-1] ** 0.5)
        # (1, H, S_q, S_kv)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if text_seq_lengths is not None:
            S_kv = k.shape[2]
            mask = torch.zeros(S_kv, dtype=torch.bool, device=scores.device)
            keep = sum(int(L) for L in text_seq_lengths)
            mask[:keep] = True
            scores = scores.masked_fill(~mask[None, None, None], float('-inf'))
        probs = scores.softmax(dim=-1)
        return probs[0]  # (H, S_q, S_kv)

    # ------------------------------------------------------------------
    # Main + internal training forward.
    # ------------------------------------------------------------------
    def forward_train_gf_internal(self, input_dict):
        """Combined GF + internal forward.

        Honors all the existing GF hooks (``return_hidden_layers``,
        ``align_layer_idx``, ``align_video_tokens_only``) on the **shared
        trunk + main remainder** (i.e. main path D_f). The internal branch
        only adds an extra (latent_pred, action_pred) tuple for
        deep-supervision.

        Visualization hooks (only used by vis path / eval):
            input_dict.get('capture_cross_attn', False)
            input_dict.get('capture_layers', list[int])  -> dict
        """
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        # ---- token embedding + concat (same as model_spatial_forcing) ----
        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text').flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        hidden_states = torch.cat(
            [
                latent_hidden_states,
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
            ],
            dim=1,
        )

        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict['noisy_latents'].shape[-2],
            latent_dict['noisy_latents'].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict['noisy_latents'].shape[-2],
            action_dict['noisy_latents'].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ]

        FlexAttnFunc.init_mask(
            latent_dict['noisy_latents'].shape,
            action_dict['noisy_latents'].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict['window_size'],
            patch_size=self.patch_size,
            device=hidden_states.device,
        )

        # ---- GF alignment hooks ----
        return_hidden_layers = bool(input_dict.get('return_hidden_layers', False))
        align_layer_idx = input_dict.get('align_layer_idx', None)
        align_video_tokens_only = bool(input_dict.get('align_video_tokens_only', True))
        requested_layers = set()
        if align_layer_idx is not None:
            if isinstance(align_layer_idx, (list, tuple, set)):
                requested_layers = {int(layer_idx) for layer_idx in align_layer_idx}
            else:
                requested_layers = {int(align_layer_idx)}
        captured_hidden_states = {}

        # ---- Cross-stream alignment hooks (capture *both* video and action) ----
        # Independent from align_layer_idx so the two objectives can run on
        # different layers without interfering. Set ``xstream_layers`` to a
        # list[int] of block indices; the model returns a dict
        #   ``xstream_hidden_states[block_idx] = {'video': (B, Lv, C), 'action': (B, La, C)}``
        # The slices follow the same split_list as forward_train so the call
        # sites do not need to know about padding.
        xstream_layers = input_dict.get('xstream_layers', None)
        xstream_set = set()
        if xstream_layers is not None:
            xstream_set = {int(i) for i in xstream_layers}
        captured_xstream = {}

        def _maybe_capture_xstream(block_idx, h):
            if block_idx not in xstream_set:
                return
            n_video = split_list[0]
            action_start = split_list[0] + split_list[1]
            n_action = split_list[2]
            video_tokens = h[:, :n_video]
            action_tokens = h[:, action_start:action_start + n_action]
            captured_xstream[block_idx] = {
                'video': rearrange(
                    video_tokens, '1 (b l) c -> b l c', b=batch_size
                ),
                'action': rearrange(
                    action_tokens, '1 (b l) c -> b l c', b=batch_size
                ),
            }

        # ---- Cross-attention probability capture (visualization) ----
        capture_cross_attn = bool(input_dict.get('capture_cross_attn', False))
        capture_layers = input_dict.get('capture_layers', None)
        capture_set = set()
        if capture_cross_attn:
            if capture_layers is None:
                capture_set = set(range(len(self.blocks)))
            else:
                capture_set = {int(i) for i in capture_layers}
        captured_attn_probs = {}

        def _do_capture_for_block(block, h_in, block_idx):
            """Run norm2(h_in) and the manual cross-attn probs (no side-effects)."""
            normed = block.norm2(h_in.float()).type_as(h_in)
            probs = self._manual_cross_attn_probs(
                block, normed, text_hidden_states,
                text_seq_lengths=input_dict.get('text_seq_lengths', None),
            )
            captured_attn_probs[block_idx] = probs.detach()

        # ---- Shared trunk: blocks[0 : internal_depth] ----
        internal_depth = self.internal_depth if self.enable_internal else len(self.blocks)
        for block_idx, block in enumerate(self.blocks[:internal_depth]):
            if capture_cross_attn and block_idx in capture_set:
                _do_capture_for_block(block, hidden_states, block_idx)
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )
            if return_hidden_layers and block_idx in requested_layers:
                if align_video_tokens_only:
                    video_tokens = hidden_states[:, :split_list[0]]
                    captured_hidden_states[block_idx] = rearrange(
                        video_tokens,
                        '1 (b l) c -> b l c',
                        b=batch_size,
                    )
                else:
                    captured_hidden_states[block_idx] = hidden_states
            _maybe_capture_xstream(block_idx, hidden_states)

        # Snapshot at the fork point so internal branch starts from same tensor.
        forked_hidden = hidden_states

        # ---- Main path: blocks[internal_depth :] ----
        for offset, block in enumerate(self.blocks[internal_depth:]):
            block_idx = internal_depth + offset
            if capture_cross_attn and block_idx in capture_set:
                _do_capture_for_block(block, hidden_states, block_idx)
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )
            if return_hidden_layers and block_idx in requested_layers:
                if align_video_tokens_only:
                    video_tokens = hidden_states[:, :split_list[0]]
                    captured_hidden_states[block_idx] = rearrange(
                        video_tokens,
                        '1 (b l) c -> b l c',
                        b=batch_size,
                    )
                else:
                    captured_hidden_states[block_idx] = hidden_states
            _maybe_capture_xstream(block_idx, hidden_states)

        # ---- Main head (D_f) ----
        main_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
        main_shift, main_scale = rearrange(main_sst, 'b l n c -> b n l c').chunk(2, dim=1)
        main_shift = main_shift.to(hidden_states.device).squeeze(1)
        main_scale = main_scale.to(hidden_states.device).squeeze(1)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1.0 + main_scale) + main_shift
        ).type_as(hidden_states)
        main_lat, _, main_act, _, _ = hidden_states.split(split_list, dim=1)
        main_lat = self.proj_out(main_lat)
        main_lat = rearrange(
            main_lat,
            '1 (b l) (n c) -> b (l n) c',
            n=self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            b=batch_size,
        )
        main_act = self.action_proj_out(main_act)
        main_act = rearrange(main_act, '1 (b l) c -> b l c', b=batch_size)

        # ---- Internal path (D_i): forked_hidden → internal_blocks → internal_head ----
        int_lat = int_act = None
        if self.enable_internal:
            internal_hidden = forked_hidden
            for block in self.internal_blocks:
                internal_hidden = block(
                    internal_hidden,
                    text_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    update_cache=False,
                )
            int_sst = self.internal_scale_shift_table[None] + temb[:, :, None, ...]
            int_shift, int_scale = rearrange(int_sst, 'b l n c -> b n l c').chunk(2, dim=1)
            int_shift = int_shift.to(internal_hidden.device).squeeze(1)
            int_scale = int_scale.to(internal_hidden.device).squeeze(1)
            internal_hidden = (
                self.internal_norm_out(internal_hidden.float()) * (1.0 + int_scale) + int_shift
            ).type_as(internal_hidden)
            int_lat, _, int_act, _, _ = internal_hidden.split(split_list, dim=1)
            int_lat = self.internal_proj_out(int_lat)
            int_lat = rearrange(
                int_lat,
                '1 (b l) (n c) -> b (l n) c',
                n=self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
                b=batch_size,
            )
            int_act = self.internal_action_proj_out(int_act)
            int_act = rearrange(int_act, '1 (b l) c -> b l c', b=batch_size)

        # ---- Pack output ----
        out = {
            'pred': (main_lat, main_act),
            'internal_pred': (int_lat, int_act) if self.enable_internal else None,
        }
        if return_hidden_layers:
            if align_layer_idx is not None and not isinstance(align_layer_idx, (list, tuple, set)):
                out['align_hidden_states'] = captured_hidden_states.get(int(align_layer_idx))
            else:
                out['align_hidden_states'] = captured_hidden_states
        if capture_cross_attn:
            out['attn_probs'] = captured_attn_probs
            out['split_list'] = split_list
            out['batch_size'] = batch_size
        if xstream_layers is not None:
            out['xstream_hidden_states'] = captured_xstream
        return out

    def forward_train(self, input_dict):
        """Override base forward_train: dispatch to the GF+internal flow.

        Trainer uses ``train_mode=True`` which routes here via ``forward``
        (inherited from base), which already calls ``forward_train``.
        """
        out = self.forward_train_gf_internal(input_dict)
        # Backward-compatible plain-tuple return when no extra hooks are on
        # AND no internal head AND no attn capture AND no cross-stream capture.
        if (
            not self.enable_internal
            and out.get('align_hidden_states') is None
            and 'attn_probs' not in out
            and 'xstream_hidden_states' not in out
        ):
            return out['pred']
        return out

    # ------------------------------------------------------------------
    # Inference forward through the internal branch (D_i).
    #
    # Mirrors the base ``forward()`` path (single-stream video OR action,
    # KV-Cache-aware) but only walks ``blocks[:internal_depth] +
    # internal_blocks`` and finishes with the internal head. Used by the
    # GF+Internal inference server when ``internal_infer_mode`` selects
    # the short path or the IG extrapolation.
    # ------------------------------------------------------------------
    def forward_internal(
        self,
        input_dict,
        update_cache=0,
        cache_name='pos',
        action_mode=False,
    ):
        if not self.enable_internal:
            raise RuntimeError(
                "forward_internal called on a model built without "
                "enable_internal=True."
            )

        # Same input prep as the base inference forward (no batch flatten
        # because here the loops are token-wise, KV-Cache handles state).
        if action_mode:
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'], 'b c f h w -> b (f h w) c'
            )
            latent_hidden_states = self.action_embedder(latent_hidden_states)
        else:
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)

        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"]
        )

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]

        pach_scale_h, pach_scale_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )
        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h)
            * (input_dict['noisy_latents'].shape[-1] // pach_scale_w),
            dim=1,
        )
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        # Shared trunk: blocks[0 : internal_depth].
        for block in self.blocks[: self.internal_depth]:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        # Internal-only blocks.
        for block in self.internal_blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        # Internal head AdaLN.
        int_sst = self.internal_scale_shift_table[None] + temb[:, :, None, ...]
        int_shift, int_scale = rearrange(int_sst, 'b l n c -> b n l c').chunk(2, dim=1)
        int_shift = int_shift.to(latent_hidden_states.device).squeeze(1)
        int_scale = int_scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (
            self.internal_norm_out(latent_hidden_states.float()) * (1.0 + int_scale)
            + int_shift
        ).type_as(latent_hidden_states)

        if action_mode:
            return self.internal_action_proj_out(latent_hidden_states)
        out = self.internal_proj_out(latent_hidden_states)
        out = rearrange(
            out,
            'b l (n c) -> b (l n) c',
            n=self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
        )
        return out
