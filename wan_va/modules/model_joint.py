# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Joint parallel denoising extension for WanTransformer3DModel.
#
# Key design: override model.forward() so that joint-mode calls still go
# through model.__call__(), which triggers FSDP2 pre/post-forward hooks
# for proper DTensor unshard/reshard.  A special '__joint_mode__' key in
# input_dict dispatches to the joint code path.

import math
import torch
import torch.nn.functional as F
from einops import rearrange


def _forward_joint_impl(
    self,
    video_input,
    action_input,
    update_cache=0,
    cache_name="pos",
):
    """
    Joint forward: video + action tokens in a single attention pass.

    Mirrors training-time noise→noise same-chunk interaction, closing
    the train-inference gap of the original serial pipeline.
    """
    ps = self.patch_size

    # 1. Token embedding
    v_lat = video_input['noisy_latents']
    video_hidden = rearrange(
        v_lat,
        'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
        p1=ps[0], p2=ps[1], p3=ps[2])
    video_hidden = self.patch_embedding_mlp(video_hidden)

    a_lat = action_input['noisy_latents']
    action_hidden = rearrange(a_lat, 'b c f h w -> b (f h w) c')
    action_hidden = self.action_embedder(action_hidden)

    video_len = video_hidden.shape[1]
    hidden_states = torch.cat([video_hidden, action_hidden], dim=1)

    # 2. Shared text embedding
    text_hidden_states = self.condition_embedder.text_embedder(
        video_input['text_emb'])

    # 3. Rotary position embedding
    full_grid_id = torch.cat(
        [video_input['grid_id'], action_input['grid_id']], dim=2)
    rotary_emb = self.rope(full_grid_id)[:, :, None]

    # 4. Time embedding (different condition embedders per modality)
    v_ts = torch.repeat_interleave(
        video_input['timesteps'],
        (v_lat.shape[-2] // ps[1]) * (v_lat.shape[-1] // ps[2]),
        dim=1)
    v_temb, v_tp = self.condition_embedder(v_ts, dtype=hidden_states.dtype)
    v_tp = v_tp.unflatten(2, (6, -1))

    a_ts = torch.repeat_interleave(
        action_input['timesteps'],
        a_lat.shape[-2] * a_lat.shape[-1],
        dim=1)
    a_temb, a_tp = self.condition_embedder_action(
        a_ts, dtype=hidden_states.dtype)
    a_tp = a_tp.unflatten(2, (6, -1))

    temb = torch.cat([v_temb, a_temb], dim=1)
    timestep_proj = torch.cat([v_tp, a_tp], dim=1)

    # 5. Transformer blocks (with KV cache)
    for block in self.blocks:
        hidden_states = block(
            hidden_states, text_hidden_states, timestep_proj,
            rotary_emb, update_cache=update_cache, cache_name=cache_name)

    # 6. Output norm
    temb_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
    shift, scale = rearrange(temb_sst, 'b l n c -> b n l c').chunk(2, dim=1)
    shift = shift.to(hidden_states.device).squeeze(1)
    scale = scale.to(hidden_states.device).squeeze(1)
    hidden_states = (
        self.norm_out(hidden_states.float()) * (1. + scale) + shift
    ).type_as(hidden_states)

    # 7. Split & output heads
    video_out = hidden_states[:, :video_len]
    action_out = hidden_states[:, video_len:]

    video_out = self.proj_out(video_out)
    video_out = rearrange(
        video_out, 'b l (n c) -> b (l n) c', n=math.prod(ps))

    action_out = self.action_proj_out(action_out)

    return video_out, action_out


def patch_model_with_joint_forward(model):
    """Override ``forward`` so joint calls go through ``model.__call__()``
    and FSDP2 hooks fire correctly (DTensor unshard/reshard).

    Callers use::

        result = model(
            {'__joint_mode__': True,
             'video_input': v_dict, 'action_input': a_dict},
            update_cache=..., cache_name=...)
    """
    _orig_forward = model.forward

    def _patched_forward(
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        if isinstance(input_dict, dict) and input_dict.get('__joint_mode__'):
            return _forward_joint_impl(
                model,
                input_dict['video_input'],
                input_dict['action_input'],
                update_cache=update_cache,
                cache_name=cache_name,
            )
        return _orig_forward(
            input_dict,
            update_cache=update_cache,
            cache_name=cache_name,
            action_mode=action_mode,
            train_mode=train_mode,
        )

    model.forward = _patched_forward
    return model
