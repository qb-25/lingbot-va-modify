import torch
import torch.nn.functional as F
from einops import rearrange

from .model import WanTransformer3DModel as BaseWanTransformer3DModel
from .model import FlexAttnFunc


class WanTransformer3DModel(BaseWanTransformer3DModel):
    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')
        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

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

        for block_idx, block in enumerate(self.blocks):
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

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1. + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = hidden_states.split(split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            '1 (b l) (n c) -> b (l n) c',
            n=self.patch_size[0] * self.patch_size[1] * self.patch_size[2],
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states, '1 (b l) c -> b l c', b=batch_size)

        if not return_hidden_layers:
            return latent_hidden_states, action_hidden_states

        if align_layer_idx is not None and not isinstance(align_layer_idx, (list, tuple, set)):
            captured_hidden_states = captured_hidden_states.get(int(align_layer_idx))

        return {
            'pred': (latent_hidden_states, action_hidden_states),
            'align_hidden_states': captured_hidden_states,
        }
