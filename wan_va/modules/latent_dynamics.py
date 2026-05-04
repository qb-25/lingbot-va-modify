import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ProjectionGaussianRegularizer(nn.Module):
    """Lightweight SIGReg-style surrogate using random 1D projections.

    We do not implement the exact Epps-Pulley statistic here. Instead, we
    encourage projected latent marginals to match a unit Gaussian by penalizing
    the first four moments across random directions.
    """

    def __init__(self, num_projections=256, eps=1e-6):
        super().__init__()
        self.num_projections = num_projections
        self.eps = eps

    def forward(self, states):
        states = states.float().reshape(-1, states.shape[-1])
        if states.shape[0] < 2:
            return states.new_tensor(0.0)

        directions = torch.randn(
            self.num_projections,
            states.shape[-1],
            device=states.device,
            dtype=states.dtype,
        )
        directions = F.normalize(directions, dim=-1)
        projected = states @ directions.t()

        mean = projected.mean(dim=0)
        centered = projected - mean
        std = centered.pow(2).mean(dim=0).add(self.eps).sqrt()
        normalized = centered / std
        skew = normalized.pow(3).mean(dim=0)
        kurt = normalized.pow(4).mean(dim=0) - 3.0

        reg = (
            mean.pow(2).mean()
            + (std - 1.0).pow(2).mean()
            + skew.pow(2).mean()
            + kurt.pow(2).mean()
        )
        return reg


class LatentDynamicsBranch(nn.Module):
    """Compact state encoder + action-conditioned transition model."""

    def __init__(
        self,
        latent_channels,
        action_dim,
        action_per_frame,
        state_dim=256,
        hidden_dim=512,
        num_projections=256,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.action_dim = action_dim
        self.action_per_frame = action_per_frame
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        action_input_dim = action_dim * action_per_frame

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(latent_channels),
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_input_dim),
            nn.Linear(action_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.gaussian_reg = ProjectionGaussianRegularizer(num_projections=num_projections)

    def encode_states(self, clean_latents):
        pooled = clean_latents.float().mean(dim=(-1, -2)).permute(0, 2, 1)
        return self.state_encoder(pooled)

    def encode_actions(self, clean_actions):
        flat_actions = rearrange(clean_actions.float(), "b c f n w -> b f (c n w)")
        return self.action_encoder(flat_actions)

    def predict_next(self, state_seq, action_seq):
        if state_seq.shape[1] < 2:
            return state_seq.new_zeros(state_seq.shape[0], 0, state_seq.shape[-1])

        transition_input = torch.cat([state_seq[:, :-1], action_seq[:, :-1]], dim=-1)
        delta = self.transition(transition_input)
        return state_seq[:, :-1] + delta

    def rollout(self, initial_state, action_seq):
        cur_state = initial_state
        predictions = []
        for step in range(action_seq.shape[1]):
            delta = self.transition(torch.cat([cur_state, action_seq[:, step]], dim=-1))
            cur_state = cur_state + delta
            predictions.append(cur_state)
        if not predictions:
            return initial_state.new_zeros(initial_state.shape[0], 0, initial_state.shape[-1])
        return torch.stack(predictions, dim=1)
