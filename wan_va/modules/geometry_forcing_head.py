# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Geometry Forcing alignment heads.

Implements the two objectives from
"Geometry Forcing: Marrying Video Diffusion and 3D Representation
for Consistent World Modeling" (Wu et al., 2025):

1. **AngularProjector** f_phi: maps the diffusion hidden states to the teacher
   (VGGT) feature dimension for cosine-direction alignment.
2. **ScalePredictor**  g_phi: given the *unit-normalized* output of f_phi,
   regresses the un-normalized teacher features (MSE), so that the
   magnitude / scale of geometric features is preserved without destabilizing
   the student representation.

For Wan-VA we use one (f_phi, g_phi) pair per selected student layer
(ModuleList), following the "independent per-layer predictor" choice.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAlignProjectionHead(nn.Module):
    """Lightweight MLP projector used for both f_phi and g_phi.

    (in_dim) -> Norm -> Linear -> GELU -> Linear -> (out_dim)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        self.norm = nn.BatchNorm1d(in_dim) if use_bn else nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (..., in_dim) -> (..., out_dim). Norm is applied on the last dim."""
        original_shape = tokens.shape
        flat_tokens = tokens.reshape(-1, original_shape[-1]).float()
        flat_tokens = self.norm(flat_tokens)
        flat_tokens = self.proj(flat_tokens)
        return flat_tokens.reshape(*original_shape[:-1], flat_tokens.shape[-1])


class GeometryForcingHead(nn.Module):
    """Per-layer Angular + Scale heads.

    Args:
        student_dims:  list of hidden sizes of the selected student layers
                       (usually identical across DiT blocks).
        teacher_dims:  list of hidden sizes of the selected teacher layers
                       (from VGGT aggregator). Must match 1-to-1 with
                       student_dims.
        hidden_dim:    bottleneck width for the MLPs.
        use_bn:        use BatchNorm1d in the Angular/Scale heads (default),
                       otherwise LayerNorm.
        use_scale:     whether to instantiate the Scale predictors (g_phi).
                       When False, Scale alignment loss is not used and the
                       module is equivalent to a multi-layer REPA / Angular
                       projector.
    """

    def __init__(
        self,
        student_dims: List[int],
        teacher_dims: List[int],
        hidden_dim: int = 2048,
        use_bn: bool = True,
        use_scale: bool = True,
    ):
        super().__init__()
        assert len(student_dims) == len(teacher_dims), (
            f"student/teacher layer count mismatch: "
            f"{len(student_dims)} vs {len(teacher_dims)}"
        )
        self.num_layers = len(student_dims)
        self.use_scale = use_scale

        # f_phi: student -> teacher dim  (Angular)
        self.angular_projectors = nn.ModuleList(
            [
                TokenAlignProjectionHead(
                    in_dim=s_dim, hidden_dim=hidden_dim, out_dim=t_dim, use_bn=use_bn
                )
                for s_dim, t_dim in zip(student_dims, teacher_dims)
            ]
        )

        # g_phi: normalized f_phi(h) -> raw teacher feature  (Scale)
        if use_scale:
            self.scale_predictors = nn.ModuleList(
                [
                    TokenAlignProjectionHead(
                        in_dim=t_dim,
                        hidden_dim=hidden_dim,
                        out_dim=t_dim,
                        use_bn=use_bn,
                    )
                    for t_dim in teacher_dims
                ]
            )
        else:
            self.scale_predictors = None

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def project(self, student_tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Run f_phi on the `layer_idx`-th layer tokens. Returns shape preserved,
        with last dim changed to teacher_dim."""
        return self.angular_projectors[layer_idx](student_tokens)

    def predict_scale(
        self, normalized_proj: torch.Tensor, layer_idx: int
    ) -> Optional[torch.Tensor]:
        """Run g_phi on unit-normalized f_phi(h). Returns un-normalized
        teacher-dim features, or None if Scale alignment is disabled."""
        if not self.use_scale or self.scale_predictors is None:
            return None
        return self.scale_predictors[layer_idx](normalized_proj)


# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------
def angular_loss(
    student_proj: torch.Tensor,
    teacher: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Angular (cosine) alignment loss.

    Args:
        student_proj:  (..., D_teacher)   - output of f_phi
        teacher:       (..., D_teacher)   - target feature
        weights:       (...,)             - per-token weight (sigma-based)
    Returns:
        scalar loss = sum(w * (1 - cos)) / sum(w)
    """
    s = F.normalize(student_proj.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    cosine = (s * t).sum(dim=-1)  # (...)
    loss = ((1.0 - cosine) * weights).sum() / (weights.sum() + 1e-6)
    cos_val = (cosine * weights).sum() / (weights.sum() + 1e-6)
    return loss, cos_val.detach()


def scale_loss(
    scale_pred: torch.Tensor,
    teacher: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Scale alignment MSE loss on un-normalized features.

    Args:
        scale_pred:  (..., D_teacher)   - output of g_phi(normalize(f_phi(h)))
        teacher:     (..., D_teacher)   - target (un-normalized) feature
        weights:     (...,)             - per-token weight (broadcast over D)
    Returns:
        scalar MSE weighted by `weights`
    """
    diff = (scale_pred.float() - teacher.float()) ** 2  # (..., D)
    diff = diff.mean(dim=-1)  # (...)
    return (diff * weights).sum() / (weights.sum() + 1e-6)
