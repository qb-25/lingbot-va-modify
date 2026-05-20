#!/usr/bin/bash
#
# Wan-VA + Geometry Forcing + Internal head training.
#
# Combines:
#   * Geometry Forcing (Wu et al., 2025): multi-layer Angular + Scale
#     alignment with frozen VGGT teacher.
#   * Internal Guidance (Zhou et al., 2025): a parallel internal head
#     forked at ``internal_depth`` (default 24/30) for deep supervision.
#
# Optional ENV:
#   ENABLE_WANDB=1, WANDB_API_KEY, WANDB_PROJECT, WANDB_RUN_NAME
#   LINGBOT_RESUME_FROM=/abs/path/checkpoint_step_xxxx
#
# The trainer also dumps periodic cross-attention overlays under
# ``train_out/.../attn_vis/attn_step_NNNNNNN.png`` for inspection.

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29517"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing_internal"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

WANDB_ARGS=()
if [ "${ENABLE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS+=(--enable-wandb)
  if [ -n "${WANDB_RUN_NAME:-}" ]; then
    WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
fi

export TOKENIZERS_PARALLELISM=false

# Make ``import vggt`` resolve to /mnt/nas/qianbin/vggt/vggt/ (the source repo
# of facebookresearch/VGGT), needed by the GF teacher loader. Override
# WAN_VGGT_REPO if you keep the clone elsewhere.
WAN_VGGT_REPO=${WAN_VGGT_REPO:-"/mnt/nas/qianbin/vggt"}
if [ -d "${WAN_VGGT_REPO}/vggt" ]; then
  export PYTHONPATH="${WAN_VGGT_REPO}:${PYTHONPATH:-}"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --local-ranks-filter=${LOG_RANK} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_vggt_geometry_forcing_internal \
        --config-name ${CONFIG_NAME} \
        "${WANDB_ARGS[@]}" $overrides
