#!/usr/bin/bash
#
# Wan-VA + Geometry Forcing (Angular + Scale alignment) training.
# Paper: Wu et al., 2025 "Geometry Forcing: Marrying Video Diffusion and
#        3D Representation for Consistent World Modeling".
#
# Weights & Biases (optional): before calling this script, set e.g.
#   export ENABLE_WANDB=1
#   export WANDB_API_KEY='...'
#   export WANDB_PROJECT=lingbotva
#   export WANDB_TEAM_NAME=...      # optional
#   export WANDB_RUN_NAME=my_run    # optional
#
# Resume (optional): export LINGBOT_RESUME_FROM=/abs/path/to/checkpoint_step_xxxx

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29515"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing"}

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
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --local-ranks-filter=${LOG_RANK} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_vggt_geometry_forcing --config-name ${CONFIG_NAME} "${WANDB_ARGS[@]}" $overrides
