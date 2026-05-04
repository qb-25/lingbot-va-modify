#!/usr/bin/bash
# Quick debug run: 50 samples, 300 steps, to validate VGGT loss convergence.
# If loss steadily decreases on this small set, the training pipeline is working.
#
# Weights & Biases (optional): set ENABLE_WANDB=1 and provide credentials via env
# (recommended: DLC secret / env injection, never commit API keys).
#   export ENABLE_WANDB=1
#   export WANDB_API_KEY="..."        # required for upload
#   export WANDB_PROJECT="lingbotva"  # optional
#   export WANDB_TEAM_NAME="..."      # optional, team entity

set -x
umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29503"}
CONFIG_NAME="robotwin_train_vggt_debug"

export TOKENIZERS_PARALLELISM=false

WANDB_ARGS=()
if [ "${ENABLE_WANDB:-0}" = "1" ]; then
  WANDB_ARGS+=(--enable-wandb)
  if [ -n "${WANDB_RUN_NAME:-}" ]; then
    WANDB_ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
  fi
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_vggt --config-name ${CONFIG_NAME} "${WANDB_ARGS[@]}" "$@"
