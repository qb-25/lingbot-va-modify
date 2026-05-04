#!/usr/bin/bash
# Weights & Biases (optional): same as run_va_posttrain_vggt_debug.sh — set ENABLE_WANDB=1 and WANDB_API_KEY.

set -x
umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29513"}
CONFIG_NAME="robotwin_train_vggt_spatial_forcing_debug"

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
    -m wan_va.train_vggt_spatial_forcing --config-name ${CONFIG_NAME} "${WANDB_ARGS[@]}" "$@"
