#!/usr/bin/bash
#
# Weights & Biases (optional): before calling this script, set e.g.
#   export ENABLE_WANDB=1
#   export WANDB_API_KEY='...'   # if you must put it in the DLC command, prefix the whole line (risk: logs/history leak)
#   export WANDB_PROJECT=lingbotva
#   export WANDB_TEAM_NAME=...   # optional
#   export WANDB_RUN_NAME=my_run # optional

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29502"}
PORT=${PORT:-"1107"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt"}

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

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_vggt --config-name ${config_name} "${WANDB_ARGS[@]}" $overrides
