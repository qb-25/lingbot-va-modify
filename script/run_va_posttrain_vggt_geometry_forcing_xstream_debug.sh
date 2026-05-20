#!/usr/bin/bash
# Single-GPU debug entry for GF + Cross-Stream Alignment.
# Usage:  NGPU=1 bash script/run_va_posttrain_vggt_geometry_forcing_xstream_debug.sh

set -x
umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29520"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing_xstream_debug"}

export TOKENIZERS_PARALLELISM=false

WAN_VGGT_REPO=${WAN_VGGT_REPO:-"/mnt/nas/qianbin/vggt"}
if [ -d "${WAN_VGGT_REPO}/vggt" ]; then
  export PYTHONPATH="${WAN_VGGT_REPO}:${PYTHONPATH:-}"
fi

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
    -m wan_va.train_vggt_geometry_forcing_xstream \
        --config-name ${CONFIG_NAME} \
        "${WANDB_ARGS[@]}" "$@"
