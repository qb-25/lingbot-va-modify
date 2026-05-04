#!/usr/bin/bash
#
# REPA-VAE backbone + VGGT / Wan VAE encoder alignment training.
#
# Resume: export LINGBOT_RESUME_FROM=/path/to/checkpoint_step_NNNN (directory that contains transformer/).
# TensorBoard: by default under ${SAVE_ROOT}/tb_logs; set LINGBOT_TB_LOG_DIR to force (e.g. on NAS with quota).
# W&B: optional ENABLE_WANDB=1 and WANDB_API_KEY in environment (do not commit secrets).

set -x
umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29516"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_repa_vae"}

SAVE_ROOT=${SAVE_ROOT:-"/mnt/nas/qianbin/train_repa"}
LINGBOT_RESUME_FROM=${LINGBOT_RESUME_FROM:-"/mnt/nas/qianbin/train_repa/vggt_repa_vae/checkpoints/checkpoint_step_10000"}
LINGBOT_TB_LOG_DIR=${LINGBOT_TB_LOG_DIR:-"${SAVE_ROOT}/tb_logs"}

export LINGBOT_RESUME_FROM
export LINGBOT_TB_LOG_DIR

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
    -m wan_va.train_vggt_repa_vae \
    --config-name "${CONFIG_NAME}" \
    --save-root "${SAVE_ROOT}" \
    "${WANDB_ARGS[@]}" \
    $overrides
