#!/usr/bin/bash
#
# Wan-VA + Geometry Forcing + Cross-Stream Alignment training.
#
# Combines:
#   * Geometry Forcing (Wu et al., 2025): multi-layer Angular + Scale
#     alignment with a frozen VGGT teacher.
#   * Cross-Stream Alignment: at a chosen DiT block (default 25, NOT a
#     GF layer), pool video and action tokens to per-frame embeddings
#     and apply a symmetric InfoNCE objective with all-gather across
#     ranks and sigma-aware gating.
#
# Also: dump the cross-attention heatmap grid every attn_vis_interval
# steps (default 2000) under ${save_root}/attn_vis/.
#
# Optional ENV:
#   ENABLE_WANDB=1, WANDB_API_KEY, WANDB_PROJECT, WANDB_RUN_NAME
#   LINGBOT_RESUME_FROM=/abs/path/checkpoint_step_xxxx
#   WAN_VGGT_REPO=/mnt/nas/qianbin/vggt   (override if vggt sources are elsewhere)

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29519"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing_xstream"}

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

# Make ``import vggt`` resolve to /mnt/nas/qianbin/vggt/vggt/ (the source
# repo of facebookresearch/VGGT) — needed by the GF teacher loader.
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
    -m wan_va.train_vggt_geometry_forcing_xstream \
        --config-name ${CONFIG_NAME} \
        "${WANDB_ARGS[@]}" $overrides
