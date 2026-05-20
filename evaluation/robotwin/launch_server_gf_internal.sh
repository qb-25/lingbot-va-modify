#!/usr/bin/bash
# 单卡 debug 启动 GF+Internal 推理服务。
set -x
umask 007

PORT=${PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
GPU=${GPU:-0}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing_internal"}

LOG_DIR=${LOG_DIR:-"./logs"}
mkdir -p "$LOG_DIR"
SAVE_ROOT=${SAVE_ROOT:-"./visualization/"}
mkdir -p "$SAVE_ROOT"

if [ -n "${EVAL_CHECKPOINT:-}" ]; then
  export wan22_pretrained_model_name_or_path="${EVAL_CHECKPOINT}"
elif [ -z "${wan22_pretrained_model_name_or_path:-}" ]; then
  echo "ERROR: please set EVAL_CHECKPOINT or wan22_pretrained_model_name_or_path"
  exit 1
fi

WAN_VGGT_REPO=${WAN_VGGT_REPO:-"/mnt/nas/qianbin/vggt"}
if [ -d "${WAN_VGGT_REPO}/vggt" ]; then
  export PYTHONPATH="${WAN_VGGT_REPO}:${PYTHONPATH:-}"
fi
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=${GPU} \
python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server_gf_internal.py \
    --config-name "$CONFIG_NAME" \
    --save_root "$SAVE_ROOT" \
    --port "$PORT"
