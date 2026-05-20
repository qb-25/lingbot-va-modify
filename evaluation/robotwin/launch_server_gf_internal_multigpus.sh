#!/usr/bin/bash
# 多卡 GF+Internal 推理服务启动脚本（每卡 1 个进程，端口与 GPU 一一对应）。
#
# 关键 ENV：
#   EVAL_CHECKPOINT=/abs/path/to/checkpoint_step_NNN
#       checkpoint 根目录（要包含 transformer/）。
#   CONFIG_NAME=robotwin_train_vggt_geometry_forcing_internal
#       想用 main-only / short-path / ig-extrapolate 中的哪种推理模式，
#       直接改 cfg 里的 `internal_infer_mode`，或设置 INTERNAL_INFER_MODE=...
#       让脚本帮你 export。
#   WAN_VGGT_REPO=/mnt/nas/qianbin/vggt
#       facebookresearch/VGGT 源码路径（GF teacher 需要）。
#   START_PORT=29556  MASTER_PORT=29661   websocket / torch.distributed master port
#   NGPU=8                                 总进程数
#
# 推理模式速选：
#   INTERNAL_INFER_MODE=main_only       behave like base wan_va_server
#   INTERNAL_INFER_MODE=short_path      step-skipping 加速
#   INTERNAL_INFER_MODE=ig_extrapolate  论文 IG 外推（需要同时设
#                                         INTERNAL_GUIDANCE_SCALE）

set -x
umask 007

START_PORT=${START_PORT:-29556}
MASTER_PORT=${MASTER_PORT:-29661}
NGPU=${NGPU:-8}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_vggt_geometry_forcing_internal"}

LOG_DIR=${LOG_DIR:-"./logs"}
mkdir -p "$LOG_DIR"
SAVE_ROOT=${SAVE_ROOT:-"./visualization/"}
mkdir -p "$SAVE_ROOT"

batch_time=$(date +%Y%m%d_%H%M%S)

# Required: checkpoint to evaluate.
if [ -n "${EVAL_CHECKPOINT:-}" ]; then
  export wan22_pretrained_model_name_or_path="${EVAL_CHECKPOINT}"
elif [ -z "${wan22_pretrained_model_name_or_path:-}" ]; then
  echo "ERROR: please set EVAL_CHECKPOINT or wan22_pretrained_model_name_or_path"
  exit 1
fi

# VGGT source path (so `import vggt` works inside the GF+Internal loader).
WAN_VGGT_REPO=${WAN_VGGT_REPO:-"/mnt/nas/qianbin/vggt"}
if [ -d "${WAN_VGGT_REPO}/vggt" ]; then
  export PYTHONPATH="${WAN_VGGT_REPO}:${PYTHONPATH:-}"
fi

# Inference-mode quick override.
if [ -n "${INTERNAL_INFER_MODE:-}" ]; then
  export internal_infer_mode="${INTERNAL_INFER_MODE}"
fi
if [ -n "${INTERNAL_GUIDANCE_SCALE:-}" ]; then
  export internal_guidance_scale="${INTERNAL_GUIDANCE_SCALE}"
fi

export TOKENIZERS_PARALLELISM=false

for ((i=0; i<NGPU; i++)); do
  CURRENT_PORT=$((START_PORT + i))
  CURRENT_MASTER_PORT=$((MASTER_PORT + i))

  LOG_FILE="${LOG_DIR}/gf_internal_server_${i}_${batch_time}.log"
  echo "[GPU $i] PORT=$CURRENT_PORT MASTER_PORT=$CURRENT_MASTER_PORT log=$LOG_FILE"

  CUDA_VISIBLE_DEVICES=$i \
  nohup python -m torch.distributed.run \
      --nproc_per_node 1 \
      --master_port "$CURRENT_MASTER_PORT" \
      wan_va/wan_va_server_gf_internal.py \
      --config-name "$CONFIG_NAME" \
      --save_root "$SAVE_ROOT" \
      --port "$CURRENT_PORT" \
      > "$LOG_FILE" 2>&1 &
  sleep 2
done

echo "All ${NGPU} GF+Internal server instances launched."
wait
