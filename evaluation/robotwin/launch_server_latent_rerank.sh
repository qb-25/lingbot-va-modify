START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

SAVE_ROOT=${SAVE_ROOT:-visualization_latent_rerank/}
mkdir -p "$SAVE_ROOT"

CKPT_PATH=${CKPT_PATH:-$(ls -d train_out/latent_dynamics/checkpoints/checkpoint_step_* 2>/dev/null | sort -V | tail -n 1)}
LATENT_DYNAMICS_CKPT_PATH=${LATENT_DYNAMICS_CKPT_PATH:-$CKPT_PATH}
ACTION_RERANK_NUM_CANDIDATES=${ACTION_RERANK_NUM_CANDIDATES:-4}
ACTION_RERANK_TOPK=${ACTION_RERANK_TOPK:-2}

if [ -z "$CKPT_PATH" ]; then
    echo "CKPT_PATH is empty. Please set CKPT_PATH to a latent-dynamics checkpoint root." >&2
    exit 1
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server_latent_rerank.py \
    --config-name robotwin \
    --port "$START_PORT" \
    --save_root "$SAVE_ROOT" \
    --ckpt-path "$CKPT_PATH" \
    --latent-dynamics-ckpt-path "$LATENT_DYNAMICS_CKPT_PATH" \
    --action-rerank-num-candidates "$ACTION_RERANK_NUM_CANDIDATES" \
    --action-rerank-topk "$ACTION_RERANK_TOPK"
