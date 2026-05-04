#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29504"}
PORT=${PORT:-"1108"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_flow"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
# 可选：团队/组织 entity；勿使用占位符。不设置则使用 wandb login 的默认账号（个人 run 通常无需设置）
# export WANDB_TEAM_NAME="your-real-wandb-entity"
export WANDB_PROJECT="${WANDB_PROJECT:-lingbotva-flow}"

## Optional dependency:
##   pip install "ptlflow>=0.4.1" "lightning<2.7" jsonargparse loguru

num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train_flow --config-name ${config_name} $overrides
