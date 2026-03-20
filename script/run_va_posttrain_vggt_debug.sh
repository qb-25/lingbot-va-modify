#!/usr/bin/bash
# Quick debug run: 50 samples, 300 steps, to validate VGGT loss convergence.
# If loss steadily decreases on this small set, the training pipeline is working.

set -x
umask 007

NGPU=${NGPU:-"1"}
MASTER_PORT=${MASTER_PORT:-"29503"}
CONFIG_NAME="robotwin_train_vggt_debug"

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run \
    --nproc_per_node=${NGPU} \
    --master_port ${MASTER_PORT} \
    --tee 3 \
    -m wan_va.train_vggt --config-name ${CONFIG_NAME}
