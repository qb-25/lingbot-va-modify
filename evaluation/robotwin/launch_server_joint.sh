#!/bin/bash
# Joint Parallel Denoising server launcher
# Uses wan_va_server_joint.py instead of wan_va_server.py
# Video and action are denoised simultaneously, closing the train-inference gap.

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server_joint.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root
