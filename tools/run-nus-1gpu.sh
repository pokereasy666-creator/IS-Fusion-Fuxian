#!/bin/bash

# Single-GPU training script for NVIDIA A30
export PYTHONPATH=$PYTHONPATH:$(cd "$(dirname "$0")/.."; pwd)
export TORCH_CUDA_ARCH_LIST="8.0"
export FORCE_CUDA="1"

TASK_DESC=$1
CONFIG=configs/isfusion/isfusion_0075voxel_1gpu.py

CUDA_VISIBLE_DEVICES=0 \
python $(dirname "$0")/train.py $CONFIG \
    --gpu-ids 0 \
    --extra_tag $TASK_DESC
