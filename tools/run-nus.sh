#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(cd "$(dirname "$0")/.."; pwd)

TASK_DESC=$1
GPUS=${2:-8}
PORT=$((8000 + RANDOM %57535))


CONFIG=configs/isfusion/isfusion_0075voxel.py

torchrun --nproc_per_node=$GPUS --master_port ${PORT} $(dirname "$0")/train.py --launcher pytorch $CONFIG \
--extra_tag $TASK_DESC
