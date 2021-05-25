#!/bin/bash
SCRIPT='mnist.py'

TRAIN_PARAMS='--epochs=1 --batch_size=100 --val_interval=100 --disp_interval=50 --deterministic --num_workers=5'
OPTIM_PARAMS='--lr=0.01'
DIST_PARAMS='--ddp=all_reduce --sync_freq=5 --gradient_accumulation'
DIST_PARAMS='--ddp=pytorch --sync_freq=5 --gradient_accumulation'
DIST_PARAMS='--ddp=DistributedGradientParallel'
DIST_PARAMS='--ddp=NetworkDataParallel --graph_type=exponential --n_peers=1 --sync_freq=10'
# DIST_PARAMS='--DDP=sgp --graph_type=exponential --n_peers=1 --sync_freq=10 --async_op'
eval "PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
