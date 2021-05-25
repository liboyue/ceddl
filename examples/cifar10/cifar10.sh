#!/bin/bash
SCRIPT='cifar10.py'

TRAIN_PARAMS='--epochs=30 --batch_size=2 --val_interval=200 --disp_interval=20 --num_workers=5'
OPTIM_PARAMS='--lr=0.001 --momentum=0.9'


DIST_PARAMS='--ddp=pytorch'
eval "PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"

DIST_PARAMS='--ddp=all_reduce --sync_freq=10'
eval "PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"


# DIST_PARAMS='--graph_type=exponential --n_peers=1 --sync_freq=1'
