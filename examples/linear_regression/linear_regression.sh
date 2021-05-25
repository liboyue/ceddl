#!/bin/bash
SCRIPT='linear_regression.py'

OPTIM_PARAMS='--lr=0.5'
TRAIN_PARAMS='--epochs=1 --batch_size=1000 \
    --val_interval=100 --disp_interval=100 \
    --dim=20 --n_samples=4000 --condition_number=10 --noise_variance=1 --deterministic'

DIST_PARAMS='--ddp=sgp --graph_type=exponential --n_peers=1 --sync_freq=1'
# DIST_PARAMS='--ddp=DistributedDataParallel'
# DIST_PARAMS='--ddp=DistributedGradientParallel'
DIST_PARAMS='--ddp=NetworkDataParallel'

PARAMS="${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}"
eval "PARAMS='${PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
