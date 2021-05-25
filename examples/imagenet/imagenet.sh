#!/bin/bash
SCRIPT='imagenet.py'

# lr = n * batch_size / 256 / 10 as in https://arxiv.org/abs/1706.02677
TRAIN_PARAMS='--model=resnet50 --epochs=90 --batch_size=256 --val_interval=1000 --disp_interval=100 --num_workers=5'
OPTIM_PARAMS='--optimizer=sgd --lr=0.4 --weight_decay=0.0001 --momentum=0.9'

echo 'Running exp 1: SGD + PyTorch DDP'
DIST_PARAMS='--ddp=pytorch'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 2.1: SGD + DistributedGradientParallel without fp16_grads'
DIST_PARAMS='--ddp=DistributedGradientParallel'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 2.2: SGD + DistributedGradientParallel with fp16_grads'
DIST_PARAMS='--ddp=DistributedGradientParallel --fp16_grads'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 3.1: SGD + DistributedDataParallel sync_freq=1 graph_type=complete'
DIST_PARAMS='--ddp=DistributedDataParallel --sync_freq=1 --graph_type=complete'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 3.2: SGD + DistributedDataParallel sync_freq=10 graph_type=complete'
DIST_PARAMS='--ddp=DistributedDataParallel --sync_freq=10 --graph_type=complete --n_peers=1'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 3.3: SGD + SparseDistributedDataParallel sync_freq=1 graph_type=exponential'
DIST_PARAMS='--ddp=SparseDistributedDataParallel --sync_freq=1 --graph_type=exponential --n_peers=1'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 3.4: SGD + SparseDistributedDataParallel sync_freq=10 graph_type=exponential'
DIST_PARAMS='--ddp=SparseDistributedDataParallel --sync_freq=10 --graph_type=exponential -n_peers=1'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 4.1: LAMB + PyTorch DDP'
OPTIM_PARAMS='--optimizer=lamb --lr=0.0070710678118654745 --weight_decay=0.0005'
DIST_PARAMS='--ddp=pytorch'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD

echo 'Running exp 4.2: LAMB + DistributedDataParallel sync_freq=1 graph_type=complete'
OPTIM_PARAMS='--optimizer=lamb --lr=0.2 --weight_decay=0.0005'
DIST_PARAMS='--ddp=DistributedDataParallel --sync_freq=1 --graph_type=complete'
CMD="PARAMS='${OPTIM_PARAMS} ${TRAIN_PARAMS} ${DIST_PARAMS}' SCRIPT=${SCRIPT} ../run.sh ${@} --output_dir=experiments/$(date +'%y-%m-%d_%H:%M:%S')"
eval $CMD
