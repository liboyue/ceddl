#!/bin/bash
TRAIN_PARAMS='--lr=0.001 --epochs=5 --batch_size=4 --momentum=0.9 --val_interval=2000 --disp_interval=2000 --deterministic'
DIST_PARAMS='--graph_type=complete --sync_freq=1'
ENVS='MASTER_ADDR=127.0.0.1 \
MASTER_PORT=23456 \
RANK=0 \
WORLD_SIZE=1 \
LOCAL_RANK=0 \
WORLD_LOCAL_SIZE=1 \
WORLD_NODE_RANK=0'

eval "echo 'Running reference script';
python cifar10_tutorial.py &> reference.log;
echo 'Reference script done'"

eval "echo 'Running distributed exp';
${ENVS} python cifar10.py ${TRAIN_PARAMS} ${DIST_PARAMS} &> sgp.log;
echo 'Distributed exp done'"

eval "echo 'Running all reduce exp';
${ENVS} python cifar10.py --all_reduce ${TRAIN_PARAMS} &> all_reduce.log;
echo 'All reduce exp done'"

cat sgp.log | grep Validation | awk '{print $9}' | tail -n +2 > sgp_acc.txt
cat all_reduce.log | grep Validation | awk '{print $9}' | tail -n +2 > all_reduce_acc.txt
cat reference.log | grep Accuracy | awk '{print $10}' > reference_acc.txt

A=`diff all_reduce_acc.txt reference_acc.txt`
B=`diff sgp_acc.txt reference_acc.txt`

if [[ -z "$A" && -z "$B" ]]; then
    echo 'The implementation is consistent with the reference code!!! Yay!!!'
else
    echo 'The implementation is not consistent with the reference code!'
fi
