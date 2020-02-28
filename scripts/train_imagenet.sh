#!/bin/bash


network='resnet'
lr=0.2
use_horovod=1
batch_per_gpu=64
warmup_lr=0.1
warm_epoch=5
gpus="0,1,2,3"
dataset='imagenet'
data_dir='./imagenet_data_new'
benchmark=1
data_nthreads=4


python3 train.py \
    --network ${network} \
    --lr ${lr} \
    --use_horovod ${use_horovod} \
    --batch_per_gpu ${batch_per_gpu} \
    --warmup_lr ${warmup_lr} \
    --warm_epoch ${warm_epoch} \
    --gpus ${gpus} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --benchmark ${benchmark} \
    --data_nthreads ${data_nthreads}
