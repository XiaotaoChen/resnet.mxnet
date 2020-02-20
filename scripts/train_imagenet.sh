#!/bin/bash


network='resnet'
lr=0.2
use_horovod=1
batch_per_gpu=64
warmup_lr=0.1
warmup_epoch=5
gpus="0,1,2,3,4,5"
dataset='imagenet'
data_dir='./rec_data'

python3 train.py \
    --network ${network} \
    --lr ${lr} \
    --use_horovod ${use_horovod} \
    --batch_per_gpu ${batch_per_gpu} \
    --warmup_lr ${warmup_lr} \
    --warmup_epoch ${warmup_epoch} \
    --gpus ${gpus} \
    --dataset ${dataset} \
    --data_dir ${data_dir}
