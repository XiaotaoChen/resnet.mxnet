#!/bin/bash

workspace=/mnt/truenas/upload/xiaotao.chen/Repositories/resnet.mxnet
cd ${workspace}

network='resnet'
model_prefix='4node'
lr=0.8
use_horovod=1
batch_per_gpu=64
warmup_lr=0.1
warm_epoch=5
gpus="0,1,2,3,4,5,6,7"
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
    --data_nthreads ${data_nthreads} \
    --model_prefix ${model_prefix}
