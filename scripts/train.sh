#!/bin/bash
workspace='/mnt/truenas/scratch/xiaotao.chen/Repositories/tmp/resnet.mxnet'
cd ${workspace}

if [ $# -ne 2 ]; then
   echo "input argument counts is $#, which requires 2, means network, kv type, to exit"
   exit
fi
network=$1
kv_type=$2

python3 example_train.py \
    --network ${network} \
    --kv_type ${kv_type}
