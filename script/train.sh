#!/bin/bash

export PYTHONPATH="/mnt/tscpfs/xiaotao.chen/Repositories/mxnet_zjq/python:${PYTHONPATH}"
workspace="/mnt/tscpfs/xiaotao.chen/Repositories/resnet.mxnet"

network='resnet'
benchmark=0
quant_mod='minmax'

cd ${workspace}
python train.py \
    --network ${network} \
    --benchmark ${benchmark} \
    --quant_mod ${quant_mod}
