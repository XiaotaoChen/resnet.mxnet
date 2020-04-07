#!/bin/bash

# model_prefix="/mnt/truenas/scratch/xiaotao.chen/outputs/infra/resnet/trained_model/resnet"
# epoch=90

# python3 test.py \
#    --model_prefix ${model_prefix} \
#    --model_load_epoch ${epoch}


python3 test.py \
    --platform "mxnet" \
    --data_type "fp32"