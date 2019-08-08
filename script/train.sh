#!/bin/bash

export PYTHONPATH="/mnt/tscpfs/xiaotao.chen/Repositories/mxnet_zjq/python:${PYTHONPATH}"
#export PYTHONPATH="/mnt/truenas/scratch/xiaotao.chen/Repositories/incubator-mxnet/python:${PYTHONPATH}"
workspace="/mnt/tscpfs/xiaotao.chen/Repositories/resnet.mxnet"


cd ${workspace}
python train.py