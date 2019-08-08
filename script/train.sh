#!/bin/bash

export PYTHONPATH="/mnt/tscpfs/xiaotao.chen/Repositories/mxnet_zjq/python:${PYTHONPATH}"
workspace="/mnt/tscpfs/xiaotao.chen/Repositories/resnet.mxnet"


cd ${workspace}
python train.py