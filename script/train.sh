#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/truenas/scratch/xiaotao.chen/Repositories/incubator-mxnet/lib
export PYTHONPATH=$PYTHONPATH:/mnt/truenas/scratch/xiaotao.chen/Repositories/incubator-mxnet/python

python3 train.py
