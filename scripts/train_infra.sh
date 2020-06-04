#!/bin/bash
eval $(cd && .tspkg/bin/tsp --env)

workspace="/root/resnet.mxnet"

cd ${workspace}

python3 infra_train.py