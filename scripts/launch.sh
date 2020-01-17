#!/bin/bash
if [ $# -ne 4 ]; then
   echo "input argument counts is $#, which requires 4, means num node, hostfile, single script, network, to exit"
   exit
fi

workspace=$(pwd)
num_node=$1
hostfile=$2
single_script=$3
network=$4


export OMP_NUM_THREADS=1
#export KMP_AFFINITY granularity=fine,noduplicates,compact,1,0

python3 /root/incubator-mxnet/tools/launch.py \
       -n ${num_node} \
       -H ${hostfile} \
       --launcher ssh \
       ${single_script} ${network} dist_sync
