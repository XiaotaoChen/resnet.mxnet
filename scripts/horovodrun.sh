#!/bin/bash
if [ $# -ne 4 ]; then
   echo "input argument counts is $#, which requires 4, means num proc, hostfile, single script, network, to exit"
   exit
fi

workspace=$(pwd)
num_proc=$1
hostfile=$2
single_script=$3
network=$4

/root/3rdparty/openmpi4.0/bin/mpirun -np ${num_proc} \
    --allow-run-as-root \
    --npernode 8 \
    --hostfile ${hostfile} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH \
    -x OMP_NUM_THREADS=1 \
    -mca pml ob1 \
    -mca btl ^openib \
    ${single_script} ${network} hvd

