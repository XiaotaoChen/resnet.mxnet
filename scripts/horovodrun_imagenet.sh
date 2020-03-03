#!/bin/bash
if [ $# -ne 3 ]; then
   echo "input argument counts is $#, which requires 3, means num proc, hostfile, single script to exit"
   exit
fi

workspace=$(pwd)
num_proc=$1
hostfile=$2
single_script=$3

/root/3rdparty/openmpi4.0/bin/mpirun -np ${num_proc} \
    --allow-run-as-root \
    --npernode 8 \
    --hostfile ${hostfile} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH \
    -x OMP_NUM_THREADS=4 \
    -mca btl_tcp_if_include 10.10.240.0/24 \
    -mca pml ob1 \
    -mca btl ^openib \
    ${single_script}

# /root/3rdparty/openmpi4.0/bin/mpirun -np ${num_proc} \
#     --allow-run-as-root \
#     --npernode 8 \
#     --hostfile ${hostfile} \
#     -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_SHM_DISABLE=1 \
#     -x LD_LIBRARY_PATH \
#     -x PATH \
#     -x PYTHONPATH \
#     -x OMP_NUM_THREADS=4 \
#     -mca btl_tcp_if_include 10.10.240.0/24 \
#     -mca pml ob1 \
#     -mca btl ^openib \
#     ${single_script}

# /root/3rdparty/openmpi4.0/bin/mpirun -np ${num_proc} \
#     --allow-run-as-root \
#     --npernode 8 \
#     --hostfile ${hostfile} \
#     -bind-to none -map-by slot \
#     -x NCCL_DEBUG=INFO \
#     -x HOROVOD_AUTOTUNE=1 \
#     -x HOROVOD_AUTOTUNE_LOG=/tmp/autotune_log.csv \
#     -x LD_LIBRARY_PATH \
#     -x PATH \
#     -x PYTHONPATH \
#     -x OMP_NUM_THREADS=4 \
#     -mca btl_tcp_if_include 10.10.240.0/24 \
#     -mca pml ob1 \
#     -mca btl ^openib \
#     ${single_script}
