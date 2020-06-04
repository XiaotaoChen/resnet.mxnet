#!/bin/bash
if [ $# -ne 3 ]; then
   echo "input argument counts is $#, which requires 3, means number of worker, hostfile, single script, to exit"
   exit
fi

num_worker=$1
hostfile=$2
single_script=$3
workspace=$(pwd)
num_proc=$[${num_worker}*8]

if [ -f "${hostfile}" ]; then
    echo "${hostfile} is existd, remove it"
    rm ${hostfile}
fi

./scripts/parser_hostname.sh ${num_worker} ${hostfile}

if [ ! -f "${hostfile}" ]; then
    echo "parser host name error, can't generate hostfile: ${hostfile}"
    exit
fi

/root/3rdparty/openmpi4.0/bin/mpirun -np ${num_proc} \
    --allow-run-as-root \
    --npernode 8 \
    --hostfile ${hostfile} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SHM_DISABLE=1 \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -x PYTHONPATH \
    -x OMP_NUM_THREADS=1 \
    -mca pml ob1 \
    -mca btl ^openib \
    ${single_script}

