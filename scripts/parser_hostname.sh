#!/bin/bash

if [ $# -ne 2 ]; then
   echo "input argument counts is $#, which requires 2, means num worker, hostfile path, to exit"
   exit
fi

num_worker=$1
hostfile=$2


python3 core/utils/parser_file.py \
    --num_worker ${num_worker} \
    --hostfile ${hostfile}

