#!/bin/bash

GPU_ID=0
INITIAL_GPU_MB=500
REALLOC_GPU_MB=100
OUTPUT_FILE="gpu_usage.txt"
REALLOC_TIMES=3

FLAGS_initial_gpu_memory_in_mb=$INITIAL_GPU_MB \
FLAGS_reallocate_gpu_memory_in_mb=$REALLOC_GPU_MB \
FLAGS_benchmark=true \
python init_realloc_flags_test.py --gpu_id=$GPU_ID \
    --flag_init_gpu_mb=$INITIAL_GPU_MB \
    --flag_realloc_gpu_mb=$REALLOC_GPU_MB \
    --output_file=$OUTPUT_FILE \
    --num_realloc=$REALLOC_TIMES

