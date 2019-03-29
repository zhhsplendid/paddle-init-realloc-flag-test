import argparse
import commands
import os
import shlex
import subprocess
import time
import threading

import numpy as np

import paddle
from paddle import fluid

# Collect GPU memory usage for each following milli-seconds
COLLECT_GPU_MEM_USAGE_LOOP_MS = 100

# GPU will initially occupied by CUDA and Paddle about 98 MB
INIT_GPU_OCCUP_USAGE_MB = 98

# Left some space in initial or realloc chunk.
LEFT_GPU_MB = 2

# 1 MB data is 1024 * 1024 / 8 numpy float64 data
MB_TO_NUM_NP = 1024 * 1024 / 8


def parse_args():
    parser = argparse.ArgumentParser("Test init and re-alloc FLAGS")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="The GPU Card Id. (default: %(default)d)")
    parser.add_argument(
        "--flag_init_gpu_mb",
        type=int,
        default=500,
        help="PaddlePaddle's FLAGS_initial_gpu_memory_in_mb")
    parser.add_argument(
        "--flag_realloc_gpu_mb",
        type=int,
        default=100,
        help="PaddlePaddle's FLAGS_reallocate_gpu_memory_in_mb")
    parser.add_argument(
        "--output_file",
        type=str,
        default="gpu_usage.txt",
        help="The name of output file recording GPU memory usage")
    parser.add_argument(
        "--num_realloc",
        type=int,
        default=3,
        help="The number of re-allocation")
    return parser.parse_args()


def paddle_new_tensor(gpu_id, init_flag, realloc_flag, num_realloc):
    init_mb = init_flag - LEFT_GPU_MB
    realloc_mb = realloc_flag - LEFT_GPU_MB

    tensor = fluid.Tensor()
    tensor.set(np.random.rand(init_mb * MB_TO_NUM_NP), fluid.CUDAPlace(gpu_id))
    print("Init alloc %d MB, gpu usage report from fluid: %d" %
          (init_mb, fluid.core.get_mem_usage(gpu_id)))
    # Sleep and wait for nvidia-smi subprocess to collect GPU usage
    time.sleep(COLLECT_GPU_MEM_USAGE_LOOP_MS / 1000)

    # We don't use for loop when re-alloc, else the tensor will be recycled
    re_tensor = [fluid.Tensor() for i in range(num_realloc)]
    for i in range(num_realloc):
        re_tensor[i].set(
            np.random.rand(realloc_mb * MB_TO_NUM_NP), fluid.CUDAPlace(gpu_id))
        print("Re-alloc %d MB, gpu usage report from fluid: %d" %
              (realloc_mb, fluid.core.get_mem_usage(gpu_id)))
        # Sleep and wait for nvidia-smi subprocess to collect GPU usage
        time.sleep(COLLECT_GPU_MEM_USAGE_LOOP_MS / 1000)
    time.sleep(COLLECT_GPU_MEM_USAGE_LOOP_MS / 1000)


def gpu_monitor_subproc_from_nvidia(gpu_id, filename):
    """
  Collect the GPU memory usage from nvidia-smi command. This function starts a
  subprocess to run the command and output used memory if non-zero into file.
  Note: you should kill the subprocess returned by this function

  param:
    gpu_id: int, the id of the gpu device to monitor
    filename: string, the filename to output gpu usage
  return:
    the subprocess 
  """
    outfile = open(filename, "w")
    command = "nvidia-smi --id=%s --query-compute-apps=used_memory --format=csv \
      -lms %d" % (gpu_id, COLLECT_GPU_MEM_USAGE_LOOP_MS)
    p = subprocess.Popen(shlex.split(command), stdout=outfile)
    return p


if __name__ == '__main__':
    args = parse_args()
    gpu_monitor_proc = gpu_monitor_subproc_from_nvidia(args.gpu_id,
                                                       args.output_file)
    paddle_new_tensor(args.gpu_id, args.flag_init_gpu_mb,
                      args.flag_realloc_gpu_mb, args.num_realloc)
    gpu_monitor_proc.kill()
    print("Please see %s file for GPU usage report from nivida-smi" %
          (args.output_file))
