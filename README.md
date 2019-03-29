# Test PaddlePaddle's Initial and Re-alloc Flags

On shell run `sh run.sh`

Then open file gpu_usage.txt, which records the gpu usage by nvidia-smi.

Expected result is GPU was allocated 500MB and then re-alloc 100 MB for three times.

You can modify the GPU id, initial size, re-alloc size, re-alloc times  and output file by modifying run.sh

