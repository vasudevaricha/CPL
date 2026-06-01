# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !nvidia-smi  # (magic command commented out)

# %% [code] Cell 2
from numba import cuda
import numpy as np

# %% [code] Cell 3
import torch

print(torch.cuda.get_device_name(0))

# %% [code] Cell 4
import torch

props = torch.cuda.get_device_properties(0)
print(props.max_threads_per_block)

# %% [code] Cell 5
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

# %% [code] Cell 6
import torch

props = torch.cuda.get_device_properties(0)

print("GPU:", props.name)
print("Max threads/block:", props.max_threads_per_block)
print("Multiprocessors:", props.multi_processor_count)

# %% [code] Cell 7
from numba import cuda
import numpy as np

# CUDA kernel
@cuda.jit
def reduce_sum(arr):
    tid = cuda.threadIdx.x

    # Parallel reduction
    stride = 256
    while stride > 0:
        if tid < stride:
            arr[tid] += arr[tid + stride]
        cuda.syncthreads()
        stride //= 2

# Host data: numbers 1 to 512
data = np.arange(1, 513, dtype=np.int32)

# Copy to GPU
d_data = cuda.to_device(data)

# Launch kernel: 1 block, 512 threads
reduce_sum[1, 512](d_data)

# Copy back result
result = d_data.copy_to_host()

print("Sum =", result[0])
