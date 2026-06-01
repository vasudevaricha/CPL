# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !nvidia-smi  # (magic command commented out)

# %% [markdown] Cell 2
# Precautions to be taken in Cuda.
# Numerical computations in cuda occurs through numba.First computation goes to CPU and then it gives further to GPU.
# 

# %% [code] Cell 3
from numba import cuda
import numpy as np


# %% [code] Cell 4
# -------------------------------
# CUDA Kernel
# -------------------------------
@cuda.jit
def add_arrays(a, b, c):
  i = cuda.grid(1) # Global thread index
  if i < a.size:
    c[i] = a[i] + b[i]
# -------------------------------
# Host Code
# -------------------------------
n = 1024
# Create input arrays on CPU

a = np.arange(n, dtype=np.float32)
b = np.arange(n, dtype=np.float32)

# Allocate output array

c = np.zeros(n, dtype=np.float32)

# Copy arrays to GPU

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# CUDA configuration

threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
# Launch kernel

add_arrays[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy result back to CPU

c = d_c.copy_to_host()
# Print result

print("First 10 results:", c[:10])

# %% [code] Cell 6
# -------------------------------
# CUDA Kernel for 2D arrays
# -------------------------------
@cuda.jit
def add_arrays_2d(a, b, c):
  x, y = cuda.grid(2) # Global thread index for 2D
  if x < a.shape[0] and y < a.shape[1]:
    c[x, y] = a[x, y] + b[x, y]

# -------------------------------
# Host Code for 2D arrays
# -------------------------------

rows = 5
cols = 5

# Create input 2D arrays on CPU
a_2d = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
b_2d = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)

# Allocate output 2D array

c_2d= np.zeros((rows, cols), dtype=np.float32)

# Copy arrays to GPU
d_a_2d = cuda.to_device(a_2d)
d_b_2d = cuda.to_device(b_2d)
d_c_2d = cuda.to_device(c_2d)

# CUDA configuration for 2D

threads_per_block_2d_new = (16, 16) # Example: 16x16 threads per block
blocks_per_grid_x_new = (rows + threads_per_block_2d_new[0] - 1) // threads_per_block_2d_new[0]
blocks_per_grid_y_new = (cols + threads_per_block_2d_new[1] - 1) // threads_per_block_2d_new[1]
blocks_per_grid_2d_new = (blocks_per_grid_x_new, blocks_per_grid_y_new)

# Launch kernel
add_arrays_2d[blocks_per_grid_2d_new, threads_per_block_2d_new](d_a_2d, d_b_2d, d_c_2d)

# Copy result back to CPU
c_2d = d_c_2d.copy_to_host()

# Print result
print("First 5x5 results from new 2D addition:\n", c_2d[:5, :5])
