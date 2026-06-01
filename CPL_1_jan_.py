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
@cuda.jit
def hello_kernel():
  # Thread and block indices
  tx=cuda.threadIdx.x
  bx=cuda.blockIdx.x
  bdim=cuda.blockDim.x


  # Global thread ID
  gid = bx*bdim+tx
  print("hellp from block ",bx,"thread ",tx,"Global id ", gid)

# %% [code] Cell 4
blocks =2
threads_per_block=4
hello_kernel[blocks,threads_per_block]()
cuda.synchronize()
