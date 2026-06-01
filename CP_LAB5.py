# -*- coding: utf-8 -*-
"""
Converted from IPYNB to PY
"""

# %% [code] Cell 1
# !nvidia-smi  # (magic command commented out)


# %% [code] Cell 2
import numpy as np
from numba import cuda

@cuda.jit
def bitonic_sort_step(values, j, k):
    i = cuda.grid(1)

    if i < values.size:
        ixj = i ^ j

        if ixj > i:
            # Ascending order
            if (i & k) == 0:
                if values[i] > values[ixj]:
                    temp = values[i]
                    values[i] = values[ixj]
                    values[ixj] = temp
            # Descending order
            else:
                if values[i] < values[ixj]:
                    temp = values[i]
                    values[i] = values[ixj]
                    values[ixj] = temp


def bitonic_sort_gpu(arr):
    n = arr.size

    d_arr = cuda.to_device(arr)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    k = 2
    while k <= n:
        j = k >> 1
        while j > 0:
            bitonic_sort_step[blocks, threads_per_block](d_arr, j, k)
            cuda.synchronize()
            j >>= 1
        k <<= 1

    return d_arr.copy_to_host()


if __name__ == "__main__":
    arr = np.array([3, 7, 4, 8, 6, 2, 1, 5], dtype=np.int32)

    print("Original array:")
    print(arr)

    sorted_arr = bitonic_sort_gpu(arr)

    print("Sorted array:")
    print(sorted_arr)
