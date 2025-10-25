import torch
import triton.language as tl
import triton


@triton.jit
def _add(z_ptr, x_ptr, y_ptr, N):
    # same as torch.arange
    offsets = tl.arange(0, 1024)

    # create 1024 pointers to X, Y, Z
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    z_ptrs = z_ptr + offsets

    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)
    z = x + y
    tl.store(z_ptrs, z)

