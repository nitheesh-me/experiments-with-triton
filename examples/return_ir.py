import torch
import triton
import triton.language as tl


@triton.jit
def return_ir(BLOCK_SIZE: tl.constexpr):
    range = tl.arange(0, 8)
    return

# @triton.jit
# def add_kernel(x_ptr,  # *Pointer* to first input vector.
#                y_ptr,  # *Pointer* to second input vector.
#                output_ptr,  # *Pointer* to output vector.
#                n_elements,
#                BLOCK_SIZE: tl.constexpr,
#                ):

#     pid = tl.program_id(axis=0)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)

#     mask = offsets < n_elements

#     x = tl.load(x_ptr + offsets, mask=mask)
#     y = tl.load(y_ptr + offsets, mask=mask)
#     output = x + y
#     tl.store(output_ptr + offsets, output, mask=mask)

if __name__ == '__main__':
    # If run directly, also print the IR to simulate modules that print at import
    # x = torch.full_like(torch.zeros(10), 1.0)
    grid = lambda meta: (triton.cdiv(0, meta['BLOCK_SIZE']), )
    # # triton_kernel=add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # torch.cuda.synchronize()
    tt_obj = return_ir[grid](BLOCK_SIZE=1024)
    # breakpoint()
    "Hello"