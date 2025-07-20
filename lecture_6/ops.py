import torch
import triton
import triton.language as tl


# gelu
def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    blk_start = pid*BLOCK_SIZE

    offsets = blk_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask)

    a = 0.79788456 * (x + 0.044715 * x * x * x)

    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    numel = x.numel()
    block_size = 1024
    num_blks = triton.cdiv(numel, block_size)

    triton_gelu_kernel[(num_blks,)](x, y, numel, BLOCK_SIZE=block_size)  # 这里直接用字面量

    return y

# softmax

def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)

def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape
    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]
    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]
    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)
    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)
    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]
    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y

@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)

def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)
    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row
    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y



