import torch
import triton
import triton.language as tl

'''
    rms = sqrt(mean(x²) + eps)
    output[i] = x[i] / rms * weight[i]
'''

@triton.jit
def rmsnorm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一行 hidden state
    row = tl.program_id(axis=0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    input_offsets = row * input_row_stride + offsets
    output_offsets = row * output_row_stride + offsets

    # 使用 float32 累加，提高数值稳定性
    values = tl.load(
        input_ptr + input_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # mean(x^2)
    square_sum = tl.sum(values * values, axis=0)
    mean_square = square_sum / hidden_size

    # 1 / sqrt(mean(x^2) + eps)
    inverse_rms = tl.rsqrt(mean_square + eps)

    normalized = values * inverse_rms
    result = normalized * weight

    tl.store(
        output_ptr + output_offsets,
        result,
        mask=mask,
    )

def rmsnorm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert input_tensor.is_cuda
    assert weight.is_cuda
    assert input_tensor.ndim == 2
    assert weight.ndim == 1
    assert input_tensor.shape[-1] == weight.numel()
    assert input_tensor.is_contiguous()
    assert weight.is_contiguous()
    assert input_tensor.device == weight.device

    num_rows, hidden_size = input_tensor.shape
    output = torch.empty_like(input_tensor)

    if num_rows == 0 or hidden_size == 0:
        return output

    block_size = triton.next_power_of_2(hidden_size)

    assert block_size <= 65536

    num_warps = 4
    if block_size >= 2048:
        num_warps = 8

    grid = (num_rows,)

    rmsnorm_kernel[grid](
        input_tensor,
        weight,
        output,
        input_tensor.stride(0),
        output.stride(0),
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output
