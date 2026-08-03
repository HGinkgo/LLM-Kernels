import torch
import triton
import triton.language as tl


@triton.jit
def elementwise_add_kernel(
    input_a_ptr,
    input_b_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(axis=0)

    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < num_elements

    input_a = tl.load(input_a_ptr + offsets, mask=mask)
    input_b = tl.load(input_b_ptr + offsets, mask=mask)

    output = input_a + input_b

    tl.store(output_ptr + offsets, output, mask=mask)


def elementwise_add(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
) -> torch.Tensor:
    assert input_a.is_cuda
    assert input_b.is_cuda
    assert input_a.shape == input_b.shape
    assert input_a.dtype == input_b.dtype
    assert input_a.device == input_b.device
    assert input_a.is_contiguous()
    assert input_b.is_contiguous()

    output = torch.empty_like(input_a)
    num_elements = output.numel()

    if num_elements == 0:
        return output

    block_size = 256
    grid = (triton.cdiv(num_elements, block_size),)

    elementwise_add_kernel[grid](
        input_a,
        input_b,
        output,
        num_elements,
        BLOCK_SIZE=block_size,
    )

    return output
