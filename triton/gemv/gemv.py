import torch
import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    matrix_ptr,
    vector_ptr,
    output_ptr,
    matrix_row_stride,
    matrix_col_stride,
    vector_stride,
    output_stride,
    num_columns,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一个输出行
    row = tl.program_id(axis=0)

    column_offsets = tl.arange(0, BLOCK_SIZE)
    mask = column_offsets < num_columns

    matrix_offsets = (
        row * matrix_row_stride +
        column_offsets * matrix_col_stride
    )

    vector_offsets = column_offsets * vector_stride

    matrix_values = tl.load(
        matrix_ptr + matrix_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    vector_values = tl.load(
        vector_ptr + vector_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    output_value = tl.sum(
        matrix_values * vector_values,
        axis=0,
    )

    tl.store(
        output_ptr + row * output_stride,
        output_value,
    )

def gemv(
    matrix: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    assert matrix.is_cuda
    assert vector.is_cuda
    assert matrix.ndim == 2
    assert vector.ndim == 1
    assert matrix.shape[1] == vector.shape[0]
    assert matrix.is_contiguous()
    assert vector.is_contiguous()
    assert matrix.device == vector.device

    num_rows, num_columns = matrix.shape
    output = torch.empty(
        (num_rows,),
        device=matrix.device,
        dtype=matrix.dtype,
    )

    if num_rows == 0 or num_columns == 0:
        return output

    block_size = triton.next_power_of_2(num_columns)

    assert block_size <= 65536

    num_warps = 4
    if block_size >= 2048:
        num_warps = 8

    grid = (num_rows,)

    gemv_kernel[grid](
        matrix,
        vector,
        output,
        matrix.stride(0),
        matrix.stride(1),
        vector.stride(0),
        output.stride(0),
        num_columns,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output
