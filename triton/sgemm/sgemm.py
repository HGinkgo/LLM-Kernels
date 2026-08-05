import argparse
import statistics

import torch
import triton
import triton.language as tl

@triton.jit
def sgemm_kernel_v1(
    matrix_a_ptr,
    matrix_b_ptr,
    matrix_c_ptr,
    num_rows,
    num_columns,
    num_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 当前 program 负责哪个输出 tile
    program_m = tl.program_id(axis=0)
    program_n = tl.program_id(axis=1)

    row_offsets = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k_offsets = tl.arange(0, BLOCK_K)

    # 每个 program 使用一个 BLOCK_M × BLOCK_N 累加器
    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    # 沿 K 维分块
    for k_block in range(0, tl.cdiv(num_k, BLOCK_K)):
        k_offsets = k_block * BLOCK_K + local_k_offsets

        # A tile: [BLOCK_M, BLOCK_K]
        matrix_a_offsets = (
            row_offsets[:, None] * stride_am +
            k_offsets[None, :] * stride_ak
        )

        # B tile: [BLOCK_K, BLOCK_N]
        matrix_b_offsets = (
            k_offsets[:, None] * stride_bk +
            column_offsets[None, :] * stride_bn
        )

        matrix_a_mask = (
            (row_offsets[:, None] < num_rows) &
            (k_offsets[None, :] < num_k)
        )

        matrix_b_mask = (
            (k_offsets[:, None] < num_k) &
            (column_offsets[None, :] < num_columns)
        )

        matrix_a_tile = tl.load(
            matrix_a_ptr + matrix_a_offsets,
            mask=matrix_a_mask,
            other=0.0,
        )

        matrix_b_tile = tl.load(
            matrix_b_ptr + matrix_b_offsets,
            mask=matrix_b_mask,
            other=0.0,
        )

        accumulator += tl.dot(
            matrix_a_tile,
            matrix_b_tile,
            input_precision="ieee",
        )

    matrix_c_offsets = (
        row_offsets[:, None] * stride_cm +
        column_offsets[None, :] * stride_cn
    )

    matrix_c_mask = (
        (row_offsets[:, None] < num_rows) &
        (column_offsets[None, :] < num_columns)
    )

    tl.store(
        matrix_c_ptr + matrix_c_offsets,
        accumulator,
        mask=matrix_c_mask,
    )

@triton.jit
def sgemm_kernel_v2(
    matrix_a_ptr,
    matrix_b_ptr,
    matrix_c_ptr,
    num_rows,
    num_columns,
    num_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    program_id = tl.program_id(axis=0)

    num_programs_m = tl.cdiv(num_rows, BLOCK_M)
    num_programs_n = tl.cdiv(num_columns, BLOCK_N)

    # 一组包含 GROUP_SIZE_M 行 tile，以及所有 N 方向 tile。
    num_programs_per_group = GROUP_SIZE_M * num_programs_n
    group_id = program_id // num_programs_per_group

    first_program_m = group_id * GROUP_SIZE_M

    # 最后一组可能不足 GROUP_SIZE_M 行。
    group_size_m = min(
        num_programs_m - first_program_m,
        GROUP_SIZE_M,
    )

    program_id_in_group = program_id % num_programs_per_group

    # M 方向变化更快：
    # (m0,n0), (m1,n0), ..., (m7,n0), (m0,n1), ...
    program_m = (
        first_program_m
        + program_id_in_group % group_size_m
    )
    program_n = program_id_in_group // group_size_m

    row_offsets = (
        program_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )
    column_offsets = (
        program_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )
    local_k_offsets = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for k_block in range(0, tl.cdiv(num_k, BLOCK_K)):
        k_offsets = (
            k_block * BLOCK_K
            + local_k_offsets
        )

        matrix_a_offsets = (
            row_offsets[:, None] * stride_am
            + k_offsets[None, :] * stride_ak
        )
        matrix_b_offsets = (
            k_offsets[:, None] * stride_bk
            + column_offsets[None, :] * stride_bn
        )

        matrix_a_mask = (
            (row_offsets[:, None] < num_rows)
            & (k_offsets[None, :] < num_k)
        )
        matrix_b_mask = (
            (k_offsets[:, None] < num_k)
            & (column_offsets[None, :] < num_columns)
        )

        matrix_a_tile = tl.load(
            matrix_a_ptr + matrix_a_offsets,
            mask=matrix_a_mask,
            other=0.0,
        )
        matrix_b_tile = tl.load(
            matrix_b_ptr + matrix_b_offsets,
            mask=matrix_b_mask,
            other=0.0,
        )

        accumulator += tl.dot(
            matrix_a_tile,
            matrix_b_tile,
            input_precision="ieee",
        )

    matrix_c_offsets = (
        row_offsets[:, None] * stride_cm
        + column_offsets[None, :] * stride_cn
    )
    matrix_c_mask = (
        (row_offsets[:, None] < num_rows)
        & (column_offsets[None, :] < num_columns)
    )

    tl.store(
        matrix_c_ptr + matrix_c_offsets,
        accumulator,
        mask=matrix_c_mask,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 16,
                "GROUP_SIZE_M": 8,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["num_rows", "num_columns", "num_k"],
)
@triton.jit
def sgemm_kernel_v3(
    matrix_a_ptr,
    matrix_b_ptr,
    matrix_c_ptr,
    num_rows,
    num_columns,
    num_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # V3 保留 V2 的 grouped ordering，只把 tile 参数交给 autotune 选择。
    program_id = tl.program_id(axis=0)

    num_programs_m = tl.cdiv(num_rows, BLOCK_M)
    num_programs_n = tl.cdiv(num_columns, BLOCK_N)
    num_programs_per_group = GROUP_SIZE_M * num_programs_n

    group_id = program_id // num_programs_per_group
    first_program_m = group_id * GROUP_SIZE_M
    group_size_m = min(
        num_programs_m - first_program_m,
        GROUP_SIZE_M,
    )
    program_id_in_group = program_id % num_programs_per_group

    program_m = (
        first_program_m
        + program_id_in_group % group_size_m
    )
    program_n = program_id_in_group // group_size_m

    row_offsets = (
        program_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )
    column_offsets = (
        program_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )
    local_k_offsets = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros(
        (BLOCK_M, BLOCK_N),
        dtype=tl.float32,
    )

    for k_block in range(0, tl.cdiv(num_k, BLOCK_K)):
        k_offsets = (
            k_block * BLOCK_K
            + local_k_offsets
        )

        matrix_a_offsets = (
            row_offsets[:, None] * stride_am
            + k_offsets[None, :] * stride_ak
        )
        matrix_b_offsets = (
            k_offsets[:, None] * stride_bk
            + column_offsets[None, :] * stride_bn
        )

        matrix_a_mask = (
            (row_offsets[:, None] < num_rows)
            & (k_offsets[None, :] < num_k)
        )
        matrix_b_mask = (
            (k_offsets[:, None] < num_k)
            & (column_offsets[None, :] < num_columns)
        )

        matrix_a_tile = tl.load(
            matrix_a_ptr + matrix_a_offsets,
            mask=matrix_a_mask,
            other=0.0,
        )
        matrix_b_tile = tl.load(
            matrix_b_ptr + matrix_b_offsets,
            mask=matrix_b_mask,
            other=0.0,
        )

        accumulator += tl.dot(
            matrix_a_tile,
            matrix_b_tile,
            input_precision="ieee",
        )

    matrix_c_offsets = (
        row_offsets[:, None] * stride_cm
        + column_offsets[None, :] * stride_cn
    )
    matrix_c_mask = (
        (row_offsets[:, None] < num_rows)
        & (column_offsets[None, :] < num_columns)
    )

    tl.store(
        matrix_c_ptr + matrix_c_offsets,
        accumulator,
        mask=matrix_c_mask,
    )




def sgemm_v1(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
) -> torch.Tensor:
    assert matrix_a.is_cuda
    assert matrix_b.is_cuda
    assert matrix_a.ndim == 2
    assert matrix_b.ndim == 2
    assert matrix_a.shape[1] == matrix_b.shape[0]
    assert matrix_a.dtype == torch.float32
    assert matrix_b.dtype == torch.float32
    assert matrix_a.device == matrix_b.device
    assert matrix_a.is_contiguous()
    assert matrix_b.is_contiguous()

    num_rows, num_k = matrix_a.shape
    _, num_columns = matrix_b.shape

    matrix_c = torch.empty(
        (num_rows, num_columns),
        device=matrix_a.device,
        dtype=torch.float32,
    )

    if num_rows == 0 or num_columns == 0:
        return matrix_c

    if num_k == 0:
        matrix_c.zero_()
        return matrix_c

    block_m = 32
    block_n = 32
    block_k = 16

    grid = (
        triton.cdiv(num_rows, block_m),
        triton.cdiv(num_columns, block_n),
    )

    sgemm_kernel_v1[grid](
        matrix_a,
        matrix_b,
        matrix_c,
        num_rows,
        num_columns,
        num_k,
        matrix_a.stride(0),
        matrix_a.stride(1),
        matrix_b.stride(0),
        matrix_b.stride(1),
        matrix_c.stride(0),
        matrix_c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )

    return matrix_c


def sgemm_v2(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
) -> torch.Tensor:
    assert matrix_a.is_cuda
    assert matrix_b.is_cuda
    assert matrix_a.ndim == 2
    assert matrix_b.ndim == 2
    assert matrix_a.shape[1] == matrix_b.shape[0]
    assert matrix_a.dtype == torch.float32
    assert matrix_b.dtype == torch.float32
    assert matrix_a.device == matrix_b.device
    assert matrix_a.is_contiguous()
    assert matrix_b.is_contiguous()

    num_rows, num_k = matrix_a.shape
    _, num_columns = matrix_b.shape

    matrix_c = torch.empty(
        (num_rows, num_columns),
        device=matrix_a.device,
        dtype=torch.float32,
    )

    if num_rows == 0 or num_columns == 0:
        return matrix_c

    if num_k == 0:
        matrix_c.zero_()
        return matrix_c

    block_m = 32
    block_n = 32
    block_k = 16
    group_size_m = 8

    num_programs_m = triton.cdiv(num_rows, block_m)
    num_programs_n = triton.cdiv(num_columns, block_n)
    grid = (num_programs_m * num_programs_n,)

    sgemm_kernel_v2[grid](
        matrix_a,
        matrix_b,
        matrix_c,
        num_rows,
        num_columns,
        num_k,
        matrix_a.stride(0),
        matrix_a.stride(1),
        matrix_b.stride(0),
        matrix_b.stride(1),
        matrix_c.stride(0),
        matrix_c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=4,
    )

    return matrix_c


def sgemm_v3(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
) -> torch.Tensor:
    assert matrix_a.is_cuda
    assert matrix_b.is_cuda
    assert matrix_a.ndim == 2
    assert matrix_b.ndim == 2
    assert matrix_a.shape[1] == matrix_b.shape[0]
    assert matrix_a.dtype == torch.float32
    assert matrix_b.dtype == torch.float32
    assert matrix_a.device == matrix_b.device
    assert matrix_a.is_contiguous()
    assert matrix_b.is_contiguous()

    num_rows, num_k = matrix_a.shape
    _, num_columns = matrix_b.shape

    matrix_c = torch.empty(
        (num_rows, num_columns),
        device=matrix_a.device,
        dtype=torch.float32,
    )

    if num_rows == 0 or num_columns == 0:
        return matrix_c

    if num_k == 0:
        matrix_c.zero_()
        return matrix_c

    # 每个候选配置使用自己的 BLOCK_M/BLOCK_N 计算 grid。
    def grid(meta):
        return (
            triton.cdiv(num_rows, meta["BLOCK_M"])
            * triton.cdiv(num_columns, meta["BLOCK_N"]),
        )

    sgemm_kernel_v3[grid](
        matrix_a,
        matrix_b,
        matrix_c,
        num_rows,
        num_columns,
        num_k,
        matrix_a.stride(0),
        matrix_a.stride(1),
        matrix_b.stride(0),
        matrix_b.stride(1),
        matrix_c.stride(0),
        matrix_c.stride(1),
    )

    return matrix_c


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.numel() == 0:
        return 0.0
    return (actual - expected).abs().max().item()


def benchmark_ms(
    launcher,
    warmup: int,
    samples: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        launcher()

    torch.cuda.synchronize()
    timings_ms = []

    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            launcher()
        stop.record()
        stop.synchronize()

        timings_ms.append(start.elapsed_time(stop) / iterations)

    timings_ms.sort()
    p90_index = (samples * 9 + 9) // 10 - 1

    return {
        "median_ms": statistics.median(timings_ms),
        "min_ms": timings_ms[0],
        "p90_ms": timings_ms[p90_index],
        "max_ms": timings_ms[-1],
    }


def run_correctness_tests() -> None:
    test_shapes = [
        (1, 1, 1),
        (7, 13, 5),
        (32, 32, 16),
        (33, 33, 17),
        (65, 97, 31),
        (1003, 257, 129),
    ]
    implementations = [
        ("Triton SGEMM V1", sgemm_v1),
        ("Triton SGEMM V2", sgemm_v2),
        ("Triton SGEMM V3", sgemm_v3),
    ]

    torch.manual_seed(42)

    for num_rows, num_columns, num_k in test_shapes:
        matrix_a = torch.randn(
            (num_rows, num_k),
            device="cuda",
            dtype=torch.float32,
        )
        matrix_b = torch.randn(
            (num_k, num_columns),
            device="cuda",
            dtype=torch.float32,
        )
        expected = torch.matmul(matrix_a, matrix_b)

        for label, implementation in implementations:
            actual = implementation(matrix_a, matrix_b)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.cuda.synchronize()

            print(
                f"{label}, shape=({num_rows}, {num_columns}, {num_k}), "
                f"max_abs_err={max_abs_error(actual, expected):.8f}, "
                "correctness=pass"
            )

    matrix_a = torch.empty((3, 0), device="cuda", dtype=torch.float32)
    matrix_b = torch.empty((0, 4), device="cuda", dtype=torch.float32)
    expected = torch.zeros((3, 4), device="cuda", dtype=torch.float32)

    for label, implementation in implementations:
        actual = implementation(matrix_a, matrix_b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"{label}, shape=(3, 4, 0), correctness=pass")


def print_benchmark_stats(
    label: str,
    stats: dict[str, float],
    num_rows: int,
    num_columns: int,
    num_k: int,
) -> None:
    operations = 2.0 * num_rows * num_columns * num_k
    tflops = operations / (stats["median_ms"] * 1e9)

    print(
        f"{label} median(ms)={stats['median_ms']:.4f}, "
        f"min(ms)={stats['min_ms']:.4f}, "
        f"p90(ms)={stats['p90_ms']:.4f}, "
        f"max(ms)={stats['max_ms']:.4f}, "
        f"TFLOPS={tflops:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    run_correctness_tests()

    if args.no_benchmark:
        return

    num_rows = 1003
    num_columns = 257
    num_k = 129
    warmup = 20
    samples = 20
    iterations = 100

    matrix_a = torch.randn(
        (num_rows, num_k),
        device="cuda",
        dtype=torch.float32,
    )
    matrix_b = torch.randn(
        (num_k, num_columns),
        device="cuda",
        dtype=torch.float32,
    )

    implementations = [
        ("Triton SGEMM V1", lambda: sgemm_v1(matrix_a, matrix_b)),
        ("Triton SGEMM V2", lambda: sgemm_v2(matrix_a, matrix_b)),
        ("Triton SGEMM V3", lambda: sgemm_v3(matrix_a, matrix_b)),
        ("PyTorch matmul", lambda: torch.matmul(matrix_a, matrix_b)),
    ]

    print(
        f"benchmark_shape=({num_rows}, {num_columns}, {num_k}), "
        "dtype=float32"
    )

    for label, launcher in implementations:
        stats = benchmark_ms(launcher, warmup, samples, iterations)
        print_benchmark_stats(
            label,
            stats,
            num_rows,
            num_columns,
            num_k,
        )

    print(
        f"benchmark warmup={warmup}, samples={samples}, "
        f"repetitions/sample={iterations}"
    )


if __name__ == "__main__":
    main()
