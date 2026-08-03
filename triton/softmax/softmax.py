import argparse
import statistics

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    num_columns,
    BLOCK_SIZE: tl.constexpr,
):
    # 一个 program 负责一行
    row = tl.program_id(axis=0)

    column_offsets = tl.arange(0, BLOCK_SIZE)
    mask = column_offsets < num_columns

    input_offsets = (row * input_row_stride + column_offsets)

    # 无效位置读取为 -inf，不影响后面的 max
    values = tl.load(
        input_ptr + input_offsets,
        mask = mask,
        other = -float("inf"),
    ).to(tl.float32)

    row_max = tl.max(values, axis=0)

    numerator = tl.exp(values - row_max)
    denominator = tl.sum(numerator, axis=0)

    probailities = numerator / denominator

    output_offsets = (
        row * output_row_stride +
        column_offsets
    )

    tl.store(
        output_ptr + output_offsets,
        probailities,
        mask=mask,
    )


def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    assert input_tensor.is_cuda
    assert input_tensor.ndim == 2
    assert input_tensor.is_contiguous()

    num_rows, num_columns = input_tensor.shape
    output = torch.empty_like(input_tensor)

    if num_rows == 0 or num_columns == 0:
        return output

    # tl.arange 的范围通常设置为 2 的幂
    block_size = triton.next_power_of_2(num_columns)

    # 当前教学版本只处理适中的行长度
    assert block_size <= 65536

    num_warps = 4
    if block_size >= 2048:
        num_warps = 8

    grid = (num_rows,)

    softmax_kernel[grid](
        input_tensor,
        output,
        input_tensor.stride(0),
        output.stride(0),
        num_columns,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )

    return output


def max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual - expected).abs().max().item()


def benchmark_ms(launcher, warmup: int, samples: int, iterations: int) -> dict[str, float]:
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
        (1, 1),
        (7, 257),
        (32, 511),
        (64, 1025),
    ]

    torch.manual_seed(42)

    for num_rows, num_columns in test_shapes:
        input_tensor = torch.randn(
            (num_rows, num_columns),
            device="cuda",
            dtype=torch.float32,
        )

        actual = softmax(input_tensor)
        expected = torch.softmax(input_tensor, dim=-1)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.cuda.synchronize()

        print(
            f"shape=({num_rows}, {num_columns}), "
            f"max_abs_err={max_abs_error(actual, expected):.8f}, correctness=pass"
        )


def print_benchmark_stats(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label} median(ms)={stats['median_ms']:.4f}, "
        f"min(ms)={stats['min_ms']:.4f}, "
        f"p90(ms)={stats['p90_ms']:.4f}, "
        f"max(ms)={stats['max_ms']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    run_correctness_tests()

    if args.no_benchmark:
        return

    benchmark_shape = (2048, 257)
    warmup = 20
    samples = 20
    iterations = 100

    input_tensor = torch.randn(
        benchmark_shape,
        device="cuda",
        dtype=torch.float32,
    )

    triton_stats = benchmark_ms(
        lambda: softmax(input_tensor),
        warmup,
        samples,
        iterations,
    )
    torch_stats = benchmark_ms(
        lambda: torch.softmax(input_tensor, dim=-1),
        warmup,
        samples,
        iterations,
    )

    print(f"benchmark_shape={benchmark_shape}, dtype=float32")
    print_benchmark_stats("Triton Softmax", triton_stats)
    print_benchmark_stats("PyTorch Softmax", torch_stats)
    print(
        f"benchmark warmup={warmup}, samples={samples}, "
        f"repetitions/sample={iterations}"
    )


if __name__ == "__main__":
    main()
