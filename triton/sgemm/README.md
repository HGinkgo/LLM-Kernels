# Triton SGEMM

本目录实现 float32 矩阵乘法：

```text
A: [M, K]
B: [K, N]
C: [M, N]
C = A @ B
```

## 版本

- V1：使用二维 grid，一个 program 计算一个 `BLOCK_M × BLOCK_N` 的 C tile。
- V2：使用一维 grid 和 grouped ordering，让相邻 program 更可能复用相同的 B tile，
  提高 L2 Cache 命中率。
- V3：保留 V2 的 grouped ordering，通过 `triton.autotune` 为不同矩阵形状选择 tile、
  warp 和 pipeline 参数。

两个版本当前均使用：

```text
BLOCK_M=32
BLOCK_N=32
BLOCK_K=16
num_warps=4
```

V2 额外使用 `GROUP_SIZE_M=8`。V3 将以下参数作为候选配置：

```text
(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
(32, 32, 16, 4, 2)
(64, 32, 32, 4, 2)
(32, 64, 32, 4, 2)
(64, 64, 16, 8, 2)
```

V3 的 wrapper 使用 `grid(meta)`，因此每个候选配置会按照自己的 tile 大小计算 program
数量。`key=["num_rows", "num_columns", "num_k"]` 表示 Triton 会按输入形状缓存调优结果。

## 正确性

测试环境为 Windows + WSL 2、RTX 4060 Ti（`sm_89`）、float32。GPU 结果与
`torch.matmul` 比较，容差为 `rtol=1e-4, atol=1e-4`。

| Shape `(M, N, K)` | V1 max absolute error | V2 max absolute error |
| --- | ---: | ---: |
| `(1, 1, 1)` | 0.00000000 | 0.00000000 |
| `(7, 13, 5)` | 0.00000024 | 0.00000024 |
| `(32, 32, 16)` | 0.00000119 | 0.00000119 |
| `(33, 33, 17)` | 0.00000191 | 0.00000191 |
| `(65, 97, 31)` | 0.00000572 | 0.00000572 |
| `(1003, 257, 129)` | 0.00003052 | 0.00003052 |

另外测试了 `(M, N, K)=(3, 4, 0)`，V1/V2 均正确返回全零矩阵。

## 性能

benchmark shape 为 `(M, N, K)=(1003, 257, 129)`。每轮预热 20 次，采集 20 个
样本，每个样本重复 100 次，只使用 CUDA Event 统计 GPU 时间。完整测试重复三轮，表中
为三轮各自 Median/P90 的中位数。

| 实现 | Median（ms） | P90（ms） | Median TFLOPS |
| --- | ---: | ---: | ---: |
| Triton SGEMM V1 | 0.0339 | 0.0396 | 1.959 |
| Triton SGEMM V2 | 0.0336 | 0.0509 | 1.982 |
| Triton SGEMM V3 | 0.0462 | 0.0844 | 1.441 |
| PyTorch matmul | 0.0246 | 0.0304 | 2.696 |

当前尺寸较小且不规则，kernel 延迟容易受到 GPU 时钟和系统负载影响。V2 的 grouped
ordering 在本轮结果中与 V1 接近。V3 在这个尺寸上选择结果会有波动，且本轮慢于 V1/V2；
autotune 不保证所有输入尺寸都比手动参数更快，候选配置需要覆盖目标 workload。

autotune 的第一次调用包含候选配置测量开销，正式 benchmark 必须先预热。不同矩阵形状会
分别触发调优和缓存。

## 运行

从仓库根目录执行：

```bash
source .venv/bin/activate
python triton/sgemm/sgemm.py
```

只执行正确性测试：

```bash
python triton/sgemm/sgemm.py --no-benchmark
```
