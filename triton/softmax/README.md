# Triton Softmax

本目录实现二维连续 Tensor 沿最后一维的 row-wise Softmax。当前教学版本使用一个 Triton
program 处理一行，通过 `tl.max` 和 `tl.sum` 完成稳定 softmax 归约。

## 数据与 program 映射

```text
Input/Output: [num_rows, num_columns]
grid:         (num_rows,)
program 0:    row 0
program 1:    row 1
...
```

`BLOCK_SIZE` 使用 `triton.next_power_of_2(num_columns)`。当列数为 257 时，一个 program
生成 512 个逻辑位置，尾部位置由 mask 屏蔽。无效位置按 `-inf` 加载，因此不影响最大值，
经过指数运算后也不会影响 softmax 分母。

## 正确性

测试环境：Windows + WSL 2、RTX 4060 Ti（`sm_89`）、PyTorch 2.13.0+cu130、
Triton 3.7.1、float32。

GPU 结果与 `torch.softmax(input, dim=-1)` 比较：

| Shape | Max absolute error |
| --- | ---: |
| `(1, 1)` | 0.00000000 |
| `(7, 257)` | 0.00000000 |
| `(32, 511)` | 0.00000001 |
| `(64, 1025)` | 0.00000001 |

257、511 和 1025 均覆盖非 2 的幂列数及 mask 尾部。

## 性能

benchmark shape 为 `(2048, 257)`。预热 20 次，采集 20 个样本，每个样本重复 100 次，
只使用 CUDA Event 统计 GPU 时间。

| 实现 | Median（ms） | P90（ms） |
| --- | ---: | ---: |
| Triton Softmax | 0.0250 | 0.0284 |
| PyTorch Softmax | 0.0136 | 0.0203 |

当前 Triton 实现以学习 program mapping、mask 和 reduction 为目标，尚未使用 persistent
program、autotune 或针对不同列数的专用配置。短 kernel 的绝对延迟会受到 GPU 时钟和
系统负载影响，应主要比较同一轮测试中的相对结果。

## 运行

从仓库根目录执行：

```bash
source .venv/bin/activate
python triton/softmax/softmax.py
```

只执行正确性测试：

```bash
python triton/softmax/softmax.py --no-benchmark
```
