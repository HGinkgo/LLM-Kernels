# SGEMM

矩阵乘法（`C = A × B`）的渐进式 CUDA 实现。该目录以 row-major `float` 矩阵为例，重点
练习线程映射、shared memory tiling、寄存器分块、向量化访存和软件流水。

## 优化路径

- v1：一个线程计算输出矩阵的一个元素。
- v2：使用 shared memory 对 A、B 进行 block tiling。
- v3：引入一维 thread tile，一个线程计算 `TM x 1` 个输出。
- v4：引入二维 thread tile，一个线程计算 `TM x TN` 个输出。
- v5：使用寄存器缓存从 shared memory 读取的 A、B fragment。
- v6：使用 `float4` 向量化搬运 global memory 数据，尾部使用标量边界路径。
- v7：使用两套 shared memory buffer，并在计算当前 K tile 时以寄存器预取下一轮数据。

当前 V4-V7 参数如下：

```text
M = 1003, N = 257, K = 129
BM = 32, BN = 32, BK = 16
TM = 4, TN = 4
block = 8 x 8
```

该 shape 同时覆盖了 M、N、K 方向的非整除边界。

## 正确性与内存检查

V4-V7 均与 CPU reference 比较，当前测试的最大绝对误差为 `0`。V7 已通过：

```text
compute-sanitizer --tool memcheck
ERROR SUMMARY: 0 errors
```

## 性能结果

### 历史基准：RTX 3090

以下是原始 RTX 3090（`sm_86`）数据，用于观察 V1-V5 的优化路径；不可与其他 GPU 的
绝对延迟直接比较。测试 shape 为 `M=1003, N=257, K=129`。

| 版本 | 时间（ms） | 说明 |
| --- | ---: | --- |
| v1 | 0.0373 | 每线程计算一个输出 |
| v2 | 0.0467 | shared memory 分块，小尺寸下同步开销较明显 |
| v3 | 0.0227 | `TM x 1` thread tile |
| v4 | 0.0168 | `TM x TN` thread tile |
| v5 | 0.0194 | 显式寄存器 fragment，当前参数下略慢于 v4 |

### 当前基准：RTX 4060 Ti

环境：Windows + WSL 2、RTX 4060 Ti、driver 610.74、CUDA 13.3、Release、`sm_89`。
对 V4-V7 分别运行 5 个独立进程；每个进程预热 100 次，采集 20 个样本，每个样本启动
kernel 1000 次。下表的“中位延迟”是 5 个进程中位延迟的中位数，P90 同理。

| 版本 | 中位延迟（ms） | P90（ms） | GFLOPS | 相对 V4 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| v4 | 0.0242 | 0.0290 | 2749.57 | 1.000x | pass |
| v5 | 0.0286 | 0.0297 | 2329.05 | 0.846x | pass |
| v6 | 0.0219 | 0.0271 | 3030.14 | 1.105x | pass |
| v7 | 0.0270 | 0.0278 | 2462.01 | 0.896x | pass |

该次测量中，V6 最快。V7 的寄存器和 shared memory 资源开销更高，在当前小 shape 与测量
状态下没有体现收益；后续应在稳定 GPU 时钟、更多 shape 和 Nsight Compute 指标下再判断
其优化效果。

## 构建与运行

以下命令从仓库根目录执行。RTX 4060 Ti 使用 CUDA 架构 `89`：

```bash
cmake -S cuda -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda --target sgemm -j
```

运行 V4-V7 的正确性测试与 benchmark：

```bash
./build/cuda/bin/sgemm --kernel all
```

只测试某个版本的正确性：

```bash
./build/cuda/bin/sgemm --kernel v7 --no-benchmark
```

支持的 `--kernel` 值为 `v4`、`v5`、`v6`、`v7` 和 `all`。`all` 为默认值。

进行内存错误检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/cuda/bin/sgemm --kernel v7 --no-benchmark
```
