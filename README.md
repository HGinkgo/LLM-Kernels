# LLM-Kernels

面向 LLM 推理的 CUDA/Triton 算子学习仓库，记录从基础实现到性能优化的渐进过程。
项目主要用于 AI Infra 岗位的算子手写、性能分析和问题定位练习，不以直接用于生产环境
为目标。

当前开发与测试环境为 Windows + WSL 2、RTX 4060 Ti（`sm_89`）。

## 目录

- `cuda/`：按算子分类的 CUDA 实现。
- `triton/`：代表性算子的 Triton 实现与 PyTorch baseline。

已完成的算子通常包含核函数、CPU reference、正确性测试、benchmark 和优化记录。kernel
的线程映射、数据布局、边界处理及核心优化以手写和理解为主，测试与工程代码允许使用 AI
辅助。

## 算子状态

| 类别 | 算子 | 状态 |
| --- | --- | --- |
| CUDA Reduction | Reduce Sum V1-V4、Reduce Max | 已完成 |
| CUDA Softmax | 1D Softmax V1-V2、Matrix Softmax V1-V4 | 已完成 |
| CUDA Elementwise | Add V1-V3 | 已完成 |
| CUDA Memory | Matrix Transpose V1-V4 | 已完成 |
| CUDA Normalization | RMSNorm V1-V4、LayerNorm V1-V4 | 已完成 |
| CUDA Linear Algebra | GEMV V1-V3、SGEMM V1-V7 | 已完成 kernel 版本 |
| CUDA LLM | RoPE V1-V2 | 已完成 |
| CUDA Attention | Naive Attention | 已完成 |
| CUDA Attention | 简化 FlashAttention V1-like | 已完成 |
| Triton | Elementwise Add、Matrix Softmax | 已完成基础版本 |

## SGEMM 优化路径

SGEMM 当前包含 V1-V7，逐步引入以下优化：

1. 一个线程计算一个输出元素。
2. 使用 shared memory 完成 block tiling。
3. 使用一维 thread tile，让一个线程计算 `TM x 1` 个输出。
4. 使用二维 thread tile，让一个线程计算 `TM x TN` 个输出。
5. 使用寄存器缓存 A、B fragment。
6. 使用 `float4` 向量化搬运 global memory 数据。
7. 使用两套 shared memory buffer 和寄存器预取下一轮 K tile。

当前 V4-V7 的测试 shape 为 `M=1003, N=257, K=129`，包含非整除边界；tile 参数为
`BM=32, BN=32, BK=16, TM=4, TN=4`。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU
- CUDA Toolkit，包含 `nvcc`
- CMake 3.24 或更高版本
- 可选：Compute Sanitizer 和 Nsight Compute

配置时应显式指定目标 GPU 的 CUDA 架构。RTX 4060 Ti 对应 `89`：

```bash
cmake -S cuda -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

## 构建与运行

以下命令均从仓库根目录执行。

构建全部已配置 target：

```bash
cmake --build build/cuda -j
```

也可以只构建当前算子：

```bash
cmake --build build/cuda --target sgemm -j
```

所有 CUDA 可执行文件输出到 `build/cuda/bin/`。例如：

```bash
./build/cuda/bin/gemv
./build/cuda/bin/sgemm --kernel all
./build/cuda/bin/sgemm --kernel v7 --no-benchmark
./build/cuda/bin/attention_flash --no-benchmark
```

SGEMM 支持选择 `v4`、`v5`、`v6`、`v7` 或 `all`。默认先与 CPU reference 比较
正确性，再进行 benchmark；`--no-benchmark` 只执行正确性测试。

## 测试与性能记录

当前 SGEMM benchmark 使用 CUDA Event 计时：预热 100 次，采集 20 组样本，每组重复
启动 kernel 1000 次，报告 median、min、P90、max 和 GFLOPS。Host/device memcpy 不计入
纯 kernel 耗时。

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/cuda/bin/sgemm --kernel v7 --no-benchmark
```

各算子目录中的 README 用于记录优化路径、测试 shape、正确性与性能结果。性能数据只用于
比较同一硬件、同一环境中不同版本的相对变化；短 kernel 的绝对耗时会受到 GPU 时钟和
系统负载影响。

## 代码风格

- `.clang-format` 定义 C/C++/CUDA 格式，使用 4 空格缩进。
- `.editorconfig` 定义 LF 换行、文件末尾换行等基础编辑器行为。

## License

本项目使用 MIT License，详见 `LICENSE`。
