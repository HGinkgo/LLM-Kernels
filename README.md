# LLM-Kernels

面向 LLM 推理的 CUDA/Triton 算子学习仓库，记录从基础实现到性能优化的演进过程。
当前主要在 RTX 3090（`sm_86`）上开发和测试，不以生产环境直接使用为目标。

## 目录

- `cuda/`：按算子分类的 CUDA 实现。
- `triton/`：Triton 学习目录，目前为占位内容。

已完成的算子在各自目录中保留核函数、CPU 正确性参考、benchmark 主程序和优化记录。

## 算子状态

| 类别 | 算子 | 状态 |
| --- | --- | --- |
| CUDA Reduce | Sum、Max、1D Softmax、Matrix Softmax | 已完成 |
| CUDA Elementwise | Add | 已完成 |
| CUDA Memory | Matrix Transpose | 已完成 |
| CUDA Normalization | RMSNorm | 已完成 v1-v4 |
| CUDA Normalization | LayerNorm | 已完成 v1-v4 |
| CUDA LLM | RoPE、GEMV、SGEMM、FlashAttention | 待实现 |
| Triton | RMSNorm、RoPE、FlashAttention | 待实现 |

## 环境要求

- CUDA Toolkit，包含 `nvcc`
- CMake 3.24 或更高版本
- 支持目标 CUDA 架构的 NVIDIA GPU

CUDA CMake 默认使用 `CMAKE_CUDA_ARCHITECTURES=86`，对应 RTX 3090。其他 GPU
可以在配置时通过 `-DCMAKE_CUDA_ARCHITECTURES=<arch>` 覆盖。

## 构建

以下命令均从仓库根目录执行。

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda -j
```

所有已完成的 CUDA 算子都会输出到同一个目录：

```bash
./build/cuda/bin/sum_reduce_sum
./build/cuda/bin/max_reduce_max
./build/cuda/bin/softmax_softmax
./build/cuda/bin/softmax_matrix_softmax_matrix
./build/cuda/bin/elementwise_add
./build/cuda/bin/transpose
./build/cuda/bin/rmsnorm
./build/cuda/bin/layernorm
```

## 性能记录

已完成算子的 README 记录优化路径、测试 shape、正确性与性能结果。性能数据来自特定硬件和
软件环境，仅用于观察同一算子不同版本的相对变化；短 kernel 的绝对耗时会受到 GPU 时钟和
系统负载影响。

## 代码风格

- `.clang-format` 定义 C/C++/CUDA 格式，使用 4 空格缩进。
- `.editorconfig` 定义 LF 换行、文件末尾换行等基础编辑器行为。

## License

本项目使用 MIT License，详见 `LICENSE`。
