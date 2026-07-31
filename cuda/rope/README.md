# RoPE

Rotary Position Embedding 的渐进式 CUDA 实现。RoPE 根据 token position 旋转 Q、K 的
二维元素对，使 Attention 的点积同时包含内容相关性与相对位置信息。

## 数据布局

```text
input/output: [batch, seq_len, num_heads, head_dim]
cos/sin:      [seq_len, head_dim / 2]
```

当前实现使用相邻元素配对：`(0, 1)`、`(2, 3)`、`(4, 5)`。`head_dim` 必须为偶数。

## 优化路径

- v1：一个线程处理一对相邻元素，使用标量读取和写回。
- v2：使用 `float2` 向量化读取和写回，数学与线程映射保持不变。

V1 对每一对元素执行：

```text
y0 = x0 * cos(theta) - x1 * sin(theta)
y1 = x0 * sin(theta) + x1 * cos(theta)
```

sin/cos cache 由 CPU 预计算。Q 和 K 可以分别调用同一个通用 kernel，V 不应用 RoPE。

## 正确性与性能

测试环境：Windows + WSL 2、RTX 4060 Ti、driver 610.74、CUDA 13.3、Release、`sm_89`。

测试 shape：

```text
batch = 2
seq_len = 1027
num_heads = 12
head_dim = 126
block = 256
```

该 shape 的 grid 尾部非整除，并且 `head_dim` 不是 4 的倍数。结果与独立的 CPU reference
比较，V1/V2 的最大绝对误差均为 `1.2e-7`，Compute Sanitizer memcheck 均报告 0
errors。

benchmark 预热 100 次，采集 20 个样本，每个样本启动 kernel 1000 次。下表统计 5 个
独立进程结果的中位数：

| 版本 | 中位延迟（ms） | P90（ms） | 有效带宽（GB/s） | 相对 V1 | 最大绝对误差 | 正确性 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v1 | 0.0241 | 0.0295 | 1544.63 | 1.000x | 0.00000012 | pass |
| v2 | 0.0237 | 0.0298 | 1572.48 | 1.017x | 0.00000012 | pass |

有效带宽按 input 读取、output 写回和 sin/cos 逻辑读取量计算。重复 benchmark 时数据可被
L2 cache 复用，因此该指标用于比较 RoPE 各版本，不代表显存 DRAM 的实际带宽。

SASS 中，V1 对输入和输出使用两条标量 `LDG.E`/`STG.E`，V2 使用一条
`LDG.E.64`/`STG.E.64`。V1 使用 18 个寄存器，V2 使用 14 个寄存器，二者均没有 shared
memory、local memory 或 spill。当前 shape 下 V2 的中位延迟约提升 1.7%，但 P90 没有
改善，因此只能认为向量化略有收益。

## 构建与运行

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda --target rope -j
./build/cuda/bin/rope --kernel all
```

只执行正确性测试：

```bash
./build/cuda/bin/rope --kernel v2 --no-benchmark
```

支持的 `--kernel` 值为 `v1`、`v2` 和 `all`，默认值为 `all`。

执行内存错误检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/cuda/bin/rope --kernel v2 --no-benchmark
```
