# Attention / FlashAttention

本目录实现 LLM 推理中的 causal self-attention。当前包含 Naive Attention 和不保存完整
Score/Probability 矩阵的教学版 FlashAttention V1-like 前向实现。

## 数据布局

```text
Q/K/V/Output: [batch, seq_len, num_heads, head_dim]
Score/Prob:    [batch, num_heads, query_position, key_position]
```

标准 Multi-Head Attention 中：

```text
hidden_size = num_heads * head_dim
```

不同 batch、不同 head 相互独立；每个 head 内的所有 Query token 与所有 Key token 计算
`seq_len x seq_len` 的关系矩阵。

## Naive Attention

`flash_attention_naive.cu` 使用三个独立 kernel：

1. `attention_qk_v1`：沿 `head_dim` 计算 Q/K 点积，乘以
   `1 / sqrt(head_dim)`，并应用 causal mask。
2. `attention_softmax_v1`：一个 block 处理一行 Score，沿 `key_position` 完成稳定
   softmax。
3. `attention_pv_v1`：沿 `key_position` 使用 Probability 对 V 做加权求和。

完整数据流：

```text
Q/K [B,S,H,D]
    -> QK^T / sqrt(D) + causal mask
    -> Score [B,H,S,S]
    -> Softmax
    -> Probability [B,H,S,S]
    -> Probability * V
    -> Output [B,S,H,D]
```

该版本显式保存 Score 和 Probability，空间复杂度为 `O(B * H * S^2)`，仅作为正确性
baseline 和 FlashAttention 的学习起点。

## 教学版 FlashAttention V1-like

`flash_attention_tiled.cu` 使用一个 block 处理一个 `(batch, head, query_position)`，将
Q 保存在寄存器中，按 tile 将 K/V 搬到 shared memory，并使用 online softmax 直接更新
输出累加器。整个计算过程不落地完整的 `S x S` Score/Probability 矩阵：

```text
Q -> registers
K/V tile -> shared memory
QK / sqrt(D) -> online softmax
online weighted V accumulation
    -> Output
```

当前实现为教学版本，配置固定为：

```text
HEAD_DIM      = 64
BLOCK_THREADS = 128
KEY_TILE      = 32
causal        = true
dtype         = float32
```

该版本的重点是展示 tiled attention 和 online softmax，尚未使用 warp shuffle、Tensor
Core、float4 加载或 double buffering，因此小规模输入下可能慢于 Naive baseline。

## 正确性与性能

测试环境：Windows + WSL 2、RTX 4060 Ti、CUDA 13.3、Release、`sm_89`。

```text
B = 2
S = 257
H = 4
D = 64
block = 256
```

`S=257` 覆盖非整除 grid 和 softmax 循环边界。GPU 三个阶段分别与独立 CPU reference
比较：

```text
Score max_abs_err       = 0.00000024
Probability max_abs_err = 0.00000007
Output max_abs_err      = 0.00000021
Memcheck                = 0 errors
Racecheck               = 0 hazards
Synccheck               = 0 errors
```

benchmark 预热 20 次，采集 20 个样本，每个样本重复 100 次。下表是 5 个独立进程结果
的中位数：

| 阶段 | Median（ms） | P90（ms） |
| --- | ---: | ---: |
| QK + Scale + Mask | 0.2343 | 0.2426 |
| Softmax | 0.0291 | 0.0346 |
| PV | 0.0641 | 0.0755 |
| 完整 Pipeline | 0.3354 | 0.3433 |

当前实现中 QK 约占端到端中位延迟的 70%，是最主要的耗时阶段。该结论只适用于当前
shape 和朴素线程映射。

FlashAttention 教学版本使用相同输入形状进行测试：

```text
Output max_abs_err = 0.00000019
Memcheck           = 0 errors
Racecheck          = 0 hazards
Synccheck          = 0 errors
```

benchmark 预热 20 次，采集 20 个样本，每个样本重复 100 次：

| Kernel | Median（ms） | P90（ms） |
| --- | ---: | ---: |
| FlashAttention V1-like | 1.0113 | 1.0174 |

## 构建与运行

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda --target attention_flash -j
./build/cuda/bin/attention_flash
```

运行 Naive baseline：

```bash
cmake --build build/cuda --target attention_naive -j
./build/cuda/bin/attention_naive
```

只执行正确性测试：

```bash
./build/cuda/bin/attention_flash --no-benchmark
```

执行内存错误检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/cuda/bin/attention_flash --no-benchmark
compute-sanitizer --tool racecheck --error-exitcode 1 \
    ./build/cuda/bin/attention_flash --no-benchmark
compute-sanitizer --tool synccheck --error-exitcode 1 \
    ./build/cuda/bin/attention_flash --no-benchmark
```

## 下一步

后续可以将当前教学版本优化为一个 warp 负责一个 query，使用 warp-level reduction 和
向量化加载，再逐步尝试 query tile 与 double buffering。
