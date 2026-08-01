#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_utils.cuh"

template <int HEAD_DIM, int BLOCK_THREADS, int KEY_TILE>
__global__ void flash_attention_forward_v1(const float* query, const float* key, const float* value,
                                           float* output, int batch_size, int seq_len,
                                           int num_heads) {
    constexpr int VALUES_PER_THREAD = (HEAD_DIM + BLOCK_THREADS - 1) / BLOCK_THREADS;

    static_assert((BLOCK_THREADS & (BLOCK_THREADS - 1)) == 0,
                  "BLOCK_THREADS must be a power of two");

    __shared__ float shared_key[KEY_TILE * HEAD_DIM];
    __shared__ float shared_value[KEY_TILE * HEAD_DIM];
    __shared__ float partial_dot[BLOCK_THREADS];

    const int tid = threadIdx.x;

    // 一个 block 负责一个 batch、一个 head、一个 query token
    int row = blockIdx.x;

    int query_position = row % seq_len;
    row /= seq_len;

    int head = row % num_heads;
    int batch = row / num_heads;

    int64_t query_offset =
        (static_cast<int64_t>(batch) * seq_len + query_position) * num_heads * HEAD_DIM +
        static_cast<int64_t>(head) * HEAD_DIM;

    float query_fragment[VALUES_PER_THREAD];
    float output_acc[VALUES_PER_THREAD];

// 每个线程加载自己负责的 Q 元素
#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        int dim = tid + i * BLOCK_THREADS;

        if (dim < HEAD_DIM) {
            query_fragment[i] = query[query_offset + dim];
        } else {
            query_fragment[i] = 0.0f;
        }

        output_acc[i] = 0.0f;
    }

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;

    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // 依次处理 K/V tile
    for (int tile_start = 0; tile_start < seq_len; tile_start += KEY_TILE) {
        // causal attention 后面的 tile 不需要处理
        if (tile_start > query_position) {
            break;
        }

        // 搬运 K/V tile 到 shared memory
        for (int index = tid; index < KEY_TILE * HEAD_DIM; index += BLOCK_THREADS) {
            int local_key = index / HEAD_DIM;
            int dim = index % HEAD_DIM;
            int key_position = tile_start + local_key;

            if (key_position < seq_len) {
                int64_t kv_offset =
                    (static_cast<int64_t>(batch) * seq_len + key_position) * num_heads * HEAD_DIM +
                    static_cast<int64_t>(head) * HEAD_DIM + dim;

                shared_key[index] = key[kv_offset];
                shared_value[index] = value[kv_offset];
            } else {
                shared_key[index] = 0.0f;
                shared_value[index] = 0.0f;
            }
        }

        __syncthreads();

        // 依次处理 tile 中的每一个 key
        for (int local_key = 0; local_key < KEY_TILE; ++local_key) {
            int key_position = tile_start + local_key;

            if (key_position >= seq_len || key_position > query_position) {
                break;
            }

            // 每个线程计算 dot product 的一部分
            float local_dot = 0.0f;

#pragma unroll
            for (int i = 0; i < VALUES_PER_THREAD; ++i) {
                int dim = tid + i * BLOCK_THREADS;

                if (dim < HEAD_DIM) {
                    local_dot += query_fragment[i] * shared_key[local_key * HEAD_DIM + dim];
                }
            }

            partial_dot[tid] = local_dot;
            __syncthreads();

            // block reduction，得到 Q·K
            for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    partial_dot[tid] += partial_dot[tid + stride];
                }

                __syncthreads();
            }

            float score = partial_dot[0] * scale;
            // 下一轮会覆盖 partial_dot，先确保所有线程都读完归约结果。
            __syncthreads();

            // online softmax
            float new_max = fmaxf(row_max, score);

            float old_weight = row_max == -CUDART_INF_F ? 0.0f : expf(row_max - new_max);

            float current_weight = expf(score - new_max);

            float new_sum = old_weight * row_sum + current_weight;

// 更新输出向量累加器
#pragma unroll
            for (int i = 0; i < VALUES_PER_THREAD; ++i) {
                int dim = tid + i * BLOCK_THREADS;

                if (dim < HEAD_DIM) {
                    float v = shared_value[local_key * HEAD_DIM + dim];

                    output_acc[i] = old_weight * output_acc[i] + current_weight * v;
                }
            }

            row_max = new_max;
            row_sum = new_sum;
        }

        // 防止下一轮覆盖 shared memory 时仍有线程读取旧数据
        __syncthreads();
    }

    // 写回 O
    int64_t output_offset =
        (static_cast<int64_t>(batch) * seq_len + query_position) * num_heads * HEAD_DIM +
        static_cast<int64_t>(head) * HEAD_DIM;

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; ++i) {
        int dim = tid + i * BLOCK_THREADS;

        if (dim < HEAD_DIM) {
            output[output_offset + dim] = row_sum > 0.0f ? output_acc[i] / row_sum : 0.0f;
        }
    }
}

namespace {

constexpr int kBatchSize = 2;
constexpr int kSeqLen = 257;
constexpr int kNumHeads = 4;
constexpr int kHeadDim = 64;
constexpr int kBlockThreads = 128;
constexpr int kKeyTile = 32;
constexpr int kWarmup = 20;
constexpr int kIterations = 100;
constexpr int kBenchmarkSamples = 20;
constexpr float kCorrectnessTolerance = 5.0e-4f;

int64_t qkv_offset(int batch, int position, int head, int dim, int seq_len, int num_heads,
                  int head_dim) {
    return ((static_cast<int64_t>(batch) * seq_len + position) * num_heads + head) * head_dim +
           dim;
}

void flash_attention_host(const float* query, const float* key, const float* value, float* output,
                          int batch_size, int seq_len, int num_heads, int head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len, -std::numeric_limits<float>::infinity());

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int query_position = 0; query_position < seq_len; ++query_position) {
            for (int head = 0; head < num_heads; ++head) {
                const int64_t query_base =
                    qkv_offset(batch, query_position, head, 0, seq_len, num_heads, head_dim);

                float row_max = -std::numeric_limits<float>::infinity();

                // 独立 CPU reference：先生成这一行的 score，再做稳定 softmax。
                for (int key_position = 0; key_position <= query_position; ++key_position) {
                    const int64_t key_base =
                        qkv_offset(batch, key_position, head, 0, seq_len, num_heads, head_dim);

                    float dot = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        dot += query[query_base + dim] * key[key_base + dim];
                    }

                    scores[key_position] = dot * scale;
                    row_max = std::max(row_max, scores[key_position]);
                }

                float row_sum = 0.0f;
                for (int key_position = 0; key_position <= query_position; ++key_position) {
                    row_sum += std::exp(scores[key_position] - row_max);
                }

                for (int dim = 0; dim < head_dim; ++dim) {
                    float weighted_sum = 0.0f;

                    for (int key_position = 0; key_position <= query_position; ++key_position) {
                        const int64_t value_index =
                            qkv_offset(batch, key_position, head, dim, seq_len, num_heads, head_dim);
                        const float probability =
                            std::exp(scores[key_position] - row_max) / row_sum;
                        weighted_sum += probability * value[value_index];
                    }

                    const int64_t output_index =
                        qkv_offset(batch, query_position, head, dim, seq_len, num_heads, head_dim);
                    output[output_index] = weighted_sum;
                }
            }
        }
    }
}

float max_abs_error(const float* actual, const float* expected, int64_t count) {
    float max_error = 0.0f;

    for (int64_t index = 0; index < count; ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            return std::numeric_limits<float>::infinity();
        }

        max_error = std::max(max_error, std::fabs(actual[index] - expected[index]));
    }

    return max_error;
}

struct BenchmarkStats {
    float median_ms;
    float min_ms;
    float p90_ms;
    float max_ms;
};

template <typename KernelLauncher>
BenchmarkStats benchmark_kernel_ms(KernelLauncher&& launcher, int warmup, int iterations) {
    for (int index = 0; index < warmup; ++index) {
        launcher();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::array<float, kBenchmarkSamples> samples{};

    for (float& sample_ms : samples) {
        CUDA_CHECK(cudaEventRecord(start));

        for (int index = 0; index < iterations; ++index) {
            launcher();
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        sample_ms = total_ms / static_cast<float>(iterations);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(samples.begin(), samples.end());

    BenchmarkStats stats{};
    stats.median_ms =
        (samples[kBenchmarkSamples / 2 - 1] + samples[kBenchmarkSamples / 2]) * 0.5f;
    stats.min_ms = samples.front();
    stats.p90_ms = samples[(kBenchmarkSamples * 9 + 9) / 10 - 1];
    stats.max_ms = samples.back();
    return stats;
}

void print_benchmark_stats(const char* label, const BenchmarkStats& stats) {
    std::printf("%s median(ms) = %.4f, min(ms) = %.4f, p90(ms) = %.4f, max(ms) = %.4f\n", label,
                stats.median_ms, stats.min_ms, stats.p90_ms, stats.max_ms);
}

void print_usage(const char* program) {
    std::fprintf(stderr, "Usage: %s [--no-benchmark]\n", program);
}

bool parse_options(int argc, char** argv, bool* run_benchmark) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--no-benchmark") == 0) {
            *run_benchmark = false;
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\n", argv[index]);
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool run_benchmark = true;
    if (!parse_options(argc, argv, &run_benchmark)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int64_t tensor_elements =
        static_cast<int64_t>(kBatchSize) * kSeqLen * kNumHeads * kHeadDim;
    const size_t tensor_bytes = static_cast<size_t>(tensor_elements) * sizeof(float);

    std::vector<float> h_query(tensor_elements);
    std::vector<float> h_key(tensor_elements);
    std::vector<float> h_value(tensor_elements);
    std::vector<float> h_output(tensor_elements);
    std::vector<float> h_output_reference(tensor_elements);

    for (int64_t index = 0; index < tensor_elements; ++index) {
        h_query[index] = static_cast<float>((index * 17) % 101 - 50) * 0.02f;
        h_key[index] = static_cast<float>((index * 13) % 79 - 39) * 0.015f;
        h_value[index] = static_cast<float>((index * 7) % 67 - 33) * 0.025f;
    }

    flash_attention_host(h_query.data(), h_key.data(), h_value.data(), h_output_reference.data(),
                         kBatchSize, kSeqLen, kNumHeads, kHeadDim);

    float* d_query = nullptr;
    float* d_key = nullptr;
    float* d_value = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_key, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_value, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), tensor_bytes, cudaMemcpyHostToDevice));

    const int rows = kBatchSize * kSeqLen * kNumHeads;
    const dim3 grid(rows);
    const dim3 block(kBlockThreads);

    auto launch_flash_attention = [&]() {
        flash_attention_forward_v1<kHeadDim, kBlockThreads, kKeyTile>
            <<<grid, block>>>(d_query, d_key, d_value, d_output, kBatchSize, kSeqLen, kNumHeads);
    };

    launch_flash_attention();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, tensor_bytes, cudaMemcpyDeviceToHost));

    const float output_error =
        max_abs_error(h_output.data(), h_output_reference.data(), tensor_elements);
    const bool pass = output_error <= kCorrectnessTolerance;

    std::printf("B = %d, S = %d, H = %d, D = %d, block = %d, key_tile = %d\n", kBatchSize,
                kSeqLen, kNumHeads, kHeadDim, kBlockThreads, kKeyTile);
    std::printf("output_max_abs_err = %.8f\n", output_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    if (run_benchmark) {
        const BenchmarkStats stats =
            benchmark_kernel_ms(launch_flash_attention, kWarmup, kIterations);
        print_benchmark_stats("FlashAttention", stats);
        std::printf("benchmark warmup = %d, samples = %d, repetitions/sample = %d\n", kWarmup,
                    kBenchmarkSamples, kIterations);
    }

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_output));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
