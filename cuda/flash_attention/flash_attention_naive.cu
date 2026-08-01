/*
    Q/K/V/Output: [batch, seq_len, num_heads, head_dim]
    Score/Prob:    [batch, num_heads, seq_len, seq_len]
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <math_constants.h>

#include "cuda_utils.cuh"
#include "utils.cuh"

__global__ void attention_qk_v1(const float* query, const float* key, float* score, int batch_size,
                                int seq_len, int num_heads, int head_dim) {
    int64_t score_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_score = static_cast<int64_t>(batch_size) * num_heads * seq_len * seq_len;

    if (score_index >= total_score) {
        return;
    }

    int key_position = score_index % seq_len;

    int64_t remaining = score_index / seq_len;

    int query_position = remaining % seq_len;
    remaining /= seq_len;

    int head = remaining % num_heads;
    int batch = remaining / num_heads;

    int64_t output_offset =
        ((static_cast<int64_t>(batch) * num_heads + head) * seq_len + query_position) * seq_len +
        key_position;

    // Causal Attention 不能查看未来 token
    if (key_position > query_position) {
        score[output_offset] = -CUDART_INF_F;
        return;
    }

    int64_t query_offset =
        ((static_cast<int64_t>(batch) * seq_len + query_position) * num_heads + head) * head_dim;

    int64_t key_offset =
        ((static_cast<int64_t>(batch) * seq_len + key_position) * num_heads + head) * head_dim;

    float sum = 0.0f;

    for (int dim = 0; dim < head_dim; ++dim) {
        sum += query[query_offset + dim] * key[key_offset + dim];
    }

    float scale = rsqrtf(static_cast<float>(head_dim));

    score[output_offset] = sum * scale;
}

__global__ void attention_softmax_v1(const float* score, float* probability, int rows,
                                     int seq_len) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;
    int thread_id = threadIdx.x;

    if (row >= rows) {
        return;
    }

    int64_t row_offset = static_cast<int64_t>(row) * seq_len;

    // 第一轮：计算这一行的最大值。
    float local_max = -CUDART_INF_F;

    for (int column = thread_id; column < seq_len; column += blockDim.x) {
        local_max = fmaxf(local_max, score[row_offset + column]);
    }

    shared_data[thread_id] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_data[thread_id] = fmaxf(shared_data[thread_id], shared_data[thread_id + stride]);
        }

        __syncthreads();
    }

    float row_max = shared_data[0];
    __syncthreads();

    // 第二轮：计算 exp(score - max) 及其总和。
    float local_sum = 0.0f;

    for (int column = thread_id; column < seq_len; column += blockDim.x) {
        float exp_value = expf(score[row_offset + column] - row_max);

        probability[row_offset + column] = exp_value;
        local_sum += exp_value;
    }

    shared_data[thread_id] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (thread_id < stride) {
            shared_data[thread_id] += shared_data[thread_id + stride];
        }

        __syncthreads();
    }

    float row_sum = shared_data[0];

    // 第三轮：归一化。
    for (int column = thread_id; column < seq_len; column += blockDim.x) {
        probability[row_offset + column] /= row_sum;
    }
}

__global__ void attention_pv_v1(const float* probability, const float* value, float* output,
                                int batch_size, int seq_len, int num_heads, int head_dim) {
    int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    int64_t total_outputs = static_cast<int64_t>(batch_size) * seq_len * num_heads * head_dim;

    if (output_index >= total_outputs) {
        return;
    }

    int dim = output_index % head_dim;

    int64_t remaining = output_index / head_dim;

    int head = remaining % num_heads;
    remaining /= num_heads;

    int query_position = remaining % seq_len;
    int batch = remaining / seq_len;

    int64_t probability_offset =
        ((static_cast<int64_t>(batch) * num_heads + head) * seq_len + query_position) * seq_len;

    float sum = 0.0f;

    for (int key_position = 0; key_position < seq_len; ++key_position) {
        int64_t value_offset =
            ((static_cast<int64_t>(batch) * seq_len + key_position) * num_heads + head) * head_dim +
            dim;

        sum += probability[probability_offset + key_position] * value[value_offset];
    }

    output[output_index] = sum;
}

int64_t qkv_offset(int batch, int position, int head, int dim, int seq_len, int num_heads,
                   int head_dim) {
    return ((static_cast<int64_t>(batch) * seq_len + position) * num_heads + head) * head_dim + dim;
}

int64_t score_offset(int batch, int head, int query_position, int key_position, int seq_len,
                     int num_heads) {
    return ((static_cast<int64_t>(batch) * num_heads + head) * seq_len + query_position) * seq_len +
           key_position;
}

void attention_qk_host(const float* query, const float* key, float* score, int batch_size,
                       int seq_len, int num_heads, int head_dim) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < num_heads; ++head) {
            for (int query_position = 0; query_position < seq_len; ++query_position) {
                for (int key_position = 0; key_position < seq_len; ++key_position) {
                    int64_t output_index =
                        score_offset(batch, head, query_position, key_position, seq_len, num_heads);

                    if (key_position > query_position) {
                        score[output_index] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float sum = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        int64_t query_index = qkv_offset(batch, query_position, head, dim, seq_len,
                                                         num_heads, head_dim);
                        int64_t key_index = qkv_offset(batch, key_position, head, dim, seq_len,
                                                       num_heads, head_dim);
                        sum += query[query_index] * key[key_index];
                    }

                    score[output_index] = sum * scale;
                }
            }
        }
    }
}

void attention_softmax_host(const float* score, float* probability, int batch_size, int seq_len,
                            int num_heads) {
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < num_heads; ++head) {
            for (int query_position = 0; query_position < seq_len; ++query_position) {
                float row_max = -std::numeric_limits<float>::infinity();

                for (int key_position = 0; key_position < seq_len; ++key_position) {
                    int64_t index =
                        score_offset(batch, head, query_position, key_position, seq_len, num_heads);
                    row_max = std::max(row_max, score[index]);
                }

                float row_sum = 0.0f;
                for (int key_position = 0; key_position < seq_len; ++key_position) {
                    int64_t index =
                        score_offset(batch, head, query_position, key_position, seq_len, num_heads);
                    float exp_value = std::exp(score[index] - row_max);
                    probability[index] = exp_value;
                    row_sum += exp_value;
                }

                for (int key_position = 0; key_position < seq_len; ++key_position) {
                    int64_t index =
                        score_offset(batch, head, query_position, key_position, seq_len, num_heads);
                    probability[index] /= row_sum;
                }
            }
        }
    }
}

void attention_pv_host(const float* probability, const float* value, float* output, int batch_size,
                       int seq_len, int num_heads, int head_dim) {
    int total_outputs = batch_size * seq_len * num_heads * head_dim;

    for (int64_t output_index = 0; output_index < total_outputs; ++output_index) {
        int dim = output_index % head_dim;
        int64_t remaining = output_index / head_dim;
        int head = remaining % num_heads;
        remaining /= num_heads;
        int query_position = remaining % seq_len;
        int batch = remaining / seq_len;

        float sum = 0.0f;
        for (int key_position = 0; key_position < seq_len; ++key_position) {
            int64_t probability_index =
                score_offset(batch, head, query_position, key_position, seq_len, num_heads);
            int64_t value_index =
                qkv_offset(batch, key_position, head, dim, seq_len, num_heads, head_dim);
            sum += probability[probability_index] * value[value_index];
        }

        output[output_index] = sum;
    }
}

float max_abs_error(const float* actual, const float* expected, int64_t count) {
    float max_error = 0.0f;

    for (int64_t index = 0; index < count; ++index) {
        if (std::isinf(actual[index]) && std::isinf(expected[index]) &&
            std::signbit(actual[index]) == std::signbit(expected[index])) {
            continue;
        }
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            return INFINITY;
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
    constexpr int SAMPLE_COUNT = 20;

    for (int index = 0; index < warmup; ++index) {
        launcher();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::array<float, SAMPLE_COUNT> samples{};
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
    constexpr int P90_INDEX = (SAMPLE_COUNT * 9 + 9) / 10 - 1;

    BenchmarkStats stats{};
    stats.median_ms = (samples[SAMPLE_COUNT / 2 - 1] + samples[SAMPLE_COUNT / 2]) * 0.5f;
    stats.min_ms = samples.front();
    stats.p90_ms = samples[P90_INDEX];
    stats.max_ms = samples.back();
    return stats;
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

int main(int argc, char** argv) {
    constexpr int BATCH_SIZE = 2;
    constexpr int SEQ_LEN = 257;
    constexpr int NUM_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP = 20;
    constexpr int ITERATIONS = 100;

    bool run_benchmark = true;
    if (!parse_options(argc, argv, &run_benchmark)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int64_t tensor_elements = static_cast<int64_t>(BATCH_SIZE) * SEQ_LEN * NUM_HEADS * HEAD_DIM;
    int64_t score_elements = static_cast<int64_t>(BATCH_SIZE) * NUM_HEADS * SEQ_LEN * SEQ_LEN;
    size_t tensor_bytes = static_cast<size_t>(tensor_elements) * sizeof(float);
    size_t score_bytes = static_cast<size_t>(score_elements) * sizeof(float);

    std::vector<float> h_query(tensor_elements);
    std::vector<float> h_key(tensor_elements);
    std::vector<float> h_value(tensor_elements);
    std::vector<float> h_score(score_elements);
    std::vector<float> h_probability(score_elements);
    std::vector<float> h_output(tensor_elements);
    std::vector<float> h_score_reference(score_elements);
    std::vector<float> h_probability_reference(score_elements);
    std::vector<float> h_output_reference(tensor_elements);

    for (int64_t index = 0; index < tensor_elements; ++index) {
        h_query[index] = static_cast<float>((index * 17) % 101 - 50) * 0.02f;
        h_key[index] = static_cast<float>((index * 13) % 79 - 39) * 0.015f;
        h_value[index] = static_cast<float>((index * 7) % 67 - 33) * 0.025f;
    }

    attention_qk_host(h_query.data(), h_key.data(), h_score_reference.data(), BATCH_SIZE, SEQ_LEN,
                      NUM_HEADS, HEAD_DIM);
    attention_softmax_host(h_score_reference.data(), h_probability_reference.data(), BATCH_SIZE,
                           SEQ_LEN, NUM_HEADS);
    attention_pv_host(h_probability_reference.data(), h_value.data(), h_output_reference.data(),
                      BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);

    float* d_query = nullptr;
    float* d_key = nullptr;
    float* d_value = nullptr;
    float* d_score = nullptr;
    float* d_probability = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_query, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_key, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_value, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_probability, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));

    CUDA_CHECK(cudaMemcpy(d_query, h_query.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value, h_value.data(), tensor_bytes, cudaMemcpyHostToDevice));

    int score_grid = static_cast<int>((score_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int rows = BATCH_SIZE * NUM_HEADS * SEQ_LEN;
    int output_grid = static_cast<int>((tensor_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dim3 block(BLOCK_SIZE);
    dim3 qk_grid(score_grid);
    dim3 softmax_grid(rows);
    dim3 pv_grid(output_grid);

    auto launch_qk = [&]() {
        attention_qk_v1<<<qk_grid, block>>>(d_query, d_key, d_score, BATCH_SIZE, SEQ_LEN, NUM_HEADS,
                                            HEAD_DIM);
    };
    auto launch_softmax = [&]() {
        attention_softmax_v1<<<softmax_grid, block, BLOCK_SIZE * sizeof(float)>>>(
            d_score, d_probability, rows, SEQ_LEN);
    };
    auto launch_pv = [&]() {
        attention_pv_v1<<<pv_grid, block>>>(d_probability, d_value, d_output, BATCH_SIZE, SEQ_LEN,
                                            NUM_HEADS, HEAD_DIM);
    };
    auto launch_pipeline = [&]() {
        launch_qk();
        launch_softmax();
        launch_pv();
    };

    launch_qk();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_score.data(), d_score, score_bytes, cudaMemcpyDeviceToHost));

    launch_softmax();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(
        cudaMemcpy(h_probability.data(), d_probability, score_bytes, cudaMemcpyDeviceToHost));

    launch_pv();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, tensor_bytes, cudaMemcpyDeviceToHost));

    float score_error = max_abs_error(h_score.data(), h_score_reference.data(), score_elements);
    float probability_error =
        max_abs_error(h_probability.data(), h_probability_reference.data(), score_elements);
    float output_error = max_abs_error(h_output.data(), h_output_reference.data(), tensor_elements);
    bool pass = score_error < 1e-5f && probability_error < 1e-5f && output_error < 1e-5f;

    std::printf("B = %d, S = %d, H = %d, D = %d, block = %u\n", BATCH_SIZE, SEQ_LEN, NUM_HEADS,
                HEAD_DIM, block.x);
    std::printf("score = [B,H,Q,K] = [%d,%d,%d,%d], probability = same\n", BATCH_SIZE, NUM_HEADS,
                SEQ_LEN, SEQ_LEN);
    std::printf("score_max_abs_err = %.8f\n", score_error);
    std::printf("probability_max_abs_err = %.8f\n", probability_error);
    std::printf("output_max_abs_err = %.8f\n", output_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    if (run_benchmark) {
        BenchmarkStats qk_stats = benchmark_kernel_ms(launch_qk, WARMUP, ITERATIONS);
        BenchmarkStats softmax_stats = benchmark_kernel_ms(launch_softmax, WARMUP, ITERATIONS);
        BenchmarkStats pv_stats = benchmark_kernel_ms(launch_pv, WARMUP, ITERATIONS);
        BenchmarkStats pipeline_stats = benchmark_kernel_ms(launch_pipeline, WARMUP, ITERATIONS);

        auto print_stats = [](const char* label, const BenchmarkStats& stats) {
            std::printf("%s median(ms) = %.4f, min(ms) = %.4f, p90(ms) = %.4f, max(ms) = %.4f\n",
                        label, stats.median_ms, stats.min_ms, stats.p90_ms, stats.max_ms);
        };

        print_stats("QK", qk_stats);
        print_stats("Softmax", softmax_stats);
        print_stats("PV", pv_stats);
        print_stats("Pipeline", pipeline_stats);
    }

    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_value));
    CUDA_CHECK(cudaFree(d_score));
    CUDA_CHECK(cudaFree(d_probability));
    CUDA_CHECK(cudaFree(d_output));

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
