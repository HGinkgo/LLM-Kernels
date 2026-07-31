#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_utils.cuh"
#include "utils.cuh"

__global__ void rope_v1(const float* input, float* output,
                        const float* cos_cache, const float* sin_cache,
                        int batch_size, int seq_len, int num_heads,
                        int head_dim) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;

  int pairs_per_head = head_dim / 2;
  int total_pairs = batch_size * seq_len * num_heads * pairs_per_head;

  if (pair_index >= total_pairs) {
    return;
  }

  // 当前线程负责 head 中的第几个元素对
  int pair_in_head = pair_index % pairs_per_head;

  // 当前线程属于哪个 [batch, sequence, head]
  int token_head_index = pair_index / pairs_per_head;

  // 去掉 head 维度，得到 [batch, sequence] 的扁平索引
  int token_index = token_head_index / num_heads;

  // 每个 batch 中，position 都是 0 到 seq_len - 1
  int position = token_index % seq_len;

  // input 的最后一维连续存储
  int input_offset = token_head_index * head_dim + pair_in_head * 2;

  // 每个 position 保存 head_dim / 2 组 sin/cos
  int cache_offset = position * pairs_per_head + pair_in_head;

  float x0 = input[input_offset];
  float x1 = input[input_offset + 1];

  float cos_value = cos_cache[cache_offset];
  float sin_value = sin_cache[cache_offset];

  output[input_offset] = x0 * cos_value - x1 * sin_value;
  output[input_offset + 1] = x0 * sin_value + x1 * cos_value;
}

__global__ void rope_v2(const float* input, float* output,
                        const float* cos_cache, const float* sin_cache,
                        int batch_size, int seq_len, int num_heads,
                        int head_dim) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;

  int pairs_per_head = head_dim / 2;
  int total_pairs = batch_size * seq_len * num_heads * pairs_per_head;

  if (pair_index >= total_pairs) {
    return;
  }

  int pair_in_head = pair_index % pairs_per_head;
  int token_head_index = pair_index / pairs_per_head;

  int token_index = token_head_index / num_heads;
  int position = token_index % seq_len;

  // 将原始 float 偏移除以 2，转成 float2 下标。
  int vector_offset = token_head_index * pairs_per_head + pair_in_head;

  int cache_offset = position * pairs_per_head + pair_in_head;

  const float2* input_vec = reinterpret_cast<const float2*>(input);

  float2* output_vec = reinterpret_cast<float2*>(output);

  float2 x = input_vec[vector_offset];

  float cos_value = cos_cache[cache_offset];
  float sin_value = sin_cache[cache_offset];

  output_vec[vector_offset] = make_float2(x.x * cos_value - x.y * sin_value,
                                          x.x * sin_value + x.y * cos_value);
}

void build_rope_cache(float* cos_cache, float* sin_cache, int seq_len,
                      int head_dim) {
  constexpr float BASE = 10000.0f;
  int pairs_per_head = head_dim / 2;

  for (int position = 0; position < seq_len; ++position) {
    for (int pair = 0; pair < pairs_per_head; ++pair) {
      float exponent =
          -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
      float inverse_frequency = std::pow(BASE, exponent);
      float angle = static_cast<float>(position) * inverse_frequency;
      int cache_offset = position * pairs_per_head + pair;

      cos_cache[cache_offset] = std::cos(angle);
      sin_cache[cache_offset] = std::sin(angle);
    }
  }
}

void rope_host(const float* input, float* output, const float* cos_cache,
               const float* sin_cache, int batch_size, int seq_len,
               int num_heads, int head_dim) {
  int pairs_per_head = head_dim / 2;

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int position = 0; position < seq_len; ++position) {
      for (int head = 0; head < num_heads; ++head) {
        int token_head_index = (batch * seq_len + position) * num_heads + head;

        for (int pair = 0; pair < pairs_per_head; ++pair) {
          int input_offset = token_head_index * head_dim + pair * 2;
          int cache_offset = position * pairs_per_head + pair;
          float x0 = input[input_offset];
          float x1 = input[input_offset + 1];
          float cos_value = cos_cache[cache_offset];
          float sin_value = sin_cache[cache_offset];

          output[input_offset] = x0 * cos_value - x1 * sin_value;
          output[input_offset + 1] = x0 * sin_value + x1 * cos_value;
        }
      }
    }
  }
}

float max_abs_error(const float* actual, const float* expected, int count) {
  float max_error = 0.0f;

  for (int index = 0; index < count; ++index) {
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
BenchmarkStats benchmark_kernel_ms(KernelLauncher&& launcher, int warmup,
                                   int iterations) {
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
  stats.median_ms =
      (samples[SAMPLE_COUNT / 2 - 1] + samples[SAMPLE_COUNT / 2]) * 0.5f;
  stats.min_ms = samples.front();
  stats.p90_ms = samples[P90_INDEX];
  stats.max_ms = samples.back();
  return stats;
}

enum class KernelSelection {
    kV1,
    kV2,
    kAll,
};

struct BenchmarkOptions {
    KernelSelection selection = KernelSelection::kAll;
    bool run_benchmark = true;
};

void print_usage(const char* program) {
    std::fprintf(stderr, "Usage: %s [--kernel v1|v2|all] [--no-benchmark]\n", program);
}

bool parse_options(int argc, char** argv, BenchmarkOptions* options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--no-benchmark") == 0) {
            options->run_benchmark = false;
            continue;
        }

        if (std::strcmp(argv[index], "--kernel") == 0) {
            if (++index >= argc) {
                std::fprintf(stderr, "Missing value after --kernel\n");
                return false;
            }

            if (std::strcmp(argv[index], "v1") == 0) {
                options->selection = KernelSelection::kV1;
            } else if (std::strcmp(argv[index], "v2") == 0) {
                options->selection = KernelSelection::kV2;
            } else if (std::strcmp(argv[index], "all") == 0) {
                options->selection = KernelSelection::kAll;
            } else {
                std::fprintf(stderr, "Invalid kernel: %s\n", argv[index]);
                return false;
            }
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\n", argv[index]);
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    constexpr int BATCH_SIZE = 2;
    constexpr int SEQ_LEN = 1027;
    constexpr int NUM_HEADS = 12;
    constexpr int HEAD_DIM = 126;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP = 100;
    constexpr int ITERATIONS = 1000;

    static_assert(HEAD_DIM % 2 == 0, "RoPE requires an even head dimension");

    BenchmarkOptions options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int pairs_per_head = HEAD_DIM / 2;
    int total_elements = BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM;
    int total_pairs = BATCH_SIZE * SEQ_LEN * NUM_HEADS * pairs_per_head;
    int cache_elements = SEQ_LEN * pairs_per_head;

    size_t tensor_bytes = static_cast<size_t>(total_elements) * sizeof(float);
    size_t cache_bytes = static_cast<size_t>(cache_elements) * sizeof(float);

    std::vector<float> h_input(total_elements);
    std::vector<float> h_output(total_elements);
    std::vector<float> h_reference(total_elements);
    std::vector<float> h_cos_cache(cache_elements);
    std::vector<float> h_sin_cache(cache_elements);

    for (int index = 0; index < total_elements; ++index) {
        h_input[index] = static_cast<float>((index * 17) % 101 - 50) * 0.02f;
    }

    build_rope_cache(h_cos_cache.data(), h_sin_cache.data(), SEQ_LEN, HEAD_DIM);
    rope_host(h_input.data(), h_reference.data(), h_cos_cache.data(), h_sin_cache.data(),
              BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_cos_cache = nullptr;
    float* d_sin_cache = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&d_cos_cache, cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_sin_cache, cache_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), tensor_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_cos_cache, h_cos_cache.data(), cache_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_sin_cache, h_sin_cache.data(), cache_bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil_div(total_pairs, BLOCK_SIZE));

    double logical_bytes = static_cast<double>(total_elements) * sizeof(float) * 2.0 +
                           static_cast<double>(total_pairs) * sizeof(float) * 2.0;

    std::printf("B = %d, S = %d, H = %d, D = %d, block = %u, grid = %u\n", BATCH_SIZE,
                SEQ_LEN, NUM_HEADS, HEAD_DIM, block.x, grid.x);

    auto run_kernel = [&](const char* label, auto&& launcher) {
        launcher();
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, tensor_bytes, cudaMemcpyDeviceToHost));

        float max_error = max_abs_error(h_output.data(), h_reference.data(), total_elements);
        bool pass = max_error < 1e-5f;

        std::printf("kernel = %s\n", label);
        if (options.run_benchmark) {
            BenchmarkStats stats = benchmark_kernel_ms(launcher, WARMUP, ITERATIONS);
            double effective_bandwidth_gb_s =
                logical_bytes / (static_cast<double>(stats.median_ms) * 1.0e6);

            std::printf(
                "median(ms) = %.4f, min(ms) = %.4f, p90(ms) = %.4f, max(ms) = %.4f\n",
                stats.median_ms, stats.min_ms, stats.p90_ms, stats.max_ms);
            std::printf("effective_bandwidth(GB/s) = %.2f\n", effective_bandwidth_gb_s);
        }
        std::printf("max_abs_err = %.8f\n", max_error);
        std::printf("correctness = %s\n", pass ? "pass" : "fail");
        return pass;
    };

    bool all_pass = true;
    if (options.selection == KernelSelection::kV1 || options.selection == KernelSelection::kAll) {
        auto launch_v1 = [&]() {
            rope_v1<<<grid, block>>>(d_input, d_output, d_cos_cache, d_sin_cache, BATCH_SIZE,
                                     SEQ_LEN, NUM_HEADS, HEAD_DIM);
        };
        all_pass = run_kernel("V1", launch_v1) && all_pass;
    }

    if (options.selection == KernelSelection::kV2 || options.selection == KernelSelection::kAll) {
        auto launch_v2 = [&]() {
            rope_v2<<<grid, block>>>(d_input, d_output, d_cos_cache, d_sin_cache, BATCH_SIZE,
                                     SEQ_LEN, NUM_HEADS, HEAD_DIM);
        };
        all_pass = run_kernel("V2", launch_v2) && all_pass;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_cos_cache));
    CUDA_CHECK(cudaFree(d_sin_cache));

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
