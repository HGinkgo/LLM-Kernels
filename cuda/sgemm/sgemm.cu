#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_utils.cuh"
#include "utils.cuh"

__device__ __forceinline__ float4 load_float4_safe(const float* source, int valid_count) {
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool aligned = (reinterpret_cast<std::uintptr_t>(source) & 0xF) == 0;

    if (valid_count == 4 && aligned) {
        return *reinterpret_cast<const float4*>(source);
    }

    if (valid_count > 0)
        value.x = source[0];
    if (valid_count > 1)
        value.y = source[1];
    if (valid_count > 2)
        value.z = source[2];
    if (valid_count > 3)
        value.w = source[3];

    return value;
}

__global__ void sgemm_v1(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;

    for (int index = 0; index < k; ++index) {
        float value_a = matrix_a[row * k + index];
        float value_b = matrix_b[index * n + col];

        sum = fmaf(value_a, value_b, sum);
    }

    matrix_c[row * n + col] = sum;
}

template <int TILE_SIZE>
__global__ void sgemm_v2(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;
    int tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tile_count; ++tile) {
        int matrix_a_col = tile * TILE_SIZE + tx;
        int matrix_b_row = tile * TILE_SIZE + ty;

        if (row < m && matrix_a_col < k) {
            tile_a[ty][tx] = matrix_a[row * k + matrix_a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        if (matrix_b_row < k && col < n) {
            tile_b[ty][tx] = matrix_b[matrix_b_row * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int index = 0; index < TILE_SIZE; ++index) {
            sum = fmaf(tile_a[ty][index], tile_b[index][tx], sum);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        matrix_c[row * n + col] = sum;
    }
}

/*
    BM:M 维度的 block tile 大小 32
    BN:N 维度的 block tile 大小 32
    BK:K 维度每轮处理的长度
    TM:thread 级别的 tile       4
*/
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v3(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN * (BM / TM) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int thread_count = blockDim.x * blockDim.y;

    int col = blockIdx.x * BN + tx;
    int row_base = blockIdx.y * BM + ty * TM;

    float sums[TM];

#pragma unroll
    for (int index = 0; index < TM; ++index) {
        sums[index] = 0.0f;
    }

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        // 协作加载 A 的 BM × BK tile
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            if (global_row < m && global_col < k) {
                tile_a[tile_row][tile_col] = matrix_a[global_row * k + global_col];
            } else {
                tile_a[tile_row][tile_col] = 0.0f;
            }
        }

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            if (global_row < k && global_col < n) {
                tile_b[tile_row][tile_col] = matrix_b[global_row * n + global_col];
            } else {
                tile_b[tile_row][tile_col] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            float value_b = tile_b[inner][tx];
#pragma unroll
            for (int index = 0; index < TM; ++index) {
                float value_a = tile_a[ty * TM + index][inner];
                sums[index] = fmaf(value_a, value_b, sums[index]);
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int index = 0; index < TM; ++index) {
        int row = row_base + index;

        if (row < m && col < n) {
            matrix_c[row * n + col] = sums[index];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int thread_col = threadIdx.x;
    int thread_row = threadIdx.y;

    int tid = thread_row * blockDim.x + thread_col;
    int thread_count = blockDim.x * blockDim.y;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sums[TM][TN] = {};

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            tile_a[tile_row][tile_col] =
                global_row < m && global_col < k ? matrix_a[global_row * k + global_col] : 0.0f;
        } // 搬运a

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            tile_b[tile_row][tile_col] =
                global_row < k && global_col < n ? matrix_b[global_row * n + global_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sums[row][col] = fmaf(tile_a[thread_row * TM + row][inner],
                                          tile_b[inner][thread_col * TN + col], sums[row][col]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sums[row][col];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int thread_col = threadIdx.x;
    int thread_row = threadIdx.y;
    int tid = thread_row * blockDim.x + thread_col;
    int thread_count = blockDim.x * blockDim.y;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sums[TM][TN] = {};
    float fragment_a[TM];
    float fragment_b[TN];

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            tile_a[tile_row][tile_col] =
                global_row < m && global_col < k ? matrix_a[global_row * k + global_col] : 0.0f;
        }

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            tile_b[tile_row][tile_col] =
                global_row < k && global_col < n ? matrix_b[global_row * n + global_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
                fragment_a[row] = tile_a[thread_row * TM + row][inner];
            }
#pragma unroll
            for (int col = 0; col < TN; ++col) {
                fragment_b[col] = tile_b[inner][thread_col * TN + col];
            }
#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sums[row][col] = fmaf(fragment_a[row], fragment_b[col], sums[row][col]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sums[row][col];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v6(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    constexpr int A_VECTORS_PER_ROW = BK / 4;
    constexpr int B_VECTORS_PER_ROW = BN / 4;
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int tid = blockDim.x * thread_row + thread_col;
    int thread_count = blockDim.x * blockDim.y;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sum[TM][TN] = {};
    float fragment_a[TM];
    float fragment_b[TN];

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        for (int vector_index = tid; vector_index < BM * A_VECTORS_PER_ROW;
             vector_index += thread_count) {
            int tile_row = vector_index / A_VECTORS_PER_ROW;
            int tile_col = vector_index % A_VECTORS_PER_ROW * 4;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (global_row < m && global_col < k) {
                int valid_count = k - global_col;
                valid_count = valid_count < 4 ? valid_count : 4;

                values = load_float4_safe(&matrix_a[global_row * k + global_col], valid_count);
            }

            tile_a[tile_row][tile_col] = values.x;
            tile_a[tile_row][tile_col + 1] = values.y;
            tile_a[tile_row][tile_col + 2] = values.z;
            tile_a[tile_row][tile_col + 3] = values.w;
        }

        for (int vector_index = tid; vector_index < BK * B_VECTORS_PER_ROW;
             vector_index += thread_count) {
            int tile_row = vector_index / B_VECTORS_PER_ROW;
            int tile_col = vector_index % B_VECTORS_PER_ROW * 4;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (global_row < k && global_col < n) {
                int valid_count = n - global_col;
                valid_count = valid_count < 4 ? valid_count : 4;

                values = load_float4_safe(&matrix_b[global_row * n + global_col], valid_count);
            }

            tile_b[tile_row][tile_col] = values.x;
            tile_b[tile_row][tile_col + 1] = values.y;
            tile_b[tile_row][tile_col + 2] = values.z;
            tile_b[tile_row][tile_col + 3] = values.w;
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
                fragment_a[row] = tile_a[thread_row * TM + row][inner];
            }

#pragma unroll
            for (int col = 0; col < TN; ++col) {
                fragment_b[col] = tile_b[inner][thread_col * TN + col];
            }

#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sum[row][col] = fmaf(fragment_a[row], fragment_b[col], sum[row][col]);
                }
            }
        }

        __syncthreads();
    }
    // 写回 C，这一版继续使用标量写回
#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sum[row][col];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v7(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    constexpr int VECTOR_WIDTH = 4;
    constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);

    constexpr int A_VECTORS_PER_ROW = BK / VECTOR_WIDTH;
    constexpr int B_VECTORS_PER_ROW = BN / VECTOR_WIDTH;

    constexpr int A_VECTOR_COUNT = BM * BK / VECTOR_WIDTH;
    constexpr int B_VECTOR_COUNT = BK * BN / VECTOR_WIDTH;

    constexpr int A_PREFETCH_COUNT = A_VECTOR_COUNT / BLOCK_THREADS;
    constexpr int B_PREFETCH_COUNT = B_VECTOR_COUNT / BLOCK_THREADS;

    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert(BK % VECTOR_WIDTH == 0);
    static_assert(BN % VECTOR_WIDTH == 0);
    static_assert(BLOCK_THREADS <= 1024);

    static_assert(A_VECTOR_COUNT >= BLOCK_THREADS);
    static_assert(B_VECTOR_COUNT >= BLOCK_THREADS);
    static_assert(A_VECTOR_COUNT % BLOCK_THREADS == 0);
    static_assert(B_VECTOR_COUNT % BLOCK_THREADS == 0);

    __shared__ float tile_a[2][BM][BK];
    __shared__ float tile_b[2][BK][BN];

    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int tid = thread_row * blockDim.x + thread_col;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sum[TM][TN] = {};
    float fragment_a[TM];
    float fragment_b[TN];

    // 下一块 global tile 暂存在寄存器中。
    float4 prefetched_a[A_PREFETCH_COUNT];
    float4 prefetched_b[B_PREFETCH_COUNT];

    int tile_count = (k + BK - 1) / BK;

    // ---------------------------------------------------------
    // 预加载第 0 个 K tile 到 shared buffer 0
    // ---------------------------------------------------------

    for (int vector_index = tid; vector_index < A_VECTOR_COUNT; vector_index += BLOCK_THREADS) {
        int tile_row = vector_index / A_VECTORS_PER_ROW;
        int tile_col = vector_index % A_VECTORS_PER_ROW * VECTOR_WIDTH;

        int global_row = blockIdx.y * BM + tile_row;
        int global_col = tile_col;

        float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (global_row < m && global_col < k) {
            int valid_count = k - global_col;
            valid_count = valid_count < VECTOR_WIDTH ? valid_count : VECTOR_WIDTH;

            values = load_float4_safe(&matrix_a[global_row * k + global_col], valid_count);
        }

        tile_a[0][tile_row][tile_col] = values.x;
        tile_a[0][tile_row][tile_col + 1] = values.y;
        tile_a[0][tile_row][tile_col + 2] = values.z;
        tile_a[0][tile_row][tile_col + 3] = values.w;
    }

    for (int vector_index = tid; vector_index < B_VECTOR_COUNT; vector_index += BLOCK_THREADS) {
        int tile_row = vector_index / B_VECTORS_PER_ROW;
        int tile_col = vector_index % B_VECTORS_PER_ROW * VECTOR_WIDTH;

        int global_row = tile_row;
        int global_col = blockIdx.x * BN + tile_col;

        float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (global_row < k && global_col < n) {
            int valid_count = n - global_col;
            valid_count = valid_count < VECTOR_WIDTH ? valid_count : VECTOR_WIDTH;

            values = load_float4_safe(&matrix_b[global_row * n + global_col], valid_count);
        }

        tile_b[0][tile_row][tile_col] = values.x;
        tile_b[0][tile_row][tile_col + 1] = values.y;
        tile_b[0][tile_row][tile_col + 2] = values.z;
        tile_b[0][tile_row][tile_col + 3] = values.w;
    }

    __syncthreads();

    // ---------------------------------------------------------
    // 双缓冲主循环
    // ---------------------------------------------------------

    for (int tile = 0; tile < tile_count; ++tile) {
        int current_buffer = tile & 1;
        int next_buffer = current_buffer ^ 1;

        int next_tile = tile + 1;
        bool has_next_tile = next_tile < tile_count;

        // -----------------------------------------------------
        // global(next tile) -> prefetch registers
        // -----------------------------------------------------

        if (has_next_tile) {
#pragma unroll
            for (int load = 0; load < A_PREFETCH_COUNT; ++load) {
                int vector_index = tid + load * BLOCK_THREADS;

                int tile_row = vector_index / A_VECTORS_PER_ROW;

                int tile_col = vector_index % A_VECTORS_PER_ROW * VECTOR_WIDTH;

                int global_row = blockIdx.y * BM + tile_row;

                int global_col = next_tile * BK + tile_col;

                prefetched_a[load] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                if (global_row < m && global_col < k) {
                    int valid_count = k - global_col;

                    valid_count = valid_count < VECTOR_WIDTH ? valid_count : VECTOR_WIDTH;

                    prefetched_a[load] =
                        load_float4_safe(&matrix_a[global_row * k + global_col], valid_count);
                }
            }

#pragma unroll
            for (int load = 0; load < B_PREFETCH_COUNT; ++load) {
                int vector_index = tid + load * BLOCK_THREADS;

                int tile_row = vector_index / B_VECTORS_PER_ROW;

                int tile_col = vector_index % B_VECTORS_PER_ROW * VECTOR_WIDTH;

                int global_row = next_tile * BK + tile_row;

                int global_col = blockIdx.x * BN + tile_col;

                prefetched_b[load] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

                if (global_row < k && global_col < n) {
                    int valid_count = n - global_col;

                    valid_count = valid_count < VECTOR_WIDTH ? valid_count : VECTOR_WIDTH;

                    prefetched_b[load] =
                        load_float4_safe(&matrix_b[global_row * n + global_col], valid_count);
                }
            }
        }

        // -----------------------------------------------------
        // shared(current tile) -> fragments -> FMA
        // -----------------------------------------------------

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
                fragment_a[row] = tile_a[current_buffer][thread_row * TM + row][inner];
            }

#pragma unroll
            for (int col = 0; col < TN; ++col) {
                fragment_b[col] = tile_b[current_buffer][inner][thread_col * TN + col];
            }

#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sum[row][col] = fmaf(fragment_a[row], fragment_b[col], sum[row][col]);
                }
            }
        }

        // -----------------------------------------------------
        // prefetch registers -> shared(next buffer)
        // -----------------------------------------------------

        if (has_next_tile) {
#pragma unroll
            for (int load = 0; load < A_PREFETCH_COUNT; ++load) {
                int vector_index = tid + load * BLOCK_THREADS;

                int tile_row = vector_index / A_VECTORS_PER_ROW;

                int tile_col = vector_index % A_VECTORS_PER_ROW * VECTOR_WIDTH;

                float4 values = prefetched_a[load];

                tile_a[next_buffer][tile_row][tile_col] = values.x;

                tile_a[next_buffer][tile_row][tile_col + 1] = values.y;

                tile_a[next_buffer][tile_row][tile_col + 2] = values.z;

                tile_a[next_buffer][tile_row][tile_col + 3] = values.w;
            }

#pragma unroll
            for (int load = 0; load < B_PREFETCH_COUNT; ++load) {
                int vector_index = tid + load * BLOCK_THREADS;

                int tile_row = vector_index / B_VECTORS_PER_ROW;

                int tile_col = vector_index % B_VECTORS_PER_ROW * VECTOR_WIDTH;

                float4 values = prefetched_b[load];

                tile_b[next_buffer][tile_row][tile_col] = values.x;

                tile_b[next_buffer][tile_row][tile_col + 1] = values.y;

                tile_b[next_buffer][tile_row][tile_col + 2] = values.z;

                tile_b[next_buffer][tile_row][tile_col + 3] = values.w;
            }

            // 保证下一轮开始前：
            // 1. 所有线程已完成当前 buffer 的计算；
            // 2. 所有线程已写完 next buffer。
            __syncthreads();
        }
    }

    // ---------------------------------------------------------
    // 标量写回 C
    // ---------------------------------------------------------

#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sum[row][col];
            }
        }
    }
}

void sgemm_host(const float* matrix_a, const float* matrix_b, float* matrix_c, int m, int n,
                int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int index = 0; index < k; ++index) {
                sum = fmaf(matrix_a[row * k + index], matrix_b[index * n + col], sum);
            }
            matrix_c[row * n + col] = sum;
        }
    }
}

float max_abs_error(const float* actual, const float* expected, int count) {
    float max_error = 0.0f;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            return INFINITY;
        }
        max_error = fmaxf(max_error, std::fabs(actual[index] - expected[index]));
    }
    return max_error;
}

struct BenchmarkStats {
    float median_ms;
    float min_ms;
    float max_ms;
    float p90_ms;
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
    stats.max_ms = samples.back();
    stats.p90_ms = samples[P90_INDEX];
    return stats;
}

enum class KernelSelection {
    kV4,
    kV5,
    kV6,
    kV7,
    kV4V5,
    kAll,
};

struct BenchmarkOptions {
    KernelSelection selection = KernelSelection::kAll;
    bool run_benchmark = true;
};

void print_usage(const char* program) {
    std::fprintf(stderr, "Usage: %s [--kernel v4|v5|v6|v7|all] [--no-benchmark]\n", program);
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

            if (std::strcmp(argv[index], "v4") == 0) {
                options->selection = KernelSelection::kV4;
            } else if (std::strcmp(argv[index], "v5") == 0) {
                options->selection = KernelSelection::kV5;
            } else if (std::strcmp(argv[index], "v6") == 0) {
                options->selection = KernelSelection::kV6;
            } else if (std::strcmp(argv[index], "v7") == 0) {
                options->selection = KernelSelection::kV7;
            } else if (std::strcmp(argv[index], "both") == 0) {
                options->selection = KernelSelection::kV4V5;
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
    constexpr int M = 1003;
    constexpr int N = 257;
    constexpr int K = 129;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int WARMUP = 100;
    constexpr int ITERATIONS = 1000;

    BenchmarkOptions options;
    if (!parse_options(argc, argv, &options)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const int matrix_a_elements = M * K;
    const int matrix_b_elements = K * N;
    const int matrix_c_elements = M * N;
    const size_t matrix_a_bytes = static_cast<size_t>(matrix_a_elements) * sizeof(float);
    const size_t matrix_b_bytes = static_cast<size_t>(matrix_b_elements) * sizeof(float);
    const size_t matrix_c_bytes = static_cast<size_t>(matrix_c_elements) * sizeof(float);

    float* h_matrix_a = static_cast<float*>(std::malloc(matrix_a_bytes));
    float* h_matrix_b = static_cast<float*>(std::malloc(matrix_b_bytes));
    float* h_matrix_c = static_cast<float*>(std::malloc(matrix_c_bytes));
    float* h_reference = static_cast<float*>(std::malloc(matrix_c_bytes));
    if (h_matrix_a == nullptr || h_matrix_b == nullptr || h_matrix_c == nullptr ||
        h_reference == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_matrix_a);
        std::free(h_matrix_b);
        std::free(h_matrix_c);
        std::free(h_reference);
        return EXIT_FAILURE;
    }

    for (int index = 0; index < matrix_a_elements; ++index) {
        h_matrix_a[index] = static_cast<float>((index * 17) % 101 - 50) * 0.02f;
    }
    for (int index = 0; index < matrix_b_elements; ++index) {
        h_matrix_b[index] = static_cast<float>((index * 13) % 79 - 39) * 0.015f;
    }
    sgemm_host(h_matrix_a, h_matrix_b, h_reference, M, N, K);

    float* d_matrix_a = nullptr;
    float* d_matrix_b = nullptr;
    float* d_matrix_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_matrix_a, matrix_a_bytes));
    CUDA_CHECK(cudaMalloc(&d_matrix_b, matrix_b_bytes));
    CUDA_CHECK(cudaMalloc(&d_matrix_c, matrix_c_bytes));
    CUDA_CHECK(cudaMemcpy(d_matrix_a, h_matrix_a, matrix_a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_b, h_matrix_b, matrix_b_bytes, cudaMemcpyHostToDevice));

    dim3 block(BN / TN, BM / TM);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    std::printf("M = %d, N = %d, K = %d, BLOCK = %u x %u\n", M, N, K, block.x, block.y);

    auto run_kernel = [&](const char* label, auto&& launcher) {
        CUDA_CHECK(cudaMemset(d_matrix_c, 0, matrix_c_bytes));
        launcher();
        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h_matrix_c, d_matrix_c, matrix_c_bytes, cudaMemcpyDeviceToHost));

        float max_error = max_abs_error(h_matrix_c, h_reference, matrix_c_elements);
        bool pass = max_error < 1e-5f;

        std::printf("kernel = %s\n", label);
        if (options.run_benchmark) {
            BenchmarkStats stats = benchmark_kernel_ms(launcher, WARMUP, ITERATIONS);
            double gflops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                            static_cast<double>(K) / (static_cast<double>(stats.median_ms) * 1.0e6);
            std::printf("median(ms) = %.4f, min(ms) = %.4f, p90(ms) = %.4f, max(ms) = %.4f\n",
                        stats.median_ms, stats.min_ms, stats.p90_ms, stats.max_ms);
            std::printf("performance(GFLOPS) = %.2f\n", gflops);
        }
        std::printf("max_abs_err = %.8f\n", max_error);
        std::printf("correctness = %s\n", pass ? "pass" : "fail");
        return pass;
    };

    bool all_pass = true;
    if (options.selection == KernelSelection::kV4 || options.selection == KernelSelection::kV4V5 ||
        options.selection == KernelSelection::kAll) {
        auto launch_v4 = [&]() {
            sgemm_v4<BM, BN, BK, TM, TN>
                <<<grid, block>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, N, K);
        };
        all_pass = run_kernel("V4", launch_v4) && all_pass;
    }
    if (options.selection == KernelSelection::kV5 || options.selection == KernelSelection::kV4V5 ||
        options.selection == KernelSelection::kAll) {
        auto launch_v5 = [&]() {
            sgemm_v5<BM, BN, BK, TM, TN>
                <<<grid, block>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, N, K);
        };
        all_pass = run_kernel("V5", launch_v5) && all_pass;
    }
    if (options.selection == KernelSelection::kV6 || options.selection == KernelSelection::kAll) {
        auto launch_v6 = [&]() {
            sgemm_v6<BM, BN, BK, TM, TN>
                <<<grid, block>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, N, K);
        };
        all_pass = run_kernel("V6", launch_v6) && all_pass;
    }
    if (options.selection == KernelSelection::kV7 || options.selection == KernelSelection::kAll) {
        auto launch_v7 = [&]() {
            sgemm_v7<BM, BN, BK, TM, TN>
                <<<grid, block>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, N, K);
        };
        all_pass = run_kernel("V7", launch_v7) && all_pass;
    }

    CUDA_CHECK(cudaFree(d_matrix_a));
    CUDA_CHECK(cudaFree(d_matrix_b));
    CUDA_CHECK(cudaFree(d_matrix_c));
    std::free(h_matrix_a);
    std::free(h_matrix_b);
    std::free(h_matrix_c);
    std::free(h_reference);

    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
