cmake_minimum_required(VERSION 3.24)

project(LLM-Kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

add_library(cuda_options INTERFACE)
target_include_directories(cuda_options INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce/include
)
target_compile_options(cuda_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

function(add_cuda_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE cuda_options)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RUNTIME_LIBRARY Shared
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endfunction()

add_cuda_executable(sum_reduce_sum reduce/sum/reduce_sum.cu)
add_cuda_executable(max_reduce_max reduce/max/reduce_max.cu)
add_cuda_executable(softmax_softmax reduce/softmax/softmax.cu)
add_cuda_executable(softmax_matrix_softmax_matrix reduce/softmax_matrix/softmax_matrix.cu)
add_cuda_executable(elementwise_add elementwise/add.cu)
add_cuda_executable(transpose transpose/transpose.cu)
add_cuda_executable(rmsnorm rmsnorm/rmsnorm.cu)
add_cuda_executable(layernorm layernorm/layernorm.cu)
add_cuda_executable(gemv gemv/gemv.cu)
add_cuda_executable(sgemm sgemm/sgemm.cu)
add_cuda_executable(rope rope/rope.cu)
add_cuda_executable(attention_naive flash_attention/flash_attention_naive.cu)
add_cuda_executable(attention_flash flash_attention/flash_attention_tiled.cu)
