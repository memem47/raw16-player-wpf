#include "cuda_denoise.h"
#include <cuda_runtime.h>
#include <cstdio>

extern "C" void launch_box3x3_u16(
    const uint16_t* d_src, uint16_t* d_dst, int w, int h, cudaStream_t stream);

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) return (int)err; \
} while(0)

extern "C" int cuda_box3x3_u16(
    const uint16_t* src_host,
    uint16_t* dst_host,
    int width,
    int height,
    float* elapsed_ms)
{
    if (!src_host || !dst_host || width <= 0 || height <= 0) return -1;

    const size_t n = (size_t)width * (size_t)height;
    const size_t bytes = n * sizeof(uint16_t);

    uint16_t* d_src = nullptr;
    uint16_t* d_dst = nullptr;

    cudaStream_t stream = nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_dst, bytes));

    // ここから計測（end-to-end：H2D + kernel + D2H）
    CUDA_CHECK(cudaEventRecord(ev0, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_src, src_host, bytes, cudaMemcpyHostToDevice, stream));
    launch_box3x3_u16(d_src, d_dst, width, height, stream);
    CUDA_CHECK(cudaMemcpyAsync(dst_host, d_dst, bytes, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    if (elapsed_ms)
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        *elapsed_ms = ms;
    }

    // 後始末
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);

    return 0; // success
}
