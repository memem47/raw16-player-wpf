#include <cuda_runtime.h>
#include <cstdint>

static __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void box3x3_u16_kernel(const uint16_t* src, uint16_t* dst, int w, int h)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;

    int x0 = clampi(x - 1, 0, w - 1);
    int x1 = x;
    int x2 = clampi(x + 1, 0, w - 1);

    int y0 = clampi(y - 1, 0, h - 1);
    int y1 = y;
    int y2 = clampi(y + 1, 0, h - 1);

    int sum = 0;
    sum += src[y0 * w + x0]; sum += src[y0 * w + x1]; sum += src[y0 * w + x2];
    sum += src[y1 * w + x0]; sum += src[y1 * w + x1]; sum += src[y1 * w + x2];
    sum += src[y2 * w + x0]; sum += src[y2 * w + x1]; sum += src[y2 * w + x2];

    dst[y * w + x] = (uint16_t)((sum + 4) / 9);
}

extern "C" void launch_box3x3_u16(
    const uint16_t* d_src,
    uint16_t* d_dst,
    int w,
    int h,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    box3x3_u16_kernel << <grid, block, 0, stream >> > (d_src, d_dst, w, h);
}
