#include "pch.h"
#include "Denoise.h"

#include <chrono>
#include <algorithm>

using namespace System;
using namespace ImageProcCli;

static inline int clampi(int v, int lo, int hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

array<UInt16>^ CpuFilters::Box3x3(array<UInt16>^ src, int width, int height, double% elapsedMs)
{
    if (src == nullptr) throw gcnew ArgumentNullException("src");
    if (width <= 0 || height <= 0) throw gcnew ArgumentException("invalid width/height");
    if (src->Length != width * height) throw gcnew ArgumentException("src length mismatch");

    auto dst = gcnew array<UInt16>(src->Length);

    // pinして生ポインタで高速化（CPU比較の基準にする）
    pin_ptr<UInt16> pSrc = &src[0];
    pin_ptr<UInt16> pDst = &dst[0];
    UInt16* s = pSrc;
    UInt16* d = pDst;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; y++)
    {
        int y0 = clampi(y - 1, 0, height - 1);
        int y1 = y;
        int y2 = clampi(y + 1, 0, height - 1);

        for (int x = 0; x < width; x++)
        {
            int x0 = clampi(x - 1, 0, width - 1);
            int x1 = x;
            int x2 = clampi(x + 1, 0, width - 1);

            // 3x3 sum（ushortなのでintで加算）
            int sum = 0;
            sum += s[y0 * width + x0]; sum += s[y0 * width + x1]; sum += s[y0 * width + x2];
            sum += s[y1 * width + x0]; sum += s[y1 * width + x1]; sum += s[y1 * width + x2];
            sum += s[y2 * width + x0]; sum += s[y2 * width + x1]; sum += s[y2 * width + x2];

            // 平均（丸め）
            d[y * width + x] = (UInt16)((sum + 4) / 9);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    elapsedMs = ms.count();

    return dst;
}
