#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

    // end-to-end のGPU処理（H2D + kernel + D2H）を実行して dst に出す
    // elapsed_ms には CUDA Event で測った時間（ms）を返す
    int cuda_box3x3_u16(
        const uint16_t* src_host,
        uint16_t* dst_host,
        int width,
        int height,
        float* elapsed_ms);

#ifdef __cplusplus
}
#endif
