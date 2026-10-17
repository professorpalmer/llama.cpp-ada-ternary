#pragma once

// Turing (sm_75) MMQ table for PTQ1_0 (experiment). The other types use the Ampere table.
// GGML_CUDA_TURING_PTQ1_VARIANT (mmq-turing-variant.h): 0 = the Ampere table; 1 = occupancy 2; 2 = occupancy 2, I 64.
#include "mmq-turing-variant.h"

#if GGML_CUDA_TURING_PTQ1_VARIANT == 1
#define PTQ1_T_NT 256
#define PTQ1_T_OC 2
#define PTQ1_T_I  128
#elif GGML_CUDA_TURING_PTQ1_VARIANT == 2
#define PTQ1_T_NT 256
#define PTQ1_T_OC 2
#define PTQ1_T_I  64
#endif

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_turing(ggml_type type, int J, bool fallback) {
#if GGML_CUDA_TURING_PTQ1_VARIANT > 0
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_PTQ1_0, PTQ1_T_NT, PTQ1_T_OC, PTQ1_T_I, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
#endif
    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}
