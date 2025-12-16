static inline void gpt_rs_ukernel_partial(const float* ap, const float* bp, float* c,
                                          size_t ldc, size_t kc, size_t mr, size_t nr,
                                          int accum) {
    float tmp[GPTRS_MR * GPTRS_NR];
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            tmp[i * GPTRS_NR + j] = accum ? c[i * ldc + j] : 0.0f;
        }
    }
    for (size_t p = 0; p < kc; ++p) {
        const float* b_row = bp + p * GPTRS_NR;
        const float* a_row = ap + p * GPTRS_MR;
        for (size_t i = 0; i < mr; ++i) {
            const float a_val = a_row[i];
            float* dst = tmp + i * GPTRS_NR;
            for (size_t j = 0; j < nr; ++j) {
                dst[j] += a_val * b_row[j];
            }
        }
    }
    for (size_t i = 0; i < mr; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            c[i * ldc + j] = tmp[i * GPTRS_NR + j];
        }
    }
}

static inline void gpt_rs_ukernel_masked(const float* ap,
                                         const float* bp,
                                         float* c,
                                         size_t ldc,
                                         size_t kc,
                                         size_t mr,
                                         size_t nr,
                                         const float* bias,
                                         int init_mode) {
    const __mmask16 mask = (nr >= GPTRS_NR) ? 0xFFFFu : (uint16_t)((1u << nr) - 1u);
    for (size_t i = 0; i < mr; ++i) {
        __m512 acc;
        if (init_mode == GPTRS_INIT_ACCUM) {
            acc = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, c + i * ldc);
        } else if (init_mode == GPTRS_INIT_BIAS && bias) {
            acc = _mm512_mask_loadu_ps(_mm512_setzero_ps(), mask, bias);
        } else {
            acc = _mm512_setzero_ps();
        }
        for (size_t p = 0; p < kc; ++p) {
            const float* a_row = ap + p * GPTRS_MR;
            const __m512 b_row = _mm512_load_ps(bp + p * GPTRS_NR);
            const __m512 a_val = _mm512_set1_ps(a_row[i]);
            acc = _mm512_fmadd_ps(a_val, b_row, acc);
        }
        _mm512_mask_storeu_ps(c + i * ldc, mask, acc);
    }
}
