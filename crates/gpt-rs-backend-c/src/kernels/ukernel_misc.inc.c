#if GPTRS_HAS_AVX512
static inline void gpt_rs_ukernel_masked(const float* ap,
                                         const float* bp,
                                         size_t b_stride,
                                         float* c,
                                         size_t ldc,
                                         size_t kc,
                                         size_t mr,
                                         size_t nr,
                                         const float* bias,
                                         int init_mode) {
    const __mmask16 mask = (__mmask16)gpt_rs_tail_mask(nr);
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
            const __m512 b_row = _mm512_load_ps(bp + p * b_stride);
            const __m512 a_val = _mm512_set1_ps(a_row[i]);
            acc = _mm512_fmadd_ps(a_val, b_row, acc);
        }
        _mm512_mask_storeu_ps(c + i * ldc, mask, acc);
    }
}
#endif
