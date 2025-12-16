static inline void gpt_rs_ukernel_6x32_zero(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();
    const float* a = ap;
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (size_t p = 0; p < kc; ++p) {
        if (p + GPTRS_PREFETCH_DIST < kc) {
            GPTRS_PREFETCH(bp0 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp1 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 b0 = _mm512_load_ps(bp0);
        const __m512 b1 = _mm512_load_ps(bp1);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, b0, c00);
        c01 = _mm512_fmadd_ps(va, b1, c01);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, b0, c10);
        c11 = _mm512_fmadd_ps(va, b1, c11);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, b0, c20);
        c21 = _mm512_fmadd_ps(va, b1, c21);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, b0, c30);
        c31 = _mm512_fmadd_ps(va, b1, c31);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, b0, c40);
        c41 = _mm512_fmadd_ps(va, b1, c41);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, b0, c50);
        c51 = _mm512_fmadd_ps(va, b1, c51);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
}

static inline void gpt_rs_ukernel_6x32_accum(const float* ap,
                                             const float* bp0,
                                             const float* bp1,
                                             size_t b_stride,
                                             float* c,
                                             size_t ldc,
                                             size_t kc) {
    __m512 c00 = _mm512_loadu_ps(c);
    __m512 c01 = _mm512_loadu_ps(c + 16);
    __m512 c10 = _mm512_loadu_ps(c + ldc);
    __m512 c11 = _mm512_loadu_ps(c + ldc + 16);
    __m512 c20 = _mm512_loadu_ps(c + 2 * ldc);
    __m512 c21 = _mm512_loadu_ps(c + 2 * ldc + 16);
    __m512 c30 = _mm512_loadu_ps(c + 3 * ldc);
    __m512 c31 = _mm512_loadu_ps(c + 3 * ldc + 16);
    __m512 c40 = _mm512_loadu_ps(c + 4 * ldc);
    __m512 c41 = _mm512_loadu_ps(c + 4 * ldc + 16);
    __m512 c50 = _mm512_loadu_ps(c + 5 * ldc);
    __m512 c51 = _mm512_loadu_ps(c + 5 * ldc + 16);
    const float* a = ap;
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (size_t p = 0; p < kc; ++p) {
        if (p + GPTRS_PREFETCH_DIST < kc) {
            GPTRS_PREFETCH(bp0 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp1 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 b0 = _mm512_load_ps(bp0);
        const __m512 b1 = _mm512_load_ps(bp1);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, b0, c00);
        c01 = _mm512_fmadd_ps(va, b1, c01);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, b0, c10);
        c11 = _mm512_fmadd_ps(va, b1, c11);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, b0, c20);
        c21 = _mm512_fmadd_ps(va, b1, c21);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, b0, c30);
        c31 = _mm512_fmadd_ps(va, b1, c31);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, b0, c40);
        c41 = _mm512_fmadd_ps(va, b1, c41);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, b0, c50);
        c51 = _mm512_fmadd_ps(va, b1, c51);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
}

static inline void gpt_rs_ukernel_6x32_bias(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc,
                                            const float* bias) {
    const __m512 b0 = _mm512_loadu_ps(bias);
    const __m512 b1 = _mm512_loadu_ps(bias + 16);
    __m512 c00 = b0;
    __m512 c01 = b1;
    __m512 c10 = b0;
    __m512 c11 = b1;
    __m512 c20 = b0;
    __m512 c21 = b1;
    __m512 c30 = b0;
    __m512 c31 = b1;
    __m512 c40 = b0;
    __m512 c41 = b1;
    __m512 c50 = b0;
    __m512 c51 = b1;
    const float* a = ap;
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (size_t p = 0; p < kc; ++p) {
        if (p + GPTRS_PREFETCH_DIST < kc) {
            GPTRS_PREFETCH(bp0 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp1 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 w0 = _mm512_load_ps(bp0);
        const __m512 w1 = _mm512_load_ps(bp1);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, w0, c00);
        c01 = _mm512_fmadd_ps(va, w1, c01);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, w0, c10);
        c11 = _mm512_fmadd_ps(va, w1, c11);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, w0, c20);
        c21 = _mm512_fmadd_ps(va, w1, c21);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, w0, c30);
        c31 = _mm512_fmadd_ps(va, w1, c31);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, w0, c40);
        c41 = _mm512_fmadd_ps(va, w1, c41);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, w0, c50);
        c51 = _mm512_fmadd_ps(va, w1, c51);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
}
