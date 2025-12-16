static inline void gpt_rs_ukernel_6x64_zero(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            const float* bp3,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc) {
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c02 = _mm512_setzero_ps();
    __m512 c03 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c12 = _mm512_setzero_ps();
    __m512 c13 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c22 = _mm512_setzero_ps();
    __m512 c23 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c32 = _mm512_setzero_ps();
    __m512 c33 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps();
    __m512 c41 = _mm512_setzero_ps();
    __m512 c42 = _mm512_setzero_ps();
    __m512 c43 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps();
    __m512 c51 = _mm512_setzero_ps();
    __m512 c52 = _mm512_setzero_ps();
    __m512 c53 = _mm512_setzero_ps();
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
            GPTRS_PREFETCH(bp2 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp3 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 b0 = _mm512_load_ps(bp0);
        const __m512 b1 = _mm512_load_ps(bp1);
        const __m512 b2 = _mm512_load_ps(bp2);
        const __m512 b3 = _mm512_load_ps(bp3);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, b0, c00);
        c01 = _mm512_fmadd_ps(va, b1, c01);
        c02 = _mm512_fmadd_ps(va, b2, c02);
        c03 = _mm512_fmadd_ps(va, b3, c03);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, b0, c10);
        c11 = _mm512_fmadd_ps(va, b1, c11);
        c12 = _mm512_fmadd_ps(va, b2, c12);
        c13 = _mm512_fmadd_ps(va, b3, c13);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, b0, c20);
        c21 = _mm512_fmadd_ps(va, b1, c21);
        c22 = _mm512_fmadd_ps(va, b2, c22);
        c23 = _mm512_fmadd_ps(va, b3, c23);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, b0, c30);
        c31 = _mm512_fmadd_ps(va, b1, c31);
        c32 = _mm512_fmadd_ps(va, b2, c32);
        c33 = _mm512_fmadd_ps(va, b3, c33);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, b0, c40);
        c41 = _mm512_fmadd_ps(va, b1, c41);
        c42 = _mm512_fmadd_ps(va, b2, c42);
        c43 = _mm512_fmadd_ps(va, b3, c43);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, b0, c50);
        c51 = _mm512_fmadd_ps(va, b1, c51);
        c52 = _mm512_fmadd_ps(va, b2, c52);
        c53 = _mm512_fmadd_ps(va, b3, c53);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
        bp2 += b_stride;
        bp3 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + 32, c02);
    _mm512_storeu_ps(c + 48, c03);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + ldc + 32, c12);
    _mm512_storeu_ps(c + ldc + 48, c13);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 2 * ldc + 32, c22);
    _mm512_storeu_ps(c + 2 * ldc + 48, c23);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 3 * ldc + 32, c32);
    _mm512_storeu_ps(c + 3 * ldc + 48, c33);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 4 * ldc + 32, c42);
    _mm512_storeu_ps(c + 4 * ldc + 48, c43);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
    _mm512_storeu_ps(c + 5 * ldc + 32, c52);
    _mm512_storeu_ps(c + 5 * ldc + 48, c53);
}

static inline void gpt_rs_ukernel_6x64_accum(const float* ap,
                                             const float* bp0,
                                             const float* bp1,
                                             const float* bp2,
                                             const float* bp3,
                                             size_t b_stride,
                                             float* c,
                                             size_t ldc,
                                             size_t kc) {
    __m512 c00 = _mm512_loadu_ps(c);
    __m512 c01 = _mm512_loadu_ps(c + 16);
    __m512 c02 = _mm512_loadu_ps(c + 32);
    __m512 c03 = _mm512_loadu_ps(c + 48);
    __m512 c10 = _mm512_loadu_ps(c + ldc);
    __m512 c11 = _mm512_loadu_ps(c + ldc + 16);
    __m512 c12 = _mm512_loadu_ps(c + ldc + 32);
    __m512 c13 = _mm512_loadu_ps(c + ldc + 48);
    __m512 c20 = _mm512_loadu_ps(c + 2 * ldc);
    __m512 c21 = _mm512_loadu_ps(c + 2 * ldc + 16);
    __m512 c22 = _mm512_loadu_ps(c + 2 * ldc + 32);
    __m512 c23 = _mm512_loadu_ps(c + 2 * ldc + 48);
    __m512 c30 = _mm512_loadu_ps(c + 3 * ldc);
    __m512 c31 = _mm512_loadu_ps(c + 3 * ldc + 16);
    __m512 c32 = _mm512_loadu_ps(c + 3 * ldc + 32);
    __m512 c33 = _mm512_loadu_ps(c + 3 * ldc + 48);
    __m512 c40 = _mm512_loadu_ps(c + 4 * ldc);
    __m512 c41 = _mm512_loadu_ps(c + 4 * ldc + 16);
    __m512 c42 = _mm512_loadu_ps(c + 4 * ldc + 32);
    __m512 c43 = _mm512_loadu_ps(c + 4 * ldc + 48);
    __m512 c50 = _mm512_loadu_ps(c + 5 * ldc);
    __m512 c51 = _mm512_loadu_ps(c + 5 * ldc + 16);
    __m512 c52 = _mm512_loadu_ps(c + 5 * ldc + 32);
    __m512 c53 = _mm512_loadu_ps(c + 5 * ldc + 48);
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
            GPTRS_PREFETCH(bp2 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp3 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 b0 = _mm512_load_ps(bp0);
        const __m512 b1 = _mm512_load_ps(bp1);
        const __m512 b2 = _mm512_load_ps(bp2);
        const __m512 b3 = _mm512_load_ps(bp3);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, b0, c00);
        c01 = _mm512_fmadd_ps(va, b1, c01);
        c02 = _mm512_fmadd_ps(va, b2, c02);
        c03 = _mm512_fmadd_ps(va, b3, c03);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, b0, c10);
        c11 = _mm512_fmadd_ps(va, b1, c11);
        c12 = _mm512_fmadd_ps(va, b2, c12);
        c13 = _mm512_fmadd_ps(va, b3, c13);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, b0, c20);
        c21 = _mm512_fmadd_ps(va, b1, c21);
        c22 = _mm512_fmadd_ps(va, b2, c22);
        c23 = _mm512_fmadd_ps(va, b3, c23);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, b0, c30);
        c31 = _mm512_fmadd_ps(va, b1, c31);
        c32 = _mm512_fmadd_ps(va, b2, c32);
        c33 = _mm512_fmadd_ps(va, b3, c33);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, b0, c40);
        c41 = _mm512_fmadd_ps(va, b1, c41);
        c42 = _mm512_fmadd_ps(va, b2, c42);
        c43 = _mm512_fmadd_ps(va, b3, c43);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, b0, c50);
        c51 = _mm512_fmadd_ps(va, b1, c51);
        c52 = _mm512_fmadd_ps(va, b2, c52);
        c53 = _mm512_fmadd_ps(va, b3, c53);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
        bp2 += b_stride;
        bp3 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + 32, c02);
    _mm512_storeu_ps(c + 48, c03);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + ldc + 32, c12);
    _mm512_storeu_ps(c + ldc + 48, c13);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 2 * ldc + 32, c22);
    _mm512_storeu_ps(c + 2 * ldc + 48, c23);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 3 * ldc + 32, c32);
    _mm512_storeu_ps(c + 3 * ldc + 48, c33);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 4 * ldc + 32, c42);
    _mm512_storeu_ps(c + 4 * ldc + 48, c43);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
    _mm512_storeu_ps(c + 5 * ldc + 32, c52);
    _mm512_storeu_ps(c + 5 * ldc + 48, c53);
}

static inline void gpt_rs_ukernel_6x64_bias(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            const float* bp3,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc,
                                            const float* bias) {
    const __m512 b0 = _mm512_loadu_ps(bias);
    const __m512 b1 = _mm512_loadu_ps(bias + 16);
    const __m512 b2 = _mm512_loadu_ps(bias + 32);
    const __m512 b3 = _mm512_loadu_ps(bias + 48);
    __m512 c00 = b0;
    __m512 c01 = b1;
    __m512 c02 = b2;
    __m512 c03 = b3;
    __m512 c10 = b0;
    __m512 c11 = b1;
    __m512 c12 = b2;
    __m512 c13 = b3;
    __m512 c20 = b0;
    __m512 c21 = b1;
    __m512 c22 = b2;
    __m512 c23 = b3;
    __m512 c30 = b0;
    __m512 c31 = b1;
    __m512 c32 = b2;
    __m512 c33 = b3;
    __m512 c40 = b0;
    __m512 c41 = b1;
    __m512 c42 = b2;
    __m512 c43 = b3;
    __m512 c50 = b0;
    __m512 c51 = b1;
    __m512 c52 = b2;
    __m512 c53 = b3;
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
            GPTRS_PREFETCH(bp2 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(bp3 + GPTRS_PREFETCH_DIST * b_stride);
            GPTRS_PREFETCH(a + GPTRS_PREFETCH_DIST * GPTRS_MR);
        }
        const __m512 w0 = _mm512_load_ps(bp0);
        const __m512 w1 = _mm512_load_ps(bp1);
        const __m512 w2 = _mm512_load_ps(bp2);
        const __m512 w3 = _mm512_load_ps(bp3);
        __m512 va = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(va, w0, c00);
        c01 = _mm512_fmadd_ps(va, w1, c01);
        c02 = _mm512_fmadd_ps(va, w2, c02);
        c03 = _mm512_fmadd_ps(va, w3, c03);
        va = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(va, w0, c10);
        c11 = _mm512_fmadd_ps(va, w1, c11);
        c12 = _mm512_fmadd_ps(va, w2, c12);
        c13 = _mm512_fmadd_ps(va, w3, c13);
        va = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(va, w0, c20);
        c21 = _mm512_fmadd_ps(va, w1, c21);
        c22 = _mm512_fmadd_ps(va, w2, c22);
        c23 = _mm512_fmadd_ps(va, w3, c23);
        va = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(va, w0, c30);
        c31 = _mm512_fmadd_ps(va, w1, c31);
        c32 = _mm512_fmadd_ps(va, w2, c32);
        c33 = _mm512_fmadd_ps(va, w3, c33);
        va = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(va, w0, c40);
        c41 = _mm512_fmadd_ps(va, w1, c41);
        c42 = _mm512_fmadd_ps(va, w2, c42);
        c43 = _mm512_fmadd_ps(va, w3, c43);
        va = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(va, w0, c50);
        c51 = _mm512_fmadd_ps(va, w1, c51);
        c52 = _mm512_fmadd_ps(va, w2, c52);
        c53 = _mm512_fmadd_ps(va, w3, c53);
        a += GPTRS_MR;
        bp0 += b_stride;
        bp1 += b_stride;
        bp2 += b_stride;
        bp3 += b_stride;
    }
    _mm512_storeu_ps(c, c00);
    _mm512_storeu_ps(c + 16, c01);
    _mm512_storeu_ps(c + 32, c02);
    _mm512_storeu_ps(c + 48, c03);
    _mm512_storeu_ps(c + ldc, c10);
    _mm512_storeu_ps(c + ldc + 16, c11);
    _mm512_storeu_ps(c + ldc + 32, c12);
    _mm512_storeu_ps(c + ldc + 48, c13);
    _mm512_storeu_ps(c + 2 * ldc, c20);
    _mm512_storeu_ps(c + 2 * ldc + 16, c21);
    _mm512_storeu_ps(c + 2 * ldc + 32, c22);
    _mm512_storeu_ps(c + 2 * ldc + 48, c23);
    _mm512_storeu_ps(c + 3 * ldc, c30);
    _mm512_storeu_ps(c + 3 * ldc + 16, c31);
    _mm512_storeu_ps(c + 3 * ldc + 32, c32);
    _mm512_storeu_ps(c + 3 * ldc + 48, c33);
    _mm512_storeu_ps(c + 4 * ldc, c40);
    _mm512_storeu_ps(c + 4 * ldc + 16, c41);
    _mm512_storeu_ps(c + 4 * ldc + 32, c42);
    _mm512_storeu_ps(c + 4 * ldc + 48, c43);
    _mm512_storeu_ps(c + 5 * ldc, c50);
    _mm512_storeu_ps(c + 5 * ldc + 16, c51);
    _mm512_storeu_ps(c + 5 * ldc + 32, c52);
    _mm512_storeu_ps(c + 5 * ldc + 48, c53);
}
