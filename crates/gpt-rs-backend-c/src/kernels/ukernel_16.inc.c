static inline void gpt_rs_ukernel_1x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc) {
    __m512 c0 = _mm512_setzero_ps();
    const float* a0 = ap;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 a = _mm512_set1_ps(*a0);
        c0 = _mm512_fmadd_ps(a, b0, c0);
        a0 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
}

static inline void gpt_rs_ukernel_1x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc) {
    __m512 c0 = _mm512_loadu_ps(c);
    const float* a0 = ap;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 a = _mm512_set1_ps(*a0);
        c0 = _mm512_fmadd_ps(a, b0, c0);
        a0 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
}

static inline void gpt_rs_ukernel_2x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    const float* a0 = ap;
    const float* a1 = ap + 1;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
}

static inline void gpt_rs_ukernel_2x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc) {
    __m512 c0 = _mm512_loadu_ps(c);
    __m512 c1 = _mm512_loadu_ps(c + ldc);
    const float* a0 = ap;
    const float* a1 = ap + 1;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
}

static inline void gpt_rs_ukernel_4x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    const float* a0 = ap;
    const float* a1 = ap + 1;
    const float* a2 = ap + 2;
    const float* a3 = ap + 3;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        const __m512 va3 = _mm512_set1_ps(*a3);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        c2 = _mm512_fmadd_ps(va2, b0, c2);
        c3 = _mm512_fmadd_ps(va3, b0, c3);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        a2 += GPTRS_MR;
        a3 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
    _mm512_storeu_ps(c + 3 * ldc, c3);
}

static inline void gpt_rs_ukernel_4x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc) {
    __m512 c0 = _mm512_loadu_ps(c);
    __m512 c1 = _mm512_loadu_ps(c + ldc);
    __m512 c2 = _mm512_loadu_ps(c + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(c + 3 * ldc);
    const float* a0 = ap;
    const float* a1 = ap + 1;
    const float* a2 = ap + 2;
    const float* a3 = ap + 3;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        const __m512 va3 = _mm512_set1_ps(*a3);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        c2 = _mm512_fmadd_ps(va2, b0, c2);
        c3 = _mm512_fmadd_ps(va3, b0, c3);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        a2 += GPTRS_MR;
        a3 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
    _mm512_storeu_ps(c + 3 * ldc, c3);
}

static inline void gpt_rs_ukernel_6x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc) {
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps();
    __m512 c5 = _mm512_setzero_ps();
    const float* a0 = ap;
    const float* a1 = ap + 1;
    const float* a2 = ap + 2;
    const float* a3 = ap + 3;
    const float* a4 = ap + 4;
    const float* a5 = ap + 5;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        const __m512 va3 = _mm512_set1_ps(*a3);
        const __m512 va4 = _mm512_set1_ps(*a4);
        const __m512 va5 = _mm512_set1_ps(*a5);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        c2 = _mm512_fmadd_ps(va2, b0, c2);
        c3 = _mm512_fmadd_ps(va3, b0, c3);
        c4 = _mm512_fmadd_ps(va4, b0, c4);
        c5 = _mm512_fmadd_ps(va5, b0, c5);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        a2 += GPTRS_MR;
        a3 += GPTRS_MR;
        a4 += GPTRS_MR;
        a5 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
    _mm512_storeu_ps(c + 3 * ldc, c3);
    _mm512_storeu_ps(c + 4 * ldc, c4);
    _mm512_storeu_ps(c + 5 * ldc, c5);
}

static inline void gpt_rs_ukernel_6x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc) {
    __m512 c0 = _mm512_loadu_ps(c);
    __m512 c1 = _mm512_loadu_ps(c + ldc);
    __m512 c2 = _mm512_loadu_ps(c + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(c + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(c + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(c + 5 * ldc);
    const float* a0 = ap;
    const float* a1 = ap + 1;
    const float* a2 = ap + 2;
    const float* a3 = ap + 3;
    const float* a4 = ap + 4;
    const float* a5 = ap + 5;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        const __m512 va3 = _mm512_set1_ps(*a3);
        const __m512 va4 = _mm512_set1_ps(*a4);
        const __m512 va5 = _mm512_set1_ps(*a5);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        c2 = _mm512_fmadd_ps(va2, b0, c2);
        c3 = _mm512_fmadd_ps(va3, b0, c3);
        c4 = _mm512_fmadd_ps(va4, b0, c4);
        c5 = _mm512_fmadd_ps(va5, b0, c5);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        a2 += GPTRS_MR;
        a3 += GPTRS_MR;
        a4 += GPTRS_MR;
        a5 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
    _mm512_storeu_ps(c + 3 * ldc, c3);
    _mm512_storeu_ps(c + 4 * ldc, c4);
    _mm512_storeu_ps(c + 5 * ldc, c5);
}

static inline void gpt_rs_ukernel_6x16_bias(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc, const float* bias) {
    const __m512 b = _mm512_loadu_ps(bias);
    __m512 c0 = b;
    __m512 c1 = b;
    __m512 c2 = b;
    __m512 c3 = b;
    __m512 c4 = b;
    __m512 c5 = b;
    const float* a0 = ap;
    const float* a1 = ap + 1;
    const float* a2 = ap + 2;
    const float* a3 = ap + 3;
    const float* a4 = ap + 4;
    const float* a5 = ap + 5;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va0 = _mm512_set1_ps(*a0);
        const __m512 va1 = _mm512_set1_ps(*a1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        const __m512 va3 = _mm512_set1_ps(*a3);
        const __m512 va4 = _mm512_set1_ps(*a4);
        const __m512 va5 = _mm512_set1_ps(*a5);
        c0 = _mm512_fmadd_ps(va0, b0, c0);
        c1 = _mm512_fmadd_ps(va1, b0, c1);
        c2 = _mm512_fmadd_ps(va2, b0, c2);
        c3 = _mm512_fmadd_ps(va3, b0, c3);
        c4 = _mm512_fmadd_ps(va4, b0, c4);
        c5 = _mm512_fmadd_ps(va5, b0, c5);
        a0 += GPTRS_MR;
        a1 += GPTRS_MR;
        a2 += GPTRS_MR;
        a3 += GPTRS_MR;
        a4 += GPTRS_MR;
        a5 += GPTRS_MR;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
    _mm512_storeu_ps(c + ldc, c1);
    _mm512_storeu_ps(c + 2 * ldc, c2);
    _mm512_storeu_ps(c + 3 * ldc, c3);
    _mm512_storeu_ps(c + 4 * ldc, c4);
    _mm512_storeu_ps(c + 5 * ldc, c5);
}
