/*
 * y[b][m][n] = sum_k x[b][m][k] * w[b][n][k]
 *
 * Linear projection in the PyTorch layout, optionally batched. Rows of x and w are contiguous
 * along K. `sxb` and `ldx` are the batch and row strides of x, and `swb` and `ldw` are those of w.
 * y is contiguous f32 [batch, M, N].
 *
 * The kernels compute MR x NR tiles of dot products with 16-lane partial sums. They reduce the
 * partial sums horizontally at the end of each K block. bf16 operands are widened to f32 in
 * registers, which is exact. So the results are f32 products and sums of the stored values.
 * Threads split N, so each weight row streams from memory exactly once per call.
 */

/* NR: 4 x 6 accumulators plus 6 weight vectors use 30 of the 32 zmm registers. KC: one K block of
 * the 6 weight rows is 24 KiB as f32. It stays in L1D while the rows of x stream past it. */
enum { GPTRS_LIN_NR = 6, GPTRS_LIN_KC = 1024 };

/* Activation footprint above which the drivers switch to the K-outer loop order. x shares the
 * per-core L2 with the weight rows that stream through it, so x may take half of the L2. */
#define GPTRS_LIN_X_CACHE_BYTES ((size_t)GPTRS_L2_BYTES / 2)

#if GPTRS_HAS_AVX512

/* Operand loaders. `NAME(p)` reads one vector. `NAME##_tail(p, left)` reads the last `left`
 * elements. */
static inline __m512 gpt_rs_lin_f32(const float* p) {
    return _mm512_loadu_ps(p);
}

static inline __m512 gpt_rs_lin_f32_tail(const float* p, size_t left) {
    return _mm512_maskz_loadu_ps((__mmask16)gpt_rs_tail_mask(left), p);
}

static inline __m512 gpt_rs_lin_bf16(const uint16_t* p) {
    const __m256i raw = _mm256_loadu_si256((const __m256i*)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

static inline __m512 gpt_rs_lin_bf16_tail(const uint16_t* p, size_t left) {
    const __m256i raw = _mm256_maskz_loadu_epi16((__mmask16)gpt_rs_tail_mask(left), p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

static inline __m512 gpt_rs_lin_fma(__m512 acc, __m512 x, __m512 w) {
    return _mm512_fmadd_ps(x, w, acc);
}

/* Full MR x GPTRS_LIN_NR tile over one K block. */
#define GPTRS_DEFINE_LINEAR_TILE(NAME, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD, MR)         \
    static inline void NAME(const XTYPE* GPTRS_RESTRICT x, size_t ldx,                       \
                            const WTYPE* GPTRS_RESTRICT w, size_t ldw,                       \
                            float* GPTRS_RESTRICT y, size_t ldy, size_t kc, int accumulate) { \
        __m512 acc[MR][GPTRS_LIN_NR];                                                        \
        for (int i = 0; i < MR; ++i) {                                                       \
            for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                         \
                acc[i][j] = _mm512_setzero_ps();                                             \
            }                                                                                \
        }                                                                                    \
        size_t p = 0;                                                                        \
        for (; p + (STEP) <= kc; p += (STEP)) {                                              \
            VEC wv[GPTRS_LIN_NR];                                                            \
            for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                         \
                wv[j] = LOADW(w + (size_t)j * ldw + p);                                      \
            }                                                                                \
            for (int i = 0; i < MR; ++i) {                                                   \
                const VEC xv = LOADX(x + (size_t)i * ldx + p);                               \
                for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                     \
                    acc[i][j] = MADD(acc[i][j], xv, wv[j]);                                  \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        if (p < kc) {                                                                        \
            const size_t left = kc - p;                                                      \
            VEC wv[GPTRS_LIN_NR];                                                            \
            for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                         \
                wv[j] = LOADW##_tail(w + (size_t)j * ldw + p, left);                         \
            }                                                                                \
            for (int i = 0; i < MR; ++i) {                                                   \
                const VEC xv = LOADX##_tail(x + (size_t)i * ldx + p, left);                  \
                for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                     \
                    acc[i][j] = MADD(acc[i][j], xv, wv[j]);                                  \
                }                                                                            \
            }                                                                                \
        }                                                                                    \
        for (int i = 0; i < MR; ++i) {                                                       \
            for (int j = 0; j < GPTRS_LIN_NR; ++j) {                                         \
                const float sum = _mm512_reduce_add_ps(acc[i][j]);                           \
                float* dst = y + (size_t)i * ldy + j;                                        \
                *dst = accumulate ? *dst + sum : sum;                                        \
            }                                                                                \
        }                                                                                    \
    }

/*
 * One dot-product kernel family. Each VEC operand holds STEP elements of K, and MADD accumulates
 * them into 16 f32 lanes.
 *
 * - `_tile<MR>` covers MR (1..4) rows by a full GPTRS_LIN_NR-wide weight panel.
 * - `_edge` covers any mr <= 4 rows by nr <= GPTRS_LIN_NR columns.
 * - `_panel` covers all m rows of one panel over one K block.
 *
 * Leading dimensions are explicit, so batched callers can pass strided operands. With
 * `accumulate`, the kernels add into y instead of storing.
 */
#define GPTRS_DEFINE_LINEAR_FAMILY(PREFIX, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD)         \
    GPTRS_DEFINE_LINEAR_TILE(PREFIX##_tile1, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD, 1)  \
    GPTRS_DEFINE_LINEAR_TILE(PREFIX##_tile2, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD, 2)  \
    GPTRS_DEFINE_LINEAR_TILE(PREFIX##_tile3, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD, 3)  \
    GPTRS_DEFINE_LINEAR_TILE(PREFIX##_tile4, XTYPE, WTYPE, VEC, STEP, LOADX, LOADW, MADD, 4)  \
    static inline void PREFIX##_edge(const XTYPE* GPTRS_RESTRICT x, size_t ldx,              \
                                     const WTYPE* GPTRS_RESTRICT w, size_t ldw,              \
                                     float* GPTRS_RESTRICT y, size_t ldy, size_t kc,         \
                                     size_t mr, size_t nr, int accumulate) {                 \
        for (size_t j = 0; j < nr; ++j) {                                                    \
            const WTYPE* wr = w + j * ldw;                                                   \
            for (size_t i = 0; i < mr; ++i) {                                                \
                const XTYPE* xr = x + i * ldx;                                               \
                __m512 acc = _mm512_setzero_ps();                                            \
                size_t p = 0;                                                                \
                for (; p + (STEP) <= kc; p += (STEP)) {                                      \
                    acc = MADD(acc, LOADX(xr + p), LOADW(wr + p));                           \
                }                                                                            \
                if (p < kc) {                                                                \
                    acc = MADD(acc, LOADX##_tail(xr + p, kc - p),                            \
                               LOADW##_tail(wr + p, kc - p));                                \
                }                                                                            \
                const float sum = _mm512_reduce_add_ps(acc);                                 \
                float* dst = y + i * ldy + j;                                                \
                *dst = accumulate ? *dst + sum : sum;                                        \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
    static inline void PREFIX##_panel(const XTYPE* GPTRS_RESTRICT x, size_t ldx,             \
                                      const WTYPE* GPTRS_RESTRICT w, size_t ldw,             \
                                      float* GPTRS_RESTRICT y, size_t ldy, size_t m,         \
                                      size_t nr, size_t kc, int accumulate) {                \
        if (nr != GPTRS_LIN_NR) {                                                            \
            for (size_t m0 = 0; m0 < m; m0 += 4) {                                           \
                PREFIX##_edge(x + m0 * ldx, ldx, w, ldw, y + m0 * ldy, ldy, kc,              \
                              GPTRS_MIN((size_t)4, m - m0), nr, accumulate);                 \
            }                                                                                \
            return;                                                                          \
        }                                                                                    \
        size_t m0 = 0;                                                                       \
        for (; m0 + 4 <= m; m0 += 4) {                                                       \
            PREFIX##_tile4(x + m0 * ldx, ldx, w, ldw, y + m0 * ldy, ldy, kc, accumulate);    \
        }                                                                                    \
        const XTYPE* xt = x + m0 * ldx;                                                      \
        float* yt = y + m0 * ldy;                                                            \
        switch (m - m0) {                                                                    \
        case 3: PREFIX##_tile3(xt, ldx, w, ldw, yt, ldy, kc, accumulate); break;             \
        case 2: PREFIX##_tile2(xt, ldx, w, ldw, yt, ldy, kc, accumulate); break;             \
        case 1: PREFIX##_tile1(xt, ldx, w, ldw, yt, ldy, kc, accumulate); break;             \
        default: break;                                                                      \
        }                                                                                    \
    }

/*
 * Threads claim contiguous ranges of NR-column weight panels. When the activations outgrow the L2
 * cache, the driver sweeps K blocks outermost. Each thread then reuses one m x KC block of x
 * across all of its panels. Both loop orders sum each output in the same order.
 */
#define GPTRS_DEFINE_LINEAR_DRIVER(NAME, XTYPE, WTYPE, PANEL, NR, KC)                          \
    static void NAME(const XTYPE* GPTRS_RESTRICT x, const WTYPE* GPTRS_RESTRICT w,            \
                     float* GPTRS_RESTRICT y, size_t batch, size_t m, size_t n, size_t k,     \
                     size_t sxb, size_t ldx, size_t swb, size_t ldw) {                        \
        if (batch == 0 || m == 0 || n == 0) {                                                \
            return;                                                                          \
        }                                                                                    \
        if (k == 0) {                                                                        \
            memset(y, 0, batch * m * n * sizeof(float));                                     \
            return;                                                                          \
        }                                                                                    \
        const size_t panels = (n + (NR) - 1) / (NR);                                         \
        const size_t blocks = (k + (KC) - 1) / (KC);                                         \
        const size_t team =                                                                  \
            batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK ? (size_t)gpt_rs_c_team_size() : 1;  \
        const size_t parts =                                                                 \
            GPTRS_MIN(panels, (team * GPTRS_TASKS_PER_THREAD + batch - 1) / batch);          \
        const int k_outer = m * k * sizeof(XTYPE) > GPTRS_LIN_X_CACHE_BYTES;                  \
        _Pragma("omp parallel for schedule(dynamic, 1) if (team > 1)")                       \
        for (size_t task = 0; task < batch * parts; ++task) {                                \
            const size_t bi = task / parts;                                                  \
            const size_t nb0 = panels * (task % parts) / parts;                              \
            const size_t count = panels * (task % parts + 1) / parts - nb0;                  \
            for (size_t step = 0; step < count * blocks; ++step) {                           \
                const size_t n0 = (nb0 + (k_outer ? step % count : step / blocks)) * (NR);  \
                const size_t kb = (k_outer ? step / count : step % blocks) * (KC);           \
                PANEL(x + bi * sxb + kb, ldx, w + bi * swb + n0 * ldw + kb, ldw,             \
                      y + bi * m * n + n0, n, m, GPTRS_MIN((size_t)(NR), n - n0),            \
                      GPTRS_MIN((size_t)(KC), k - kb), kb != 0);                             \
            }                                                                                \
        }                                                                                    \
    }

GPTRS_DEFINE_LINEAR_FAMILY(gpt_rs_lin_f32, float, float, __m512, 16, gpt_rs_lin_f32,
                           gpt_rs_lin_f32, gpt_rs_lin_fma)
GPTRS_DEFINE_LINEAR_FAMILY(gpt_rs_lin_f32_bf16, float, uint16_t, __m512, 16, gpt_rs_lin_f32,
                           gpt_rs_lin_bf16, gpt_rs_lin_fma)
GPTRS_DEFINE_LINEAR_DRIVER(gpt_rs_c_linear_nt_f32, float, float, gpt_rs_lin_f32_panel,
                           GPTRS_LIN_NR, GPTRS_LIN_KC)
GPTRS_DEFINE_LINEAR_DRIVER(gpt_rs_c_linear_nt_f32_bf16, float, uint16_t, gpt_rs_lin_f32_bf16_panel,
                           GPTRS_LIN_NR, GPTRS_LIN_KC)

#else

/* Portable fallback for bf16 weights. It computes one output per (row, column) pair, and threads
 * split the columns. */
#define GPTRS_DEFINE_LINEAR_SCALAR(NAME, XTYPE, LOADX)                                         \
    static void NAME(const XTYPE* GPTRS_RESTRICT x, const uint16_t* GPTRS_RESTRICT w,         \
                     float* GPTRS_RESTRICT y, size_t batch, size_t m, size_t n, size_t k,     \
                     size_t sxb, size_t ldx, size_t swb, size_t ldw) {                        \
        _Pragma("omp parallel for schedule(static) if (batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK)") \
        for (size_t col = 0; col < batch * n; ++col) {                                       \
            const size_t bi = col / n;                                                       \
            const size_t j = col % n;                                                        \
            const uint16_t* wr = w + bi * swb + j * ldw;                                     \
            for (size_t i = 0; i < m; ++i) {                                                 \
                const XTYPE* xr = x + bi * sxb + i * ldx;                                    \
                float acc = 0.0f;                                                            \
                _Pragma("omp simd reduction(+:acc)")                                         \
                for (size_t p = 0; p < k; ++p) {                                             \
                    acc += LOADX(xr[p]) * gpt_rs_bf16_to_f32(wr[p]);                         \
                }                                                                            \
                y[(bi * m + i) * n + j] = acc;                                               \
            }                                                                                \
        }                                                                                    \
    }

#define GPTRS_LIN_SCALAR_F32(v) (v)

GPTRS_DEFINE_LINEAR_SCALAR(gpt_rs_c_linear_nt_f32_bf16, float, GPTRS_LIN_SCALAR_F32)
GPTRS_DEFINE_LINEAR_SCALAR(gpt_rs_c_linear_nt_bf16_bf16, uint16_t, gpt_rs_bf16_to_f32)

static void gpt_rs_c_linear_nt_f32(const float* GPTRS_RESTRICT x, const float* GPTRS_RESTRICT w,
                                   float* GPTRS_RESTRICT y, size_t batch, size_t m, size_t n,
                                   size_t k, size_t sxb, size_t ldx, size_t swb, size_t ldw) {
    gpt_rs_c_matmul_f32(x, w, y, batch, m, n, k, sxb, ldx, 1, swb, 1, ldw, NULL, NULL);
}

#endif

/*
 * y[b][m][n] = sum_k x[b][m][k] * w[b][n][k] with bf16 x and w, f32 accumulation and output.
 *
 * bf16 x bf16 products are exact in f32. With AVX512-BF16, the tiles use vdpbf16ps, which does 32
 * products per instruction. vdpbf16ps treats denormal inputs as zero and flushes denormal results.
 * Without AVX512-BF16, the kernels widen the operands to f32 and multiply them with FMA.
 */
#if GPTRS_HAS_AVX512 && defined(__AVX512BF16__)

static inline __m512bh gpt_rs_lin_bh(const uint16_t* p) {
    return (__m512bh)_mm512_loadu_si512((const void*)p);
}

static inline __m512bh gpt_rs_lin_bh_tail(const uint16_t* p, size_t left) {
    return (__m512bh)_mm512_maskz_loadu_epi16((__mmask32)gpt_rs_tail_mask(left), p);
}

static inline __m512 gpt_rs_lin_dpbf16(__m512 acc, __m512bh x, __m512bh w) {
    return _mm512_dpbf16_ps(acc, x, w);
}

GPTRS_DEFINE_LINEAR_FAMILY(gpt_rs_lin_dp, uint16_t, uint16_t, __m512bh, 32, gpt_rs_lin_bh,
                           gpt_rs_lin_bh, gpt_rs_lin_dpbf16)

/*
 * Compute-bound path for many rows. For each K block, the kernel packs each 32-column weight panel
 * into VNNI pairs. One 32-bit lane holds the pair `wp[p][j] = {w[j][2p], w[j][2p + 1]}`. An 8 x 32
 * outer-product micro-kernel then broadcasts pairs of x and uses vdpbf16ps to accumulate 16
 * outputs per register. So this path needs no horizontal reductions. Products are exact. Only the
 * f32 summation order differs from the dot-product path.
 */

/* NR: two zmm of outputs per row. MR: 8 x 2 independent accumulators cover the vdpbf16ps latency
 * on two FMA pipes. KC: a packed K block of one panel is 256 pairs x 32 lanes, or 32 KiB. It stays
 * in L1D. */
enum {
    GPTRS_LIN_PACK_NR = 32,
    GPTRS_LIN_PACK_MR = 8,
    GPTRS_LIN_PACK_KC = 512
};

/* Packs `kc` columns of 32 weight rows into `wp[kc / 2][32]`. `kc` is even, and the row stride
 * is `k`. */
static inline void gpt_rs_lin_pack_panel(const uint16_t* GPTRS_RESTRICT w, size_t k, size_t kc,
                                         uint32_t* GPTRS_RESTRICT wp) {
    const size_t pairs = kc / 2;
    for (size_t half = 0; half < GPTRS_LIN_PACK_NR / 16; ++half) {
        for (size_t p0 = 0; p0 < pairs; p0 += 16) {
            const size_t left = pairs - p0;
            const __mmask16 mask = (__mmask16)gpt_rs_tail_mask(left);
            __m512i r[16];
            for (int j = 0; j < 16; ++j) {
                const uint32_t* row = (const uint32_t*)(w + (half * 16 + (size_t)j) * k) + p0;
                r[j] = _mm512_maskz_loadu_epi32(mask, row);
            }
            gpt_rs_transpose16(r);
            const size_t rows = GPTRS_MIN((size_t)16, left);
            for (size_t q = 0; q < rows; ++q) {
                _mm512_storeu_si512((void*)(wp + (p0 + q) * GPTRS_LIN_PACK_NR + half * 16), r[q]);
            }
        }
    }
}

#define GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(NAME, MR)                                           \
    static inline void NAME(const uint16_t* GPTRS_RESTRICT x, size_t ldx,                     \
                            const uint32_t* GPTRS_RESTRICT wp, size_t pairs,                  \
                            float* GPTRS_RESTRICT y, size_t ldy, int accumulate) {            \
        __m512 acc[MR][2];                                                                    \
        for (int i = 0; i < MR; ++i) {                                                        \
            acc[i][0] = _mm512_setzero_ps();                                                  \
            acc[i][1] = _mm512_setzero_ps();                                                  \
        }                                                                                     \
        for (size_t p = 0; p < pairs; ++p) {                                                  \
            const __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(wp + p * 32));     \
            const __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(wp + p * 32 + 16)); \
            for (int i = 0; i < MR; ++i) {                                                    \
                int32_t pair;                                                                 \
                memcpy(&pair, x + (size_t)i * ldx + 2 * p, sizeof(pair));                     \
                const __m512bh a = (__m512bh)_mm512_set1_epi32(pair);                         \
                acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a, b0);                               \
                acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a, b1);                               \
            }                                                                                 \
        }                                                                                     \
        for (int i = 0; i < MR; ++i) {                                                        \
            float* dst = y + (size_t)i * ldy;                                                 \
            if (accumulate) {                                                                 \
                acc[i][0] = _mm512_add_ps(acc[i][0], _mm512_loadu_ps(dst));                   \
                acc[i][1] = _mm512_add_ps(acc[i][1], _mm512_loadu_ps(dst + 16));              \
            }                                                                                 \
            _mm512_storeu_ps(dst, acc[i][0]);                                                 \
            _mm512_storeu_ps(dst + 16, acc[i][1]);                                            \
        }                                                                                     \
    }

GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr1, 1)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr2, 2)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr3, 3)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr4, 4)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr5, 5)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr6, 6)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr7, 7)
GPTRS_DEFINE_LINEAR_PACKED_UKERNEL(gpt_rs_lin_packed_ukr8, 8)

/* Computes one weight panel over one K block of even length. A partial panel (nr < 32) uses the
 * dot-product kernels. */
static inline void gpt_rs_lin_packed_panel(const uint16_t* GPTRS_RESTRICT x, size_t ldx,
                                           const uint16_t* GPTRS_RESTRICT w, size_t ldw,
                                           float* GPTRS_RESTRICT y, size_t ldy, size_t m,
                                           size_t nr, size_t kc, int accumulate) {
    if (nr < GPTRS_LIN_PACK_NR) {
        for (size_t n0 = 0; n0 < nr; n0 += GPTRS_LIN_NR) {
            gpt_rs_lin_dp_panel(x, ldx, w + n0 * ldw, ldw, y + n0, ldy, m,
                                GPTRS_MIN((size_t)GPTRS_LIN_NR, nr - n0), kc, accumulate);
        }
        return;
    }
    uint32_t wp[(GPTRS_LIN_PACK_KC / 2) * GPTRS_LIN_PACK_NR] __attribute__((aligned(64)));
    const size_t pairs = kc / 2;
    gpt_rs_lin_pack_panel(w, ldw, kc, wp);
    size_t m0 = 0;
    for (; m0 + GPTRS_LIN_PACK_MR <= m; m0 += GPTRS_LIN_PACK_MR) {
        gpt_rs_lin_packed_ukr8(x + m0 * ldx, ldx, wp, pairs, y + m0 * ldy, ldy, accumulate);
    }
    const uint16_t* xt = x + m0 * ldx;
    float* yt = y + m0 * ldy;
    switch (m - m0) {
    case 7: gpt_rs_lin_packed_ukr7(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 6: gpt_rs_lin_packed_ukr6(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 5: gpt_rs_lin_packed_ukr5(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 4: gpt_rs_lin_packed_ukr4(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 3: gpt_rs_lin_packed_ukr3(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 2: gpt_rs_lin_packed_ukr2(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    case 1: gpt_rs_lin_packed_ukr1(xt, ldx, wp, pairs, yt, ldy, accumulate); break;
    default: break;
    }
}

GPTRS_DEFINE_LINEAR_DRIVER(gpt_rs_lin_dp_rows, uint16_t, uint16_t, gpt_rs_lin_dp_panel,
                           GPTRS_LIN_NR, GPTRS_LIN_KC)
GPTRS_DEFINE_LINEAR_DRIVER(gpt_rs_lin_packed_rows, uint16_t, uint16_t, gpt_rs_lin_packed_panel,
                           GPTRS_LIN_PACK_NR, GPTRS_LIN_PACK_KC)

static void gpt_rs_c_linear_nt_bf16_bf16(const uint16_t* GPTRS_RESTRICT x,
                                         const uint16_t* GPTRS_RESTRICT w,
                                         float* GPTRS_RESTRICT y,
                                         size_t batch,
                                         size_t m,
                                         size_t n,
                                         size_t k,
                                         size_t sxb,
                                         size_t ldx,
                                         size_t swb,
                                         size_t ldw) {
    /* A vdpbf16ps does twice the products of an f32 FMA. So the per-output reductions weigh twice
     * as much as in f32, and packing pays off from half the rows. */
    if (m >= GPTRS_LIN_PACK_MIN_ROWS / 2 && k % 2 == 0) {
        gpt_rs_lin_packed_rows(x, w, y, batch, m, n, k, sxb, ldx, swb, ldw);
    } else {
        gpt_rs_lin_dp_rows(x, w, y, batch, m, n, k, sxb, ldx, swb, ldw);
    }
}

#elif GPTRS_HAS_AVX512

GPTRS_DEFINE_LINEAR_FAMILY(gpt_rs_lin_bb, uint16_t, uint16_t, __m512, 16, gpt_rs_lin_bf16,
                           gpt_rs_lin_bf16, gpt_rs_lin_fma)
GPTRS_DEFINE_LINEAR_DRIVER(gpt_rs_c_linear_nt_bf16_bf16, uint16_t, uint16_t, gpt_rs_lin_bb_panel,
                           GPTRS_LIN_NR, GPTRS_LIN_KC)

#endif
