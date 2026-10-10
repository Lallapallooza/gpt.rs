/*
 * c[b][i][j] = bias[j] + sum_p a[b * sab + i * sam + p * sak] * b[b * sbb + p * sbk + j * sbn]
 *
 * gpt_rs_c_matmul_SUFFIX multiplies operands of type T. c is contiguous f32 [batch, m, n]. Packing
 * absorbs the operand layouts and widens the operands to f32, which is exact, so the products and
 * sums are f32. When `b_panels` is not NULL, it holds b packed as
 * [n / GPTRS_PANEL_N, k, GPTRS_PANEL_N] f32 panels, batch is 1, and b, sbb, sbk and sbn are unused.
 * `bias` may be NULL.
 */
#if GPTRS_HAS_AVX512
_Static_assert(GPTRS_NC % GPTRS_PANEL_N == 0, "column blocks start on a packed panel");

/*
 * A single row of a (m == 1) reuses no element of b, so the kernel reads unit-stride rows of b in
 * place. The threads split the columns, in blocks whose outputs fit in L1.
 *
 * Otherwise each thread claims GPTRS_TASKS_PER_THREAD output tiles and packs the panels of a and b
 * that its tile needs. Each tile is a whole number of micro-panels and about as tall as it is
 * wide, which minimizes the panels that more than one tile packs.
 */
#define GPTRS_DEFINE_MATMUL(SUFFIX, T)                                                        \
    static inline void gpt_rs_c_matmul_##SUFFIX(                                              \
        const T* GPTRS_RESTRICT a,                                                            \
        const T* GPTRS_RESTRICT b,                                                            \
        float* GPTRS_RESTRICT c,                                                              \
        size_t batch,                                                                         \
        size_t m,                                                                             \
        size_t n,                                                                             \
        size_t k,                                                                             \
        size_t sab,                                                                           \
        size_t sam,                                                                           \
        size_t sak,                                                                           \
        size_t sbb,                                                                           \
        size_t sbk,                                                                           \
        size_t sbn,                                                                           \
        const float* GPTRS_RESTRICT b_panels,                                                 \
        const float* bias                                                                     \
    ) {                                                                                       \
        if (batch == 0 || m == 0 || n == 0) {                                                 \
            return;                                                                           \
        }                                                                                     \
        if (k == 0) {                                                                         \
            for (size_t row = 0; row < batch * m; ++row) {                                    \
                for (size_t j = 0; j < n; ++j) {                                              \
                    c[row * n + j] = bias ? bias[j] : 0.0f;                                   \
                }                                                                             \
            }                                                                                 \
            return;                                                                           \
        }                                                                                     \
        if (m == 1 && sbn == 1 && !b_panels) {                                                \
            const int parallel = batch * n * k >= GPTRS_PARALLEL_MIN_WORK;                    \
            const size_t threads = parallel ? (size_t)gpt_rs_c_team_size() : 1;               \
            const size_t share = ((n + threads - 1) / threads + 15) / 16 * 16;                \
            const size_t nb = GPTRS_MIN(GPTRS_GEMV_NC, share);                                \
            const size_t chunks = (n + nb - 1) / nb;                                          \
            _Pragma("omp parallel for schedule(static) if (parallel)")                        \
            for (size_t t = 0; t < batch * chunks; ++t) {                                     \
                const size_t bi = t / chunks;                                                 \
                const size_t j0 = t % chunks * nb;                                            \
                gpt_rs_gemv_##SUFFIX##_rows(a + bi * sab, sak, b + bi * sbb + j0, sbk,        \
                                            c + bi * n + j0, GPTRS_MIN(nb, n - j0), k,        \
                                            bias ? bias + j0 : NULL);                         \
            }                                                                                 \
            return;                                                                           \
        }                                                                                     \
        const size_t kc_block = gpt_rs_choose_kc(k);                                          \
        const int parallel = batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK;                    \
        const size_t team = parallel ? (size_t)gpt_rs_c_team_size() : 1;                      \
        const size_t per_batch = (team * GPTRS_TASKS_PER_THREAD + batch - 1) / batch;         \
        const size_t side = (size_t)ceil(sqrt((double)m * (double)n / (double)per_batch));    \
        const size_t tile_m =                                                                 \
            GPTRS_MIN(m, (GPTRS_MIN(side, m) + GPTRS_MR - 1) / GPTRS_MR * GPTRS_MR);          \
        const size_t tile_n = GPTRS_MIN(                                                      \
            n, (GPTRS_MIN(side, n) + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N * GPTRS_PANEL_N);     \
        const size_t m_tiles = (m + tile_m - 1) / tile_m;                                     \
        const size_t n_tiles = (n + tile_n - 1) / tile_n;                                     \
        const size_t tiles = batch * m_tiles * n_tiles;                                       \
        _Pragma("omp parallel for schedule(dynamic, 1) if (parallel && tiles > 1)")           \
        for (size_t t = 0; t < tiles; ++t) {                                                  \
            float* apack = NULL;                                                              \
            float* bpack = NULL;                                                              \
            if (!gpt_rs_get_scratch_with_capacity((size_t)GPTRS_MC * kc_block,                \
                                                  (size_t)GPTRS_NC * kc_block, &apack,        \
                                                  &bpack)) {                                  \
                abort();                                                                      \
            }                                                                                 \
            const size_t bi = t / (m_tiles * n_tiles);                                        \
            const size_t i0 = t / n_tiles % m_tiles * tile_m;                                 \
            const size_t j0 = t % n_tiles * tile_n;                                           \
            const size_t i1 = GPTRS_MIN(m, i0 + tile_m);                                      \
            const size_t j1 = GPTRS_MIN(n, j0 + tile_n);                                      \
            const T* ab = a + bi * sab;                                                       \
            const T* bb = b_panels ? NULL : b + bi * sbb;                                     \
            float* cb = c + bi * m * n;                                                       \
            for (size_t jc = j0; jc < j1; jc += GPTRS_NC) {                                   \
                const size_t nc = GPTRS_MIN(GPTRS_NC, j1 - jc);                               \
                for (size_t pc = 0; pc < k; pc += kc_block) {                                 \
                    const size_t kc = GPTRS_MIN(kc_block, k - pc);                            \
                    const float* bp = bpack;                                                  \
                    size_t panel_stride = kc * GPTRS_PANEL_N;                                 \
                    if (b_panels) {                                                           \
                        bp = b_panels + jc * k + pc * GPTRS_PANEL_N;                          \
                        panel_stride = k * GPTRS_PANEL_N;                                     \
                    } else {                                                                  \
                        gpt_rs_pack_b_##SUFFIX(bb + pc * sbk + jc * sbn, sbk, sbn, kc, nc,    \
                                               bpack);                                        \
                    }                                                                         \
                    const int init_mode = (pc == 0)                                           \
                        ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)                          \
                        : GPTRS_INIT_ACCUM;                                                   \
                    for (size_t ic = i0; ic < i1; ic += GPTRS_MC) {                           \
                        const size_t mc = GPTRS_MIN(GPTRS_MC, i1 - ic);                       \
                        gpt_rs_pack_a_##SUFFIX(ab + ic * sam + pc * sak, sam, sak, kc, mc,    \
                                               apack);                                        \
                        gpt_rs_compute_block(apack, bp, panel_stride, cb, n, ic, mc, jc, nc,  \
                                             kc, bias, init_mode);                            \
                    }                                                                         \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }

GPTRS_DEFINE_MATMUL(f32, float)
GPTRS_DEFINE_MATMUL(bf16, uint16_t)

/*
 * NHWC convolution padded with zeros, plus `bias` when it is not NULL. The weight is
 * [k_h * k_w * c_in, c_out], or packed as [c_out / GPTRS_PANEL_N, k_h * k_w * c_in, GPTRS_PANEL_N]
 * when `w_packed` is set.
 */
static inline void gpt_rs_c_conv2d_nhwc_f32(
    const float* GPTRS_RESTRICT input,
    const float* GPTRS_RESTRICT weight,
    const float* GPTRS_RESTRICT bias,
    float* GPTRS_RESTRICT out,
    size_t n,
    size_t in_h,
    size_t in_w,
    size_t c_in,
    size_t out_h,
    size_t out_w,
    size_t c_out,
    size_t k_h,
    size_t k_w,
    size_t stride_h,
    size_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t pad_top,
    size_t pad_left,
    int w_packed
) {
    const size_t m = out_h * out_w;
    const size_t k = k_h * k_w * c_in;
    if (n == 0 || m == 0 || k == 0 || c_out == 0) {
        return;
    }

    if (k_h == 1 && k_w == 1
        && stride_h == 1 && stride_w == 1
        && dilation_h == 1 && dilation_w == 1
        && pad_top == 0 && pad_left == 0
        && out_h == in_h && out_w == in_w) {
        gpt_rs_c_matmul_f32(input, weight, out, 1, n * m, c_out, c_in, 0, c_in, 1, 0, c_out, 1,
                            w_packed ? weight : NULL, bias);
        return;
    }

    /* The kernel reads the whole weight as one K block, which it packs unless `w_packed`. */
    const float* bp = w_packed ? weight : NULL;
    const size_t panels = (c_out + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N;
    if (!bp) {
        float* apack = NULL;
        float* bpack = NULL;
        if (!gpt_rs_get_scratch_with_capacity(1, panels * GPTRS_PANEL_N * k, &apack, &bpack)) {
            abort();
        }
        gpt_rs_pack_b_f32(weight, c_out, 1, k, c_out, bpack);
        bp = bpack;
    }
    /*
     * A task is a block of up to GPTRS_MC output pixels of one image times a run of whole panels
     * of output channels. The panels split only as far as the team needs tasks. Each output sums
     * over all of K in one task, so the split leaves every sum as the serial loop computes it.
     */
    const size_t m_blocks = (m + GPTRS_MC - 1) / GPTRS_MC;
    const int parallel = n * m * c_out * k >= GPTRS_PARALLEL_MIN_WORK;
    const size_t team = parallel ? (size_t)gpt_rs_c_team_size() : 1;
    const size_t blocks = n * m_blocks;
    const size_t parts =
        GPTRS_MIN(panels, (team * GPTRS_TASKS_PER_THREAD + blocks - 1) / blocks);
    const int init_mode = bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO;
    _Pragma("omp parallel for schedule(dynamic, 1) if (team > 1 && blocks * parts > 1)")
    for (size_t task = 0; task < blocks * parts; ++task) {
        /* Each thread packs its rows of patches into its own scratch. This call asks for a
         * one-element second buffer, so it never reallocates the calling thread's second buffer,
         * which holds the packed weight that the other threads read. */
        float* apack = NULL;
        float* unused = NULL;
        if (!gpt_rs_get_scratch_with_capacity((size_t)GPTRS_MC * k, 1, &apack, &unused)) {
            abort();
        }
        const size_t part = task % parts;
        const size_t block = task / parts;
        const size_t batch = block / m_blocks;
        const size_t ic = block % m_blocks * GPTRS_MC;
        const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
        const size_t jc = panels * part / parts * GPTRS_PANEL_N;
        const size_t nc = GPTRS_MIN(c_out, panels * (part + 1) / parts * GPTRS_PANEL_N) - jc;
        gpt_rs_pack_a_conv(input, batch, in_h, in_w, c_in, out_w, k_h, k_w, stride_h, stride_w,
                           dilation_h, dilation_w, pad_top, pad_left, 0, k, ic, mc, apack);
        gpt_rs_compute_block(apack, bp + jc * k, k * GPTRS_PANEL_N, out + batch * m * c_out,
                             c_out, ic, mc, jc, nc, k, bias, init_mode);
    }
}
#else
#define GPTRS_DEFINE_MATMUL(SUFFIX, T, WIDEN)                                                 \
    static inline void gpt_rs_c_matmul_##SUFFIX(                                              \
        const T* GPTRS_RESTRICT a,                                                            \
        const T* GPTRS_RESTRICT b,                                                            \
        float* GPTRS_RESTRICT c,                                                              \
        size_t batch,                                                                         \
        size_t m,                                                                             \
        size_t n,                                                                             \
        size_t k,                                                                             \
        size_t sab,                                                                           \
        size_t sam,                                                                           \
        size_t sak,                                                                           \
        size_t sbb,                                                                           \
        size_t sbk,                                                                           \
        size_t sbn,                                                                           \
        const float* GPTRS_RESTRICT b_panels,                                                 \
        const float* bias                                                                     \
    ) {                                                                                       \
        const int parallel = batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK;                    \
        _Pragma("omp parallel for schedule(static) if (parallel)")                            \
        for (size_t col = 0; col < batch * n; ++col) {                                        \
            const size_t bi = col / n;                                                        \
            const size_t j = col % n;                                                         \
            const float* panel = b_panels                                                     \
                ? b_panels + j / GPTRS_PANEL_N * k * GPTRS_PANEL_N + j % GPTRS_PANEL_N        \
                : NULL;                                                                       \
            const T* b_col = b_panels ? NULL : b + bi * sbb + j * sbn;                        \
            for (size_t i = 0; i < m; ++i) {                                                  \
                const T* a_row = a + bi * sab + i * sam;                                      \
                float acc = 0.0f;                                                             \
                _Pragma("omp simd reduction(+:acc)")                                          \
                for (size_t p = 0; p < k; ++p) {                                              \
                    const float bv =                                                          \
                        panel ? panel[p * GPTRS_PANEL_N] : WIDEN(b_col[p * sbk]);             \
                    acc += WIDEN(a_row[p * sak]) * bv;                                        \
                }                                                                             \
                c[(bi * m + i) * n + j] = bias ? bias[j] + acc : acc;                         \
            }                                                                                 \
        }                                                                                     \
    }

GPTRS_DEFINE_MATMUL(f32, float, gpt_rs_f32_to_f32)
GPTRS_DEFINE_MATMUL(bf16, uint16_t, gpt_rs_bf16_to_f32)

static inline void gpt_rs_c_conv2d_nhwc_f32(
    const float* GPTRS_RESTRICT input,
    const float* GPTRS_RESTRICT weight,
    const float* GPTRS_RESTRICT bias,
    float* GPTRS_RESTRICT out,
    size_t n,
    size_t in_h,
    size_t in_w,
    size_t c_in,
    size_t out_h,
    size_t out_w,
    size_t c_out,
    size_t k_h,
    size_t k_w,
    size_t stride_h,
    size_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t pad_top,
    size_t pad_left,
    int w_packed
) {
    if (n == 0 || out_h == 0 || out_w == 0 || c_out == 0) {
        return;
    }
    const size_t k = k_h * k_w * c_in;

    const int64_t in_h_i = (int64_t)in_h;
    const int64_t in_w_i = (int64_t)in_w;
    const int64_t pad_top_i = (int64_t)pad_top;
    const int64_t pad_left_i = (int64_t)pad_left;
    const int64_t stride_h_i = (int64_t)stride_h;
    const int64_t stride_w_i = (int64_t)stride_w;
    const int64_t dilation_h_i = (int64_t)dilation_h;
    const int64_t dilation_w_i = (int64_t)dilation_w;

    /* Rows of the output split across the team. Each output sums in the same order whatever the
     * split. */
    const int parallel = n * out_h * out_w * c_out * k >= GPTRS_PARALLEL_MIN_WORK;
    _Pragma("omp parallel for schedule(static) if (parallel)")
    for (size_t row = 0; row < n * out_h; ++row) {
        const size_t batch = row / out_h;
        const size_t oh = row % out_h;
        {
            const int64_t base_h = (int64_t)oh * stride_h_i - pad_top_i;
            for (size_t ow = 0; ow < out_w; ++ow) {
                const int64_t base_w = (int64_t)ow * stride_w_i - pad_left_i;
                float* out_ptr =
                    out + (((batch * out_h + oh) * out_w + ow) * c_out);
                for (size_t oc = 0; oc < c_out; ++oc) {
                    /* Element (p, oc) of the weight is w_oc[p * sp]. */
                    const float* w_oc = w_packed
                        ? weight + oc / GPTRS_PANEL_N * k * GPTRS_PANEL_N + oc % GPTRS_PANEL_N
                        : weight + oc;
                    const size_t sp = w_packed ? GPTRS_PANEL_N : c_out;
                    float acc = bias ? bias[oc] : 0.0f;
                    for (size_t kh = 0; kh < k_h; ++kh) {
                        const int64_t ih = base_h + (int64_t)kh * dilation_h_i;
                        if (ih < 0 || ih >= in_h_i) {
                            continue;
                        }
                        for (size_t kw = 0; kw < k_w; ++kw) {
                            const int64_t iw = base_w + (int64_t)kw * dilation_w_i;
                            if (iw < 0 || iw >= in_w_i) {
                                continue;
                            }
                            const size_t input_base =
                                (((batch * in_h + (size_t)ih) * in_w + (size_t)iw) * c_in);
                            const size_t p0 = (kh * k_w + kw) * c_in;
                            for (size_t ic = 0; ic < c_in; ++ic) {
                                acc += input[input_base + ic] * w_oc[(p0 + ic) * sp];
                            }
                        }
                    }
                    out_ptr[oc] = acc;
                }
            }
        }
    }
}
#endif

/*
 * Grouped NHWC convolution padded with zeros, plus `bias` when it is not NULL. The weight is
 * [k_h, k_w, c_in / groups, c_out]: output channel c belongs to group g = c / (c_out / groups),
 * which reads input channels g * (c_in / groups) onward. A depthwise convolution is the case
 * groups == c_in.
 *
 * Each output pixel accumulates in place, tap by tap and input channel by input channel. With one
 * output channel per group, the innermost loop runs across the groups, so its vectors span
 * channels as NHWC stores them. Otherwise it runs across the output channels of a group. Rows of
 * the output split across the team, and each output sums in the same order whatever the split.
 */
static inline void gpt_rs_c_conv2d_grouped_nhwc_f32(
    const float* GPTRS_RESTRICT input,
    const float* GPTRS_RESTRICT weight,
    const float* GPTRS_RESTRICT bias,
    float* GPTRS_RESTRICT out,
    size_t n,
    size_t in_h,
    size_t in_w,
    size_t c_in,
    size_t out_h,
    size_t out_w,
    size_t c_out,
    size_t groups,
    size_t k_h,
    size_t k_w,
    size_t stride_h,
    size_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t pad_top,
    size_t pad_left
) {
    if (n == 0 || out_h == 0 || out_w == 0 || c_out == 0) {
        return;
    }
    const size_t cg_in = c_in / groups;
    const size_t cg_out = c_out / groups;
    const int parallel =
        n * out_h * out_w * c_out * k_h * k_w * cg_in >= GPTRS_PARALLEL_MIN_WORK;
    _Pragma("omp parallel for schedule(static) if (parallel)")
    for (size_t row = 0; row < n * out_h; ++row) {
        const size_t batch = row / out_h;
        const int64_t base_h = (int64_t)(row % out_h * stride_h) - (int64_t)pad_top;
        for (size_t ow = 0; ow < out_w; ++ow) {
            const int64_t base_w = (int64_t)(ow * stride_w) - (int64_t)pad_left;
            float* GPTRS_RESTRICT o = out + (row * out_w + ow) * c_out;
            _Pragma("omp simd")
            for (size_t c = 0; c < c_out; ++c) {
                o[c] = bias ? bias[c] : 0.0f;
            }
            for (size_t kh = 0; kh < k_h; ++kh) {
                const int64_t ih = base_h + (int64_t)(kh * dilation_h);
                if (ih < 0 || ih >= (int64_t)in_h) {
                    continue;
                }
                for (size_t kw = 0; kw < k_w; ++kw) {
                    const int64_t iw = base_w + (int64_t)(kw * dilation_w);
                    if (iw < 0 || iw >= (int64_t)in_w) {
                        continue;
                    }
                    const float* GPTRS_RESTRICT x =
                        input + ((batch * in_h + (size_t)ih) * in_w + (size_t)iw) * c_in;
                    const float* GPTRS_RESTRICT w = weight + (kh * k_w + kw) * cg_in * c_out;
                    for (size_t i = 0; i < cg_in; ++i) {
                        const float* GPTRS_RESTRICT wi = w + i * c_out;
                        if (cg_out == 1) {
                            _Pragma("omp simd")
                            for (size_t g = 0; g < groups; ++g) {
                                o[g] += x[g * cg_in + i] * wi[g];
                            }
                            continue;
                        }
                        for (size_t g = 0; g < groups; ++g) {
                            const float xv = x[g * cg_in + i];
                            float* GPTRS_RESTRICT og = o + g * cg_out;
                            const float* GPTRS_RESTRICT wg = wi + g * cg_out;
                            _Pragma("omp simd")
                            for (size_t j = 0; j < cg_out; ++j) {
                                og[j] += xv * wg[j];
                            }
                        }
                    }
                }
            }
        }
    }
}
