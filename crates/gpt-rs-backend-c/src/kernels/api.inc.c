/*
 * c[b][i][j] = bias[j] + sum_p a[b * sab + i * sam + p * sak] * b[b * sbb + p * sbk + j * sbn]
 *
 * c is contiguous [batch, m, n]. Packing absorbs the operand layouts. `cache` is NULL or holds a
 * prepacked copy of b that every batch shares. `bias` may be NULL.
 */
#if GPTRS_HAS_AVX512
_Static_assert(GPTRS_NC % GPTRS_PANEL_N == 0, "column blocks start on a packed panel");

static inline void gpt_rs_c_matmul_f32(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t batch,
    size_t m,
    size_t n,
    size_t k,
    size_t sab,
    size_t sam,
    size_t sak,
    size_t sbb,
    size_t sbk,
    size_t sbn,
    gpt_rs_bpack_cache* cache,
    const float* bias
) {
    if (batch == 0 || m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        for (size_t row = 0; row < batch * m; ++row) {
            for (size_t j = 0; j < n; ++j) {
                c[row * n + j] = bias ? bias[j] : 0.0f;
            }
        }
        return;
    }

    /* A single row of a reuses no element of b, so the kernel reads unit-stride rows of b in
     * place. */
    if (m == 1 && sbn == 1) {
        const size_t chunks = (n + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N;
        #pragma omp parallel for schedule(static) if (batch * n * k >= GPTRS_PARALLEL_MIN_WORK)
        for (size_t t = 0; t < batch * chunks; ++t) {
            const size_t bi = t / chunks;
            const size_t j0 = t % chunks * GPTRS_PANEL_N;
            gpt_rs_gemv_f32_cols(a + bi * sab, sak, b + bi * sbb + j0, sbk, c + bi * n + j0,
                                 GPTRS_MIN((size_t)GPTRS_PANEL_N, n - j0), k,
                                 bias ? bias + j0 : NULL);
        }
        return;
    }

    const size_t kc_block = gpt_rs_choose_kc(k);
    const float* cached =
        cache && gpt_rs_bpack_cache_prepare(cache, b, n, k, sbk, sbn) ? cache->bpack : NULL;

    /* Each thread claims GPTRS_TASKS_PER_THREAD output tiles and packs the panels of a and b that
     * its tile needs. Each tile is a whole number of micro-panels and about as tall as it is wide,
     * which minimises the panels that more than one tile packs. */
    const int parallel = batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK;
    const size_t team = parallel ? (size_t)gpt_rs_c_team_size() : 1;
    const size_t per_batch = (team * GPTRS_TASKS_PER_THREAD + batch - 1) / batch;
    const size_t side = (size_t)ceil(sqrt((double)m * (double)n / (double)per_batch));
    const size_t tile_m = GPTRS_MIN(m, (GPTRS_MIN(side, m) + GPTRS_MR - 1) / GPTRS_MR * GPTRS_MR);
    const size_t tile_n =
        GPTRS_MIN(n, (GPTRS_MIN(side, n) + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N * GPTRS_PANEL_N);
    const size_t m_tiles = (m + tile_m - 1) / tile_m;
    const size_t n_tiles = (n + tile_n - 1) / tile_n;
    const size_t tiles = batch * m_tiles * n_tiles;
    #pragma omp parallel for schedule(dynamic, 1) if (parallel && tiles > 1)
    for (size_t t = 0; t < tiles; ++t) {
        float* apack = NULL;
        float* bpack = NULL;
        if (!gpt_rs_get_scratch_with_capacity((size_t)GPTRS_MC * kc_block,
                                              (size_t)GPTRS_NC * kc_block, &apack, &bpack)) {
            abort();
        }
        const size_t bi = t / (m_tiles * n_tiles);
        const size_t i0 = t / n_tiles % m_tiles * tile_m;
        const size_t j0 = t % n_tiles * tile_n;
        const size_t i1 = GPTRS_MIN(m, i0 + tile_m);
        const size_t j1 = GPTRS_MIN(n, j0 + tile_n);
        const float* ab = a + bi * sab;
        const float* bb = b + bi * sbb;
        float* cb = c + bi * m * n;
        for (size_t jc = j0; jc < j1; jc += GPTRS_NC) {
            const size_t nc = GPTRS_MIN(GPTRS_NC, j1 - jc);
            for (size_t pc = 0; pc < k; pc += kc_block) {
                const size_t kc = GPTRS_MIN(kc_block, k - pc);
                const float* bp = bpack;
                size_t panel_stride = kc * GPTRS_PANEL_N;
                if (cached) {
                    bp = cached + jc * k + pc * GPTRS_PANEL_N;
                    panel_stride = k * GPTRS_PANEL_N;
                } else {
                    gpt_rs_pack_b(bb + pc * sbk + jc * sbn, sbk, sbn, kc, nc, bpack);
                }
                const int init_mode = (pc == 0)
                    ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)
                    : GPTRS_INIT_ACCUM;
                for (size_t ic = i0; ic < i1; ic += GPTRS_MC) {
                    const size_t mc = GPTRS_MIN(GPTRS_MC, i1 - ic);
                    gpt_rs_pack_a(ab + ic * sam + pc * sak, sam, sak, kc, mc, apack);
                    gpt_rs_compute_block(apack, bp, panel_stride, cb, n, ic, mc, jc, nc, kc, bias,
                                         init_mode);
                }
            }
        }
    }
}

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
    gpt_rs_bpack_cache* cache
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
                            cache, bias);
        return;
    }

    /* The whole weight [k, c_out] is packed as a single K block. Without a cache, this happens
     * once per call. */
    const float* bp =
        cache && gpt_rs_bpack_cache_prepare(cache, weight, c_out, k, c_out, 1) ? cache->bpack : NULL;
    const size_t panels = (c_out + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N;
    float* apack = NULL;
    float* bpack = NULL;
    if (!gpt_rs_get_scratch_with_capacity((size_t)GPTRS_MC * k,
                                          bp ? 1 : panels * GPTRS_PANEL_N * k, &apack, &bpack)) {
        abort();
    }
    if (!bp) {
        gpt_rs_pack_b(weight, c_out, 1, k, c_out, bpack);
        bp = bpack;
    }
    const int init_mode = bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO;
    for (size_t batch = 0; batch < n; ++batch) {
        float* out_b = out + batch * m * c_out;
        for (size_t ic = 0; ic < m; ic += GPTRS_MC) {
            const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
            gpt_rs_pack_a_conv(input, batch, in_h, in_w, c_in, out_w, k_h, k_w, stride_h,
                               stride_w, dilation_h, dilation_w, pad_top, pad_left, 0, k, ic, mc,
                               apack);
            gpt_rs_compute_block(apack, bp, k * GPTRS_PANEL_N, out_b, c_out, ic, mc, 0, c_out, k,
                                 bias, init_mode);
        }
    }
}
#else
static inline void gpt_rs_c_matmul_f32(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t batch,
    size_t m,
    size_t n,
    size_t k,
    size_t sab,
    size_t sam,
    size_t sak,
    size_t sbb,
    size_t sbk,
    size_t sbn,
    gpt_rs_bpack_cache* cache,
    const float* bias
) {
    (void)cache;
    #pragma omp parallel for schedule(static) if (batch * m * n * k >= GPTRS_PARALLEL_MIN_WORK)
    for (size_t col = 0; col < batch * n; ++col) {
        const size_t bi = col / n;
        const size_t j = col % n;
        const float* b_col = b + bi * sbb + j * sbn;
        for (size_t i = 0; i < m; ++i) {
            const float* a_row = a + bi * sab + i * sam;
            float acc = 0.0f;
            #pragma omp simd reduction(+:acc)
            for (size_t p = 0; p < k; ++p) {
                acc += a_row[p * sak] * b_col[p * sbk];
            }
            c[(bi * m + i) * n + j] = bias ? bias[j] + acc : acc;
        }
    }
}

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
    gpt_rs_bpack_cache* cache
) {
    (void)cache;
    if (n == 0 || out_h == 0 || out_w == 0 || c_out == 0) {
        return;
    }

    const int64_t in_h_i = (int64_t)in_h;
    const int64_t in_w_i = (int64_t)in_w;
    const int64_t pad_top_i = (int64_t)pad_top;
    const int64_t pad_left_i = (int64_t)pad_left;
    const int64_t stride_h_i = (int64_t)stride_h;
    const int64_t stride_w_i = (int64_t)stride_w;
    const int64_t dilation_h_i = (int64_t)dilation_h;
    const int64_t dilation_w_i = (int64_t)dilation_w;

    for (size_t batch = 0; batch < n; ++batch) {
        for (size_t oh = 0; oh < out_h; ++oh) {
            const int64_t base_h = (int64_t)oh * stride_h_i - pad_top_i;
            for (size_t ow = 0; ow < out_w; ++ow) {
                const int64_t base_w = (int64_t)ow * stride_w_i - pad_left_i;
                float* out_ptr =
                    out + (((batch * out_h + oh) * out_w + ow) * c_out);
                for (size_t oc = 0; oc < c_out; ++oc) {
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
                            const size_t weight_base =
                                (((kh * k_w + kw) * c_in) * c_out);
                            for (size_t ic = 0; ic < c_in; ++ic) {
                                acc += input[input_base + ic]
                                    * weight[weight_base + ic * c_out + oc];
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
