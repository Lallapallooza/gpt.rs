static inline void gpt_rs_c_matmul_f32_cached_b_impl(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t m,
    size_t n,
    size_t k,
    gpt_rs_bpack_cache* cache,
    const float* bias
) {
    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    float* apack = NULL;
    float* bpack = NULL;
    const size_t kc_block = gpt_rs_choose_kc(k);
    const size_t apack_elems = (size_t)GPTRS_MC * kc_block;
    const size_t bpack_elems = (size_t)GPTRS_NC * kc_block;
    if (!gpt_rs_get_scratch_with_capacity(apack_elems, bpack_elems, &apack, &bpack)) {
        abort();
    }

    const float* bpack_full = NULL;
    const float* bpack32 = NULL;
    const float* bpack64 = NULL;
    if (cache && gpt_rs_bpack_cache_prepare(cache, b, n, k)) {
        bpack_full = cache->bpack;
        bpack32 = cache->bpack32;
        bpack64 = cache->bpack64;
    }

    if (m == 1 && !bias) {
        gpt_rs_c_gemv_f32(a, b, c, n, k, bpack_full, bpack);
        return;
    }

    if (bpack_full) {
        for (size_t pc = 0; pc < k; pc += kc_block) {
            const size_t kc = GPTRS_MIN(kc_block, k - pc);
            for (size_t ic = 0; ic < m; ic += GPTRS_MC) {
                const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
                gpt_rs_pack_a(a + ic * k + pc, k, kc, mc, apack);
                for (size_t jc = 0; jc < n; jc += GPTRS_NC) {
                    const size_t nc = GPTRS_MIN(GPTRS_NC, n - jc);
                    const int init_mode = (pc == 0)
                        ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)
                        : GPTRS_INIT_ACCUM;
                    gpt_rs_matmul_compute_block(
                        apack,
                        bpack_full,
                        bpack64,
                        bpack32,
                        bpack,
                        c,
                        ic,
                        mc,
                        jc,
                        nc,
                        kc,
                        n,
                        k,
                        pc,
                        bias,
                        init_mode
                    );
                }
            }
        }
    } else {
        for (size_t jc = 0; jc < n; jc += GPTRS_NC) {
            const size_t nc = GPTRS_MIN(GPTRS_NC, n - jc);
            for (size_t pc = 0; pc < k; pc += kc_block) {
                const size_t kc = GPTRS_MIN(kc_block, k - pc);
                gpt_rs_pack_b(b + pc * n + jc, n, kc, nc, bpack);
                for (size_t ic = 0; ic < m; ic += GPTRS_MC) {
                    const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
                    gpt_rs_pack_a(a + ic * k + pc, k, kc, mc, apack);
                    const int init_mode = (pc == 0)
                        ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)
                        : GPTRS_INIT_ACCUM;
                    gpt_rs_matmul_compute_block(
                        apack,
                        bpack_full,
                        bpack64,
                        bpack32,
                        bpack,
                        c,
                        ic,
                        mc,
                        jc,
                        nc,
                        kc,
                        n,
                        k,
                        pc,
                        bias,
                        init_mode
                    );
                }
            }
        }
    }
}

static inline void gpt_rs_c_matmul_f32_cached_b(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t m,
    size_t n,
    size_t k,
    gpt_rs_bpack_cache* cache
) {
    gpt_rs_c_matmul_f32_cached_b_impl(a, b, c, m, n, k, cache, NULL);
}

static inline void gpt_rs_c_matmul_f32_cached_b_bias(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    const float* GPTRS_RESTRICT bias,
    float* GPTRS_RESTRICT c,
    size_t m,
    size_t n,
    size_t k,
    gpt_rs_bpack_cache* cache
) {
    gpt_rs_c_matmul_f32_cached_b_impl(a, b, c, m, n, k, cache, bias);
}

static inline void gpt_rs_c_matmul_f32(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t m,
    size_t n,
    size_t k
) {
    gpt_rs_c_matmul_f32_cached_b(a, b, c, m, n, k, NULL);
}

static inline void gpt_rs_c_conv2d_nhwc_f32_cached_b(
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
    if (n == 0 || out_h == 0 || out_w == 0 || c_out == 0) {
        return;
    }
    const size_t m = out_h * out_w;
    const size_t k = k_h * k_w * c_in;
    if (m == 0 || k == 0) {
        return;
    }

    if (k_h == 1 && k_w == 1
        && stride_h == 1 && stride_w == 1
        && dilation_h == 1 && dilation_w == 1
        && pad_top == 0 && pad_left == 0
        && out_h == in_h && out_w == in_w) {
        for (size_t batch = 0; batch < n; ++batch) {
            float* out_b = out + batch * m * c_out;
            const float* in_b = input + batch * m * c_in;
            if (bias) {
                gpt_rs_c_matmul_f32_cached_b_bias(
                    in_b,
                    weight,
                    bias,
                    out_b,
                    m,
                    c_out,
                    c_in,
                    cache
                );
            } else {
                gpt_rs_c_matmul_f32_cached_b(in_b, weight, out_b, m, c_out, c_in, cache);
            }
        }
        return;
    }

    const size_t kc_block = k;
    float* apack = NULL;
    float* bpack = NULL;
    {
        const size_t apack_elems = (size_t)GPTRS_MC * kc_block;
        const size_t bpack_elems = (size_t)GPTRS_NC * kc_block;
        if (!gpt_rs_get_scratch_with_capacity(apack_elems, bpack_elems, &apack, &bpack)) {
            abort();
        }
    }

    const float* bpack_full = NULL;
    const float* bpack32 = NULL;
    const float* bpack64 = NULL;
    if (cache && gpt_rs_bpack_cache_prepare(cache, weight, c_out, k)) {
        bpack_full = cache->bpack;
        bpack32 = cache->bpack32;
        bpack64 = cache->bpack64;
    }

    for (size_t batch = 0; batch < n; ++batch) {
        float* out_b = out + batch * m * c_out;

        if (bpack_full) {
            for (size_t pc = 0; pc < k; pc += kc_block) {
                const size_t kc = GPTRS_MIN(kc_block, k - pc);
                const int init_mode = (pc == 0)
                    ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)
                    : GPTRS_INIT_ACCUM;
                for (size_t ic = 0; ic < m; ic += GPTRS_MC) {
                    const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
                    gpt_rs_pack_a_conv(
                        input,
                        batch,
                        in_h,
                        in_w,
                        c_in,
                        out_w,
                        k_h,
                        k_w,
                        stride_h,
                        stride_w,
                        dilation_h,
                        dilation_w,
                        pad_top,
                        pad_left,
                        pc,
                        kc,
                        ic,
                        mc,
                        apack
                    );
                    for (size_t jc = 0; jc < c_out; jc += GPTRS_NC) {
                        const size_t nc = GPTRS_MIN(GPTRS_NC, c_out - jc);
                        gpt_rs_conv2d_compute_block(
                            apack,
                            bpack_full,
                            bpack64,
                            bpack32,
                            bpack,
                            out_b,
                            ic,
                            mc,
                            jc,
                            nc,
                            kc,
                            c_out,
                            k,
                            pc,
                            bias,
                            init_mode
                        );
                    }
                }
            }
        } else {
            for (size_t jc = 0; jc < c_out; jc += GPTRS_NC) {
                const size_t nc = GPTRS_MIN(GPTRS_NC, c_out - jc);
                for (size_t pc = 0; pc < k; pc += kc_block) {
                    const size_t kc = GPTRS_MIN(kc_block, k - pc);
                    const int init_mode = (pc == 0)
                        ? (bias ? GPTRS_INIT_BIAS : GPTRS_INIT_ZERO)
                        : GPTRS_INIT_ACCUM;
                    gpt_rs_pack_b(weight + pc * c_out + jc, c_out, kc, nc, bpack);
                    for (size_t ic = 0; ic < m; ic += GPTRS_MC) {
                        const size_t mc = GPTRS_MIN(GPTRS_MC, m - ic);
                        gpt_rs_pack_a_conv(
                            input,
                            batch,
                            in_h,
                            in_w,
                            c_in,
                            out_w,
                            k_h,
                            k_w,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            pad_top,
                            pad_left,
                            pc,
                            kc,
                            ic,
                            mc,
                            apack
                        );

                        gpt_rs_conv2d_compute_block(
                            apack,
                            bpack_full,
                            bpack64,
                            bpack32,
                            bpack,
                            out_b,
                            ic,
                            mc,
                            jc,
                            nc,
                            kc,
                            c_out,
                            k,
                            pc,
                            bias,
                            init_mode
                        );
                    }
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
    size_t pad_left
) {
    gpt_rs_c_conv2d_nhwc_f32_cached_b(
        input,
        weight,
        bias,
        out,
        n,
        in_h,
        in_w,
        c_in,
        out_h,
        out_w,
        c_out,
        k_h,
        k_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_top,
        pad_left,
        NULL
    );
}
