static inline void gpt_rs_ukernel_6x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_6x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_6x16_bias(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc, const float* bias);
static inline void gpt_rs_ukernel_4x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_4x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_6x32_accum(const float* ap,
                                             const float* bp0,
                                             const float* bp1,
                                             size_t b_stride,
                                             float* c,
                                             size_t ldc,
                                             size_t kc);
static inline void gpt_rs_ukernel_6x32_zero(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc);
static inline void gpt_rs_ukernel_6x32_bias(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc,
                                            const float* bias);
static inline void gpt_rs_ukernel_6x48_accum(const float* ap,
                                             const float* bp0,
                                             const float* bp1,
                                             const float* bp2,
                                             size_t b_stride,
                                             float* c,
                                             size_t ldc,
                                             size_t kc);
static inline void gpt_rs_ukernel_6x48_zero(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc);
static inline void gpt_rs_ukernel_6x48_bias(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc,
                                            const float* bias);
static inline void gpt_rs_ukernel_6x64_accum(const float* ap,
                                             const float* bp0,
                                             const float* bp1,
                                             const float* bp2,
                                             const float* bp3,
                                             size_t b_stride,
                                             float* c,
                                             size_t ldc,
                                             size_t kc);
static inline void gpt_rs_ukernel_6x64_zero(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            const float* bp3,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc);
static inline void gpt_rs_ukernel_6x64_bias(const float* ap,
                                            const float* bp0,
                                            const float* bp1,
                                            const float* bp2,
                                            const float* bp3,
                                            size_t b_stride,
                                            float* c,
                                            size_t ldc,
                                            size_t kc,
                                            const float* bias);
static inline void gpt_rs_ukernel_2x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_2x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_1x16_accum(const float* ap, const float* bp, float* c,
                                             size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_1x16_zero(const float* ap, const float* bp, float* c,
                                            size_t ldc, size_t kc);
static inline void gpt_rs_ukernel_partial(const float* ap, const float* bp, float* c,
                                          size_t ldc, size_t kc, size_t mr, size_t nr,
                                          int accum);
static inline void gpt_rs_ukernel_masked(const float* ap,
                                         const float* bp,
                                         float* c,
                                         size_t ldc,
                                         size_t kc,
                                         size_t mr,
                                         size_t nr,
                                         const float* bias,
                                         int init_mode);

static inline void gpt_rs_conv2d_panel16(
    const float* ap,
    const float* bp,
    float* cptr,
    size_t c_out,
    size_t kc,
    size_t mr,
    const float* bias,
    int init_mode
) {
    if (mr == GPTRS_MR) {
        if (init_mode == GPTRS_INIT_ACCUM) {
            gpt_rs_ukernel_6x16_accum(ap, bp, cptr, c_out, kc);
            return;
        }
        if (init_mode == GPTRS_INIT_BIAS && bias) {
            gpt_rs_ukernel_6x16_bias(ap, bp, cptr, c_out, kc, bias);
            return;
        }
        gpt_rs_ukernel_6x16_zero(ap, bp, cptr, c_out, kc);
        return;
    }

    if (init_mode == GPTRS_INIT_ACCUM) {
        if (mr >= 4) {
            gpt_rs_ukernel_4x16_accum(ap, bp, cptr, c_out, kc);
            if (mr == 5) {
                gpt_rs_ukernel_1x16_accum(
                    ap + 4,
                    bp,
                    cptr + 4 * c_out,
                    c_out,
                    kc
                );
            } else if (mr == 6) {
                gpt_rs_ukernel_2x16_accum(
                    ap + 4,
                    bp,
                    cptr + 4 * c_out,
                    c_out,
                    kc
                );
            }
        } else if (mr >= 2) {
            gpt_rs_ukernel_2x16_accum(ap, bp, cptr, c_out, kc);
            if (mr == 3) {
                gpt_rs_ukernel_1x16_accum(
                    ap + 2,
                    bp,
                    cptr + 2 * c_out,
                    c_out,
                    kc
                );
            }
        } else if (mr == 1) {
            gpt_rs_ukernel_1x16_accum(ap, bp, cptr, c_out, kc);
        }
        return;
    }

    if (mr >= 4) {
        gpt_rs_ukernel_4x16_zero(ap, bp, cptr, c_out, kc);
        if (mr == 5) {
            gpt_rs_ukernel_1x16_zero(
                ap + 4,
                bp,
                cptr + 4 * c_out,
                c_out,
                kc
            );
        } else if (mr == 6) {
            gpt_rs_ukernel_2x16_zero(
                ap + 4,
                bp,
                cptr + 4 * c_out,
                c_out,
                kc
            );
        }
    } else if (mr >= 2) {
        gpt_rs_ukernel_2x16_zero(ap, bp, cptr, c_out, kc);
        if (mr == 3) {
            gpt_rs_ukernel_1x16_zero(
                ap + 2,
                bp,
                cptr + 2 * c_out,
                c_out,
                kc
            );
        }
    } else if (mr == 1) {
        gpt_rs_ukernel_1x16_zero(ap, bp, cptr, c_out, kc);
    }

    if (init_mode == GPTRS_INIT_BIAS && bias) {
        for (size_t i = 0; i < mr; ++i) {
            for (size_t j = 0; j < GPTRS_NR; ++j) {
                cptr[i * c_out + j] += bias[j];
            }
        }
    }
}

static inline void gpt_rs_conv2d_compute_block(
    const float* apack,
    const float* bpack_full,
    const float* bpack64,
    const float* bpack32,
    const float* bpack,
    float* out_b,
    size_t m_offset,
    size_t mc,
    size_t jc,
    size_t nc,
    size_t kc,
    size_t c_out,
    size_t k_total,
    size_t pc,
    const float* bias,
    int init_mode
) {
    const size_t mtiles = (mc + GPTRS_MR - 1) / GPTRS_MR;
    const size_t ntiles = (nc + GPTRS_NR - 1) / GPTRS_NR;
    const size_t a_panel_stride = kc * GPTRS_MR;
    const size_t b_panel_stride = kc * GPTRS_NR;
    const size_t full_tiles = mc / GPTRS_MR;
    const size_t tail_rows = mc - full_tiles * GPTRS_MR;

    size_t jb = 0;
    while (jb < ntiles) {
        const size_t j0 = jb * GPTRS_NR;
        const size_t rem_cols = nc - j0;
        if (rem_cols >= 64 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack64) {
                const size_t group = (jc + j0) / 64;
                bp_base = bpack64 + group * k_total * 64 + pc * 64;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack64 ? (k_total * 64) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack64 ? 16 : stride);
            const float* bp2 = bp_base + (bpack64 ? 32 : 2 * stride);
            const float* bp3 = bp_base + (bpack64 ? 48 : 3 * stride);
            const size_t b_stride = bpack64 ? 64 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = out_b + (m_offset + i0) * c_out + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x64_accum(ap, bp0, bp1, bp2, bp3, b_stride, cptr, c_out, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x64_bias(ap, bp0, bp1, bp2, bp3, b_stride, cptr, c_out, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x64_zero(ap, bp0, bp1, bp2, bp3, b_stride, cptr, c_out, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack64 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = out_b + (m_offset + full_tiles * GPTRS_MR) * c_out + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 2 * tail_stride, ctail + 2 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 2 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 3 * tail_stride, ctail + 3 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 3 * GPTRS_NR : NULL, init_mode);
            }
            jb += 4;
            continue;
        }
        if (rem_cols >= 48 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack64 && ((jc + j0) % 64 == 0)) {
                const size_t group = (jc + j0) / 64;
                bp_base = bpack64 + group * k_total * 64 + pc * 64;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack64 ? (k_total * 64) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack64 ? 16 : stride);
            const float* bp2 = bp_base + (bpack64 ? 32 : 2 * stride);
            const size_t b_stride = bpack64 ? 64 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = out_b + (m_offset + i0) * c_out + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x48_accum(ap, bp0, bp1, bp2, b_stride, cptr, c_out, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x48_bias(ap, bp0, bp1, bp2, b_stride, cptr, c_out, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x48_zero(ap, bp0, bp1, bp2, b_stride, cptr, c_out, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack64 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = out_b + (m_offset + full_tiles * GPTRS_MR) * c_out + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 2 * tail_stride, ctail + 2 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 2 * GPTRS_NR : NULL, init_mode);
            }
            jb += 3;
            continue;
        }
        if (rem_cols >= 32 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack32 && ((jc + j0) % 32 == 0)) {
                const size_t group = (jc + j0) / 32;
                bp_base = bpack32 + group * k_total * 32 + pc * 32;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack32 ? (k_total * 32) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack32 ? 16 : stride);
            const size_t b_stride = bpack32 ? 32 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = out_b + (m_offset + i0) * c_out + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x32_accum(ap, bp0, bp1, b_stride, cptr, c_out, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x32_bias(ap, bp0, bp1, b_stride, cptr, c_out, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x32_zero(ap, bp0, bp1, b_stride, cptr, c_out, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack32 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = out_b + (m_offset + full_tiles * GPTRS_MR) * c_out + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, c_out, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
            }
            jb += 2;
            continue;
        }
        {
            const size_t nr = GPTRS_MIN(GPTRS_NR, rem_cols);
            const float* bp = NULL;
            if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp = bpack + jb * b_panel_stride;
            }
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < mtiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const size_t mr = GPTRS_MIN(GPTRS_MR, mc - i0);
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = out_b + (m_offset + i0) * c_out + (jc + j0);
                if (nr == GPTRS_NR) {
                    gpt_rs_conv2d_panel16(ap, bp, cptr, c_out, kc, mr, bias_ptr, init_mode);
                } else {
                    gpt_rs_ukernel_masked(ap, bp, cptr, c_out, kc, mr, nr,
                                          bias_ptr, init_mode);
                }
            }
        }
        jb += 1;
    }
}

static inline void gpt_rs_matmul_compute_block(
    const float* apack,
    const float* bpack_full,
    const float* bpack64,
    const float* bpack32,
    const float* bpack,
    float* c,
    size_t m_offset,
    size_t mc,
    size_t jc,
    size_t nc,
    size_t kc,
    size_t n,
    size_t k_total,
    size_t pc,
    const float* bias,
    int init_mode
) {
    const size_t mtiles = (mc + GPTRS_MR - 1) / GPTRS_MR;
    const size_t ntiles = (nc + GPTRS_NR - 1) / GPTRS_NR;
    const size_t a_panel_stride = kc * GPTRS_MR;
    const size_t b_panel_stride = kc * GPTRS_NR;
    const size_t full_tiles = mc / GPTRS_MR;
    const size_t tail_rows = mc - full_tiles * GPTRS_MR;
    size_t jb = 0;
    while (jb < ntiles) {
        const size_t j0 = jb * GPTRS_NR;
        const size_t rem_cols = nc - j0;
        if (rem_cols >= 64 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack64) {
                const size_t group = (jc + j0) / 64;
                bp_base = bpack64 + group * k_total * 64 + pc * 64;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack64 ? (k_total * 64) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack64 ? 16 : stride);
            const float* bp2 = bp_base + (bpack64 ? 32 : 2 * stride);
            const float* bp3 = bp_base + (bpack64 ? 48 : 3 * stride);
            const size_t b_stride = bpack64 ? 64 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = c + (m_offset + i0) * n + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x64_accum(ap, bp0, bp1, bp2, bp3, b_stride, cptr, n, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x64_bias(ap, bp0, bp1, bp2, bp3, b_stride, cptr, n, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x64_zero(ap, bp0, bp1, bp2, bp3, b_stride, cptr, n, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack64 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = c + (m_offset + full_tiles * GPTRS_MR) * n + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 2 * tail_stride, ctail + 2 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 2 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 3 * tail_stride, ctail + 3 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 3 * GPTRS_NR : NULL, init_mode);
            }
            jb += 4;
            continue;
        }
        if (rem_cols >= 48 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack64 && ((jc + j0) % 64 == 0)) {
                const size_t group = (jc + j0) / 64;
                bp_base = bpack64 + group * k_total * 64 + pc * 64;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack64 ? (k_total * 64) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack64 ? 16 : stride);
            const float* bp2 = bp_base + (bpack64 ? 32 : 2 * stride);
            const size_t b_stride = bpack64 ? 64 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = c + (m_offset + i0) * n + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x48_accum(ap, bp0, bp1, bp2, b_stride, cptr, n, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x48_bias(ap, bp0, bp1, bp2, b_stride, cptr, n, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x48_zero(ap, bp0, bp1, bp2, b_stride, cptr, n, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack64 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = c + (m_offset + full_tiles * GPTRS_MR) * n + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + 2 * tail_stride, ctail + 2 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 2 * GPTRS_NR : NULL, init_mode);
            }
            jb += 3;
            continue;
        }
        if (rem_cols >= 32 && full_tiles > 0) {
            const float* bp_base = NULL;
            if (bpack32 && ((jc + j0) % 32 == 0)) {
                const size_t group = (jc + j0) / 32;
                bp_base = bpack32 + group * k_total * 32 + pc * 32;
            } else if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp_base = bpack + jb * b_panel_stride;
            }
            const float* bp0 = bp_base;
            const size_t stride = bpack32 ? (k_total * 32) : ((bpack_full ? k_total : kc) * GPTRS_NR);
            const float* bp1 = bp_base + (bpack32 ? 16 : stride);
            const size_t b_stride = bpack32 ? 32 : GPTRS_NR;
            const size_t panel_stride = (bpack_full ? k_total : kc) * GPTRS_NR;
            const float* bias_ptr = bias ? bias + jc + j0 : NULL;
            for (size_t ib = 0; ib < full_tiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = c + (m_offset + i0) * n + (jc + j0);
                if (init_mode == GPTRS_INIT_ACCUM) {
                    gpt_rs_ukernel_6x32_accum(ap, bp0, bp1, b_stride, cptr, n, kc);
                } else if (init_mode == GPTRS_INIT_BIAS) {
                    gpt_rs_ukernel_6x32_bias(ap, bp0, bp1, b_stride, cptr, n, kc, bias_ptr);
                } else {
                    gpt_rs_ukernel_6x32_zero(ap, bp0, bp1, b_stride, cptr, n, kc);
                }
            }
            if (tail_rows > 0) {
                const float* bp_tail_base = bp_base;
                size_t tail_stride = panel_stride;
                if (bpack32 && bpack_full) {
                    const size_t panel = (jc + j0) / GPTRS_NR;
                    bp_tail_base = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
                }
                const float* ap_tail = apack + full_tiles * a_panel_stride;
                float* ctail = c + (m_offset + full_tiles * GPTRS_MR) * n + (jc + j0);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base, ctail + 0 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 0 * GPTRS_NR : NULL, init_mode);
                gpt_rs_conv2d_panel16(ap_tail, bp_tail_base + tail_stride, ctail + 1 * GPTRS_NR, n, kc, tail_rows,
                                      bias ? bias + jc + j0 + 1 * GPTRS_NR : NULL, init_mode);
            }
            jb += 2;
            continue;
        }

        {
            const size_t nr = GPTRS_MIN(GPTRS_NR, rem_cols);
            const float* bp = NULL;
            if (bpack_full) {
                const size_t panel = (jc + j0) / GPTRS_NR;
                bp = bpack_full + panel * k_total * GPTRS_NR + pc * GPTRS_NR;
            } else {
                bp = bpack + jb * b_panel_stride;
            }
            for (size_t ib = 0; ib < mtiles; ++ib) {
                const size_t i0 = ib * GPTRS_MR;
                const size_t mr = GPTRS_MIN(GPTRS_MR, mc - i0);
                const float* ap = apack + ib * a_panel_stride;
                float* cptr = c + (m_offset + i0) * n + (jc + j0);
                if (nr == GPTRS_NR) {
                    gpt_rs_conv2d_panel16(ap, bp, cptr, n, kc, mr,
                                          bias ? bias + jc + j0 : NULL, init_mode);
                } else {
                    gpt_rs_ukernel_masked(ap, bp, cptr, n, kc, mr, nr,
                                          bias ? bias + jc + j0 : NULL, init_mode);
                }
            }
        }
        jb += 1;
    }
}

static inline void gpt_rs_gemv_1x16_zero(const float* a, const float* bp, float* c,
                                         size_t kc) {
    __m512 c0 = _mm512_setzero_ps();
    const float* a0 = a;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va = _mm512_set1_ps(*a0);
        c0 = _mm512_fmadd_ps(va, b0, c0);
        a0 += 1;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
}

static inline void gpt_rs_gemv_1x16_accum(const float* a, const float* bp, float* c,
                                          size_t kc) {
    __m512 c0 = _mm512_loadu_ps(c);
    const float* a0 = a;
    for (size_t p = 0; p < kc; ++p) {
        const __m512 b0 = _mm512_load_ps(bp);
        const __m512 va = _mm512_set1_ps(*a0);
        c0 = _mm512_fmadd_ps(va, b0, c0);
        a0 += 1;
        bp += GPTRS_NR;
    }
    _mm512_storeu_ps(c, c0);
}

static inline void gpt_rs_gemv_1x16_tail(const float* a, const float* bp, float* c,
                                         size_t kc, size_t nr, int accum) {
    float tmp[GPTRS_NR];
    for (size_t j = 0; j < nr; ++j) {
        tmp[j] = accum ? c[j] : 0.0f;
    }
    for (size_t p = 0; p < kc; ++p) {
        const float a_val = a[p];
        const float* b_row = bp + p * GPTRS_NR;
        for (size_t j = 0; j < nr; ++j) {
            tmp[j] += a_val * b_row[j];
        }
    }
    for (size_t j = 0; j < nr; ++j) {
        c[j] = tmp[j];
    }
}

static inline void gpt_rs_c_gemv_f32(
    const float* GPTRS_RESTRICT a,
    const float* GPTRS_RESTRICT b,
    float* GPTRS_RESTRICT c,
    size_t n,
    size_t k,
    const float* bpack_full,
    float* bpack
) {
    if (n == 0 || k == 0) {
        return;
    }

    if (bpack_full) {
        for (size_t jc = 0; jc < n; jc += GPTRS_NR) {
            const size_t nr = GPTRS_MIN(GPTRS_NR, n - jc);
            const size_t panel = jc / GPTRS_NR;
            const float* bp = bpack_full + panel * k * GPTRS_NR;
            if (nr == GPTRS_NR) {
                gpt_rs_gemv_1x16_zero(a, bp, c + jc, k);
            } else {
                gpt_rs_gemv_1x16_tail(a, bp, c + jc, k, nr, 0);
            }
        }
        return;
    }

    for (size_t jc = 0; jc < n; jc += GPTRS_NC) {
        const size_t nc = GPTRS_MIN(GPTRS_NC, n - jc);
        for (size_t pc = 0; pc < k; pc += GPTRS_KC) {
            const size_t kc = GPTRS_MIN(GPTRS_KC, k - pc);
            gpt_rs_pack_b(b + pc * n + jc, n, kc, nc, bpack);

            const size_t ntiles = (nc + GPTRS_NR - 1) / GPTRS_NR;
            for (size_t jb = 0; jb < ntiles; ++jb) {
                const size_t j0 = jb * GPTRS_NR;
                const size_t nr = GPTRS_MIN(GPTRS_NR, nc - j0);
                const float* bp = bpack + jb * kc * GPTRS_NR;
                float* cptr = c + jc + j0;
                if (nr == GPTRS_NR) {
                    if (pc == 0) {
                        gpt_rs_gemv_1x16_zero(a + pc, bp, cptr, kc);
                    } else {
                        gpt_rs_gemv_1x16_accum(a + pc, bp, cptr, kc);
                    }
                } else {
                    gpt_rs_gemv_1x16_tail(a + pc, bp, cptr, kc, nr, pc != 0);
                }
            }
        }
    }
}
