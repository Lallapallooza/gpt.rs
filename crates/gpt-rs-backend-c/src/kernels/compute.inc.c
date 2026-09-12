#if GPTRS_HAS_AVX512
/* Computes one 16-column strip of mr <= GPTRS_MR rows. Rows of b are GPTRS_PANEL_N apart. */
static inline void gpt_rs_panel16(const float* ap, const float* bp, float* c, size_t ldc,
                                  size_t kc, size_t mr, const float* bias, int init_mode) {
    const size_t bs = GPTRS_PANEL_N;
    const int accum = init_mode == GPTRS_INIT_ACCUM;
    if (mr == GPTRS_MR) {
        if (accum) {
            gpt_rs_ukernel_6x16_accum(ap, bp, bs, c, ldc, kc);
        } else if (init_mode == GPTRS_INIT_BIAS && bias) {
            gpt_rs_ukernel_6x16_bias(ap, bp, bs, c, ldc, kc, bias);
        } else {
            gpt_rs_ukernel_6x16_zero(ap, bp, bs, c, ldc, kc);
        }
        return;
    }
    /* 4-, 2- and 1-row kernels cover the mr rows. */
    size_t i = 0;
    if (mr - i >= 4) {
        if (accum) {
            gpt_rs_ukernel_4x16_accum(ap, bp, bs, c, ldc, kc);
        } else {
            gpt_rs_ukernel_4x16_zero(ap, bp, bs, c, ldc, kc);
        }
        i += 4;
    }
    if (mr - i >= 2) {
        if (accum) {
            gpt_rs_ukernel_2x16_accum(ap + i, bp, bs, c + i * ldc, ldc, kc);
        } else {
            gpt_rs_ukernel_2x16_zero(ap + i, bp, bs, c + i * ldc, ldc, kc);
        }
        i += 2;
    }
    if (mr - i >= 1) {
        if (accum) {
            gpt_rs_ukernel_1x16_accum(ap + i, bp, bs, c + i * ldc, ldc, kc);
        } else {
            gpt_rs_ukernel_1x16_zero(ap + i, bp, bs, c + i * ldc, ldc, kc);
        }
    }
    if (init_mode == GPTRS_INIT_BIAS && bias) {
        for (size_t r = 0; r < mr; ++r) {
            for (size_t j = 0; j < GPTRS_NR; ++j) {
                c[r * ldc + j] += bias[j];
            }
        }
    }
}

/* Computes one GPTRS_MR-row tile over w columns of a b panel. w is 32, 48 or 64. */
static inline void gpt_rs_tile_wide(const float* ap, const float* bp, float* c, size_t ldc,
                                    size_t kc, size_t w, const float* bias, int init_mode) {
    const size_t bs = GPTRS_PANEL_N;
    const float* b1 = bp + 16;
    const float* b2 = bp + 32;
    const float* b3 = bp + 48;
    if (init_mode == GPTRS_INIT_ACCUM) {
        if (w == 64) {
            gpt_rs_ukernel_6x64_accum(ap, bp, b1, b2, b3, bs, c, ldc, kc);
        } else if (w == 48) {
            gpt_rs_ukernel_6x48_accum(ap, bp, b1, b2, bs, c, ldc, kc);
        } else {
            gpt_rs_ukernel_6x32_accum(ap, bp, b1, bs, c, ldc, kc);
        }
    } else if (init_mode == GPTRS_INIT_BIAS) {
        if (w == 64) {
            gpt_rs_ukernel_6x64_bias(ap, bp, b1, b2, b3, bs, c, ldc, kc, bias);
        } else if (w == 48) {
            gpt_rs_ukernel_6x48_bias(ap, bp, b1, b2, bs, c, ldc, kc, bias);
        } else {
            gpt_rs_ukernel_6x32_bias(ap, bp, b1, bs, c, ldc, kc, bias);
        }
    } else {
        if (w == 64) {
            gpt_rs_ukernel_6x64_zero(ap, bp, b1, b2, b3, bs, c, ldc, kc);
        } else if (w == 48) {
            gpt_rs_ukernel_6x48_zero(ap, bp, b1, b2, bs, c, ldc, kc);
        } else {
            gpt_rs_ukernel_6x32_zero(ap, bp, b1, bs, c, ldc, kc);
        }
    }
}

/*
 * c[i][j] (+)= sum_p apack(i, p) * b(p, j) over one K block
 *
 * Computes the mc x nc block of c at row m_offset and column jc. The row stride of c is ldc.
 * `b` holds panels from gpt_rs_pack_b that start at column jc and are `panel_stride` floats
 * apart. The bias applies with GPTRS_INIT_BIAS.
 */
static inline void gpt_rs_compute_block(const float* apack, const float* b, size_t panel_stride,
                                        float* c, size_t ldc, size_t m_offset, size_t mc,
                                        size_t jc, size_t nc, size_t kc, const float* bias,
                                        int init_mode) {
    const size_t full_tiles = mc / GPTRS_MR;
    for (size_t j0 = 0; j0 < nc; j0 += GPTRS_PANEL_N) {
        const float* bp = b + j0 / GPTRS_PANEL_N * panel_stride;
        const size_t nr = GPTRS_MIN((size_t)GPTRS_PANEL_N, nc - j0);
        const float* bias_j = bias ? bias + jc + j0 : NULL;
        float* cj = c + m_offset * ldc + jc + j0;
        /* For full row tiles, one wide kernel covers all whole 16-column strips. */
        const size_t wide = full_tiles > 0 && nr >= 32 ? nr / 16 * 16 : 0;
        for (size_t ib = 0; wide && ib < full_tiles; ++ib) {
            gpt_rs_tile_wide(apack + ib * kc * GPTRS_MR, bp, cj + ib * GPTRS_MR * ldc, ldc, kc,
                             wide, bias_j, init_mode);
        }
        for (size_t s = 0; s < nr; s += 16) {
            const float* bias_s = bias_j ? bias_j + s : NULL;
            for (size_t i0 = s < wide ? full_tiles * GPTRS_MR : 0; i0 < mc; i0 += GPTRS_MR) {
                const float* ap = apack + i0 * kc;
                float* cs = cj + i0 * ldc + s;
                const size_t mr = GPTRS_MIN((size_t)GPTRS_MR, mc - i0);
                if (nr - s >= 16) {
                    gpt_rs_panel16(ap, bp + s, cs, ldc, kc, mr, bias_s, init_mode);
                } else {
                    gpt_rs_ukernel_masked(ap, bp + s, GPTRS_PANEL_N, cs, ldc, kc, mr, nr - s,
                                          bias_s, init_mode);
                }
            }
        }
    }
}

/*
 * One row of a (m == 1): c[j] = bias[j] + sum_p a[p * sak] * b[p * sbk + j] for
 * j < nc <= GPTRS_PANEL_N.
 *
 * Each element of b is used once, so the kernel streams rows of b in place and does not pack them.
 */
static inline void gpt_rs_gemv_f32_cols(const float* a, size_t sak, const float* b, size_t sbk,
                                        float* c, size_t nc, size_t k, const float* bias) {
    enum { V = GPTRS_PANEL_N / 16 };
    __mmask16 mask[V];
    __m512 acc[V];
    for (int v = 0; v < V; ++v) {
        const size_t j = (size_t)v * 16;
        mask[v] = (__mmask16)gpt_rs_tail_mask(nc > j ? nc - j : 0);
        acc[v] = bias ? _mm512_maskz_loadu_ps(mask[v], bias + j) : _mm512_setzero_ps();
    }
    for (size_t p = 0; p < k; ++p) {
        const __m512 av = _mm512_set1_ps(a[p * sak]);
        for (int v = 0; v < V; ++v) {
            const __m512 bv = _mm512_maskz_loadu_ps(mask[v], b + p * sbk + (size_t)v * 16);
            acc[v] = _mm512_fmadd_ps(av, bv, acc[v]);
        }
    }
    for (int v = 0; v < V; ++v) {
        _mm512_mask_storeu_ps(c + (size_t)v * 16, mask[v], acc[v]);
    }
}
#endif
