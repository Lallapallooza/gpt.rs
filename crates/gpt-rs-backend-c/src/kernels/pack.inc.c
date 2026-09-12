#if GPTRS_HAS_AVX512
/* In-register transpose of a 16 x 16 block of 32-bit lanes. */
static inline void gpt_rs_transpose16(__m512i r[16]) {
    __m512i t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        r[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        r[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        r[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        t[i] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0x88);
        t[i + 4] = _mm512_shuffle_i32x4(r[i], r[i + 4], 0xdd);
        t[i + 8] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0x88);
        t[i + 12] = _mm512_shuffle_i32x4(r[i + 8], r[i + 12], 0xdd);
    }
    for (int i = 0; i < 4; ++i) {
        r[i] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0x88);
        r[i + 8] = _mm512_shuffle_i32x4(t[i], t[i + 8], 0xdd);
        r[i + 4] = _mm512_shuffle_i32x4(t[i + 4], t[i + 12], 0x88);
        r[i + 12] = _mm512_shuffle_i32x4(t[i + 4], t[i + 12], 0xdd);
    }
}

/*
 * Packs the kc x nc block of b into panels of GPTRS_PANEL_N columns. Element (p, j) of b is at
 * b[p * sbk + j * sbn], and it goes to
 *
 *     bpack[j / GPTRS_PANEL_N * kc * GPTRS_PANEL_N + p * GPTRS_PANEL_N + j % GPTRS_PANEL_N].
 *
 * Columns from nc up to the next multiple of 16 are zero.
 */
static inline void gpt_rs_pack_b(const float* b, size_t sbk, size_t sbn, size_t kc, size_t nc,
                                 float* bpack) {
    for (size_t j0 = 0; j0 < nc; j0 += 16) {
        const float* src = b + j0 * sbn;
        float* dst = bpack + j0 / GPTRS_PANEL_N * kc * GPTRS_PANEL_N + j0 % GPTRS_PANEL_N;
        const size_t nr = GPTRS_MIN((size_t)16, nc - j0);
        if (sbn == 1) {
            const __mmask16 mask = (__mmask16)gpt_rs_tail_mask(nr);
            for (size_t p = 0; p < kc; ++p) {
                _mm512_store_ps(dst + p * GPTRS_PANEL_N, _mm512_maskz_loadu_ps(mask, src + p * sbk));
            }
        } else if (sbk == 1) {
            /* Columns of b are contiguous, so transpose 16 x 16 blocks. */
            for (size_t p0 = 0; p0 < kc; p0 += 16) {
                const __mmask16 mask = (__mmask16)gpt_rs_tail_mask(kc - p0);
                __m512i r[16];
                for (size_t j = 0; j < 16; ++j) {
                    r[j] = j < nr ? _mm512_maskz_loadu_epi32(mask, src + j * sbn + p0)
                                  : _mm512_setzero_si512();
                }
                gpt_rs_transpose16(r);
                for (size_t q = 0; q < GPTRS_MIN((size_t)16, kc - p0); ++q) {
                    _mm512_store_si512((void*)(dst + (p0 + q) * GPTRS_PANEL_N), r[q]);
                }
            }
        } else {
            for (size_t p = 0; p < kc; ++p) {
                for (size_t j = 0; j < 16; ++j) {
                    dst[p * GPTRS_PANEL_N + j] = j < nr ? src[p * sbk + j * sbn] : 0.0f;
                }
            }
        }
    }
}

_Static_assert(GPTRS_MR <= 16, "a packed row of a is one masked 16-lane store");

/* Packs the mc x kc block of a. Element (i, p) of a is at a[i * sam + p * sak]. */
static inline void gpt_rs_pack_a(const float* a, size_t sam, size_t sak, size_t kc, size_t mc,
                                 float* apack) {
    const size_t mtiles = (mc + GPTRS_MR - 1) / GPTRS_MR;
    for (size_t ib = 0; ib < mtiles; ++ib) {
        const size_t i0 = ib * GPTRS_MR;
        const size_t mr = GPTRS_MIN(GPTRS_MR, mc - i0);
        float* ap = apack + ib * kc * GPTRS_MR;
        size_t p = 0;
        if (sak == 1) {
            /* Packs 16 columns at a time with a register transpose of the rows. Rows past mr
             * are zero. */
            for (; p + 16 <= kc; p += 16) {
                __m512i r[16];
                for (size_t i = 0; i < 16; ++i) {
                    r[i] = i < mr ? _mm512_loadu_si512(a + (i0 + i) * sam + p)
                                  : _mm512_setzero_si512();
                }
                gpt_rs_transpose16(r);
                for (size_t q = 0; q < 16; ++q) {
                    _mm512_mask_storeu_epi32(ap + (p + q) * GPTRS_MR,
                                             (__mmask16)((1u << GPTRS_MR) - 1u), r[q]);
                }
            }
        }
        for (; p < kc; ++p) {
            float* dst = ap + p * GPTRS_MR;
            for (size_t i = 0; i < mr; ++i) {
                dst[i] = a[(i0 + i) * sam + p * sak];
            }
            for (size_t i = mr; i < GPTRS_MR; ++i) {
                dst[i] = 0.0f;
            }
        }
    }
}

static inline void gpt_rs_pack_a_conv(
    const float* input,
    size_t batch,
    size_t in_h,
    size_t in_w,
    size_t c_in,
    size_t out_w,
    size_t k_h,
    size_t k_w,
    size_t stride_h,
    size_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t pad_top,
    size_t pad_left,
    size_t k_offset,
    size_t kc,
    size_t m_offset,
    size_t mc,
    float* apack
) {
    const size_t mtiles = (mc + GPTRS_MR - 1) / GPTRS_MR;
    const size_t kw_ci = k_w * c_in;
    const float* input_b = input + batch * in_h * in_w * c_in;
    size_t kh0 = 0;
    size_t kw0 = 0;
    size_t ci0 = 0;
    if (kw_ci != 0) {
        kh0 = k_offset / kw_ci;
        size_t rem = k_offset - kh0 * kw_ci;
        if (c_in != 0) {
            kw0 = rem / c_in;
            ci0 = rem - kw0 * c_in;
        }
    }
    const int64_t k_ext_h = (int64_t)(k_h - 1) * (int64_t)dilation_h;
    const int64_t k_ext_w = (int64_t)(k_w - 1) * (int64_t)dilation_w;

    for (size_t ib = 0; ib < mtiles; ++ib) {
        const size_t i0 = ib * GPTRS_MR;
        const size_t mr = GPTRS_MIN(GPTRS_MR, mc - i0);
        float* ap = apack + ib * kc * GPTRS_MR;
        const size_t row0 = m_offset + i0;
        size_t oh = row0 / out_w;
        size_t ow = row0 - oh * out_w;
        int64_t base_h[GPTRS_MR];
        int64_t base_w[GPTRS_MR];
        int tile_interior = 1;

        size_t cur_oh = oh;
        size_t cur_ow = ow;
        for (size_t i = 0; i < mr; ++i) {
            const int64_t bh = (int64_t)cur_oh * (int64_t)stride_h - (int64_t)pad_top;
            const int64_t bw = (int64_t)cur_ow * (int64_t)stride_w - (int64_t)pad_left;
            base_h[i] = bh;
            base_w[i] = bw;
            if (bh < 0 || bw < 0
                || bh + k_ext_h >= (int64_t)in_h
                || bw + k_ext_w >= (int64_t)in_w) {
                tile_interior = 0;
            }
            ++cur_ow;
            if (cur_ow == out_w) {
                cur_ow = 0;
                ++cur_oh;
            }
        }
        for (size_t i = mr; i < GPTRS_MR; ++i) {
            base_h[i] = 0;
            base_w[i] = 0;
        }

        size_t kh = kh0;
        size_t kw = kw0;
        size_t ci = ci0;
        size_t p = 0;
        for (; kh < k_h && p < kc; ++kh) {
            const int64_t kh_d = (int64_t)kh * (int64_t)dilation_h;
            for (; kw < k_w && p < kc; ++kw) {
                const int64_t kw_d = (int64_t)kw * (int64_t)dilation_w;
                const float* src_ptr[GPTRS_MR];
                if (tile_interior) {
                    for (size_t i = 0; i < mr; ++i) {
                        const size_t ih = (size_t)(base_h[i] + kh_d);
                        const size_t iw = (size_t)(base_w[i] + kw_d);
                        src_ptr[i] = input_b + (ih * in_w + iw) * c_in;
                    }
                } else {
                    for (size_t i = 0; i < mr; ++i) {
                        const int64_t ih = base_h[i] + kh_d;
                        const int64_t iw = base_w[i] + kw_d;
                        if (ih < 0 || iw < 0
                            || ih >= (int64_t)in_h
                            || iw >= (int64_t)in_w) {
                            src_ptr[i] = NULL;
                        } else {
                            src_ptr[i] = input_b + ((size_t)ih * in_w + (size_t)iw) * c_in;
                        }
                    }
                }
                for (; ci < c_in && p < kc; ++ci, ++p) {
                    float* dst = ap + p * GPTRS_MR;
                    for (size_t i = 0; i < mr; ++i) {
                        const float* sp = src_ptr[i];
                        dst[i] = sp ? sp[ci] : 0.0f;
                    }
                    for (size_t i = mr; i < GPTRS_MR; ++i) {
                        dst[i] = 0.0f;
                    }
                }
                ci = 0;
            }
            kw = 0;
        }
    }
}
#endif

/* A persistent copy of a k x n matrix b in the panel layout of gpt_rs_pack_b. Calls reuse it. */
typedef struct {
    const float* b_ptr;
    size_t n;
    size_t k;
    float* bpack;
} gpt_rs_bpack_cache;

/* Returns whether `cache` holds b. Packs b again when b, n or k changed. */
static inline int gpt_rs_bpack_cache_prepare(gpt_rs_bpack_cache* cache, const float* b, size_t n,
                                             size_t k, size_t sbk, size_t sbn) {
#if GPTRS_HAS_AVX512
    if (cache->b_ptr == b && cache->n == n && cache->k == k && cache->bpack) {
        return 1;
    }
    const size_t panels = (n + GPTRS_PANEL_N - 1) / GPTRS_PANEL_N;
    if (panels == 0 || k == 0 || k > SIZE_MAX / sizeof(float) / GPTRS_PANEL_N / panels) {
        return 0;
    }
    float* buf = (float*)gpt_rs_aligned_malloc(panels * k * GPTRS_PANEL_N * sizeof(float));
    if (!buf) {
        return 0;
    }
    gpt_rs_pack_b(b, sbk, sbn, k, n, buf);
    gpt_rs_aligned_free(cache->bpack);
    cache->bpack = buf;
    cache->b_ptr = b;
    cache->n = n;
    cache->k = k;
    return 1;
#else
    (void)cache;
    (void)b;
    (void)n;
    (void)k;
    (void)sbk;
    (void)sbn;
    return 0;
#endif
}
