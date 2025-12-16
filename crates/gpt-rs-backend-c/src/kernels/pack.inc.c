static inline void gpt_rs_pack_b(const float* b, size_t n, size_t kc, size_t nc, float* bpack) {
    const size_t ntiles = (nc + GPTRS_NR - 1) / GPTRS_NR;
    for (size_t jb = 0; jb < ntiles; ++jb) {
        const size_t j0 = jb * GPTRS_NR;
        const size_t nr = GPTRS_MIN(GPTRS_NR, nc - j0);
        float* bp = bpack + jb * kc * GPTRS_NR;
        for (size_t p = 0; p < kc; ++p) {
            const float* brow = b + p * n + j0;
            float* dst = bp + p * GPTRS_NR;
            if (nr == GPTRS_NR) {
                const __m512 v = _mm512_loadu_ps(brow);
                _mm512_store_ps(dst, v);
            } else {
                for (size_t j = 0; j < nr; ++j) {
                    dst[j] = brow[j];
                }
                for (size_t j = nr; j < GPTRS_NR; ++j) {
                    dst[j] = 0.0f;
                }
            }
        }
    }
}

static inline void gpt_rs_pack_b_interleaved(const float* b,
                                              size_t n,
                                              size_t k,
                                              size_t group,
                                              float* bpack) {
    const size_t groups = (n + group - 1) / group;
    for (size_t gb = 0; gb < groups; ++gb) {
        const size_t j0 = gb * group;
        const size_t nr = GPTRS_MIN(group, n - j0);
        float* gp = bpack + gb * k * group;
        for (size_t p = 0; p < k; ++p) {
            const float* brow = b + p * n + j0;
            float* dst = gp + p * group;
            if (nr == group) {
                if (group == 32) {
                    const __m512 v0 = _mm512_loadu_ps(brow);
                    const __m512 v1 = _mm512_loadu_ps(brow + 16);
                    _mm512_store_ps(dst, v0);
                    _mm512_store_ps(dst + 16, v1);
                } else if (group == 64) {
                    const __m512 v0 = _mm512_loadu_ps(brow);
                    const __m512 v1 = _mm512_loadu_ps(brow + 16);
                    const __m512 v2 = _mm512_loadu_ps(brow + 32);
                    const __m512 v3 = _mm512_loadu_ps(brow + 48);
                    _mm512_store_ps(dst, v0);
                    _mm512_store_ps(dst + 16, v1);
                    _mm512_store_ps(dst + 32, v2);
                    _mm512_store_ps(dst + 48, v3);
                } else {
                    for (size_t j = 0; j < group; ++j) {
                        dst[j] = brow[j];
                    }
                }
            } else {
                for (size_t j = 0; j < nr; ++j) {
                    dst[j] = brow[j];
                }
                for (size_t j = nr; j < group; ++j) {
                    dst[j] = 0.0f;
                }
            }
        }
    }
}

static inline void gpt_rs_pack_a(const float* a, size_t k, size_t kc, size_t mc, float* apack) {
    const size_t mtiles = (mc + GPTRS_MR - 1) / GPTRS_MR;
    for (size_t ib = 0; ib < mtiles; ++ib) {
        const size_t i0 = ib * GPTRS_MR;
        const size_t mr = GPTRS_MIN(GPTRS_MR, mc - i0);
        float* ap = apack + ib * kc * GPTRS_MR;
        for (size_t p = 0; p < kc; ++p) {
            float* dst = ap + p * GPTRS_MR;
            for (size_t i = 0; i < mr; ++i) {
                dst[i] = a[(i0 + i) * k + p];
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
