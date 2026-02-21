
// Kernel scaffold (SIMD hooks can be added here).
#if defined(__AVX512F__) && defined(__FMA__)
#define GPTRS_HAS_AVX512 1
#include <immintrin.h>
#else
#define GPTRS_HAS_AVX512 0
#endif

#if defined(__GNUC__)
#define GPTRS_RESTRICT __restrict__
#define GPTRS_THREAD_LOCAL __thread
#elif defined(_MSC_VER)
#define GPTRS_THREAD_LOCAL __declspec(thread)
#else
#define GPTRS_RESTRICT
#define GPTRS_THREAD_LOCAL
#endif

#if GPTRS_HAS_AVX512
#define GPTRS_PREFETCH(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#else
#define GPTRS_PREFETCH(ptr) ((void)(ptr))
#endif

#define GPTRS_MIN(a, b) ((a) < (b) ? (a) : (b))

enum {
    GPTRS_INIT_ZERO = 0,
    GPTRS_INIT_ACCUM = 1,
    GPTRS_INIT_BIAS = 2
};

#if defined(GPTRS_C_PROFILE)
#if defined(_WIN32)
#include <windows.h>
#endif

static inline int gpt_rs_c_profile_on(void) {
    const char* env = getenv("GPTRS_PROFILE_BACKEND");
    if (!env || env[0] == '\0') {
        return 0;
    }
    char c = env[0];
    return (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y' || c == 'o' || c == 'O');
}

static inline uint64_t gpt_rs_c_now_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER counter;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    QueryPerformanceCounter(&counter);
    return (uint64_t)((counter.QuadPart * 1000000000ull) / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}
#endif

enum { GPTRS_MR = 6, GPTRS_NR = 16 };
enum { GPTRS_MC = 120, GPTRS_NC = 256, GPTRS_KC = 256 };
enum { GPTRS_PREFETCH_DIST = 8 };

static inline size_t gpt_rs_choose_kc(size_t k) {
    if (k <= 1152) {
        return k;
    }
    if (k <= 2304) {
        return 512;
    }
    return GPTRS_KC;
}

static inline void* gpt_rs_aligned_malloc(size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, 64);
#else
    void* p = NULL;
    if (posix_memalign(&p, 64, size) != 0) {
        return NULL;
    }
    return p;
#endif
}

static inline void gpt_rs_aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

typedef struct {
    float* apack;
    float* bpack;
    size_t apack_cap;
    size_t bpack_cap;
} gpt_rs_matmul_scratch;

static GPTRS_THREAD_LOCAL gpt_rs_matmul_scratch gpt_rs_tls_scratch = {0};

static inline int gpt_rs_get_scratch_with_capacity(
    size_t apack_elems,
    size_t bpack_elems,
    float** apack_out,
    float** bpack_out
) {
    if (apack_elems == 0 || bpack_elems == 0) {
        return 0;
    }
    if (apack_elems > gpt_rs_tls_scratch.apack_cap) {
        if (gpt_rs_tls_scratch.apack) {
            gpt_rs_aligned_free(gpt_rs_tls_scratch.apack);
        }
        gpt_rs_tls_scratch.apack =
            (float*)gpt_rs_aligned_malloc(apack_elems * sizeof(float));
        gpt_rs_tls_scratch.apack_cap = gpt_rs_tls_scratch.apack ? apack_elems : 0;
    }
    if (bpack_elems > gpt_rs_tls_scratch.bpack_cap) {
        if (gpt_rs_tls_scratch.bpack) {
            gpt_rs_aligned_free(gpt_rs_tls_scratch.bpack);
        }
        gpt_rs_tls_scratch.bpack =
            (float*)gpt_rs_aligned_malloc(bpack_elems * sizeof(float));
        gpt_rs_tls_scratch.bpack_cap = gpt_rs_tls_scratch.bpack ? bpack_elems : 0;
    }
    if (!gpt_rs_tls_scratch.apack || !gpt_rs_tls_scratch.bpack) {
        return 0;
    }
    *apack_out = gpt_rs_tls_scratch.apack;
    *bpack_out = gpt_rs_tls_scratch.bpack;
    return 1;
}

static inline int gpt_rs_get_scratch(float** apack_out, float** bpack_out) {
    const size_t apack_elems = (size_t)GPTRS_MC * (size_t)GPTRS_KC;
    const size_t bpack_elems = (size_t)GPTRS_NC * (size_t)GPTRS_KC;
    return gpt_rs_get_scratch_with_capacity(apack_elems, bpack_elems, apack_out, bpack_out);
}

typedef struct {
    const float* b_ptr;
    size_t n;
    size_t k;
    size_t ntiles;
    float* bpack;
    float* bpack32;
    float* bpack64;
} gpt_rs_bpack_cache;

static inline void gpt_rs_pack_b(const float* b, size_t n, size_t kc, size_t nc, float* bpack);
static inline void gpt_rs_pack_b_interleaved(const float* b,
                                              size_t n,
                                              size_t k,
                                              size_t group,
                                              float* bpack);

static inline int gpt_rs_bpack_cache_prepare(
    gpt_rs_bpack_cache* cache,
    const float* b,
    size_t n,
    size_t k
) {
#if GPTRS_HAS_AVX512
    if (!cache) {
        return 0;
    }
    if (cache->b_ptr == b && cache->n == n && cache->k == k && cache->bpack) {
        return 1;
    }
    const size_t ntiles = (n + GPTRS_NR - 1) / GPTRS_NR;
    if (ntiles == 0 || k == 0) {
        return 0;
    }
    const size_t elems = ntiles * k * GPTRS_NR;
    if (elems == 0 || elems > (SIZE_MAX / sizeof(float))) {
        return 0;
    }
    float* buf = (float*)gpt_rs_aligned_malloc(elems * sizeof(float));
    if (!buf) {
        return 0;
    }
    gpt_rs_pack_b(b, n, k, n, buf);
    if (cache->bpack) {
        gpt_rs_aligned_free(cache->bpack);
    }
    if (cache->bpack32) {
        gpt_rs_aligned_free(cache->bpack32);
    }
    if (cache->bpack64) {
        gpt_rs_aligned_free(cache->bpack64);
    }
    cache->bpack = buf;
    cache->bpack32 = NULL;
    cache->bpack64 = NULL;
    cache->b_ptr = b;
    cache->n = n;
    cache->k = k;
    cache->ntiles = ntiles;

    if (n >= 32) {
        const size_t groups32 = (n + 31) / 32;
        const size_t elems32 = groups32 * k * 32;
        if (elems32 <= (SIZE_MAX / sizeof(float))) {
            float* buf32 = (float*)gpt_rs_aligned_malloc(elems32 * sizeof(float));
            if (buf32) {
                gpt_rs_pack_b_interleaved(b, n, k, 32, buf32);
                cache->bpack32 = buf32;
            }
        }
    }
    if (n >= 48) {
        const size_t groups64 = (n + 63) / 64;
        const size_t elems64 = groups64 * k * 64;
        if (elems64 <= (SIZE_MAX / sizeof(float))) {
            float* buf64 = (float*)gpt_rs_aligned_malloc(elems64 * sizeof(float));
            if (buf64) {
                gpt_rs_pack_b_interleaved(b, n, k, 64, buf64);
                cache->bpack64 = buf64;
            }
        }
    }
    return 1;
#else
    (void)b;
    (void)n;
    (void)k;
    if (!cache) {
        return 0;
    }
    return 0;
#endif
}
