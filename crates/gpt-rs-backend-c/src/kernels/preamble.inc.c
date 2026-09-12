
// Kernel scaffold (SIMD hooks can be added here).
/* The vector kernels use AVX512F with FMA plus the BW/VL 16-bit masked loads. */
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__FMA__)
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

/* Returns a mask with the low min(left, 32) bits set. It selects the lanes of a masked vector
 * load or store over a tail. */
static inline uint32_t gpt_rs_tail_mask(size_t left) {
    return left >= 32 ? 0xffffffffu : (1u << left) - 1u;
}

/* glibc declares its libmvec vector variants only under -ffast-math. These declarations let GCC
 * vectorise elementwise loops with libmvec, whose results are within 4 ulp. Clang gets
 * -fveclib=libmvec instead. */
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__GLIBC__) && \
    !defined(__FAST_MATH__)
#if __GLIBC_PREREQ(2, 22)
__attribute__((simd("notinbranch"))) float expf(float);
__attribute__((simd("notinbranch"))) float logf(float);
#endif
#if __GLIBC_PREREQ(2, 35)
__attribute__((simd("notinbranch"))) float tanhf(float);
__attribute__((simd("notinbranch"))) float erff(float);
#endif
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Returns the thread count of the next parallel region. The entry point sets the team size. */
static inline int gpt_rs_c_team_size(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/* Tasks per thread of the parallel matmul loops. Threads claim tasks dynamically, so faster or
 * less loaded cores take more of them. A higher count gives smaller GEMM tiles, which repack more
 * shared panels. */
#define GPTRS_TASKS_PER_THREAD ((size_t)2)

/* Per-core L2 cache size. The host passes it as -DGPTRS_L2_BYTES. It is 1 MiB when unknown. */
#if !defined(GPTRS_L2_BYTES)
#define GPTRS_L2_BYTES (1 << 20)
#endif

#if defined(__GNUC__)
#define GPTRS_NOINLINE __attribute__((noinline))
#else
#define GPTRS_NOINLINE
#endif

/* bfloat16 values are stored as the upper 16 bits of an IEEE-754 binary32. */
static inline float gpt_rs_bf16_to_f32(uint16_t bits) {
    uint32_t wide = ((uint32_t)bits) << 16;
    float out;
    memcpy(&out, &wide, sizeof(out));
    return out;
}

/* Converts with round-to-nearest-even. NaNs stay quiet NaNs. */
static inline uint16_t gpt_rs_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

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

/* Per-op timing hooks. Only profiling builds (GPTRS_PROFILE_BACKEND=1) compile them in. */
#if defined(GPTRS_C_PROFILE)
#define GPTRS_OP_BEGIN(id) \
    const uint64_t gpt_rs_op_start_##id = gpt_rs_c_profile_on() ? gpt_rs_c_now_ns() : 0
#define GPTRS_OP_END(id)                                                             \
    do {                                                                             \
        if (gpt_rs_op_start_##id != 0) {                                             \
            gpt_rs_c_prof_op_ns[id] += gpt_rs_c_now_ns() - gpt_rs_op_start_##id;     \
            gpt_rs_c_prof_op_calls[id] += 1;                                         \
        }                                                                            \
    } while (0)
#else
#define GPTRS_OP_BEGIN(id) ((void)0)
#define GPTRS_OP_END(id) ((void)0)
#endif

enum { GPTRS_MR = 6, GPTRS_NR = 16 };
/* Packed b panels are as wide as the widest micro-kernel, 6 x 64. */
enum { GPTRS_PANEL_N = 64 };
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
