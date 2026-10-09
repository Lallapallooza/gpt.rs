
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
 * vectorize elementwise loops with libmvec, whose results are within 4 ulp. Clang gets
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

/* Copies `bytes` bytes. A copy counts one unit of work per 4 bytes, like an f32 element of a
 * loop, and splits across the team in cache-line-aligned shares at GPTRS_PARALLEL_MIN_WORK
 * units or more. */
static void gpt_rs_copy(void* GPTRS_RESTRICT dst, const void* GPTRS_RESTRICT src, size_t bytes) {
    const size_t team = (size_t)gpt_rs_c_team_size();
    if (team < 2 || bytes / 4 < GPTRS_PARALLEL_MIN_WORK) {
        memcpy(dst, src, bytes);
        return;
    }
    const size_t share = (bytes / team + 63) / 64 * 64;
    _Pragma("omp parallel for schedule(static)")
    for (size_t t = 0; t < team; ++t) {
        const size_t begin = t * share;
        if (begin < bytes) {
            memcpy((char*)dst + begin, (const char*)src + begin, GPTRS_MIN(share, bytes - begin));
        }
    }
}

/* Tasks per thread of the parallel matmul loops. Threads claim tasks dynamically, so faster or
 * less loaded cores take more of them. A higher count gives smaller GEMM tiles, which repack more
 * shared panels. */
#define GPTRS_TASKS_PER_THREAD ((size_t)2)

/* Per-core L1 data and L2 cache sizes. The host passes them as -DGPTRS_L1_BYTES and
 * -DGPTRS_L2_BYTES. They are 32 KiB and 1 MiB when unknown. */
#if !defined(GPTRS_L1_BYTES)
#define GPTRS_L1_BYTES (32 << 10)
#endif
#if !defined(GPTRS_L2_BYTES)
#define GPTRS_L2_BYTES (1 << 20)
#endif

/* Output columns of one single-row matmul task. They fill a quarter of L1, so they stay in L1
 * while the four rows of b stream through it. */
#define GPTRS_GEMV_NC ((size_t)GPTRS_L1_BYTES / 4 / sizeof(float) / 16 * 16)

/* Keeps a function out of line, so it gets its own register allocation. A function with a large
 * register tile spills that tile to the stack when it is inlined into a loop. */
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

/* float8_e4m3fn: 4 exponent bits with bias 7, 3 mantissa bits, no infinities, and NaN at
 * S.1111.111. The host generates the table of all 256 values. */
static inline float gpt_rs_f8e4m3_to_f32(uint8_t bits) {
    float out;
    memcpy(&out, &gpt_rs_f8e4m3_bits[bits], sizeof(out));
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

/* The identity, for kernels that read f32 operands where other instances widen narrower ones. */
static inline float gpt_rs_f32_to_f32(float value) {
    return value;
}

#if GPTRS_HAS_AVX512
/* Operand loaders. `NAME(p)` reads 16 elements and widens them to f32 lanes, which is exact.
 * `NAME##_tail(p, left)` reads the first min(left, 16) elements, and the other lanes are zero. */
static inline __m512 gpt_rs_load_f32(const float* p) {
    return _mm512_loadu_ps(p);
}

static inline __m512 gpt_rs_load_f32_tail(const float* p, size_t left) {
    return _mm512_maskz_loadu_ps((__mmask16)gpt_rs_tail_mask(left), p);
}

static inline __m512 gpt_rs_load_bf16(const uint16_t* p) {
    const __m256i raw = _mm256_loadu_si256((const __m256i*)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

static inline __m512 gpt_rs_load_bf16_tail(const uint16_t* p, size_t left) {
    const __m256i raw = _mm256_maskz_loadu_epi16((__mmask16)gpt_rs_tail_mask(left), p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16));
}

/* f8e4m3 widens exactly through f16. Its bits S.EEEE.MMM move to the f16 fields as
 * S.0EEEE.MMM0000000, which is the value times 2^-8, because the exponent biases are 7 and 15. f16
 * subnormals hold the f8e4m3 subnormals. Sign extension puts the sign in bit 15 after the shift,
 * and the mask clears its copy in bit 14. S.1111.111 is NaN. */
static inline __m512 gpt_rs_widen_f8e4m3(__m128i raw) {
    const __m256i half = _mm256_and_si256(_mm256_slli_epi16(_mm256_cvtepi8_epi16(raw), 7),
                                          _mm256_set1_epi16((short)0xbf80));
    const __m256i magnitude = _mm256_and_si256(half, _mm256_set1_epi16(0x3f80));
    const __mmask16 nan = _mm256_cmpeq_epi16_mask(magnitude, _mm256_set1_epi16(0x3f80));
    const __m512 value = _mm512_mul_ps(_mm512_cvtph_ps(half), _mm512_set1_ps(256.0f));
    return _mm512_mask_mov_ps(value, nan, _mm512_set1_ps(NAN));
}

static inline __m512 gpt_rs_load_f8e4m3(const uint8_t* p) {
    return gpt_rs_widen_f8e4m3(_mm_loadu_si128((const __m128i*)p));
}

static inline __m512 gpt_rs_load_f8e4m3_tail(const uint8_t* p, size_t left) {
    return gpt_rs_widen_f8e4m3(_mm_maskz_loadu_epi8((__mmask16)gpt_rs_tail_mask(left), p));
}
#endif

/* Keys that order floats the way IEEE 754-2019 maximum and minimum do: -0.0 below +0.0, and NaN
 * above every number for maximum and below every number for minimum. The unsigned max or min of
 * keys is exact in any order of accumulation, so reductions over keys vectorize. */
static inline uint32_t gpt_rs_f32_to_order_key(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    /* Sets the sign bit of a positive float and flips every bit of a negative one. */
    uint32_t negative = 0u - (bits >> 31);
    return bits ^ (negative | 0x80000000u);
}

static inline uint32_t gpt_rs_f32_to_max_key(float value) {
    return value != value ? UINT32_MAX : gpt_rs_f32_to_order_key(value);
}

static inline uint32_t gpt_rs_f32_to_min_key(float value) {
    return value != value ? 0u : gpt_rs_f32_to_order_key(value);
}

/* Inverts the keys. The NaN keys UINT32_MAX and 0 come back as NaN. */
static inline float gpt_rs_key_to_f32(uint32_t key) {
    uint32_t positive = key >> 31;
    uint32_t bits = key ^ ((positive - 1u) | 0x80000000u);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* The float whose bits are the AND, or the OR, of the bits of two floats. */
static inline float gpt_rs_f32_and_bits(float lhs, float rhs) {
    uint32_t a, b;
    memcpy(&a, &lhs, sizeof(a));
    memcpy(&b, &rhs, sizeof(b));
    a &= b;
    memcpy(&lhs, &a, sizeof(lhs));
    return lhs;
}

static inline float gpt_rs_f32_or_bits(float lhs, float rhs) {
    uint32_t a, b;
    memcpy(&a, &lhs, sizeof(a));
    memcpy(&b, &rhs, sizeof(b));
    a |= b;
    memcpy(&lhs, &a, sizeof(lhs));
    return lhs;
}

/* IEEE 754-2019 maximum: NaN when either operand is NaN, and -0.0 orders below +0.0. Equal
 * operands differ at most in the sign of a zero, and ANDing their bits keeps +0.0. The function
 * is written as selects, so loops over it vectorize. */
static inline float gpt_rs_maximum_f32(float lhs, float rhs) {
    float larger = lhs < rhs ? rhs : lhs;
    float result = lhs == rhs ? gpt_rs_f32_and_bits(lhs, rhs) : larger;
    return rhs != rhs ? rhs : result;
}

/* IEEE 754-2019 minimum: NaN when either operand is NaN, and -0.0 orders below +0.0. Equal
 * operands differ at most in the sign of a zero, and ORing their bits keeps -0.0. The function
 * is written as selects, so loops over it vectorize. */
static inline float gpt_rs_minimum_f32(float lhs, float rhs) {
    float smaller = lhs > rhs ? rhs : lhs;
    float result = lhs == rhs ? gpt_rs_f32_or_bits(lhs, rhs) : smaller;
    return rhs != rhs ? rhs : result;
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
