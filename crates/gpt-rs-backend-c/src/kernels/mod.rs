/// Minimum work at which kernels and generated loops split across OpenMP threads. Work counts the
/// multiply-accumulates of a matmul or the output elements of an elementwise loop. Smaller ops
/// finish before a fork/join would pay off. The kernels see it as `GPTRS_PARALLEL_MIN_WORK`.
pub const PARALLEL_MIN_WORK: usize = 1 << 16;

/// Minimum rows of x at which f32 linear-layout dots (`x [m, k] . w [n, k]`) use the packed GEMM.
/// Packing costs about one pass over the n x k weights, spread over the m rows. The dot-product
/// kernels instead pay a 16-lane horizontal reduction for each output and K block. The kernels see
/// it as `GPTRS_LIN_PACK_MIN_ROWS`.
pub const LINEAR_PACK_MIN_ROWS: usize = 64;

const KERNELS: &str = concat!(
    include_str!("preamble.inc.c"),
    include_str!("pack.inc.c"),
    include_str!("ukernel_16.inc.c"),
    include_str!("ukernel_32.inc.c"),
    include_str!("ukernel_48.inc.c"),
    include_str!("ukernel_64.inc.c"),
    include_str!("ukernel_misc.inc.c"),
    include_str!("compute.inc.c"),
    include_str!("api.inc.c"),
    include_str!("linear.inc.c")
);

pub fn emit_c_kernels() -> String {
    format!(
        "#define GPTRS_PARALLEL_MIN_WORK ((size_t){PARALLEL_MIN_WORK})\n\
         #define GPTRS_LIN_PACK_MIN_ROWS ((size_t){LINEAR_PACK_MIN_ROWS})\n{KERNELS}"
    )
}
