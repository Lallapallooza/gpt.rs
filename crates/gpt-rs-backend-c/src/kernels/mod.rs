use gpt_rs_ir::float8::F8E4M3;

/// Minimum work at which kernels and generated loops split across OpenMP threads. Work counts the
/// multiply-accumulates of a matmul or the output elements of an elementwise loop. Smaller ops
/// finish before a fork/join would pay off. The kernels see it as `GPTRS_PARALLEL_MIN_WORK`.
pub const PARALLEL_MIN_WORK: usize = 1 << 16;

/// Minimum rows of x at which f32 linear-layout dots (`x [m, k] . w [n, k]`) use the packed GEMM.
/// Packing costs about one pass over the n x k weights, spread over the m rows. The dot-product
/// kernels instead pay a 16-lane horizontal reduction for each output and K block. The kernels see
/// it as `GPTRS_LIN_PACK_MIN_ROWS`.
pub const LINEAR_PACK_MIN_ROWS: usize = 64;

/// Columns of a packed GEMM panel, the width of the widest micro-kernel (6 x 64). The GEMM reads a
/// packed `k x n` matrix as `[n / PANEL_N, k, PANEL_N]`: element `(p, j)` is at
/// `[j / PANEL_N, p, j % PANEL_N]`. The kernels see it as `GPTRS_PANEL_N`.
pub const PANEL_N: usize = 64;

const KERNELS: &str = concat!(
    include_str!("preamble.inc.c"),
    include_str!("random.inc.c"),
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
         #define GPTRS_LIN_PACK_MIN_ROWS ((size_t){LINEAR_PACK_MIN_ROWS})\n\
         #define GPTRS_PANEL_N ((size_t){PANEL_N})\n{}{KERNELS}",
        f8e4m3_table()
    )
}

/// The kernels' table `gpt_rs_f8e4m3_bits`: the f32 bit pattern of each of the 256
/// `float8_e4m3fn` values, indexed by its byte.
fn f8e4m3_table() -> String {
    let bits: Vec<String> = (0..=u8::MAX)
        .map(|byte| format!("0x{:08x}u", F8E4M3(byte).to_f32().to_bits()))
        .collect();
    format!(
        "static const uint32_t gpt_rs_f8e4m3_bits[256] = {{{}}};\n",
        bits.join(", ")
    )
}
