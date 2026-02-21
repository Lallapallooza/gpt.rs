use serde::{Deserialize, Serialize};

pub const EWISE_BINARY_KERNEL_ID: &str = "gpt_rs.triton.kernel.elementwise_binary_f32.v1";
pub const EWISE_BINARY_SYMBOL: &str = "gpt_rs_triton_ewise_binary_f32";
pub const EWISE_UNARY_KERNEL_ID: &str = "gpt_rs.triton.kernel.elementwise_unary_f32.v1";
pub const EWISE_UNARY_SYMBOL: &str = "gpt_rs_triton_ewise_unary_f32";
pub const PREPACKED_EWISE_BINARY_KERNEL: &str =
    include_str!("kernels/prepacked/elementwise_binary_f32.triton");
pub const PREPACKED_EWISE_UNARY_KERNEL: &str =
    include_str!("kernels/prepacked/elementwise_unary_f32.triton");
pub const PREPACKED_MATMUL_KERNEL: &str = include_str!("kernels/prepacked/matmul_f32.triton");
pub const PREPACKED_REDUCE_SUM_LAST_AXIS_KERNEL: &str =
    include_str!("kernels/prepacked/reduce_sum_last_axis_f32.triton");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub kind: KernelKind,
    pub source: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelKind {
    ElementwiseBinaryF32,
    ElementwiseUnaryF32,
}

pub fn elementwise_binary_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EWISE_BINARY_KERNEL_ID.to_string(),
        kind: KernelKind::ElementwiseBinaryF32,
        source: elementwise_binary_triton_source(),
        symbol: EWISE_BINARY_SYMBOL.to_string(),
    }
}

pub fn elementwise_unary_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EWISE_UNARY_KERNEL_ID.to_string(),
        kind: KernelKind::ElementwiseUnaryF32,
        source: elementwise_unary_triton_source(),
        symbol: EWISE_UNARY_SYMBOL.to_string(),
    }
}

pub fn elementwise_binary_triton_source() -> String {
    PREPACKED_EWISE_BINARY_KERNEL.to_string()
}

pub fn elementwise_unary_triton_source() -> String {
    PREPACKED_EWISE_UNARY_KERNEL.to_string()
}

pub fn prepacked_kernel_sources() -> &'static [&'static str] {
    &[
        PREPACKED_EWISE_BINARY_KERNEL,
        PREPACKED_EWISE_UNARY_KERNEL,
        PREPACKED_MATMUL_KERNEL,
        PREPACKED_REDUCE_SUM_LAST_AXIS_KERNEL,
    ]
}
