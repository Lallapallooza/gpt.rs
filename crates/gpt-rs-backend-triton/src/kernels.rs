use crate::bundle::{KernelKind, KernelSpec};

pub const EWISE_BINARY_KERNEL_ID: &str = "gpt_rs.triton.kernel.elementwise_binary_f32.v1";
pub const EWISE_BINARY_SYMBOL: &str = "gpt_rs_triton_ewise_binary_f32";

pub fn elementwise_binary_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EWISE_BINARY_KERNEL_ID.to_string(),
        kind: KernelKind::ElementwiseBinaryF32,
        source: elementwise_binary_triton_source(),
        symbol: EWISE_BINARY_SYMBOL.to_string(),
    }
}

pub fn elementwise_binary_triton_source() -> String {
    r#"// gpt_rs.kernel: elementwise_binary_f32
// gpt_rs.symbol: gpt_rs_triton_ewise_binary_f32
// args: lhs_ptr(*fp32), rhs_ptr(*fp32), out_ptr(*fp32), n(u32), op(u32)
// op mapping: 0=add,1=sub,2=mul,3=div,4=max,5=min
"#
    .to_string()
}
