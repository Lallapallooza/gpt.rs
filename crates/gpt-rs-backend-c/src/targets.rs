//! The C backend's kernel tables: which kernel reads which dtypes, and the C expressions of the
//! elementwise ops.

use gpt_rs_ir::dialects::tensor::TensorOps;
use gpt_rs_ir::ir::DType;

/// The C function that computes `x [.., M, K] . w [.., N, K]` with f32 sums, for an `x` of dtype
/// `x` and weights stored in dtype `w`, dequantized by a scale of dtype `scale` when one is given.
pub fn linear_kernel(x: DType, w: DType, scale: Option<DType>) -> Option<&'static str> {
    use DType::{Bf16, F32, F8E4M3};
    Some(match (x, w, scale) {
        (F32, Bf16, None) => "gpt_rs_c_linear_nt_f32_bf16",
        (F32, F8E4M3, None) => "gpt_rs_c_linear_nt_f32_f8e4m3",
        (F32, F8E4M3, Some(Bf16)) => "gpt_rs_c_linear_nt_f32_f8e4m3_bf16_scale",
        (F32, F8E4M3, Some(F32)) => "gpt_rs_c_linear_nt_f32_f8e4m3_f32_scale",
        (Bf16, Bf16, None) => "gpt_rs_c_linear_nt_bf16_bf16",
        (Bf16, F8E4M3, Some(Bf16)) => "gpt_rs_c_linear_nt_bf16_f8e4m3_bf16_scale",
        (Bf16, F8E4M3, Some(F32)) => "gpt_rs_c_linear_nt_bf16_f8e4m3_f32_scale",
        _ => return None,
    })
}

/// Whether the kernel can read a fused elementwise input of shape `input` for every element of
/// `out`. When `starts` is set, the input is a `slice` that the kernel covers, read at an offset
/// inside a same-rank source. Otherwise the kernel broadcasts it right-aligned.
pub fn fused_input_fits(out: &[usize], input: &[usize], starts: Option<&[usize]>) -> bool {
    match starts {
        Some(starts) => {
            input.len() == out.len()
                && starts.len() == out.len()
                && starts
                    .iter()
                    .zip(out)
                    .zip(input)
                    .all(|((start, size), dim)| {
                        start.checked_add(*size).is_some_and(|end| end <= *dim)
                    })
        }
        None => {
            input.len() <= out.len()
                && input
                    .iter()
                    .zip(&out[out.len() - input.len()..])
                    .all(|(dim, out_dim)| *dim == 1 || dim == out_dim)
        }
    }
}

/// The C expression of the elementwise `op` on the f32 expression `arg`, when `op` has one.
pub fn unary_expr(op: &TensorOps, arg: &str) -> Option<String> {
    Some(match op {
        TensorOps::Neg(_) => format!("-({arg})"),
        TensorOps::Abs(_) => format!("fabsf({arg})"),
        TensorOps::Exp(_) => format!("expf({arg})"),
        TensorOps::Log(_) => format!("logf({arg})"),
        TensorOps::Tanh(_) => format!("tanhf({arg})"),
        TensorOps::Erf(_) => format!("erff({arg})"),
        TensorOps::Rsqrt(_) => format!("1.0f / sqrtf({arg})"),
        TensorOps::Reciprocal(_) => format!("1.0f / ({arg})"),
        TensorOps::Sin(_) => format!("sinf({arg})"),
        TensorOps::Cos(_) => format!("cosf({arg})"),
        _ => return None,
    })
}

/// The C expression of the elementwise `op` on the f32 expressions `lhs` and `rhs`, when `op`
/// has one.
pub fn binary_expr(op: &TensorOps, lhs: &str, rhs: &str) -> Option<String> {
    Some(match op {
        TensorOps::Add(_) => format!("({lhs}) + ({rhs})"),
        TensorOps::Sub(_) => format!("({lhs}) - ({rhs})"),
        TensorOps::Mul(_) => format!("({lhs}) * ({rhs})"),
        TensorOps::Div(_) => format!("({lhs}) / ({rhs})"),
        TensorOps::Maximum(_) => format!("gpt_rs_maximum_f32({lhs}, {rhs})"),
        TensorOps::Minimum(_) => format!("gpt_rs_minimum_f32({lhs}, {rhs})"),
        _ => return None,
    })
}
