//! The C backend's kernel tables: which kernel reads which dtypes, and the C expressions of the
//! elementwise ops.

use gpt_rs_ir::dialects::tensor::{BinaryKind, UnaryKind};
use gpt_rs_ir::types::DType;

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

/// The C expression of `kind` applied to the f32 expression `arg`.
pub fn unary_expr(kind: UnaryKind, arg: &str) -> String {
    match kind {
        UnaryKind::Neg => format!("-({arg})"),
        UnaryKind::Abs => format!("fabsf({arg})"),
        UnaryKind::Exp => format!("expf({arg})"),
        UnaryKind::Log => format!("logf({arg})"),
        UnaryKind::Tanh => format!("tanhf({arg})"),
        UnaryKind::Erf => format!("erff({arg})"),
        UnaryKind::Rsqrt => format!("1.0f / sqrtf({arg})"),
        UnaryKind::Reciprocal => format!("1.0f / ({arg})"),
        UnaryKind::Sin => format!("sinf({arg})"),
        UnaryKind::Cos => format!("cosf({arg})"),
        UnaryKind::Not => unreachable!("not flips the bits of integers, not of f32"),
    }
}

/// The C expression of `kind` applied to the f32 expressions `lhs` and `rhs`.
pub fn binary_expr(kind: BinaryKind, lhs: &str, rhs: &str) -> String {
    match kind {
        BinaryKind::Add => format!("({lhs}) + ({rhs})"),
        BinaryKind::Sub => format!("({lhs}) - ({rhs})"),
        BinaryKind::Mul => format!("({lhs}) * ({rhs})"),
        BinaryKind::Div => format!("({lhs}) / ({rhs})"),
        BinaryKind::Maximum => format!("gpt_rs_maximum_f32({lhs}, {rhs})"),
        BinaryKind::Minimum => format!("gpt_rs_minimum_f32({lhs}, {rhs})"),
        BinaryKind::And
        | BinaryKind::Or
        | BinaryKind::Xor
        | BinaryKind::ShiftLeft
        | BinaryKind::ShiftRightLogical
        | BinaryKind::ShiftRightArithmetic => {
            unreachable!("bitwise ops compute on integers, not on f32")
        }
    }
}
