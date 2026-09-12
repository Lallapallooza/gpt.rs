//! Backend-private custom-call targets that C pipeline passes create and C codegen consumes.

use gpt_rs::backend::spec::{ElementwiseBinaryOp, ElementwiseUnaryOp};

/// NHWC f32 convolution over an im2col-shaped weight.
pub const TARGET_CONV2D_NHWC_F32_V1: &str = "gpt_rs.c.conv2d.nhwc.f32.v1";
/// Fused elementwise kernel with f32 inputs and an f32 or bf16 output. The kernel reads each input
/// broadcast or at a slice offset.
pub const TARGET_ELEMENTWISE_FUSED: &str = "gpt_rs.c.fused_elementwise.v2";
/// Linear projection `x: f32 [M, K] . w: bf16 [N, K] -> f32 [M, N]`. The kernel widens the weight
/// to f32 in registers.
pub const TARGET_LINEAR_NT_F32_BF16: &str = "gpt_rs.c.linear_nt.f32_bf16.v1";

/// Whether the kernel can read a fused elementwise input of shape `input` for every element of
/// `out`. When `starts` is set, the input is an absorbed `slice`, and the kernel reads it at an
/// offset inside a same-rank source. Otherwise the kernel broadcasts it right-aligned.
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

pub fn unary_code(op: ElementwiseUnaryOp) -> i64 {
    match op {
        ElementwiseUnaryOp::Neg => 0,
        ElementwiseUnaryOp::Abs => 1,
        ElementwiseUnaryOp::Exp => 2,
        ElementwiseUnaryOp::Log => 3,
        ElementwiseUnaryOp::Tanh => 4,
        ElementwiseUnaryOp::Erf => 5,
        ElementwiseUnaryOp::Rsqrt => 6,
        ElementwiseUnaryOp::Reciprocal => 7,
    }
}

pub fn binary_code(op: ElementwiseBinaryOp) -> i64 {
    match op {
        ElementwiseBinaryOp::Add => 0,
        ElementwiseBinaryOp::Sub => 1,
        ElementwiseBinaryOp::Mul => 2,
        ElementwiseBinaryOp::Div => 3,
        ElementwiseBinaryOp::Maximum => 4,
        ElementwiseBinaryOp::Minimum => 5,
    }
}

pub fn unary_expr_from_code(code: i64, arg: &str) -> Option<String> {
    let expr = match code {
        0 => format!("-({arg})"),
        1 => format!("fabsf({arg})"),
        2 => format!("expf({arg})"),
        3 => format!("logf({arg})"),
        4 => format!("tanhf({arg})"),
        5 => format!("erff({arg})"),
        6 => format!("1.0f / sqrtf({arg})"),
        7 => format!("1.0f / ({arg})"),
        _ => return None,
    };
    Some(expr)
}

pub fn binary_expr_from_code(code: i64, lhs: &str, rhs: &str) -> Option<String> {
    let expr = match code {
        0 => format!("({lhs}) + ({rhs})"),
        1 => format!("({lhs}) - ({rhs})"),
        2 => format!("({lhs}) * ({rhs})"),
        3 => format!("({lhs}) / ({rhs})"),
        4 => format!("({lhs}) > ({rhs}) ? ({lhs}) : ({rhs})"),
        5 => format!("({lhs}) < ({rhs}) ? ({lhs}) : ({rhs})"),
        _ => return None,
    };
    Some(expr)
}
