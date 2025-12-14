use gpt_rs::backend::spec::{ElementwiseBinaryOp, ElementwiseUnaryOp};

pub const TARGET_CONV2D_NHWC_F32_V1: &str = "gpt_rs.c.conv2d.nhwc.f32.v1";
pub const TARGET_ELEMENTWISE_FUSED_F32_V1: &str = "gpt_rs.c.fused_elementwise.f32.v1";

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
