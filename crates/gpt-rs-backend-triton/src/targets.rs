use gpt_rs::backend::spec::{ElementwiseBinaryOp, ElementwiseUnaryOp};

pub const TARGET_ELEMENTWISE_FUSED_F32_V1: &str = "gpt_rs.triton.fused_elementwise.f32.v1";
pub const FUSION_ATTR_VERSION: &str = "fusion_version";
pub const FUSION_ATTR_KIND: &str = "fusion_kind";
pub const FUSION_KIND_ELEMENTWISE_DAG_V1: &str = "elementwise_dag_v1";

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
