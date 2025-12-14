use gpt_rs::backend::spec::{
    ComparisonOp, ElementwiseBinaryOp, ElementwiseUnaryOp, Operation, ReduceKind,
};

use crate::targets::{TARGET_CONV2D_NHWC_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1};

pub(crate) const LABEL_BACKEND_MATMUL: &str = "backend.matmul";
pub(crate) const LABEL_BACKEND_DOT_GENERAL: &str = "backend.dot_general";
pub(crate) const LABEL_BACKEND_RESHAPE: &str = "backend.reshape";
pub(crate) const LABEL_BACKEND_SLICE: &str = "backend.slice";
pub(crate) const LABEL_BACKEND_TRANSPOSE: &str = "backend.transpose";
pub(crate) const LABEL_BACKEND_BROADCAST_TO: &str = "backend.broadcast_to";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_ADD: &str = "backend.elementwise_binary.add";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_SUB: &str = "backend.elementwise_binary.sub";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_MUL: &str = "backend.elementwise_binary.mul";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_DIV: &str = "backend.elementwise_binary.div";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_MAXIMUM: &str =
    "backend.elementwise_binary.maximum";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_BINARY_MINIMUM: &str =
    "backend.elementwise_binary.minimum";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_NEG: &str = "backend.elementwise_unary.neg";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_ABS: &str = "backend.elementwise_unary.abs";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_EXP: &str = "backend.elementwise_unary.exp";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_LOG: &str = "backend.elementwise_unary.log";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_TANH: &str = "backend.elementwise_unary.tanh";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_ERF: &str = "backend.elementwise_unary.erf";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_RSQRT: &str = "backend.elementwise_unary.rsqrt";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_UNARY_RECIPROCAL: &str =
    "backend.elementwise_unary.reciprocal";
pub(crate) const LABEL_BACKEND_REDUCE_SUM: &str = "backend.reduce.sum";
pub(crate) const LABEL_BACKEND_REDUCE_MAX: &str = "backend.reduce.max";
pub(crate) const LABEL_BACKEND_REDUCE_MIN: &str = "backend.reduce.min";
pub(crate) const LABEL_BACKEND_COMPARE_EQUAL: &str = "backend.compare.equal";
pub(crate) const LABEL_BACKEND_COMPARE_NOT_EQUAL: &str = "backend.compare.not_equal";
pub(crate) const LABEL_BACKEND_COMPARE_GREATER: &str = "backend.compare.greater";
pub(crate) const LABEL_BACKEND_COMPARE_GREATER_EQUAL: &str = "backend.compare.greater_equal";
pub(crate) const LABEL_BACKEND_COMPARE_LESS: &str = "backend.compare.less";
pub(crate) const LABEL_BACKEND_COMPARE_LESS_EQUAL: &str = "backend.compare.less_equal";
pub(crate) const LABEL_BACKEND_SELECT: &str = "backend.select";
pub(crate) const LABEL_BACKEND_TAKE: &str = "backend.take";
pub(crate) const LABEL_BACKEND_GATHER: &str = "backend.gather";
pub(crate) const LABEL_BACKEND_IOTA: &str = "backend.iota";
pub(crate) const LABEL_BACKEND_SCATTER_ADD: &str = "backend.scatter_add";
pub(crate) const LABEL_BACKEND_SCATTER_REDUCE: &str = "backend.scatter_reduce";
pub(crate) const LABEL_BACKEND_RNG_UNIFORM: &str = "backend.rng_uniform";
pub(crate) const LABEL_BACKEND_RNG_NORMAL: &str = "backend.rng_normal";
pub(crate) const LABEL_BACKEND_TOP_K: &str = "backend.top_k";
pub(crate) const LABEL_BACKEND_PAD: &str = "backend.pad";
pub(crate) const LABEL_BACKEND_CONCAT: &str = "backend.concat";
pub(crate) const LABEL_BACKEND_TILE: &str = "backend.tile";
pub(crate) const LABEL_BACKEND_DYNAMIC_SLICE: &str = "backend.dynamic_slice";
pub(crate) const LABEL_BACKEND_DYNAMIC_UPDATE_SLICE: &str = "backend.dynamic_update_slice";
pub(crate) const LABEL_BACKEND_CAST: &str = "backend.cast";
pub(crate) const LABEL_BACKEND_STOP_GRADIENT: &str = "backend.stop_gradient";
pub(crate) const LABEL_BACKEND_CONSTANT: &str = "backend.constant";
pub(crate) const LABEL_BACKEND_WHILE: &str = "backend.while";
pub(crate) const LABEL_BACKEND_SCAN: &str = "backend.scan";
pub(crate) const LABEL_BACKEND_COND: &str = "backend.cond";
pub(crate) const LABEL_BACKEND_EXTRACT_PATCHES: &str = "backend.extract_patches";
pub(crate) const LABEL_BACKEND_REDUCE_WINDOW: &str = "backend.reduce_window";
pub(crate) const LABEL_BACKEND_SEGMENT_REDUCE: &str = "backend.segment_reduce";
pub(crate) const LABEL_BACKEND_ARGMAX: &str = "backend.argmax";
pub(crate) const LABEL_BACKEND_QUANTIZE: &str = "backend.quantize";
pub(crate) const LABEL_BACKEND_DEQUANTIZE: &str = "backend.dequantize";
pub(crate) const LABEL_BACKEND_REQUANTIZE: &str = "backend.requantize";
pub(crate) const LABEL_BACKEND_CONV2D_NHWC: &str = "backend.conv2d_nhwc";
pub(crate) const LABEL_BACKEND_ELEMENTWISE_FUSED: &str = "backend.elementwise_fused";
pub(crate) const LABEL_BACKEND_CUSTOM_CALL: &str = "backend.custom_call";
pub(crate) const LABEL_BACKEND_OTHER: &str = "backend.other";

pub(crate) const BACKEND_LABELS: &[&str] = &[
    LABEL_BACKEND_MATMUL,
    LABEL_BACKEND_DOT_GENERAL,
    LABEL_BACKEND_RESHAPE,
    LABEL_BACKEND_SLICE,
    LABEL_BACKEND_TRANSPOSE,
    LABEL_BACKEND_BROADCAST_TO,
    LABEL_BACKEND_ELEMENTWISE_BINARY_ADD,
    LABEL_BACKEND_ELEMENTWISE_BINARY_SUB,
    LABEL_BACKEND_ELEMENTWISE_BINARY_MUL,
    LABEL_BACKEND_ELEMENTWISE_BINARY_DIV,
    LABEL_BACKEND_ELEMENTWISE_BINARY_MAXIMUM,
    LABEL_BACKEND_ELEMENTWISE_BINARY_MINIMUM,
    LABEL_BACKEND_ELEMENTWISE_UNARY_NEG,
    LABEL_BACKEND_ELEMENTWISE_UNARY_ABS,
    LABEL_BACKEND_ELEMENTWISE_UNARY_EXP,
    LABEL_BACKEND_ELEMENTWISE_UNARY_LOG,
    LABEL_BACKEND_ELEMENTWISE_UNARY_TANH,
    LABEL_BACKEND_ELEMENTWISE_UNARY_ERF,
    LABEL_BACKEND_ELEMENTWISE_UNARY_RSQRT,
    LABEL_BACKEND_ELEMENTWISE_UNARY_RECIPROCAL,
    LABEL_BACKEND_REDUCE_SUM,
    LABEL_BACKEND_REDUCE_MAX,
    LABEL_BACKEND_REDUCE_MIN,
    LABEL_BACKEND_COMPARE_EQUAL,
    LABEL_BACKEND_COMPARE_NOT_EQUAL,
    LABEL_BACKEND_COMPARE_GREATER,
    LABEL_BACKEND_COMPARE_GREATER_EQUAL,
    LABEL_BACKEND_COMPARE_LESS,
    LABEL_BACKEND_COMPARE_LESS_EQUAL,
    LABEL_BACKEND_SELECT,
    LABEL_BACKEND_TAKE,
    LABEL_BACKEND_GATHER,
    LABEL_BACKEND_IOTA,
    LABEL_BACKEND_SCATTER_ADD,
    LABEL_BACKEND_SCATTER_REDUCE,
    LABEL_BACKEND_RNG_UNIFORM,
    LABEL_BACKEND_RNG_NORMAL,
    LABEL_BACKEND_TOP_K,
    LABEL_BACKEND_PAD,
    LABEL_BACKEND_CONCAT,
    LABEL_BACKEND_TILE,
    LABEL_BACKEND_DYNAMIC_SLICE,
    LABEL_BACKEND_DYNAMIC_UPDATE_SLICE,
    LABEL_BACKEND_CAST,
    LABEL_BACKEND_STOP_GRADIENT,
    LABEL_BACKEND_CONSTANT,
    LABEL_BACKEND_WHILE,
    LABEL_BACKEND_SCAN,
    LABEL_BACKEND_COND,
    LABEL_BACKEND_EXTRACT_PATCHES,
    LABEL_BACKEND_REDUCE_WINDOW,
    LABEL_BACKEND_SEGMENT_REDUCE,
    LABEL_BACKEND_ARGMAX,
    LABEL_BACKEND_QUANTIZE,
    LABEL_BACKEND_DEQUANTIZE,
    LABEL_BACKEND_REQUANTIZE,
    LABEL_BACKEND_CONV2D_NHWC,
    LABEL_BACKEND_ELEMENTWISE_FUSED,
    LABEL_BACKEND_CUSTOM_CALL,
    LABEL_BACKEND_OTHER,
];

pub(crate) fn backend_label_from_str(label: &str) -> Option<&'static str> {
    BACKEND_LABELS.iter().copied().find(|item| *item == label)
}

#[allow(unreachable_patterns)]
pub(crate) fn backend_operation_label(op: &Operation) -> &'static str {
    match op {
        Operation::Reshape(_) => LABEL_BACKEND_RESHAPE,
        Operation::Slice(_) => LABEL_BACKEND_SLICE,
        Operation::Transpose(_) => LABEL_BACKEND_TRANSPOSE,
        Operation::BroadcastTo(_) => LABEL_BACKEND_BROADCAST_TO,
        Operation::DotGeneral(_) => LABEL_BACKEND_DOT_GENERAL,
        Operation::ElementwiseBinary(kind) => match kind {
            ElementwiseBinaryOp::Add => LABEL_BACKEND_ELEMENTWISE_BINARY_ADD,
            ElementwiseBinaryOp::Sub => LABEL_BACKEND_ELEMENTWISE_BINARY_SUB,
            ElementwiseBinaryOp::Mul => LABEL_BACKEND_ELEMENTWISE_BINARY_MUL,
            ElementwiseBinaryOp::Div => LABEL_BACKEND_ELEMENTWISE_BINARY_DIV,
            ElementwiseBinaryOp::Maximum => LABEL_BACKEND_ELEMENTWISE_BINARY_MAXIMUM,
            ElementwiseBinaryOp::Minimum => LABEL_BACKEND_ELEMENTWISE_BINARY_MINIMUM,
        },
        Operation::ElementwiseUnary(kind) => match kind {
            ElementwiseUnaryOp::Neg => LABEL_BACKEND_ELEMENTWISE_UNARY_NEG,
            ElementwiseUnaryOp::Abs => LABEL_BACKEND_ELEMENTWISE_UNARY_ABS,
            ElementwiseUnaryOp::Exp => LABEL_BACKEND_ELEMENTWISE_UNARY_EXP,
            ElementwiseUnaryOp::Log => LABEL_BACKEND_ELEMENTWISE_UNARY_LOG,
            ElementwiseUnaryOp::Tanh => LABEL_BACKEND_ELEMENTWISE_UNARY_TANH,
            ElementwiseUnaryOp::Erf => LABEL_BACKEND_ELEMENTWISE_UNARY_ERF,
            ElementwiseUnaryOp::Rsqrt => LABEL_BACKEND_ELEMENTWISE_UNARY_RSQRT,
            ElementwiseUnaryOp::Reciprocal => LABEL_BACKEND_ELEMENTWISE_UNARY_RECIPROCAL,
        },
        Operation::Reduce(spec) => match spec.kind {
            ReduceKind::Sum => LABEL_BACKEND_REDUCE_SUM,
            ReduceKind::Max => LABEL_BACKEND_REDUCE_MAX,
            ReduceKind::Min => LABEL_BACKEND_REDUCE_MIN,
        },
        Operation::Compare(spec) => match spec.op {
            ComparisonOp::Equal => LABEL_BACKEND_COMPARE_EQUAL,
            ComparisonOp::NotEqual => LABEL_BACKEND_COMPARE_NOT_EQUAL,
            ComparisonOp::Greater => LABEL_BACKEND_COMPARE_GREATER,
            ComparisonOp::GreaterEqual => LABEL_BACKEND_COMPARE_GREATER_EQUAL,
            ComparisonOp::Less => LABEL_BACKEND_COMPARE_LESS,
            ComparisonOp::LessEqual => LABEL_BACKEND_COMPARE_LESS_EQUAL,
        },
        Operation::Select => LABEL_BACKEND_SELECT,
        Operation::Take => LABEL_BACKEND_TAKE,
        Operation::Gather(_) => LABEL_BACKEND_GATHER,
        Operation::Iota(_) => LABEL_BACKEND_IOTA,
        Operation::ScatterAdd(_) => LABEL_BACKEND_SCATTER_ADD,
        Operation::ScatterReduce(_) => LABEL_BACKEND_SCATTER_REDUCE,
        Operation::RngUniform(_) => LABEL_BACKEND_RNG_UNIFORM,
        Operation::RngNormal(_) => LABEL_BACKEND_RNG_NORMAL,
        Operation::TopK(_) => LABEL_BACKEND_TOP_K,
        Operation::Pad(_) => LABEL_BACKEND_PAD,
        Operation::Concat(_) => LABEL_BACKEND_CONCAT,
        Operation::Tile(_) => LABEL_BACKEND_TILE,
        Operation::DynamicSlice(_) => LABEL_BACKEND_DYNAMIC_SLICE,
        Operation::DynamicUpdateSlice(_) => LABEL_BACKEND_DYNAMIC_UPDATE_SLICE,
        Operation::Cast(_) => LABEL_BACKEND_CAST,
        Operation::StopGradient => LABEL_BACKEND_STOP_GRADIENT,
        Operation::Constant(_) => LABEL_BACKEND_CONSTANT,
        Operation::While(_) => LABEL_BACKEND_WHILE,
        Operation::Scan(_) => LABEL_BACKEND_SCAN,
        Operation::Cond(_) => LABEL_BACKEND_COND,
        Operation::ExtractPatches(_) => LABEL_BACKEND_EXTRACT_PATCHES,
        Operation::ReduceWindow(_) => LABEL_BACKEND_REDUCE_WINDOW,
        Operation::SegmentReduce(_) => LABEL_BACKEND_SEGMENT_REDUCE,
        Operation::ArgMax(_) => LABEL_BACKEND_ARGMAX,
        Operation::Quantize(_) => LABEL_BACKEND_QUANTIZE,
        Operation::Dequantize(_) => LABEL_BACKEND_DEQUANTIZE,
        Operation::Requantize(_) => LABEL_BACKEND_REQUANTIZE,
        Operation::CustomCall(spec) => match spec.target.as_str() {
            TARGET_CONV2D_NHWC_F32_V1 => LABEL_BACKEND_CONV2D_NHWC,
            TARGET_ELEMENTWISE_FUSED_F32_V1 => LABEL_BACKEND_ELEMENTWISE_FUSED,
            _ => LABEL_BACKEND_CUSTOM_CALL,
        },
        _ => LABEL_BACKEND_OTHER,
    }
}
