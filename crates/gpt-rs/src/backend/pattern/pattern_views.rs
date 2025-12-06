use super::{filters, OperationView};
use crate::backend::{
    index::InstId,
    rewriter::ProgramRewriter,
    spec::{
        BroadcastToSpec, CastSpec, CustomCallSpec, DotGeneralSpec, ElementwiseBinaryOp,
        ElementwiseUnaryOp, ExtractPatchesSpec, Operand, Operation, ReduceKind, ReduceSpec,
        ReshapeSpec, SliceSpec, TransposeSpec, ValueId, ValueType,
    },
};

#[derive(Clone)]
pub struct CastOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: CastSpec,
}

impl CastOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Cast(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for CastOpView {
    const MATCHER: super::OperationMatcher = filters::cast;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct BroadcastOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: BroadcastToSpec,
}

impl BroadcastOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::BroadcastTo(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for BroadcastOpView {
    const MATCHER: super::OperationMatcher = filters::broadcast_to;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct ElementwiseBinaryOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub op: ElementwiseBinaryOp,
}

impl ElementwiseBinaryOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::ElementwiseBinary(op) => Some(Self {
                root,
                operands,
                result,
                result_type,
                op: *op,
            }),
            _ => None,
        }
    }
}

impl OperationView for ElementwiseBinaryOpView {
    const MATCHER: super::OperationMatcher = filters::elementwise_binary;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct ElementwiseUnaryOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub op: ElementwiseUnaryOp,
}

impl ElementwiseUnaryOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::ElementwiseUnary(op) => Some(Self {
                root,
                operands,
                result,
                result_type,
                op: *op,
            }),
            _ => None,
        }
    }
}

impl OperationView for ElementwiseUnaryOpView {
    const MATCHER: super::OperationMatcher = filters::elementwise_unary;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

macro_rules! define_elementwise_binary_view {
    ($name:ident, $filter:ident, $kind:expr) => {
        #[derive(Clone)]
        pub struct $name {
            pub root: InstId,
            pub operands: Vec<Operand>,
            pub result: ValueId,
            pub result_type: ValueType,
        }

        impl $name {
            pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                let operands = rewriter.operands(root).to_vec();
                let result = rewriter.value_of(root);
                let result_type = rewriter.type_of(result)?.clone();
                match rewriter.op(root) {
                    Operation::ElementwiseBinary(op) if *op == $kind => Some(Self {
                        root,
                        operands,
                        result,
                        result_type,
                    }),
                    _ => None,
                }
            }
        }

        impl OperationView for $name {
            const MATCHER: super::OperationMatcher = filters::$filter;

            fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                Self::new(root, rewriter)
            }
        }
    };
}

define_elementwise_binary_view!(AddOpView, add, ElementwiseBinaryOp::Add);
define_elementwise_binary_view!(SubOpView, sub, ElementwiseBinaryOp::Sub);
define_elementwise_binary_view!(MulOpView, mul, ElementwiseBinaryOp::Mul);
define_elementwise_binary_view!(DivOpView, div, ElementwiseBinaryOp::Div);
define_elementwise_binary_view!(MaximumOpView, maximum, ElementwiseBinaryOp::Maximum);
define_elementwise_binary_view!(MinimumOpView, minimum, ElementwiseBinaryOp::Minimum);

macro_rules! define_elementwise_unary_view {
    ($name:ident, $filter:ident, $kind:expr) => {
        #[derive(Clone)]
        pub struct $name {
            pub root: InstId,
            pub operands: Vec<Operand>,
            pub result: ValueId,
            pub result_type: ValueType,
        }

        impl $name {
            pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                let operands = rewriter.operands(root).to_vec();
                let result = rewriter.value_of(root);
                let result_type = rewriter.type_of(result)?.clone();
                match rewriter.op(root) {
                    Operation::ElementwiseUnary(op) if *op == $kind => Some(Self {
                        root,
                        operands,
                        result,
                        result_type,
                    }),
                    _ => None,
                }
            }
        }

        impl OperationView for $name {
            const MATCHER: super::OperationMatcher = filters::$filter;

            fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                Self::new(root, rewriter)
            }
        }
    };
}

define_elementwise_unary_view!(ExpOpView, exp, ElementwiseUnaryOp::Exp);

#[derive(Clone)]
pub struct ReshapeOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: ReshapeSpec,
}

impl ReshapeOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Reshape(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for ReshapeOpView {
    const MATCHER: super::OperationMatcher = filters::reshape;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct TransposeOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: TransposeSpec,
}

impl TransposeOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Transpose(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for TransposeOpView {
    const MATCHER: super::OperationMatcher = filters::transpose;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct SliceOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: SliceSpec,
}

#[derive(Clone)]
pub struct DotGeneralOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: DotGeneralSpec,
}

impl DotGeneralOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::DotGeneral(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for DotGeneralOpView {
    const MATCHER: super::OperationMatcher = filters::dot_general;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct ExtractPatchesOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: ExtractPatchesSpec,
}

impl ExtractPatchesOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::ExtractPatches(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for ExtractPatchesOpView {
    const MATCHER: super::OperationMatcher = filters::extract_patches;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct CustomCallOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: CustomCallSpec,
}

impl CustomCallOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::CustomCall(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for CustomCallOpView {
    const MATCHER: super::OperationMatcher = filters::custom_call;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

impl SliceOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Slice(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for SliceOpView {
    const MATCHER: super::OperationMatcher = filters::slice;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct ReduceOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: ReduceSpec,
}

impl ReduceOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Reduce(spec) => Some(Self {
                root,
                operands,
                result,
                result_type,
                spec: spec.clone(),
            }),
            _ => None,
        }
    }
}

impl OperationView for ReduceOpView {
    const MATCHER: super::OperationMatcher = filters::reduce;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

macro_rules! define_reduce_view {
    ($name:ident, $filter:ident, $kind:expr) => {
        #[derive(Clone)]
        pub struct $name {
            pub root: InstId,
            pub operands: Vec<Operand>,
            pub result: ValueId,
            pub result_type: ValueType,
            pub spec: ReduceSpec,
        }

        impl $name {
            pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                let operands = rewriter.operands(root).to_vec();
                let result = rewriter.value_of(root);
                let result_type = rewriter.type_of(result)?.clone();
                match rewriter.op(root) {
                    Operation::Reduce(spec) if spec.kind == $kind => Some(Self {
                        root,
                        operands,
                        result,
                        result_type,
                        spec: spec.clone(),
                    }),
                    _ => None,
                }
            }
        }

        impl OperationView for $name {
            const MATCHER: super::OperationMatcher = filters::$filter;

            fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
                Self::new(root, rewriter)
            }
        }
    };
}

define_reduce_view!(ReduceSumOpView, reduce_sum, ReduceKind::Sum);
define_reduce_view!(ReduceMaxOpView, reduce_max, ReduceKind::Max);
define_reduce_view!(ReduceMinOpView, reduce_min, ReduceKind::Min);
