use super::{filters, OperationView};
use crate::backend::{
    index::InstId,
    rewriter::ProgramRewriter,
    spec::{
        BroadcastToSpec, CastSpec, ConcatSpec, CustomCallSpec, DotGeneralSpec,
        DynamicUpdateSliceSpec, ElementwiseBinaryOp, ElementwiseUnaryOp, ExtractPatchesSpec,
        Operand, Operation, ReduceKind, ReduceSpec, ReduceWindowSpec, ReshapeSpec, SliceSpec,
        TransposeSpec, ValueId, ValueType,
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
pub struct AnyOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub op: Operation,
}

impl AnyOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        let op = rewriter.op(root).clone();
        Some(Self {
            root,
            operands,
            result,
            result_type,
            op,
        })
    }
}

impl OperationView for AnyOpView {
    const MATCHER: super::OperationMatcher = filters::any;

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
pub struct ConcatOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: ConcatSpec,
}

impl ConcatOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Concat(spec) => Some(Self {
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

impl OperationView for ConcatOpView {
    const MATCHER: super::OperationMatcher = filters::concat;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct TakeOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
}

impl TakeOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::Take => Some(Self {
                root,
                operands,
                result,
                result_type,
            }),
            _ => None,
        }
    }
}

impl OperationView for TakeOpView {
    const MATCHER: super::OperationMatcher = filters::take;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct DynamicUpdateSliceOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: DynamicUpdateSliceSpec,
}

impl DynamicUpdateSliceOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::DynamicUpdateSlice(spec) => Some(Self {
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

impl OperationView for DynamicUpdateSliceOpView {
    const MATCHER: super::OperationMatcher = filters::dynamic_update_slice;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

#[derive(Clone)]
pub struct ReduceWindowOpView {
    pub root: InstId,
    pub operands: Vec<Operand>,
    pub result: ValueId,
    pub result_type: ValueType,
    pub spec: ReduceWindowSpec,
}

impl ReduceWindowOpView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let operands = rewriter.operands(root).to_vec();
        let result = rewriter.value_of(root);
        let result_type = rewriter.type_of(result)?.clone();
        match rewriter.op(root) {
            Operation::ReduceWindow(spec) => Some(Self {
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

impl OperationView for ReduceWindowOpView {
    const MATCHER: super::OperationMatcher = filters::reduce_window;

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

#[derive(Clone)]
pub struct SoftmaxDecompositionView {
    pub output: DivOpView,
    pub exp_values: ExpOpView,
    pub shifted: SubOpView,
    pub sum: ReduceSumOpView,
    pub max: ReduceMaxOpView,
    pub sum_broadcast: Option<BroadcastOpView>,
    pub max_broadcast: Option<BroadcastOpView>,
}

impl SoftmaxDecompositionView {
    pub fn new(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        let output = DivOpView::new(root, rewriter)?;
        let div_lhs = value_operand(output.operands.first())?;
        let div_rhs = value_operand(output.operands.get(1))?;

        let exp_values = ExpOpView::new(rewriter.inst_of(div_lhs)?, rewriter)?;
        let exp_input = value_operand(exp_values.operands.first())?;

        let (sum, sum_broadcast) = reduce_sum_chain(div_rhs, rewriter)?;
        let sum_input = value_operand(sum.operands.first())?;
        if sum_input != exp_values.result {
            return None;
        }

        let shifted = SubOpView::new(rewriter.inst_of(exp_input)?, rewriter)?;
        let shifted_lhs = value_operand(shifted.operands.first())?;
        let shifted_rhs = value_operand(shifted.operands.get(1))?;

        let (max, max_broadcast) = reduce_max_chain(shifted_rhs, rewriter)?;
        let max_input = value_operand(max.operands.first())?;
        if shifted_lhs != max_input {
            return None;
        }

        Some(Self {
            output,
            exp_values,
            shifted,
            sum,
            max,
            sum_broadcast,
            max_broadcast,
        })
    }
}

impl OperationView for SoftmaxDecompositionView {
    const MATCHER: super::OperationMatcher = filters::div;

    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self> {
        Self::new(root, rewriter)
    }
}

fn reduce_sum_chain(
    value: ValueId,
    rewriter: &ProgramRewriter,
) -> Option<(ReduceSumOpView, Option<BroadcastOpView>)> {
    reduce_chain(value, rewriter, ReduceSumOpView::new)
}

fn reduce_max_chain(
    value: ValueId,
    rewriter: &ProgramRewriter,
) -> Option<(ReduceMaxOpView, Option<BroadcastOpView>)> {
    reduce_chain(value, rewriter, ReduceMaxOpView::new)
}

fn reduce_chain<R>(
    value: ValueId,
    rewriter: &ProgramRewriter,
    reduce_ctor: fn(InstId, &ProgramRewriter) -> Option<R>,
) -> Option<(R, Option<BroadcastOpView>)> {
    let producer = rewriter.inst_of(value)?;
    if let Some(broadcast) = BroadcastOpView::new(producer, rewriter) {
        let reduce_value = value_operand(broadcast.operands.first())?;
        let reduce_inst = rewriter.inst_of(reduce_value)?;
        let reduce = reduce_ctor(reduce_inst, rewriter)?;
        return Some((reduce, Some(broadcast)));
    }

    let reduce = reduce_ctor(producer, rewriter)?;
    Some((reduce, None))
}

fn value_operand(operand: Option<&Operand>) -> Option<ValueId> {
    match operand {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}
