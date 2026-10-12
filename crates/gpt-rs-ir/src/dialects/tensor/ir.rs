//! The `tensor` dialect's ops and their attributes.

use std::collections::BTreeMap;
use std::ops::Deref;

use crate::attr::F64;
use crate::dense::Dense;
use crate::dialects::arith;
use crate::scalar::Combiner;
use crate::traits::{
    Absorbing, Associative, Commutative, DistributesOver, Elementwise, Idempotent, IntValue,
    Involution, Movement, RightIdentity, Terminator, NEG_ZERO,
};
use crate::types::{DType, Dim, Perm, Ranked, TensorType};

// ------------------------------------------------------------------------------------------------
// Constants.

/// `tensor.constant`: a tensor whose elements are an attribute.
///
/// - The result has the dtype and the static shape of its `value` attribute.
/// - The literal payload is interpreted as row-major, little-endian scalar storage, one byte per
///   `i1` element. Backends that do not support a dtype MAY reject the program as unimplemented.
/// - In the text form, elements narrower than a byte are written in hex, packed as tensors store
///   them.
#[crate::op("tensor.constant", results(output(Tensor)), constant)]
pub struct Constant {
    pub value: Dense,
}

// ------------------------------------------------------------------------------------------------
// Elementwise ops: each result element depends only on the operand elements at its position.
// Their operands have equal shapes: nothing broadcasts implicitly.

/// `tensor.neg`, `tensor.exp`, ...: applies the function of the op to every element.
///
/// - The result has the input's shape and dtype.
#[crate::family(
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Elementwise),
    rule(crate::dialects::tensor::verify::elementwise::unary),
    eval(crate::dialects::tensor::interp::elementwise::unary)
)]
pub trait Unary {
    /// The `arith` op that it applies to each element.
    type Scalar: arith::Unary + Default;
}

/// `tensor.neg`: `-x`. Integers wrap, so the negation of the smallest signed integer is itself.
#[crate::op("tensor.neg", family(Unary, Scalar(arith::Neg)), traits(Involution))]
pub struct Neg;

/// `tensor.abs`: `abs(x)`. Integers wrap, so the absolute value of the smallest signed integer is
/// itself.
#[crate::op("tensor.abs", family(Unary, Scalar(arith::Abs)), traits(Idempotent))]
pub struct Abs;

/// `tensor.exp`: `e^x`.
#[crate::op("tensor.exp", family(Unary, Scalar(arith::Exp)))]
pub struct Exp;

/// `tensor.log`: The natural logarithm, `ln(x)`.
#[crate::op("tensor.log", family(Unary, Scalar(arith::Log)))]
pub struct Log;

/// `tensor.tanh`: The hyperbolic tangent.
#[crate::op("tensor.tanh", family(Unary, Scalar(arith::Tanh)))]
pub struct Tanh;

/// `tensor.erf`: The error function.
#[crate::op("tensor.erf", family(Unary, Scalar(arith::Erf)))]
pub struct Erf;

/// `tensor.rsqrt`: `1 / sqrt(x)`.
#[crate::op("tensor.rsqrt", family(Unary, Scalar(arith::Rsqrt)))]
pub struct Rsqrt;

/// `tensor.reciprocal`: `1 / x`.
#[crate::op("tensor.reciprocal", family(Unary, Scalar(arith::Reciprocal)))]
pub struct Reciprocal;

/// `tensor.sin`: The sine of `x` radians.
#[crate::op("tensor.sin", family(Unary, Scalar(arith::Sin)))]
pub struct Sin;

/// `tensor.cos`: The cosine of `x` radians.
#[crate::op("tensor.cos", family(Unary, Scalar(arith::Cos)))]
pub struct Cos;

/// `tensor.not`: Flips every bit; on `i1`, logical not.
#[crate::op("tensor.not", family(Unary, Scalar(arith::Not)), traits(Involution))]
pub struct Not;

/// `tensor.add`, `tensor.mul`, ...: combines two tensors of equal type element by element with the
/// function of the op: `out[i] = op(lhs[i], rhs[i])`.
///
/// - `lhs` and `rhs` MUST have the same dtype and the same shape, which the result has.
/// - Integer results wrap modulo `2^bits`, two's complement for the signed dtypes.
/// - A NaN result has no specified sign or payload.
/// - The identities of their traits hold for every float, `-0.0` included: `x + 0.0` is not `x`,
///   since `-0.0 + 0.0` is `0.0`. Floats have no absorbing element: `inf * 0` and
///   `maximum(NaN, inf)` are NaN.
#[crate::family(
    operands(lhs(Tensor), rhs(Tensor)),
    results(output(Tensor)),
    traits(Elementwise),
    rule(crate::dialects::tensor::verify::elementwise::binary),
    eval(crate::dialects::tensor::interp::elementwise::binary)
)]
pub trait Binary {
    /// The `arith` op that it applies to each pair of elements.
    type Scalar: arith::Binary + Default;
}

/// `tensor.add`: `lhs + rhs`.
#[crate::op(
    "tensor.add",
    family(Binary, Scalar(arith::Add)),
    traits(
        Commutative,
        Associative::INT,
        RightIdentity::new(NEG_ZERO, IntValue::Zero)
    )
)]
pub struct Add;

/// `tensor.sub`: `lhs - rhs`.
#[crate::op(
    "tensor.sub",
    family(Binary, Scalar(arith::Sub)),
    traits(RightIdentity::new(0.0, IntValue::Zero))
)]
pub struct Sub;

/// `tensor.mul`: `lhs * rhs`.
#[crate::op(
    "tensor.mul",
    family(Binary, Scalar(arith::Mul)),
    traits(
        Commutative,
        Associative::INT,
        RightIdentity::new(1.0, IntValue::One),
        Absorbing::int(IntValue::Zero),
        DistributesOver::int(Add)
    )
)]
pub struct Mul;

/// `tensor.div`: `lhs / rhs`. Integer division truncates toward zero, and backends MUST reject an
/// integer division by zero as a spec violation. Float division follows IEEE 754, so it can give
/// Inf or NaN.
#[crate::op(
    "tensor.div",
    family(Binary, Scalar(arith::Div)),
    traits(RightIdentity::new(1.0, IntValue::One))
)]
pub struct Div;

/// `tensor.maximum`: The larger operand. A float NaN operand gives NaN, and `-0.0` orders below
/// `0.0`.
#[crate::op(
    "tensor.maximum",
    family(Binary, Scalar(arith::Maximum)),
    traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::NEG_INFINITY, IntValue::Min),
        Absorbing::int(IntValue::Max)
    )
)]
pub struct Maximum;

/// `tensor.minimum`: The smaller operand. A float NaN operand gives NaN, and `-0.0` orders below
/// `0.0`.
#[crate::op(
    "tensor.minimum",
    family(Binary, Scalar(arith::Minimum)),
    traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::INFINITY, IntValue::Max),
        Absorbing::int(IntValue::Min)
    )
)]
pub struct Minimum;

/// `tensor.and`: Bitwise and; on `i1`, logical and.
#[crate::op(
    "tensor.and",
    family(Binary, Scalar(arith::And)),
    traits(
        Commutative,
        Associative::INT,
        Idempotent,
        RightIdentity::int(IntValue::AllOnes),
        Absorbing::int(IntValue::Zero),
        DistributesOver::int(Or)
    )
)]
pub struct And;

/// `tensor.or`: Bitwise or; on `i1`, logical or.
#[crate::op(
    "tensor.or",
    family(Binary, Scalar(arith::Or)),
    traits(
        Commutative,
        Associative::INT,
        Idempotent,
        RightIdentity::int(IntValue::Zero),
        Absorbing::int(IntValue::AllOnes),
        DistributesOver::int(And)
    )
)]
pub struct Or;

/// `tensor.xor`: Bitwise exclusive or; on `i1`, logical exclusive or.
#[crate::op(
    "tensor.xor",
    family(Binary, Scalar(arith::Xor)),
    traits(Commutative, Associative::INT, RightIdentity::int(IntValue::Zero))
)]
pub struct Xor;

/// `tensor.shift_left`: `lhs` shifted left by `rhs` bits, shifting in zeros. `rhs` reads as
/// unsigned, and a shift by at least the bit width gives 0.
#[crate::op(
    "tensor.shift_left",
    family(Binary, Scalar(arith::ShiftLeft)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftLeft;

/// `tensor.shift_right_logical`: `lhs` shifted right by `rhs` bits, shifting in zeros, into a
/// signed dtype too. `rhs` reads as unsigned, and a shift by at least the bit width gives 0.
#[crate::op(
    "tensor.shift_right_logical",
    family(Binary, Scalar(arith::ShiftRightLogical)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftRightLogical;

/// `tensor.shift_right_arithmetic`: `lhs` shifted right by `rhs` bits, shifting in copies of the
/// sign bit. `rhs` reads as unsigned, and a shift by at least the bit width gives -1 for a negative
/// `lhs`, else 0.
#[crate::op(
    "tensor.shift_right_arithmetic",
    family(Binary, Scalar(arith::ShiftRightArithmetic)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftRightArithmetic;

/// `tensor.less`, `tensor.equal`, ...: compares two tensors of equal type element by element into
/// an `i1` tensor.
///
/// - `out[i]` is 1 where the relation of the op holds between `lhs[i]` and `rhs[i]`, and 0
///   elsewhere. The result has the operands' shape.
/// - `lhs` and `rhs` MUST have the same dtype and the same shape.
/// - Unsigned integers order as unsigned numbers, and `i1` orders false below true.
/// - Floats compare as IEEE does: every relation but `not_equal` is false if either operand is NaN,
///   and `not_equal` is true.
#[crate::family(
    operands(lhs(Tensor), rhs(Tensor)),
    results(output(Tensor)),
    traits(Elementwise),
    rule(crate::dialects::tensor::verify::elementwise::compare),
    eval(crate::dialects::tensor::interp::elementwise::compare)
)]
pub trait Compare {
    /// The `arith` op that compares each pair of elements.
    type Scalar: arith::Compare + Default;
}

/// `tensor.less`: `lhs < rhs`.
#[crate::op("tensor.less", family(Compare, Scalar(arith::Less)))]
pub struct Less;

/// `tensor.less_equal`: `lhs <= rhs`.
#[crate::op("tensor.less_equal", family(Compare, Scalar(arith::LessEqual)))]
pub struct LessEqual;

/// `tensor.equal`: `lhs == rhs`.
#[crate::op("tensor.equal", family(Compare, Scalar(arith::Equal)))]
pub struct Equal;

/// `tensor.greater_equal`: `lhs >= rhs`.
#[crate::op("tensor.greater_equal", family(Compare, Scalar(arith::GreaterEqual)))]
pub struct GreaterEqual;

/// `tensor.greater`: `lhs > rhs`.
#[crate::op("tensor.greater", family(Compare, Scalar(arith::Greater)))]
pub struct Greater;

/// `tensor.not_equal`: `lhs != rhs`.
#[crate::op("tensor.not_equal", family(Compare, Scalar(arith::NotEqual)))]
pub struct NotEqual;

/// `tensor.select`: picks `on_true` where `pred` is set and `on_false` elsewhere.
///
/// - For each element: `out[i] = pred[i] ? on_true[i] : on_false[i]`.
/// - `pred` is `i1`. All operands MUST have the same shape, and `on_true` and `on_false` MUST have
///   the same dtype, which the result has.
#[crate::op(
    "tensor.select",
    operands(pred(Tensor), on_true(Tensor), on_false(Tensor)),
    results(output(Tensor)),
    traits(Elementwise)
)]
pub struct Select;

/// `tensor.cast`: converts every element to `to`.
///
/// - The shape is preserved.
/// - The conversion rules are those of `cast(x; to=dst)` in "DTypes and casting" of the
///   dialect documentation.
#[crate::op(
    "tensor.cast",
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Elementwise)
)]
pub struct Cast {
    pub to: DType,
}

/// `tensor.stop_gradient`: its input, unchanged, as a derivative barrier.
///
/// - The forward value is identical to the input.
/// - Autodiff MUST treat it as a derivative barrier.
/// - Passes MUST keep it until autodiff has run, including through lowering to lower levels when
///   autodiff runs there, so no pass folds it away. Backends that run no autodiff MAY treat it as a
///   pure no-op.
#[crate::op(
    "tensor.stop_gradient",
    operands(input(Tensor)),
    results(output(Tensor))
)]
pub struct StopGradient;

// ------------------------------------------------------------------------------------------------
// Ops that combine many elements into one.

/// `tensor.dot_general`: a contraction over `contract_*` axes, batched over `batch_*` axes, like
/// an einsum. Each result element sums the products of the `lhs` and `rhs` elements at its batch
/// and free coordinates, over every coordinate of the contracting axes.
///
/// - `batch_lhs` pairs with `batch_rhs`, and `contract_lhs` with `contract_rhs`; the axes of each
///   pair MUST have matching extents. Every axis MUST be in range, a negative axis counting from
///   the end, and an operand's axis MUST appear at most once in its batch and contracting axes.
/// - The result axes are the batch axes, in `batch_lhs` order, then the remaining non-contracting
///   axes of `lhs`, then those of `rhs`, each in axis order.
/// - The operand dtypes MAY differ: integers with integers, and floats or narrow floats with each
///   other. For example, FP8 training multiplies e5m2 gradients by e4m3 weights, and W4A8
///   multiplies int4 weights by int8 activations. Then `accum` and `out` MUST be given.
/// - It accumulates the products in `accum` and casts the sums to `out`. The accumulator policy
///   gives their defaults. A wider `accum` holds the products of narrower inputs exactly, for
///   example bf16 x bf16 with f32 accumulation.
#[crate::op(
    "tensor.dot_general",
    operands(lhs(Tensor), rhs(Tensor)),
    results(output(Tensor))
)]
pub struct DotGeneral {
    #[attr(default)]
    pub batch_lhs: Ranked<i64>,
    #[attr(default)]
    pub batch_rhs: Ranked<i64>,
    pub contract_lhs: Ranked<i64>,
    pub contract_rhs: Ranked<i64>,
    /// The accumulator dtype, when it differs from [`DType::default_accumulator`] of the operands.
    #[attr(default)]
    pub accum: Option<DType>,
    /// The result dtype, when it differs from the operands'.
    #[attr(default)]
    pub out: Option<DType>,
}

/// The attributes of the `tensor.reduce_*` ops.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReduceAttrs {
    pub axes: Ranked<i64>,
    #[attr(default)]
    pub keepdims: bool,
    /// The accumulator dtype, when it differs from [`DType::default_accumulator`] of the input.
    #[attr(default)]
    pub accum: Option<DType>,
    /// The result dtype, when it differs from the input's.
    #[attr(default)]
    pub out: Option<DType>,
}

/// `tensor.reduce_sum`, `tensor.reduce_max`, ...: reduces the input over `axes`, folding its
/// elements with the binary op and from the initial value that the op states.
///
/// - `axes` MUST be unique and in range. A negative axis counts from the end, and the order of
///   `axes` does not matter.
/// - If `keepdims` is true, the reduced axes remain with extent 1; otherwise they are removed.
/// - It accumulates in `accum` and casts the result to `out`. The accumulator policy gives their
///   defaults.
/// - When a reduced axis has extent 0, each result element is the initial value.
#[crate::family(
    operands(input(Tensor)),
    results(output(Tensor)),
    rule(crate::dialects::tensor::verify::contraction::reduce),
    eval(crate::dialects::tensor::interp::contraction::reduce)
)]
pub trait Reduce: Deref<Target = ReduceAttrs> + From<ReduceAttrs> {
    /// The elementwise op that folds the elements.
    type Combiner: Combiner;
}

/// `tensor.reduce_sum`: folds with `add`. The initial value is 0.
#[crate::op("tensor.reduce_sum", family(Reduce, Combiner(Add)))]
pub struct ReduceSum(pub ReduceAttrs);

/// `tensor.reduce_max`: folds with `maximum`. The initial value is `-inf` for floats and the
/// smallest value of the dtype for integers.
#[crate::op("tensor.reduce_max", family(Reduce, Combiner(Maximum)))]
pub struct ReduceMax(pub ReduceAttrs);

/// `tensor.reduce_min`: folds with `minimum`. The initial value is `+inf` for floats and the
/// largest value of the dtype for integers.
#[crate::op("tensor.reduce_min", family(Reduce, Combiner(Minimum)))]
pub struct ReduceMin(pub ReduceAttrs);

/// `tensor.reduce_and`: folds with `and`. The initial value has every bit set, true for `i1`, so
/// on `i1` it is "all".
#[crate::op("tensor.reduce_and", family(Reduce, Combiner(And)))]
pub struct ReduceAnd(pub ReduceAttrs);

/// `tensor.reduce_or`: folds with `or`. The initial value is 0, so on `i1` it is "any".
#[crate::op("tensor.reduce_or", family(Reduce, Combiner(Or)))]
pub struct ReduceOr(pub ReduceAttrs);

/// `tensor.argmax`: the index of the largest element along `axis`, a negative axis counting from
/// the end.
///
/// - Ties break toward the smallest index. NaN is larger than every number, as in Torch.
/// - If `keepdims` is true, `axis` remains with extent 1; otherwise it is removed. The indices
///   have the dtype `out`, `si32` or `si64`.
/// - If `axis` has extent 0, backends MUST reject the op as a spec violation.
#[crate::op("tensor.argmax", operands(input(Tensor)), results(output(Tensor)))]
pub struct ArgMax {
    pub axis: i64,
    #[attr(default)]
    pub keepdims: bool,
    /// The index dtype: `si32` or `si64`.
    pub out: DType,
}

/// The attributes of the `tensor.segment_*` ops.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentAttrs {
    pub num_segments: usize,
    #[attr(default)]
    pub indices_are_sorted: bool,
    #[attr(default)]
    pub unique: bool,
}

/// `tensor.segment_sum`, `tensor.segment_max`, ...: reduces the rows of `data` into
/// `num_segments` rows, row `i` into row `segment_ids[i]`, folding them with the binary op that the
/// op states.
///
/// - `segment_ids` holds one `si32` or `si64` id per row of `data`, along its leading axis. The
///   result has the shape of `data` with `num_segments` rows: `[num_segments, ...]`.
/// - Each id MUST lie in `[0, num_segments)`. A segment that no row reaches holds the initial
///   value of the fold, as the `tensor.reduce_*` op of the same fold states.
/// - It accumulates in the default accumulator of the dtype of `data` and returns that dtype.
/// - `indices_are_sorted` and `unique` state facts about the ids that MUST hold when set, so
///   kernels MAY rely on them.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::family(
    operands(data(Tensor), segment_ids(Tensor)),
    results(output(Tensor)),
    rule(crate::dialects::tensor::verify::contraction::segment),
    eval(crate::dialects::tensor::interp::contraction::segment)
)]
pub trait Segment: Deref<Target = SegmentAttrs> + From<SegmentAttrs> {
    /// The elementwise op that folds the rows of a segment.
    type Combiner: Combiner;
}

/// `tensor.segment_sum`: folds with `add`, from 0.
#[crate::op("tensor.segment_sum", family(Segment, Combiner(Add)))]
pub struct SegmentSum(pub SegmentAttrs);

/// `tensor.segment_max`: folds with `maximum`, from `-inf` or the smallest integer.
#[crate::op("tensor.segment_max", family(Segment, Combiner(Maximum)))]
pub struct SegmentMax(pub SegmentAttrs);

/// `tensor.segment_min`: folds with `minimum`, from `+inf` or the largest integer.
#[crate::op("tensor.segment_min", family(Segment, Combiner(Minimum)))]
pub struct SegmentMin(pub SegmentAttrs);

/// `tensor.segment_and`: folds with `and`, from every bit set.
#[crate::op("tensor.segment_and", family(Segment, Combiner(And)))]
pub struct SegmentAnd(pub SegmentAttrs);

/// `tensor.segment_or`: folds with `or`, from 0.
#[crate::op("tensor.segment_or", family(Segment, Combiner(Or)))]
pub struct SegmentOr(pub SegmentAttrs);

/// The attributes of `tensor.top_k` and `tensor.bottom_k`.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SelectKAttrs {
    pub k: usize,
    pub axis: i64,
    /// The index dtype: `si32` or `si64`.
    pub indices: DType,
}

/// `tensor.top_k`, `tensor.bottom_k`: the `k` extreme elements along `axis`, and their indices.
///
/// - A negative axis counts from the end.
/// - Of equal elements, the one with the smaller index comes first, and NaN is larger than every
///   number.
/// - The values have the input's shape with extent `k` on `axis`, and their indices along `axis`
///   have the same shape and the dtype that the attribute `indices` names, `si32` or `si64`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::family(
    operands(input(Tensor)),
    results(values(Tensor), indices(Tensor)),
    rule(crate::dialects::tensor::verify::contraction::select_k),
    eval(crate::dialects::tensor::interp::contraction::select_k)
)]
pub trait SelectK: Deref<Target = SelectKAttrs> + From<SelectKAttrs> {
    /// Whether it selects the largest elements, largest first, rather than the smallest, smallest
    /// first.
    const LARGEST: bool;
}

/// `tensor.top_k`: the `k` largest elements along `axis`, largest first.
#[crate::op("tensor.top_k", family(SelectK, LARGEST(true)))]
pub struct TopK(pub SelectKAttrs);

/// `tensor.bottom_k`: the `k` smallest elements along `axis`, smallest first.
#[crate::op("tensor.bottom_k", family(SelectK, LARGEST(false)))]
pub struct BottomK(pub SelectKAttrs);

// ------------------------------------------------------------------------------------------------
// Ops that move elements to positions their attributes fix.

/// `tensor.reshape`: the same elements in row-major order, with a new shape.
///
/// - `shape` gives every extent explicitly, and the element count MUST be preserved.
#[crate::op(
    "tensor.reshape",
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Movement::ALL)
)]
pub struct Reshape {
    pub shape: Ranked<Dim>,
}

/// `tensor.transpose`: permutes the axes. Output axis `i` is input axis `perm[i]`.
///
/// - `perm` MUST be a permutation of `[0, rank)`.
#[crate::op(
    "tensor.transpose",
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Movement::ALL)
)]
pub struct Transpose {
    pub perm: Perm,
}

/// `tensor.broadcast_to`: aligns the input with the trailing axes of `shape` and repeats it along
/// every axis where the input has extent 1 or no axis.
///
/// - The result has the extents of `shape`.
/// - Compatibility follows the right-aligned rule of "Broadcasting policy" in the dialect
///   documentation: `shape` has at least the input's rank, and each input extent is 1 or the
///   extent of `shape` on its axis. Backends MUST reject shapes that are not broadcast-compatible
///   as a spec violation.
#[crate::op(
    "tensor.broadcast_to",
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Movement::ALL)
)]
pub struct BroadcastTo {
    pub shape: Ranked<Dim>,
}

/// `tensor.slice`: the block of extents `sizes` starting at `starts`, taking every `strides[i]`-th
/// element along axis `i`.
///
/// - Along axis `i`, result element `k` is input element `starts[i] + k * strides[i]`. Without
///   `strides`, the window is contiguous: unit stride.
/// - `starts` and `sizes` have one entry per axis of the input, and `strides` is empty or has one
///   positive entry per axis.
/// - The window MUST lie in the input: on every axis `i`, `starts[i] + span <= in_shape[i]`, where
///   `span` is `(sizes[i] - 1) * strides[i] + 1`, or 0 when `sizes[i]` is 0. With unit stride,
///   that is `starts[i] + sizes[i] <= in_shape[i]`.
#[crate::op(
    "tensor.slice",
    operands(input(Tensor)),
    results(output(Tensor)),
    traits(Movement::ALL)
)]
pub struct Slice {
    pub starts: Ranked<usize>,
    pub sizes: Ranked<usize>,
    /// One step per axis, or none for steps of 1.
    #[attr(default)]
    pub strides: Ranked<usize>,
}

/// `tensor.concat`: joins the inputs along `axis`.
///
/// - Concatenates one or more inputs along `axis`, a negative axis counting from the end. The
///   result's extent on `axis` is the sum of the inputs' extents.
/// - All dtypes MUST match, and all shapes MUST match on the axes other than `axis`.
#[crate::op(
    "tensor.concat",
    operands(inputs(Variadic(Tensor))),
    results(output(Tensor)),
    traits(Movement::ALL)
)]
pub struct Concat {
    pub axis: i64,
}

/// `tensor.bitcast`: the bits of the input read as elements of `to`.
///
/// - Each row of the last axis is one stream of bits: its elements in order, each from its least
///   significant bit up, as tensors store them. The last axis holds the same bits, so its extent
///   scales by `bits(input) / bits(to)`, which MUST give a whole number; the other axes stay.
///   Dtypes of equal width keep the shape.
/// - `i1` takes no part: its elements are bytes holding 0 or 1, not bits.
/// - It reads published packings in place: GPTQ and AWQ pack eight 4-bit codes into each int32,
///   and gpt-oss stores MXFP4 blocks and their e8m0 scales as bytes.
#[crate::op("tensor.bitcast", operands(input(Tensor)), results(output(Tensor)))]
pub struct Bitcast {
    pub to: DType,
}

// ------------------------------------------------------------------------------------------------
// Ops that address elements by index. The dynamic slice ops take their starts at run time, as a
// decode step does to read and write its cache.

/// `tensor.iota`: a tensor of `shape` whose elements count up along `axis`.
///
/// - The value increases along `axis`, a negative axis counting from the end, from 0 to
///   `shape[axis] - 1`, and repeats along every other axis.
/// - Each value is its integer index converted to `dtype` as `tensor.cast` converts an integer: a
///   float rounds it once, to nearest-even, and a narrow integer saturates it.
#[crate::op("tensor.iota", results(output(Tensor)))]
pub struct Iota {
    pub shape: Ranked<Dim>,
    pub dtype: DType,
    pub axis: i64,
}

/// `tensor.take`: the rows of `table` at `indices`. The result shape is the indices' shape
/// followed by the table's trailing axes.
///
/// - Equivalent to `index_select` on axis 0, the common embedding primitive.
/// - `indices` are `si32` or `si64`, and each MUST satisfy `0 <= index < N`, where `N` is the
///   extent of the table's leading axis.
#[crate::op(
    "tensor.take",
    operands(table(Tensor), indices(Tensor)),
    results(output(Tensor)),
    traits(Movement::leading(1))
)]
pub struct Take;

/// `tensor.dynamic_slice`: the block of extents `sizes` starting at the run-time `starts`, with
/// unit stride.
///
/// - `sizes` has one extent per axis, and `starts` holds one `si32` or `si64` start per axis, of
///   shape `[rank]`.
/// - On each axis, the start is clamped into `[0, extent - size]`, so the block lies inside the
///   input.
/// - If a size exceeds its axis's extent at run time, backends MUST reject the op as a spec
///   violation.
#[crate::op(
    "tensor.dynamic_slice",
    operands(input(Tensor), starts(Tensor)),
    results(output(Tensor)),
    traits(Movement::leading(1))
)]
pub struct DynamicSlice {
    pub sizes: Ranked<usize>,
}

/// `tensor.dynamic_update_slice`: `base` with the block at the run-time `starts` replaced by
/// `update`.
///
/// - The block's extents are the static extents of `update`, which has the dtype and rank of
///   `base`.
/// - `starts` holds one `si32` or `si64` start per axis, of shape `[rank]`, clamped as in
///   `tensor.dynamic_slice`: into `[0, extent - size]`.
#[crate::op(
    "tensor.dynamic_update_slice",
    operands(base(Tensor), update(Tensor), starts(Tensor)),
    results(output(Tensor)),
    traits(Movement::leading(2))
)]
pub struct DynamicUpdateSlice;

/// `tensor.gather`: `torch.gather`. Element `i` of the result is element `i` of `input` with its
/// coordinate on `axis` replaced by `indices[i]`.
///
/// - `indices` are `si32` or `si64` and MUST have the rank of the input and its extents on every
///   axis but `axis`, a negative axis counting from the end; along `axis` they MAY differ.
/// - The result has the shape of `indices`.
/// - Each index MUST satisfy `0 <= index < in_shape[axis]`.
#[crate::op(
    "tensor.gather",
    operands(input(Tensor), indices(Tensor)),
    results(output(Tensor))
)]
pub struct Gather {
    pub axis: i64,
}

/// The attributes of the `tensor.scatter_*` ops.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScatterAttrs {
    pub axis: i64,
}

/// `tensor.scatter_add`, `tensor.scatter_max`, `tensor.scatter_min`: `torch.scatter_reduce` of the
/// same-rank form, folding each update into its element with the binary op that the op states.
///
/// - The result is `input` with element `i` of `updates` folded into the element whose coordinate
///   on `axis` is `indices[i]`, in row-major order of `updates`.
/// - `indices` are `si32` or `si64`, and `updates` MUST have their shape and the input's dtype.
///   Both MUST have the input's rank and its extents on every axis but `axis`, a negative axis
///   counting from the end; along `axis` they MAY differ.
/// - Each index MUST be in range.
/// - The elements and the updates accumulate in the default accumulator of the dtype, then return
///   the dtype.
#[crate::family(
    operands(input(Tensor), indices(Tensor), updates(Tensor)),
    results(output(Tensor)),
    rule(crate::dialects::tensor::verify::indexing::scatter),
    eval(crate::dialects::tensor::interp::indexing::scatter)
)]
pub trait Scatter: Deref<Target = ScatterAttrs> + From<ScatterAttrs> {
    /// The elementwise op that folds an update into its element.
    type Combiner: Combiner;
}

/// `tensor.scatter_add`: accumulates the sum of the element and its updates.
#[crate::op("tensor.scatter_add", family(Scatter, Combiner(Add)))]
pub struct ScatterAdd(pub ScatterAttrs);

/// `tensor.scatter_max`: accumulates the maximum of the element and its updates.
#[crate::op("tensor.scatter_max", family(Scatter, Combiner(Maximum)))]
pub struct ScatterMax(pub ScatterAttrs);

/// `tensor.scatter_min`: accumulates the minimum of the element and its updates.
#[crate::op("tensor.scatter_min", family(Scatter, Combiner(Minimum)))]
pub struct ScatterMin(pub ScatterAttrs);

/// `tensor.scatter_replace`: `input` with element `i` of `updates` written to the element whose
/// coordinate on `axis` is `indices[i]`. Of several updates to one element, the last in row-major
/// order of `updates` wins.
///
/// - The operands follow the rules of `tensor.scatter_add`, of any dtype.
#[crate::op(
    "tensor.scatter_replace",
    operands(input(Tensor), indices(Tensor), updates(Tensor)),
    results(output(Tensor))
)]
pub struct ScatterReplace(pub ScatterAttrs);

// ------------------------------------------------------------------------------------------------
// Sliding-window ops.

/// `tensor.extract_patches`: gathers every `window` patch of the padded input into its own row
/// (im2col, unfold).
///
/// - The input is channels-last, `[N, spatial..., C]`, and the result is
///   `[N, positions..., prod(window) * C]`.
/// - `window`, `strides`, `dilation` and `padding` have one entry per spatial axis. The positions
///   along each spatial axis are those of `tensor.reduce_window_*` with a base dilation of 1.
/// - A patch holds the elements of its window in row-major order of the window, `C` channels each.
///   Elements in the padding are `pad_value`, which MUST be a value of the input's dtype.
/// - Backends SHOULD fuse `extract_patches` + `dot_general` into convolution kernels.
/// - An extension op.
#[crate::op(
    "tensor.extract_patches",
    operands(input(Tensor)),
    results(output(Tensor))
)]
pub struct ExtractPatches {
    pub window: Ranked<usize>,
    pub strides: Ranked<usize>,
    pub dilation: Ranked<usize>,
    /// `(low, high)` padding per window axis.
    pub padding: Ranked<(usize, usize)>,
    pub pad_value: F64,
}

/// The attributes of the `tensor.reduce_window_*` ops.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReduceWindowAttrs {
    pub window_dims: Ranked<usize>,
    pub strides: Ranked<usize>,
    /// `(low, high)` padding per axis.
    pub padding: Ranked<(usize, usize)>,
    pub base_dilation: Ranked<usize>,
    pub window_dilation: Ranked<usize>,
    /// The accumulator dtype, when it differs from [`DType::default_accumulator`] of the input.
    #[attr(default)]
    pub accum: Option<DType>,
    /// The result dtype, when it differs from the input's.
    #[attr(default)]
    pub out: Option<DType>,
}

/// `tensor.reduce_window_sum`, `tensor.reduce_window_max`, ...: reduces every window of the
/// padded, dilated input, the pooling primitive.
///
/// - Each attribute has one entry per axis: `window_dims` sizes the window, `strides` steps it,
///   `padding` pads the input with `(low, high)` positions, and `base_dilation` and
///   `window_dilation` space the input and the window elements. All but `padding` are positive.
/// - Along an axis of extent `n`, the dilated, padded input spans
///   `padded = (n - 1) * base_dilation + 1 + low + high` positions, the dilated window
///   `span = (window - 1) * window_dilation + 1`, and the result has
///   `(padded - span) / stride + 1` positions.
/// - Padding and the holes between dilated elements add nothing to a window.
/// - It folds with the binary op and from the initial value that the op states, and accumulates
///   in `accum` as `tensor.reduce_*` does, by the accumulator policy; the result dtype is `out`,
///   by default the input's.
#[crate::family(
    operands(input(Tensor)),
    results(output(Tensor)),
    rule(crate::dialects::tensor::verify::window::reduce_window),
    eval(crate::dialects::tensor::interp::window::reduce_window)
)]
pub trait ReduceWindow: Deref<Target = ReduceWindowAttrs> + From<ReduceWindowAttrs> {
    /// The elementwise op that folds the elements of a window.
    type Combiner: Combiner;
}

/// `tensor.reduce_window_sum`: sums each window, from 0: average pooling divides it.
#[crate::op("tensor.reduce_window_sum", family(ReduceWindow, Combiner(Add)))]
pub struct ReduceWindowSum(pub ReduceWindowAttrs);

/// `tensor.reduce_window_max`: the largest element of each window, from `-inf` or the
/// smallest integer: max pooling.
#[crate::op("tensor.reduce_window_max", family(ReduceWindow, Combiner(Maximum)))]
pub struct ReduceWindowMax(pub ReduceWindowAttrs);

/// `tensor.reduce_window_min`: the smallest element of each window, from `+inf` or the
/// largest integer.
#[crate::op("tensor.reduce_window_min", family(ReduceWindow, Combiner(Minimum)))]
pub struct ReduceWindowMin(pub ReduceWindowAttrs);

/// `tensor.reduce_window_and`: folds each window with `and`, from every bit set.
#[crate::op("tensor.reduce_window_and", family(ReduceWindow, Combiner(And)))]
pub struct ReduceWindowAnd(pub ReduceWindowAttrs);

/// `tensor.reduce_window_or`: folds each window with `or`, from 0.
#[crate::op("tensor.reduce_window_or", family(ReduceWindow, Combiner(Or)))]
pub struct ReduceWindowOr(pub ReduceWindowAttrs);

// ------------------------------------------------------------------------------------------------
// Ops that draw random numbers from an explicit generator state.

/// The attributes of `tensor.rng_uniform` and `tensor.rng_normal`.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngAttrs {
    pub shape: Ranked<Dim>,
    pub dtype: DType,
}

/// `tensor.rng_uniform`, `tensor.rng_normal`: a tensor of `shape` whose elements are random, and
/// the state after them.
///
/// - The ops are pure: one state gives the same values on every backend.
/// - A state, `ui64[2]`, is a key and a counter: `[key, counter]`. Its stream is the output of
///   Philox4x32-10 (Random123). Counter block `j` is the 128-bit number `counter + j`, split into
///   four 32-bit words from the least significant one up. The key splits into two words the same
///   way. Each block gives four 32-bit words. An op reads the first words of its stream and
///   returns `[key, counter + blocks read]`, the counter wrapping modulo `2^64`.
/// - Each element takes the next `ceil(bits / 32)` words, the first one least significant, where
///   `bits` is the width of `dtype`, or the significand of a uniform float.
/// - A uniform integer is the low `bits` of its words: each value of the dtype equally likely.
/// - A uniform float with a significand of `p` bits (`bf16`: 8, `f16`: 11, `f32`: 24, `f64`: 53)
///   is the top `p` bits of its words times `2^-p`: on `[0, 1)`, and exact in its dtype.
/// - A normal float takes two uniforms `u1` and `u2` of its dtype's precision, in that order, and
///   is `sqrt(-2 ln(1 - u1)) * cos(2 pi u2)`, rounded to its dtype.
/// - Uniform `ui32` values are the draws of stochastic rounding (`tensor.quantize`).
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::family(
    operands(state(Tensor)),
    results(new_state(Tensor), output(Tensor)),
    rule(crate::dialects::tensor::verify::random::rng),
    eval(crate::dialects::tensor::interp::random::rng)
)]
pub trait Rng: Deref<Target = RngAttrs> + From<RngAttrs> {
    /// The `arith` op that draws one element.
    type Scalar: arith::Rng;
}

/// `tensor.rng_uniform`: each value of an integer dtype equally likely, or a float on `[0, 1)`.
#[crate::op("tensor.rng_uniform", family(Rng, Scalar(arith::RngUniform)))]
pub struct RngUniform(pub RngAttrs);

/// `tensor.rng_normal`: the standard normal distribution, of a float dtype.
#[crate::op("tensor.rng_normal", family(Rng, Scalar(arith::RngNormal)))]
pub struct RngNormal(pub RngAttrs);

// ------------------------------------------------------------------------------------------------
// Quantization: elements of a narrow dtype with scales, and offsets, that map them back to real
// values. "Quantization" in the dialect documentation gives the scale layout the ops share.

/// `tensor.dequantize`: `out[i] = to(q[i] * scale[i / block] + offset[i / block])`, computed in
/// f32, then rounded to nearest-even in `to`.
///
/// - `to` is `f32`, `bf16` or `f16`. The scale and the offset follow the scale layout of
///   "Quantization" in the dialect documentation. Without an offset, the sum adds nothing.
/// - `q` holds quantized elements, integers or narrow floats, or the values of a code table that
///   `tensor.take` read at the codes: NF4 weights dequantize as `take(table, codes)` and then this
///   op.
/// - The offset carries asymmetric quantization: the zero points of GPTQ and AWQ as
///   `offset = -zero_point * scale`, computed once at load time, and the block minimums of GGUF.
/// - A backend SHOULD read `q` and `scale` in place inside the op that consumes the result, such
///   as a `dot_general` with a quantized weight.
/// - Autodiff MUST differentiate it as multiplication by the scale, so that
///   `dequantize(quantize(x))` passes gradients through unchanged.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::op(
    "tensor.dequantize",
    operands(q(Tensor), scale(Tensor), offset(Optional(Tensor))),
    results(output(Tensor))
)]
pub struct Dequantize {
    pub block: Ranked<usize>,
    pub to: DType,
}

/// `tensor.quantize`: quantizes `x` into `to` with given scales and offsets, to the nearest value.
///
/// - `x` is `f32`, `bf16` or `f16`. `scale` holds static scales, for example parameters, or the
///   scales that a `tensor.block_scale_*` op computes from `x`: dynamic quantization is the two
///   ops. The scale and the offset follow the scale layout of "Quantization" in the dialect
///   documentation.
/// - `y[i] = (x[i] - offset[i / block]) / scale[i / block]`, computed in f32. A zero scale gives
///   `y[i] = 0`.
/// - `q[i] = to(y[i])`, the nearest value of `to`, the even one on a tie, saturated to the largest
///   finite value of `to`. Infinities saturate. NaN maps to NaN in a dtype that has one, and to 0
///   otherwise.
/// - Autodiff MUST differentiate it as division by the scale, ignoring the rounding: the
///   straight-through estimator of quantization-aware training.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::op(
    "tensor.quantize",
    operands(x(Tensor), scale(Tensor), offset(Optional(Tensor))),
    results(output(Tensor))
)]
pub struct Quantize {
    pub block: Ranked<usize>,
    pub to: DType,
}

/// `tensor.quantize_stochastic`: quantizes `x` as `tensor.quantize` does, rounding each `y[i]` up
/// or down at random.
///
/// - It rounds `abs(y)` to one of its two neighbors `lo` and `lo + q` in `to`, where `q` is their
///   spacing. It picks `lo + q` when the element's draw `r` satisfies
///   `r < floor((abs(y) - lo) / q * 2^32)`, else `lo`, and keeps the sign of `y`. So rounding picks
///   `lo + q` with probability `(abs(y) - lo) / q`. FP8 and FP4 training round gradients this way.
/// - `random` holds one `ui32` draw per element of `x`. The draws come from `tensor.rng_uniform`
///   with `dtype = ui32`, so a state reproduces the rounding.
/// - Scales, offsets, saturation, infinities, NaN and autodiff are those of `tensor.quantize`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::op(
    "tensor.quantize_stochastic",
    operands(x(Tensor), scale(Tensor), offset(Optional(Tensor)), random(Tensor)),
    results(output(Tensor))
)]
pub struct QuantizeStochastic {
    pub block: Ranked<usize>,
    pub to: DType,
}

/// `tensor.quantize_table`: quantizes `x` into the indices of a code table.
///
/// - `y[i]` is that of `tensor.quantize`. `q[i]` is the index of the `table` value nearest to
///   `y[i]`, the lowest index on a tie, and 0 for NaN. NF4 weights and 8-bit optimizer states
///   quantize this way.
/// - `table` is a rank-1 tensor of floats or narrow floats, and `to` an integer dtype.
/// - Autodiff is that of `tensor.quantize`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::op(
    "tensor.quantize_table",
    operands(x(Tensor), scale(Tensor), offset(Optional(Tensor)), table(Tensor)),
    results(output(Tensor))
)]
pub struct QuantizeTable {
    pub block: Ranked<usize>,
    pub to: DType,
}

/// The attributes of the ops that compute scales: `tensor.block_scale_*` and
/// `tensor.block_scale_range`.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockScaleAttrs {
    pub block: Ranked<usize>,
    /// The quantized dtype the scales map the blocks into.
    pub to: DType,
    pub scale_dtype: DType,
}

/// `tensor.block_scale_amax`, `tensor.block_scale_pow2_floor`, ...: the symmetric scale of each
/// block of `x`, for `tensor.quantize` into `to`, from the block's largest magnitude `amax`.
///
/// - `x` is `f32`, `bf16` or `f16`. The blocks, and the shape of the scales, follow the scale
///   layout of "Quantization" in the dialect documentation. The scales have dtype `scale_dtype`.
/// - `amax` ignores NaN elements. `max(to)` is the largest finite value of `to`.
/// - A power-of-two scale has its exponent clamped to `[-127, 127]`, and a block whose `amax` is 0
///   gets the smallest one, `2^-127`. `e8m0` scales need a power-of-two op.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::family(
    operands(x(Tensor)),
    results(scale(Tensor)),
    rule(crate::dialects::tensor::verify::quant::block_scale),
    eval(crate::dialects::tensor::interp::quant::block_scale)
)]
pub trait BlockScale:
    Deref<Target = BlockScaleAttrs> + From<BlockScaleAttrs> + ScaleFormula
{
    /// Whether every scale is a power of two, which `e8m0` scales need.
    const POWER_OF_TWO: bool;
}

/// What a `tensor.block_scale_*` op computes for one block.
pub trait ScaleFormula {
    /// The scale of a block whose largest magnitude is `amax`, for a quantized dtype whose largest
    /// finite value is `largest`, in f64.
    fn scale(amax: f64, largest: f64) -> f64;
}

/// `tensor.block_scale_amax`: `scale_dtype(amax / max(to))`, rounded to nearest-even, so `amax`
/// maps to the largest value of `to`; a block whose `amax` is 0 gets the scale 0. NVFP4 takes its
/// `f8e4m3` block scales this way, and per-tensor, per-channel and per-token scales are `f32` or
/// `bf16`.
#[crate::op("tensor.block_scale_amax", family(BlockScale, POWER_OF_TWO(false)))]
pub struct BlockScaleAmax(pub BlockScaleAttrs);

/// `tensor.block_scale_pow2_floor`: `2^(floor(log2(amax)) - emax)`, the microscaling (MX) rule,
/// where `emax = floor(log2(max(to)))` is the exponent of the largest power of two `to` holds
/// (`f4e2m1`: 2, `f6e2m3`: 2, `f6e3m2`: 4, `f8e4m3`: 8, `f8e5m2`: 15). Values above `max(to)`
/// times the scale saturate.
#[crate::op(
    "tensor.block_scale_pow2_floor",
    family(BlockScale, POWER_OF_TWO(true))
)]
pub struct BlockScalePow2Floor(pub BlockScaleAttrs);

/// `tensor.block_scale_pow2_ceil`: `2^ceil(log2(amax / max(to)))`, the smallest power of two that
/// maps `amax` into `to`, as DeepSeek's UE8M0 scales are.
#[crate::op("tensor.block_scale_pow2_ceil", family(BlockScale, POWER_OF_TWO(true)))]
pub struct BlockScalePow2Ceil(pub BlockScaleAttrs);

/// `tensor.block_scale_range`: the asymmetric scale and offset of each block of `x`, for
/// `tensor.quantize` into `to`.
///
/// - `x`, the blocks and the shapes are those of the `tensor.block_scale_*` ops. The scale and the
///   offset ignore NaN elements, and `scale_dtype` is a float.
/// - The scale maps the block's range `[lo, hi]`, widened to hold 0, onto the range of `to`:
///   `scale = (hi - lo) / (max(to) - min(to))`. The integer zero point is
///   `zero_point = round(min(to) - lo / scale)`, rounded to nearest-even and clamped into the range
///   of `to`, so that 0 quantizes exactly; a zero scale gives the zero point 0. The offset is
///   `offset = -zero_point * scale`, with the scale as stored. `max(to)` and `min(to)` are the
///   largest and smallest finite values of `to`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[crate::op(
    "tensor.block_scale_range",
    operands(x(Tensor)),
    results(scale(Tensor), offset(Tensor))
)]
pub struct BlockScaleRange(pub BlockScaleAttrs);

// ------------------------------------------------------------------------------------------------
// Control flow over whole tensor values: ops whose regions compute their results.

/// `tensor.yield`: ends a region of a control-flow op. Its operands are the region's results.
///
/// - Each region of a control-flow op has its own arguments and ends with a `tensor.yield`. The
///   ops compute on whole tensor values, like the rest of the dialect.
#[crate::op("tensor.yield", operands(values(Variadic(Tensor))), traits(Terminator))]
pub struct Yield;

/// `tensor.cond`: runs `then_branch` on `args` when `pred` holds and `else_branch` otherwise,
/// and gives the results that the branch yields, of the types `results`.
///
/// - `pred` is an `i1` scalar, `i1[]`.
/// - Both branches take arguments of the types of `args` and MUST yield the types `results`.
#[crate::op(
    "tensor.cond",
    operands(pred(Tensor), args(Variadic(Tensor))),
    results(outputs(Variadic(Tensor))),
    regions(then_branch(Block(Yield)), else_branch(Block(Yield)))
)]
pub struct Cond {
    pub results: Vec<TensorType>,
}

/// `tensor.while`: a loop over the carried values `init`.
///
/// - `cond` takes the carried values and yields an `i1` scalar and new carried values:
///   `cond(carry...) -> (i1[], carry...)`.
/// - If the scalar is true, `body` runs on the carried values `cond` yielded, `body(carry...) ->
///   carry...`, and the loop repeats with the values `body` yields.
/// - If the scalar is false, the results are the carried values `cond` yielded.
#[crate::op(
    "tensor.while",
    operands(init(Variadic(Tensor))),
    results(outputs(Variadic(Tensor))),
    regions(cond(Block(Yield)), body(Block(Yield)))
)]
pub struct While;

/// `tensor.scan`: a loop over the leading axis of the scanned inputs, like a functional for-loop.
///
/// - Its operands are `carry_count` carried values, then the scanned inputs, each `[T, ...]`:
///   at least one, all with the same leading extent `T`.
/// - Step `t` runs `body` on the carried values and slice `t` of each scanned input, and `body`
///   yields the next carried values, then one output of each type in `outputs`.
/// - The results are the final carried values, then each output stacked along a new leading axis
///   over the steps, `[T, ...]`.
#[crate::op(
    "tensor.scan",
    operands(inputs(Variadic(Tensor))),
    results(outputs(Variadic(Tensor))),
    regions(body(Block(Yield)))
)]
pub struct Scan {
    pub carry_count: usize,
    /// The type of each per-step output.
    pub outputs: Vec<TensorType>,
}

// ------------------------------------------------------------------------------------------------
// `tensor.custom_call`: a call into a backend kernel by name.

/// A value in a custom call's attribute map: `3`, `1.5`, `true`, `"s"`, or a list with its element
/// type, `array<i64: [1, 2]>`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum CustomAttr {
    #[attr(format = "$0")]
    I64(i64),
    #[attr(format = "$0")]
    F64(F64),
    #[attr(format = "$0")]
    Bool(bool),
    #[attr(format = "$0")]
    String(String),
    #[attr(format = "`array` `<` $0 `>`")]
    Array(CustomArray),
}

/// A list in a custom call's attribute map, after its element type.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum CustomArray {
    #[attr(format = "`i64` `` `:` $0")]
    I64(Vec<i64>),
    #[attr(format = "`f64` `` `:` $0")]
    F64(Vec<F64>),
    #[attr(format = "`bool` `` `:` $0")]
    Bool(Vec<bool>),
    #[attr(format = "`str` `` `:` $0")]
    String(Vec<String>),
}

/// `tensor.custom_call`: runs the backend kernel `target` on operands that the target defines, with
/// one result per entry of `results`. The result types are attributes, since only the rewrite that
/// creates the call knows them.
///
/// - Effects: a custom call has effects that the IR does not describe. Passes MUST NOT merge,
///   precompute or erase a custom call, even when nothing reads its results. Backends MAY cache
///   internal compiled kernels, packed weights, or plans.
/// - Portability: if a backend does not recognize `target`, it MUST reject it as unimplemented
///   (no silent fallback to a portable decomposition). The interpreter has no semantics for it.
/// - Typing: `results` is authoritative. Backends MAY validate that operand dtypes and shapes and
///   `attrs` agree with it.
/// - Stable naming: `target` strings MUST be namespaced and versioned, such as
///   `<namespace>.conv2d.nhwc.f32.v1`.
/// - Attributes: `attrs` is a small serializable map of primitives (integers, floats, bools and
///   strings) and arrays of primitives. Backends SHOULD reject unknown or malformed attributes as
///   a spec violation.
#[crate::op(
    "tensor.custom_call",
    operands(inputs(Variadic(Tensor))),
    results(outputs(Variadic(Tensor))),
    effects(opaque)
)]
pub struct CustomCall {
    pub target: String,
    #[attr(default)]
    pub attrs: BTreeMap<String, CustomAttr>,
    pub results: Vec<TensorType>,
}

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "tensor", op = Tensor)]
pub enum TensorOp {
    Constant(Constant),
    Neg(Neg),
    Abs(Abs),
    Exp(Exp),
    Log(Log),
    Tanh(Tanh),
    Erf(Erf),
    Rsqrt(Rsqrt),
    Reciprocal(Reciprocal),
    Sin(Sin),
    Cos(Cos),
    Not(Not),
    Add(Add),
    Sub(Sub),
    Mul(Mul),
    Div(Div),
    Maximum(Maximum),
    Minimum(Minimum),
    And(And),
    Or(Or),
    Xor(Xor),
    ShiftLeft(ShiftLeft),
    ShiftRightLogical(ShiftRightLogical),
    ShiftRightArithmetic(ShiftRightArithmetic),
    Less(Less),
    LessEqual(LessEqual),
    Equal(Equal),
    GreaterEqual(GreaterEqual),
    Greater(Greater),
    NotEqual(NotEqual),
    Select(Select),
    Cast(Cast),
    DotGeneral(DotGeneral),
    ReduceSum(ReduceSum),
    ReduceMax(ReduceMax),
    ReduceMin(ReduceMin),
    ReduceAnd(ReduceAnd),
    ReduceOr(ReduceOr),
    Reshape(Reshape),
    Transpose(Transpose),
    BroadcastTo(BroadcastTo),
    Slice(Slice),
    Bitcast(Bitcast),
    Concat(Concat),
    Iota(Iota),
    Take(Take),
    DynamicSlice(DynamicSlice),
    DynamicUpdateSlice(DynamicUpdateSlice),
    // The window ops hold four or five lists indexed by axis. Boxed, they leave the other ops
    // their size.
    ExtractPatches(Box<ExtractPatches>),
    ReduceWindowSum(Box<ReduceWindowSum>),
    ReduceWindowMax(Box<ReduceWindowMax>),
    ReduceWindowMin(Box<ReduceWindowMin>),
    ReduceWindowAnd(Box<ReduceWindowAnd>),
    ReduceWindowOr(Box<ReduceWindowOr>),
    CustomCall(CustomCall),
    StopGradient(StopGradient),
    ArgMax(ArgMax),
    SegmentSum(SegmentSum),
    SegmentMax(SegmentMax),
    SegmentMin(SegmentMin),
    SegmentAnd(SegmentAnd),
    SegmentOr(SegmentOr),
    TopK(TopK),
    BottomK(BottomK),
    Gather(Gather),
    ScatterAdd(ScatterAdd),
    ScatterMax(ScatterMax),
    ScatterMin(ScatterMin),
    ScatterReplace(ScatterReplace),
    RngUniform(RngUniform),
    RngNormal(RngNormal),
    Yield(Yield),
    Cond(Cond),
    While(While),
    Scan(Scan),
    Dequantize(Dequantize),
    Quantize(Quantize),
    QuantizeStochastic(QuantizeStochastic),
    QuantizeTable(QuantizeTable),
    BlockScaleAmax(BlockScaleAmax),
    BlockScalePow2Floor(BlockScalePow2Floor),
    BlockScalePow2Ceil(BlockScalePow2Ceil),
    BlockScaleRange(BlockScaleRange),
}
