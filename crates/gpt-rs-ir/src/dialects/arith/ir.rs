//! The `arith` ops and their attributes.

use std::ops::Deref;

use crate::dialects::arith::math::{BinaryMath, CompareMath, Draw, UnaryMath};
use crate::dialects::tensor::utils::class::{Class, ARITHMETIC, FLOATS, INTEGERS, LOGIC};
use crate::ir::{DType, Dense};
use crate::traits::{
    Absorbing, Associative, Commutative, DistributesOver, Idempotent, IntValue, Involution,
    NarrowInF32, RightIdentity, NEG_ZERO,
};

/// A scalar value of a dtype, written `0.5 : f32`, `-3 : si32` or `1 : i1`. It is held as a
/// rank-0 dense constant, so constant queries and folds see it as they see `tensor.constant`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = super::utils::LiteralText)]
pub struct Literal(pub(super) Dense);

/// `arith.constant`: the scalar `value`.
#[crate::op("arith.constant", results(output(Scalar)), constant)]
pub struct Constant {
    pub value: Literal,
}

/// `arith.neg`, `arith.exp`, ...: an op on one scalar, as the `tensor` op of the same name applies
/// it to each element. [`UnaryMath`] is its function.
#[crate::family(
    operands(input(Scalar)),
    results(output(Scalar)),
    traits(NarrowInF32::ROUNDED),
    rule(crate::dialects::arith::verify::unary),
    eval(crate::dialects::arith::interp::unary)
)]
pub trait Unary: UnaryMath {
    /// The dtypes it computes on.
    const CLASSES: &'static [Class];
}

/// As `tensor.neg`, on one scalar.
#[crate::op("arith.neg", family(Unary, CLASSES(ARITHMETIC)), traits(Involution))]
pub struct Neg;

/// As `tensor.abs`, on one scalar.
#[crate::op("arith.abs", family(Unary, CLASSES(ARITHMETIC)), traits(Idempotent))]
pub struct Abs;

/// As `tensor.exp`, on one scalar.
#[crate::op("arith.exp", family(Unary, CLASSES(FLOATS)))]
pub struct Exp;

/// As `tensor.log`, on one scalar.
#[crate::op("arith.log", family(Unary, CLASSES(FLOATS)))]
pub struct Log;

/// As `tensor.tanh`, on one scalar.
#[crate::op("arith.tanh", family(Unary, CLASSES(FLOATS)))]
pub struct Tanh;

/// As `tensor.erf`, on one scalar.
#[crate::op("arith.erf", family(Unary, CLASSES(FLOATS)))]
pub struct Erf;

/// As `tensor.rsqrt`, on one scalar.
#[crate::op("arith.rsqrt", family(Unary, CLASSES(FLOATS)))]
pub struct Rsqrt;

/// As `tensor.reciprocal`, on one scalar.
#[crate::op("arith.reciprocal", family(Unary, CLASSES(FLOATS)))]
pub struct Reciprocal;

/// As `tensor.sin`, on one scalar.
#[crate::op("arith.sin", family(Unary, CLASSES(FLOATS)))]
pub struct Sin;

/// As `tensor.cos`, on one scalar.
#[crate::op("arith.cos", family(Unary, CLASSES(FLOATS)))]
pub struct Cos;

/// As `tensor.not`, on one scalar.
#[crate::op("arith.not", family(Unary, CLASSES(LOGIC)), traits(Involution))]
pub struct Not;

/// `arith.add`, `arith.mul`, ...: an op on two scalars of one dtype, as the `tensor` op of the
/// same name applies it to each pair of elements. [`BinaryMath`] is its function.
#[crate::family(
    operands(lhs(Scalar), rhs(Scalar)),
    results(output(Scalar)),
    traits(NarrowInF32::ROUNDED),
    rule(crate::dialects::arith::verify::binary),
    eval(crate::dialects::arith::interp::binary)
)]
pub trait Binary: BinaryMath {
    /// The dtypes it computes on.
    const CLASSES: &'static [Class];
}

/// As `tensor.add`, on two scalars.
#[crate::op(
    "arith.add",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(
        Commutative,
        Associative::INT,
        RightIdentity::new(NEG_ZERO, IntValue::Zero)
    )
)]
pub struct Add;

/// As `tensor.sub`, on two scalars.
#[crate::op(
    "arith.sub",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(RightIdentity::new(0.0, IntValue::Zero))
)]
pub struct Sub;

/// As `tensor.mul`, on two scalars.
#[crate::op(
    "arith.mul",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(
        Commutative,
        Associative::INT,
        RightIdentity::new(1.0, IntValue::One),
        Absorbing::int(IntValue::Zero),
        DistributesOver::int(Add)
    )
)]
pub struct Mul;

/// As `tensor.div`, on two scalars.
#[crate::op(
    "arith.div",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(RightIdentity::new(1.0, IntValue::One))
)]
pub struct Div;

/// As `tensor.maximum`, on two scalars.
#[crate::op(
    "arith.maximum",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::NEG_INFINITY, IntValue::Min),
        Absorbing::int(IntValue::Max)
    )
)]
pub struct Maximum;

/// As `tensor.minimum`, on two scalars.
#[crate::op(
    "arith.minimum",
    family(Binary, CLASSES(ARITHMETIC)),
    traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::INFINITY, IntValue::Max),
        Absorbing::int(IntValue::Min)
    )
)]
pub struct Minimum;

/// As `tensor.and`, on two scalars.
#[crate::op(
    "arith.and",
    family(Binary, CLASSES(LOGIC)),
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

/// As `tensor.or`, on two scalars.
#[crate::op(
    "arith.or",
    family(Binary, CLASSES(LOGIC)),
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

/// As `tensor.xor`, on two scalars.
#[crate::op(
    "arith.xor",
    family(Binary, CLASSES(LOGIC)),
    traits(Commutative, Associative::INT, RightIdentity::int(IntValue::Zero))
)]
pub struct Xor;

/// As `tensor.shift_left`, on two scalars.
#[crate::op(
    "arith.shift_left",
    family(Binary, CLASSES(INTEGERS)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftLeft;

/// As `tensor.shift_right_logical`, on two scalars.
#[crate::op(
    "arith.shift_right_logical",
    family(Binary, CLASSES(INTEGERS)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftRightLogical;

/// As `tensor.shift_right_arithmetic`, on two scalars.
#[crate::op(
    "arith.shift_right_arithmetic",
    family(Binary, CLASSES(INTEGERS)),
    traits(RightIdentity::int(IntValue::Zero))
)]
pub struct ShiftRightArithmetic;

/// `arith.less`, `arith.equal`, ...: an `i1` that says whether the relation of the op holds
/// between two scalars of one dtype. `tensor.less`, ... apply it to each pair of elements.
/// [`CompareMath`] is its function.
#[crate::family(
    operands(lhs(Scalar), rhs(Scalar)),
    results(output(Scalar)),
    traits(NarrowInF32::COMPARED),
    rule(crate::dialects::arith::verify::compare),
    eval(crate::dialects::arith::interp::compare)
)]
pub trait Compare: CompareMath {}

/// As `tensor.less`, on two scalars.
#[crate::op("arith.less", family(Compare))]
pub struct Less;

/// As `tensor.less_equal`, on two scalars.
#[crate::op("arith.less_equal", family(Compare))]
pub struct LessEqual;

/// As `tensor.equal`, on two scalars.
#[crate::op("arith.equal", family(Compare))]
pub struct Equal;

/// As `tensor.greater_equal`, on two scalars.
#[crate::op("arith.greater_equal", family(Compare))]
pub struct GreaterEqual;

/// As `tensor.greater`, on two scalars.
#[crate::op("arith.greater", family(Compare))]
pub struct Greater;

/// As `tensor.not_equal`, on two scalars.
#[crate::op("arith.not_equal", family(Compare))]
pub struct NotEqual;

/// `arith.select`: `on_true` when `pred` is set, else `on_false`.
#[crate::op(
    "arith.select",
    operands(pred(Scalar), on_true(Scalar), on_false(Scalar)),
    results(output(Scalar))
)]
pub struct Select;

/// `arith.fma`: `lhs * rhs + acc`, rounded once to the dtype, f32 or f64. No `tensor` op computes
/// it: it enters below `tile`, where a dispatch's numerics contract lets a product and the sum it
/// enters round once.
#[crate::op(
    "arith.fma",
    operands(lhs(Scalar), rhs(Scalar), acc(Scalar)),
    results(output(Scalar))
)]
pub struct Fma;

/// `arith.cast`: the scalar converted to `to`, as `tensor.cast` converts each element.
#[crate::op("arith.cast", operands(input(Scalar)), results(output(Scalar)))]
pub struct Cast {
    pub to: DType,
}

/// `arith.bitcast`: the bits of the scalar read as a value of `to`, of the same width. Unpacking a
/// sub-byte code from a word is shifts and masks on integers, then a bitcast.
#[crate::op("arith.bitcast", operands(input(Scalar)), results(output(Scalar)))]
pub struct Bitcast {
    pub to: DType,
}

/// `arith.from_index`: the index as an element of `dtype`, converted as `tensor.iota` converts its
/// indices: as an si64 cast to `dtype`.
#[crate::op("arith.from_index", operands(input(Index)), results(output(Scalar)))]
pub struct FromIndex {
    pub dtype: DType,
}

/// `arith.to_index`: the integer as an index. Its value MUST fit `i64`.
#[crate::op("arith.to_index", operands(input(Scalar)), results(output(Index)))]
pub struct ToIndex;

/// The attributes of `arith.rng_uniform` and `arith.rng_normal`.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngAttrs {
    /// The dtype of the element.
    pub dtype: DType,
}

/// `arith.rng_uniform`, `arith.rng_normal`: element `index` of the stream that the `tensor` op of
/// the same name draws in `dtype` from the state `[key, counter]`, as that op computes it. `key`
/// and `counter` are `ui64`, and `index` an integer. [`Draw`] is its function.
#[crate::family(
    operands(key(Scalar), counter(Scalar), index(Scalar)),
    results(output(Scalar)),
    rule(crate::dialects::arith::verify::rng),
    eval(crate::dialects::arith::interp::rng)
)]
pub trait Rng: Deref<Target = RngAttrs> + From<RngAttrs> + Draw {
    /// The dtypes it draws.
    const CLASSES: &'static [Class];
}

/// As `tensor.rng_uniform`, on one element.
#[crate::op("arith.rng_uniform", family(Rng, CLASSES(ARITHMETIC)))]
pub struct RngUniform(pub RngAttrs);

/// As `tensor.rng_normal`, on one element.
#[crate::op("arith.rng_normal", family(Rng, CLASSES(FLOATS)))]
pub struct RngNormal(pub RngAttrs);

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "arith", op = Arith)]
pub enum ArithOps {
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
    Fma(Fma),
    Cast(Cast),
    Bitcast(Bitcast),
    FromIndex(FromIndex),
    ToIndex(ToIndex),
    RngUniform(RngUniform),
    RngNormal(RngNormal),
}
