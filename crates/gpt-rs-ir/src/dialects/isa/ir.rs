//! The ops of the `isa` dialect and their attributes.

use std::ops::Deref;

use crate::dialects::arith;
use crate::ir::{DType, Level, PerGroupAxis, Ranked, F64};

// ------------------------------------------------------------------------------------------------
// Lane-wise ops: `arith` on each lane.

/// `isa.neg`, `isa.exp`, ...: the `arith` op of the same name on each lane of a group of
/// registers.
#[crate::family(
    operands(input(Vector)),
    results(output(Vector)),
    rule(crate::dialects::isa::verify::unary),
    eval(crate::dialects::isa::interp::unary)
)]
pub trait Unary {
    /// The `arith` op that it applies to each lane.
    type Scalar: arith::Unary + Default;
}

/// `isa.neg`: `arith.neg` on each lane.
#[crate::op("isa.neg", family(Unary, Scalar(arith::Neg)))]
pub struct Neg;

/// `isa.abs`: `arith.abs` on each lane.
#[crate::op("isa.abs", family(Unary, Scalar(arith::Abs)))]
pub struct Abs;

/// `isa.exp`: `arith.exp` on each lane.
#[crate::op("isa.exp", family(Unary, Scalar(arith::Exp)))]
pub struct Exp;

/// `isa.log`: `arith.log` on each lane.
#[crate::op("isa.log", family(Unary, Scalar(arith::Log)))]
pub struct Log;

/// `isa.tanh`: `arith.tanh` on each lane.
#[crate::op("isa.tanh", family(Unary, Scalar(arith::Tanh)))]
pub struct Tanh;

/// `isa.erf`: `arith.erf` on each lane.
#[crate::op("isa.erf", family(Unary, Scalar(arith::Erf)))]
pub struct Erf;

/// `isa.rsqrt`: `arith.rsqrt` on each lane.
#[crate::op("isa.rsqrt", family(Unary, Scalar(arith::Rsqrt)))]
pub struct Rsqrt;

/// `isa.reciprocal`: `arith.reciprocal` on each lane.
#[crate::op("isa.reciprocal", family(Unary, Scalar(arith::Reciprocal)))]
pub struct Reciprocal;

/// `isa.sin`: `arith.sin` on each lane.
#[crate::op("isa.sin", family(Unary, Scalar(arith::Sin)))]
pub struct Sin;

/// `isa.cos`: `arith.cos` on each lane.
#[crate::op("isa.cos", family(Unary, Scalar(arith::Cos)))]
pub struct Cos;

/// `isa.not`: `arith.not` on each lane.
#[crate::op("isa.not", family(Unary, Scalar(arith::Not)))]
pub struct Not;

/// `isa.add`, `isa.mul`, ...: the `arith` op of the same name on each pair of lanes of two groups
/// of registers of one type.
#[crate::family(
    operands(lhs(Vector), rhs(Vector)),
    results(output(Vector)),
    rule(crate::dialects::isa::verify::binary),
    eval(crate::dialects::isa::interp::binary)
)]
pub trait Binary {
    /// The `arith` op that it applies to each pair of lanes.
    type Scalar: arith::Binary + Default;
}

/// `isa.add`: `arith.add` on each pair of lanes.
#[crate::op("isa.add", family(Binary, Scalar(arith::Add)))]
pub struct Add;

/// `isa.sub`: `arith.sub` on each pair of lanes.
#[crate::op("isa.sub", family(Binary, Scalar(arith::Sub)))]
pub struct Sub;

/// `isa.mul`: `arith.mul` on each pair of lanes.
#[crate::op("isa.mul", family(Binary, Scalar(arith::Mul)))]
pub struct Mul;

/// `isa.div`: `arith.div` on each pair of lanes.
#[crate::op("isa.div", family(Binary, Scalar(arith::Div)))]
pub struct Div;

/// `isa.maximum`: `arith.maximum` on each pair of lanes.
#[crate::op("isa.maximum", family(Binary, Scalar(arith::Maximum)))]
pub struct Maximum;

/// `isa.minimum`: `arith.minimum` on each pair of lanes.
#[crate::op("isa.minimum", family(Binary, Scalar(arith::Minimum)))]
pub struct Minimum;

/// `isa.and`: `arith.and` on each pair of lanes.
#[crate::op("isa.and", family(Binary, Scalar(arith::And)))]
pub struct And;

/// `isa.or`: `arith.or` on each pair of lanes.
#[crate::op("isa.or", family(Binary, Scalar(arith::Or)))]
pub struct Or;

/// `isa.xor`: `arith.xor` on each pair of lanes.
#[crate::op("isa.xor", family(Binary, Scalar(arith::Xor)))]
pub struct Xor;

/// `isa.shift_left`: `arith.shift_left` on each pair of lanes.
#[crate::op("isa.shift_left", family(Binary, Scalar(arith::ShiftLeft)))]
pub struct ShiftLeft;

/// `isa.shift_right_logical`: `arith.shift_right_logical` on each pair of lanes.
#[crate::op(
    "isa.shift_right_logical",
    family(Binary, Scalar(arith::ShiftRightLogical))
)]
pub struct ShiftRightLogical;

/// `isa.shift_right_arithmetic`: `arith.shift_right_arithmetic` on each pair of lanes.
#[crate::op(
    "isa.shift_right_arithmetic",
    family(Binary, Scalar(arith::ShiftRightArithmetic))
)]
pub struct ShiftRightArithmetic;

/// `isa.less`, `isa.equal`, ...: a mask whose flag says whether the relation of the `arith` op of
/// the same name holds between each pair of lanes of two groups of registers of one type.
#[crate::family(
    operands(lhs(Vector), rhs(Vector)),
    results(output(Vector)),
    rule(crate::dialects::isa::verify::compare),
    eval(crate::dialects::isa::interp::compare)
)]
pub trait Compare {
    /// The `arith` op that it applies to each pair of lanes.
    type Scalar: arith::Compare + Default;
}

/// `isa.less`: `arith.less` on each pair of lanes.
#[crate::op("isa.less", family(Compare, Scalar(arith::Less)))]
pub struct Less;

/// `isa.less_equal`: `arith.less_equal` on each pair of lanes.
#[crate::op("isa.less_equal", family(Compare, Scalar(arith::LessEqual)))]
pub struct LessEqual;

/// `isa.equal`: `arith.equal` on each pair of lanes.
#[crate::op("isa.equal", family(Compare, Scalar(arith::Equal)))]
pub struct Equal;

/// `isa.greater_equal`: `arith.greater_equal` on each pair of lanes.
#[crate::op("isa.greater_equal", family(Compare, Scalar(arith::GreaterEqual)))]
pub struct GreaterEqual;

/// `isa.greater`: `arith.greater` on each pair of lanes.
#[crate::op("isa.greater", family(Compare, Scalar(arith::Greater)))]
pub struct Greater;

/// `isa.not_equal`: `arith.not_equal` on each pair of lanes.
#[crate::op("isa.not_equal", family(Compare, Scalar(arith::NotEqual)))]
pub struct NotEqual;

/// `isa.select`: per lane, `on_true` where the flag of `pred` is set, else `on_false`.
#[crate::op(
    "isa.select",
    operands(pred(Vector), on_true(Vector), on_false(Vector)),
    results(output(Vector))
)]
pub struct Select;

/// `isa.fma`: `arith.fma` on each triple of lanes.
#[crate::op(
    "isa.fma",
    operands(lhs(Vector), rhs(Vector), acc(Vector)),
    results(output(Vector))
)]
pub struct Fma;

/// `isa.cast`: each lane converted to `to`, as `arith.cast` converts a scalar.
#[crate::op("isa.cast", operands(input(Vector)), results(output(Vector)))]
pub struct Cast {
    pub to: DType,
}

/// `isa.bitcast`: the bits of each lane read as a value of `to`, of the same width, as
/// `arith.bitcast` reads a scalar.
#[crate::op("isa.bitcast", operands(input(Vector)), results(output(Vector)))]
pub struct Bitcast {
    pub to: DType,
}

// ------------------------------------------------------------------------------------------------
// Registers from scalars and registers: `isa.splat`, `isa.expand` and `isa.tile_mask`.

/// `isa.splat`: the scalar in every lane of a group of registers of the extents `dims`, the lanes
/// last.
#[crate::op("isa.splat", operands(input(Scalar)), results(output(Vector)))]
pub struct Splat {
    pub dims: Ranked<usize>,
}

/// `isa.expand`: the group of registers with each axis of extent 1 repeated to its extent in
/// `dims`, which has the input's lanes and number of axes.
#[crate::op("isa.expand", operands(input(Vector)), results(output(Vector)))]
pub struct Expand {
    pub dims: Ranked<usize>,
}

/// `isa.tile_mask`: the mask of the points of a register tile that lie inside `extents`, one per
/// axis of the tile. The tile's axes are those of the group, the last one with its lanes, or the
/// lanes alone for a mask with no group axes: lane `l` of register `[i, j]` of a mask of extents
/// `[R, V, L]` is the point `[i, j * L + l]`. Its flag is set when each group axis but the last is
/// below its own extent, and the last one with the lanes below the last extent: here
/// `i < extents[0]` and `j * L + l < extents[1]`.
#[crate::op(
    "isa.tile_mask",
    operands(extents(Variadic(Index))),
    results(output(Vector))
)]
pub struct TileMask {
    pub dims: Ranked<usize>,
}

// ------------------------------------------------------------------------------------------------
// Memory: the reads `isa.load` and `isa.broadcast` and their masked forms, `isa.store`, its masked
// form, and `isa.prefetch`.

/// The attributes of the reads.
#[derive(crate::AttrGroup, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReadAttrs {
    /// The extents of the group of registers it reads, the lanes last.
    pub dims: Ranked<usize>,
    /// For each axis of the group, how far the position moves on each axis of the buffer from one
    /// register to the next.
    #[attr(default)]
    pub steps: PerGroupAxis<Ranked<i64>>,
    /// The value of a lane outside the buffer, on an axis that `checked` allows.
    #[attr(default)]
    pub pad: Option<F64>,
    /// With `pad`, per register of the group in row-major order, the axes of the buffer on which
    /// its lanes may fall outside it. An empty list allows every axis of every register.
    #[attr(default)]
    pub checked: Vec<Ranked<usize>>,
}

/// `isa.load`, `isa.broadcast`: a group of registers of extents `dims` from `buffer`, register `g`
/// from the position `base + g[0] * steps[0] + ...`. Without `pad`, every position MUST be inside
/// the buffer. With `pad v`, a lane may fall outside the buffer on the axes that `checked` names
/// for its register, or on any axis when `checked` is empty. Such a lane is `v` and reads nothing.
/// On any other axis, its position MUST be inside.
#[crate::family(
    operands(buffer(Buffer), base(Variadic(Index))),
    results(output(Vector)),
    effects(read(buffer)),
    rule(crate::dialects::isa::verify::read),
    eval(crate::dialects::isa::interp::read)
)]
pub trait Read: Deref<Target = ReadAttrs> + From<ReadAttrs> {
    /// Whether lane `l` of a register reads the element `l` further along the last axis of the
    /// buffer than the register's position, rather than every lane reading the element there.
    const ALONG: bool;
}

/// `isa.load`: lane `l` of a register is the element `l` further along the last axis of the
/// buffer than the register's position.
#[crate::op("isa.load", family(Read, ALONG(true)))]
pub struct Load(pub ReadAttrs);

/// `isa.broadcast`: every lane of a register is the element at the register's position.
#[crate::op("isa.broadcast", family(Read, ALONG(false)))]
pub struct Broadcast(pub ReadAttrs);

/// `isa.masked_load`, `isa.masked_broadcast`: the read of the same name without `masked_` on the
/// lanes whose flag in `mask`, a mask of the extents `dims`, is set. The other lanes are 0, and
/// read nothing.
#[crate::family(
    operands(mask(Vector), buffer(Buffer), base(Variadic(Index))),
    results(output(Vector)),
    effects(read(buffer)),
    rule(crate::dialects::isa::verify::masked_read),
    eval(crate::dialects::isa::interp::masked_read)
)]
pub trait MaskedRead: Deref<Target = ReadAttrs> + From<ReadAttrs> {
    /// As [`Read::ALONG`].
    const ALONG: bool;
}

/// `isa.masked_load`: `isa.load` of the lanes whose flag in `mask` is set.
#[crate::op("isa.masked_load", family(MaskedRead, ALONG(true)))]
pub struct MaskedLoad(pub ReadAttrs);

/// `isa.masked_broadcast`: `isa.broadcast` into the lanes whose flag in `mask` is set.
#[crate::op("isa.masked_broadcast", family(MaskedRead, ALONG(false)))]
pub struct MaskedBroadcast(pub ReadAttrs);

/// `isa.store`: writes the lanes of `value` into `buffer` where `isa.load` would read them.
#[crate::op(
    "isa.store",
    operands(value(Vector), buffer(Buffer), base(Variadic(Index))),
    effects(write(buffer))
)]
pub struct Store {
    #[attr(default)]
    pub steps: PerGroupAxis<Ranked<i64>>,
}

/// `isa.masked_store`: `isa.store` of the lanes whose flag in `mask` is set. The other elements
/// keep their values.
#[crate::op(
    "isa.masked_store",
    operands(mask(Vector), value(Vector), buffer(Buffer), base(Variadic(Index))),
    effects(write(buffer))
)]
pub struct MaskedStore {
    #[attr(default)]
    pub steps: PerGroupAxis<Ranked<i64>>,
}

/// `isa.prefetch`: asks the cache `level` for the element at the first lane of each register of
/// extents `dims` that `isa.load` of `steps` at `base` would read from `buffer`, which a later
/// load reads. A hint: it computes and changes nothing, and a position outside the buffer reads
/// nothing.
#[crate::op(
    "isa.prefetch",
    operands(buffer(Buffer), base(Variadic(Index))),
    effects(read(buffer))
)]
pub struct Prefetch {
    pub dims: Ranked<usize>,
    #[attr(default)]
    pub steps: PerGroupAxis<Ranked<i64>>,
    pub level: Level,
}

// ------------------------------------------------------------------------------------------------
// The dialect

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "isa", op = Isa)]
pub enum IsaOps {
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
    Splat(Splat),
    Expand(Expand),
    TileMask(TileMask),
    Load(Load),
    Broadcast(Broadcast),
    MaskedLoad(MaskedLoad),
    MaskedBroadcast(MaskedBroadcast),
    Store(Store),
    MaskedStore(MaskedStore),
    Prefetch(Prefetch),
}
