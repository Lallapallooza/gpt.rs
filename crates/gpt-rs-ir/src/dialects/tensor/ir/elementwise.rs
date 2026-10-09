//! Elementwise ops: each result element depends only on the operand elements at its position.
//! Their operands have equal shapes: nothing broadcasts implicitly.

use anyhow::{ensure, Result};

use super::class::{ensure_class, ensure_dtype_class, Class, ARITHMETIC, FLOATS, INTEGERS, LOGIC};
use crate::op::{
    fold, Fold, FoldCx, FoldOut, Folded, Folding, NoRegions, OpDef, OwnedTypes, TypeCx, Types,
};
use crate::traits::{
    Absorbing, Associative, Commutative, DistributesOver, Elementwise, Idempotent, IntValue,
    Involution, RightIdentity,
};
use crate::types::DType;
use crate::Attr;

/// The kinds of `tensor.{kind}` on one operand.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    /// `-x`. Integers wrap, so the negation of the smallest signed integer is itself.
    #[traits(Involution)]
    Neg,
    /// `abs(x)`. Integers wrap, so the absolute value of the smallest signed integer is itself.
    #[traits(Idempotent)]
    Abs,
    /// `e^x`.
    Exp,
    /// The natural logarithm, `ln(x)`.
    Log,
    /// The hyperbolic tangent.
    Tanh,
    /// The error function.
    Erf,
    /// `1 / sqrt(x)`.
    Rsqrt,
    /// `1 / x`.
    Reciprocal,
    /// The sine of `x` radians.
    Sin,
    /// The cosine of `x` radians.
    Cos,
    /// Flips every bit; on `i1`, logical not.
    #[traits(Involution)]
    Not,
}

/// `tensor.neg`, `tensor.exp`, ...: applies `kind` to every element.
///
/// - The result has the input's shape and dtype.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.{kind}",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Elementwise)
)]
pub struct Unary {
    pub kind: UnaryKind,
}

impl OpDef for Unary {
    fn result_types(
        &self,
        ins: UnaryOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<UnaryResults<OwnedTypes>> {
        let classes = match self.kind {
            UnaryKind::Neg | UnaryKind::Abs => ARITHMETIC,
            UnaryKind::Not => LOGIC,
            _ => FLOATS,
        };
        ensure_class("the input", ins.input, classes)?;
        Ok(UnaryResults {
            output: ins.input.clone(),
        })
    }
}

/// The kinds of `tensor.{kind}` on two operands. The identities of their traits hold for every
/// float, `-0.0` included: `x + 0.0` is not `x`, since `-0.0 + 0.0` is `0.0`. Floats have no
/// absorbing element: `inf * 0` and `maximum(NaN, inf)` are NaN.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryKind {
    /// `lhs + rhs`.
    #[traits(Commutative, Associative::INT, RightIdentity::new(-0.0, IntValue::Zero))]
    Add,
    /// `lhs - rhs`.
    #[traits(RightIdentity::new(0.0, IntValue::Zero))]
    Sub,
    /// `lhs * rhs`.
    #[traits(
        Commutative,
        Associative::INT,
        RightIdentity::new(1.0, IntValue::One),
        Absorbing::int(IntValue::Zero),
        DistributesOver { over: || Binary { kind: BinaryKind::Add }.into(), float: false }
    )]
    Mul,
    /// `lhs / rhs`. Integer division truncates toward zero, and backends MUST reject an integer
    /// division by zero as a spec violation. Float division follows IEEE 754, so it can give Inf
    /// or NaN.
    #[traits(RightIdentity::new(1.0, IntValue::One))]
    Div,
    /// The larger operand. A float NaN operand gives NaN, and `-0.0` orders below `0.0`.
    #[traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::NEG_INFINITY, IntValue::Min),
        Absorbing::int(IntValue::Max)
    )]
    Maximum,
    /// The smaller operand. A float NaN operand gives NaN, and `-0.0` orders below `0.0`.
    #[traits(
        Commutative,
        Associative::ALL,
        Idempotent,
        RightIdentity::new(f64::INFINITY, IntValue::Max),
        Absorbing::int(IntValue::Min)
    )]
    Minimum,
    /// Bitwise and; on `i1`, logical and.
    #[traits(
        Commutative,
        Associative::INT,
        Idempotent,
        RightIdentity::int(IntValue::AllOnes),
        Absorbing::int(IntValue::Zero),
        DistributesOver { over: || Binary { kind: BinaryKind::Or }.into(), float: false }
    )]
    And,
    /// Bitwise or; on `i1`, logical or.
    #[traits(
        Commutative,
        Associative::INT,
        Idempotent,
        RightIdentity::int(IntValue::Zero),
        Absorbing::int(IntValue::AllOnes),
        DistributesOver { over: || Binary { kind: BinaryKind::And }.into(), float: false }
    )]
    Or,
    /// Bitwise exclusive or; on `i1`, logical exclusive or.
    #[traits(Commutative, Associative::INT, RightIdentity::int(IntValue::Zero))]
    Xor,
    /// `lhs` shifted left by `rhs` bits, shifting in zeros. `rhs` reads as unsigned, and a shift by
    /// at least the bit width gives 0.
    #[traits(RightIdentity::int(IntValue::Zero))]
    ShiftLeft,
    /// `lhs` shifted right by `rhs` bits, shifting in zeros, into a signed dtype too. `rhs` reads
    /// as unsigned, and a shift by at least the bit width gives 0.
    #[traits(RightIdentity::int(IntValue::Zero))]
    ShiftRightLogical,
    /// `lhs` shifted right by `rhs` bits, shifting in copies of the sign bit. `rhs` reads as
    /// unsigned, and a shift by at least the bit width gives -1 for a negative `lhs`, else 0.
    #[traits(RightIdentity::int(IntValue::Zero))]
    ShiftRightArithmetic,
}

impl BinaryKind {
    /// The dtypes the kind computes on.
    pub fn classes(self) -> &'static [Class] {
        match self {
            BinaryKind::And | BinaryKind::Or | BinaryKind::Xor => LOGIC,
            BinaryKind::ShiftLeft
            | BinaryKind::ShiftRightLogical
            | BinaryKind::ShiftRightArithmetic => INTEGERS,
            _ => ARITHMETIC,
        }
    }
}

/// `tensor.add`, `tensor.mul`, ...: combines two tensors of equal type element by element:
/// `out[i] = kind(lhs[i], rhs[i])`.
///
/// - `lhs` and `rhs` MUST have the same dtype and the same shape, which the result has.
/// - Integer results wrap modulo `2^bits`, two's complement for the signed dtypes.
/// - A NaN result has no specified sign or payload.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.{kind}",
    operands(lhs: Tensor, rhs: Tensor),
    results(output: Tensor),
    traits(Elementwise)
)]
pub struct Binary {
    pub kind: BinaryKind,
}

impl OpDef for Binary {
    fn result_types(
        &self,
        ins: BinaryOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<BinaryResults<OwnedTypes>> {
        ensure!(
            ins.lhs == ins.rhs,
            "operands must have equal types, got {} and {}",
            ins.lhs,
            ins.rhs
        );
        ensure_class("each operand", ins.lhs, self.kind.classes())?;
        Ok(BinaryResults {
            output: ins.lhs.clone(),
        })
    }
}

/// The comparisons of `tensor.compare`.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareKind {
    /// `lhs < rhs`.
    Less,
    /// `lhs <= rhs`.
    LessEqual,
    /// `lhs == rhs`.
    Equal,
    /// `lhs >= rhs`.
    GreaterEqual,
    /// `lhs > rhs`.
    Greater,
    /// `lhs != rhs`.
    NotEqual,
}

/// `tensor.compare`: compares two tensors of equal type element by element into an `i1` tensor.
///
/// - `out[i]` is 1 where `kind` holds between `lhs[i]` and `rhs[i]`, and 0 elsewhere. The result
///   has the operands' shape.
/// - `lhs` and `rhs` MUST have the same dtype and the same shape.
/// - Unsigned integers order as unsigned numbers, and `i1` orders false below true.
/// - Floats compare as IEEE does: `less`, `less_equal`, `equal`, `greater_equal` and `greater` are
///   false if either operand is NaN, and `not_equal` is true if either operand is NaN.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.compare",
    operands(lhs: Tensor, rhs: Tensor),
    results(output: Tensor),
    traits(Elementwise)
)]
pub struct Compare {
    pub kind: CompareKind,
}

impl OpDef for Compare {
    fn result_types(
        &self,
        ins: CompareOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<CompareResults<OwnedTypes>> {
        ensure!(
            ins.lhs == ins.rhs,
            "operands must have equal types, got {} and {}",
            ins.lhs,
            ins.rhs
        );
        ensure_class("each operand", ins.lhs, super::class::ORDERED)?;
        Ok(CompareResults {
            output: ins.lhs.with_dtype(DType::I1),
        })
    }
}

/// `tensor.select`: picks `on_true` where `pred` is set and `on_false` elsewhere.
///
/// - For each element: `out[i] = pred[i] ? on_true[i] : on_false[i]`.
/// - `pred` is `i1`. All operands MUST have the same shape, and `on_true` and `on_false` MUST have
///   the same dtype, which the result has.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.select",
    operands(pred: Tensor, on_true: Tensor, on_false: Tensor),
    results(output: Tensor),
    traits(Elementwise)
)]
pub struct Select;

impl OpDef for Select {
    fn result_types(
        &self,
        ins: SelectOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<SelectResults<OwnedTypes>> {
        ensure!(
            ins.pred.dtype == DType::I1,
            "pred must be an i1 tensor, got {}",
            ins.pred
        );
        ensure!(
            ins.on_true == ins.on_false,
            "on_true and on_false must have equal types, got {} and {}",
            ins.on_true,
            ins.on_false
        );
        ensure!(
            ins.pred.dims == ins.on_true.dims,
            "pred and the branches must have equal shapes, got {} and {}",
            ins.pred,
            ins.on_true
        );
        Ok(SelectResults {
            output: ins.on_true.clone(),
        })
    }
}

/// `tensor.cast`: converts every element to `to`.
///
/// - The shape is preserved.
/// - The conversion rules are those of `cast(x; to=dst)` in "DTypes and casting" of the
///   dialect documentation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.cast",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Elementwise)
)]
pub struct Cast {
    pub to: DType,
}

impl OpDef for Cast {
    fn result_types(
        &self,
        ins: CastOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<CastResults<OwnedTypes>> {
        // e8m0 scales come from `quantize_dynamic`, which states how it rounds them.
        ensure_dtype_class(
            "to",
            self.to,
            &[Class::Bool, Class::Int, Class::Float, Class::Narrow],
        )?;
        Ok(CastResults {
            output: ins.input.with_dtype(self.to),
        })
    }
}

/// `tensor.stop_gradient`: its input, unchanged, as a derivative barrier.
///
/// - The forward value is identical to the input.
/// - Autodiff MUST treat it as a derivative barrier.
/// - Passes MUST keep it until autodiff has run, including through lowering to lower levels when
///   autodiff runs there, so no pass folds it away. Backends that run no autodiff MAY treat it as a
///   pure no-op.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op("tensor.stop_gradient", operands(input: Tensor), results(output: Tensor))]
pub struct StopGradient;

impl OpDef for StopGradient {
    fn result_types(
        &self,
        ins: StopGradientOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<StopGradientResults<OwnedTypes>> {
        Ok(StopGradientResults {
            output: ins.input.clone(),
        })
    }
}

/// `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are
/// the same value, and `on_false` when `pred` is all false.
#[fold]
impl Fold for Select {
    fn fold(
        &self,
        ins: SelectOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<SelectResults<FoldOut>> {
        let picked = if ins.pred.is_splat(true) || ins.on_true.value == ins.on_false.value {
            ins.on_true
        } else if ins.pred.is_splat(false) {
            ins.on_false
        } else {
            return None;
        };
        Some(SelectResults {
            output: Folded::Value(picked.value),
        })
    }
}

/// A cast to the input's own type is the input.
#[fold]
impl Fold for Cast {
    fn fold(
        &self,
        ins: CastOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<CastResults<FoldOut>> {
        (ins.input.ty == cx.result_type(0)).then_some(CastResults {
            output: Folded::Value(ins.input.value),
        })
    }
}
