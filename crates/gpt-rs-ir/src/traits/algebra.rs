//! Algebraic traits. Each states its equations as laws and folds what follows from it alone.

use std::fmt;

use half::{bf16, f16};

use super::{FoldResult, Law, OpRef, OpTrait, Term};
use crate::op::Folded;
use crate::types::DType;
use crate::{Doc, Op};

const A: Term = Term::Var(0);
const B: Term = Term::Var(1);
const C: Term = Term::Var(2);

fn apply<const N: usize>(terms: [Term; N]) -> Term {
    Term::Apply(terms.into())
}

/// Whether a trait that holds for integers and `i1`, and for floats only when `float` is set,
/// covers `dtype`.
fn covers(float: bool, dtype: DType) -> bool {
    dtype.int_format().is_some() || (float && dtype.is_float())
}

/// `f(a, b) = f(b, a)`. Folding puts a run-time operand first and a constant last (see
/// `Module::is_static`), so the other folds, the patterns and CSE see a constant on the right.
#[derive(Debug, Doc)]
pub struct Commutative;

impl fmt::Display for Commutative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("commutative")
    }
}

impl OpTrait for Commutative {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let &[lhs, rhs] = op.operands else {
            return None;
        };
        let rank = |v| (op.module.is_static(v), op.module.constant(v).is_some());
        (rank(lhs) > rank(rhs)).then(|| FoldResult::Operands(vec![rhs, lhs]))
    }

    fn laws(&self, arity: usize, _dtype: DType) -> Vec<Law> {
        if arity != 2 {
            return Vec::new();
        }
        vec![Law {
            name: "f(a, b) = f(b, a)",
            lhs: apply([A, B]),
            rhs: apply([B, A]),
        }]
    }
}

/// `f(f(a, b), c) = f(a, f(b, c))`, for integers, and for floats when `float` is set: float
/// addition and multiplication round differently once regrouped.
#[derive(Debug, Doc)]
pub struct Associative {
    pub float: bool,
}

impl Associative {
    pub const ALL: Self = Self { float: true };
    pub const INT: Self = Self { float: false };

    pub fn covers(&self, dtype: DType) -> bool {
        covers(self.float, dtype)
    }
}

impl fmt::Display for Associative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("associative")?;
        for_integers(self.float, f)
    }
}

/// The scope of a trait that holds for floats only when `float` is set.
fn for_integers(float: bool, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if float {
        Ok(())
    } else {
        f.write_str(" on integers")
    }
}

impl OpTrait for Associative {
    fn laws(&self, arity: usize, dtype: DType) -> Vec<Law> {
        if arity != 2 || !self.covers(dtype) {
            return Vec::new();
        }
        vec![Law {
            name: "f(f(a, b), c) = f(a, f(b, c))",
            lhs: apply([apply([A, B]), C]),
            rhs: apply([A, apply([B, C])]),
        }]
    }
}

/// An integer that a trait names in every integer dtype and `i1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntValue {
    Zero,
    One,
    Min,
    Max,
    /// Every bit set: -1 in a signed dtype, the largest value in an unsigned one, true in `i1`.
    AllOnes,
}

impl fmt::Display for IntValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            IntValue::Zero => "0",
            IntValue::One => "1",
            IntValue::Min => "min",
            IntValue::Max => "max",
            IntValue::AllOnes => "all ones",
        })
    }
}

/// A value for each dtype class a trait covers: `float` for bf16, f16, f32 and f64, `int` for the
/// integers and `i1`. `None` where the trait has no such value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerClass {
    pub float: Option<f64>,
    pub int: Option<IntValue>,
}

impl PerClass {
    /// The value as one element of `dtype`, or `None` when the trait has no value there.
    pub fn bytes(&self, dtype: DType) -> Option<Vec<u8>> {
        if let Some(int) = dtype.int_format() {
            let value = match self.int? {
                IntValue::Zero => 0,
                IntValue::One => 1,
                IntValue::Min => int.min(),
                IntValue::Max => int.max(),
                IntValue::AllOnes => int.wrap(-1),
            };
            return Some(int.to_le_bytes(value, dtype.size_in_bytes()?));
        }
        let float = self.float?;
        // Traits name only signed zeros, 1 and infinities. Each is exact in every dtype here.
        Some(match dtype {
            DType::F64 => float.to_le_bytes().to_vec(),
            DType::F32 => (float as f32).to_le_bytes().to_vec(),
            DType::Bf16 => bf16::from_f64(float).to_le_bytes().to_vec(),
            DType::F16 => f16::from_f64(float).to_le_bytes().to_vec(),
            _ => return None,
        })
    }
}

/// `f(a, e) = a`. A commutative op has `e` on the left too, since folding moves constants to the
/// right.
#[derive(Debug, Doc)]
pub struct RightIdentity(pub PerClass);

impl fmt::Display for PerClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.float, self.int) {
            (Some(float), Some(int)) => write!(f, "{float:?} on floats, {int} on integers"),
            (Some(float), None) => write!(f, "{float:?} on floats"),
            (None, Some(int)) => write!(f, "{int} on integers"),
            (None, None) => f.write_str("none"),
        }
    }
}

impl fmt::Display for RightIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "right identity {}", self.0)
    }
}

impl RightIdentity {
    pub const fn new(float: f64, int: IntValue) -> Self {
        Self(PerClass {
            float: Some(float),
            int: Some(int),
        })
    }

    /// An identity of the integers and `i1` alone.
    pub const fn int(int: IntValue) -> Self {
        Self(PerClass {
            float: None,
            int: Some(int),
        })
    }
}

impl OpTrait for RightIdentity {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let &[lhs, _] = op.operands else {
            return None;
        };
        let e = self.0.bytes(op.dtype()?)?;
        op.is_splat(1, &e)
            .then(|| FoldResult::Results(vec![Folded::Value(lhs)]))
    }

    fn laws(&self, arity: usize, dtype: DType) -> Vec<Law> {
        let Some(e) = self.0.bytes(dtype).filter(|_| arity == 2) else {
            return Vec::new();
        };
        vec![Law {
            name: "f(a, e) = a",
            lhs: apply([A, Term::Splat(e)]),
            rhs: A,
        }]
    }
}

/// `f(a, z) = z`, for the integers and `i1`.
#[derive(Debug, Doc)]
pub struct Absorbing(IntValue);

impl fmt::Display for Absorbing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "absorbing {} on integers", self.0)
    }
}

impl Absorbing {
    pub const fn int(z: IntValue) -> Self {
        Self(z)
    }

    fn bytes(&self, dtype: DType) -> Option<Vec<u8>> {
        PerClass {
            float: None,
            int: Some(self.0),
        }
        .bytes(dtype)
    }
}

impl OpTrait for Absorbing {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let &[_, rhs] = op.operands else {
            return None;
        };
        let z = self.bytes(op.dtype()?)?;
        op.is_splat(1, &z)
            .then(|| FoldResult::Results(vec![Folded::Value(rhs)]))
    }

    fn laws(&self, arity: usize, dtype: DType) -> Vec<Law> {
        let Some(z) = self.bytes(dtype).filter(|_| arity == 2) else {
            return Vec::new();
        };
        vec![Law {
            name: "f(a, z) = z",
            lhs: apply([A, Term::Splat(z.clone())]),
            rhs: Term::Splat(z),
        }]
    }
}

/// For a unary op `f(f(a)) = f(a)`, for a binary one `f(a, a) = a`.
#[derive(Debug, Doc)]
pub struct Idempotent;

impl fmt::Display for Idempotent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("idempotent")
    }
}

impl OpTrait for Idempotent {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let kept = match *op.operands {
            [operand] => op.same_op_operands(0).map(|_| operand),
            [lhs, rhs] => (lhs == rhs).then_some(lhs),
            _ => None,
        }?;
        Some(FoldResult::Results(vec![Folded::Value(kept)]))
    }

    fn laws(&self, arity: usize, _dtype: DType) -> Vec<Law> {
        match arity {
            1 => vec![Law {
                name: "f(f(a)) = f(a)",
                lhs: apply([apply([A])]),
                rhs: apply([A]),
            }],
            2 => vec![Law {
                name: "f(a, a) = a",
                lhs: apply([A, A]),
                rhs: A,
            }],
            _ => Vec::new(),
        }
    }
}

/// `f(f(a)) = a`.
#[derive(Debug, Doc)]
pub struct Involution;

impl fmt::Display for Involution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("involution")
    }
}

impl OpTrait for Involution {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let &[inner] = op.same_op_operands(0)? else {
            return None;
        };
        Some(FoldResult::Results(vec![Folded::Value(inner)]))
    }

    fn laws(&self, arity: usize, _dtype: DType) -> Vec<Law> {
        if arity != 1 {
            return Vec::new();
        }
        vec![Law {
            name: "f(f(a)) = a",
            lhs: apply([apply([A])]),
            rhs: A,
        }]
    }
}

/// `f(a, g(b, c)) = g(f(a, b), f(a, c))` for the op `g` that `over` builds, for integers, and for
/// floats when `float` is set.
#[derive(Debug, Doc)]
pub struct DistributesOver {
    pub over: fn() -> Op,
    pub float: bool,
}

impl DistributesOver {
    pub fn covers(&self, dtype: DType) -> bool {
        covers(self.float, dtype)
    }
}

impl fmt::Display for DistributesOver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "distributes over `{}`", (self.over)().name())?;
        for_integers(self.float, f)
    }
}

impl OpTrait for DistributesOver {
    fn laws(&self, arity: usize, dtype: DType) -> Vec<Law> {
        if arity != 2 || !self.covers(dtype) {
            return Vec::new();
        }
        let g = |x: Term, y: Term| Term::Other(Box::new((self.over)()), vec![x, y]);
        vec![Law {
            name: "f(a, g(b, c)) = g(f(a, b), f(a, c))",
            lhs: apply([A, g(B, C)]),
            rhs: g(apply([A, B]), apply([A, C])),
        }]
    }
}
