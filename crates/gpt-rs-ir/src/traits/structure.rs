//! Structural traits: how an op's results relate to its operands and its place in a region.

use std::fmt;

use super::{FoldResult, Law, OpTrait, Operation, Term};
use crate::ir::{DType, Dense, Folded};
use crate::Doc;

/// Each result element depends only on the operand elements at its own position, so every tensor
/// operand and result has one shape.
#[derive(Debug, Doc)]
pub struct Elementwise;

impl OpTrait for Elementwise {}

impl fmt::Display for Elementwise {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("elementwise")
    }
}

/// On floats that the IR computes on in f32 ([`DType::computes_in_f32`], bf16 and f16), the op is
/// the op on their f32 values, with its float result rounded once to the operands' dtype, or, for
/// a comparison, kept as it is. A backend without arithmetic on such floats computes the op in f32.
#[derive(Debug, Doc)]
pub struct NarrowInF32 {
    /// Whether the result is a float of the operands' dtype, which the f32 result rounds to.
    pub rounds: bool,
}

impl NarrowInF32 {
    /// An op whose result is a float of its operands' dtype.
    pub const ROUNDED: Self = Self { rounds: true };
    /// A comparison, whose result is `i1`.
    pub const COMPARED: Self = Self { rounds: false };
}

impl fmt::Display for NarrowInF32 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self.rounds {
            true => "narrow in f32, rounded once",
            false => "narrow in f32, compared",
        })
    }
}

impl OpTrait for NarrowInF32 {
    fn laws(&self, arity: usize, dtype: DType) -> Vec<Law> {
        if !dtype.computes_in_f32() {
            return Vec::new();
        }
        let mut wide = Vec::with_capacity(arity);
        let mut vars = Vec::with_capacity(arity);
        for index in 0..arity {
            wide.push(Term::Cast(DType::F32, Box::new(Term::Var(index))));
            vars.push(Term::Var(index));
        }
        let in_f32 = Term::Apply(wide);
        let (name, rhs) = match self.rounds {
            true => (
                "f(a, ..) = f(f32(a), ..), rounded to the dtype",
                Term::Cast(dtype, Box::new(in_f32)),
            ),
            false => ("f(a, ..) = f(f32(a), ..)", in_f32),
        };
        vec![Law {
            name,
            lhs: Term::Apply(vars),
            rhs,
        }]
    }
}

/// Each result element is an element of one of the data operands: the first `data` operands, or
/// all of them when `data` is `None`. The op moves elements without computing on them, so a
/// splat stays a splat. The fold builds that splat only when it holds no more elements than the
/// operands: a constant stores every element, so a broadcast splat stays a broadcast.
#[derive(Debug, Doc)]
pub struct Movement {
    pub data: Option<usize>,
}

impl Movement {
    pub const ALL: Self = Self { data: None };

    pub const fn leading(count: usize) -> Self {
        Self { data: Some(count) }
    }

    fn data<'a, T>(&self, operands: &'a [T]) -> &'a [T] {
        &operands[..self.data.unwrap_or(operands.len())]
    }
}

impl fmt::Display for Movement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.data {
            None => f.write_str("movement"),
            Some(1) => f.write_str("movement of operand 0"),
            Some(count) => write!(f, "movement of operands 0 to {}", count - 1),
        }
    }
}

impl OpTrait for Movement {
    fn fold(&self, op: &Operation) -> Option<FoldResult> {
        let [result_type] = op.results else {
            return None;
        };
        let result = result_type.as_tensor()?;
        let dims = result.static_dims()?;
        let data = self.data(op.operands);
        let operand_elements: usize = data
            .iter()
            .map(|&operand| op.module.constant(operand).map(Dense::element_count))
            .sum::<Option<usize>>()?;
        if dims.iter().product::<usize>() > operand_elements {
            return None;
        }
        let (first, rest) = data.split_first()?;
        let element = op.module.constant(*first)?.splat_bytes()?;
        let same = |&operand| {
            op.module
                .constant(operand)
                .and_then(Dense::splat_bytes)
                .is_some_and(|bytes| bytes == element)
        };
        rest.iter().all(same).then(|| {
            FoldResult::Results(vec![Folded::Constant(Dense::splat(
                result.dtype,
                dims,
                element,
            ))])
        })
    }
}

/// Ends a region: its operands are what the region yields, and it has no results.
#[derive(Debug, Doc)]
pub struct Terminator;

impl OpTrait for Terminator {}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("terminator")
    }
}

/// It is the only use of each of its operands, whose value it takes over.
#[derive(Debug, Doc)]
pub struct OnlyUse;

impl OpTrait for OnlyUse {}

impl fmt::Display for OnlyUse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("only use")
    }
}

/// Its regions read no value defined outside them.
#[derive(Debug, Doc)]
pub struct IsolatedFromAbove;

impl OpTrait for IsolatedFromAbove {}

impl fmt::Display for IsolatedFromAbove {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("isolated from above")
    }
}
