//! Structural traits: how an op's results relate to its operands and its place in a region.

use std::fmt;

use super::{FoldResult, OpRef, OpTrait};
use crate::ir::{Dense, Folded};
use crate::Doc;

/// Each result element depends only on the operand elements at its own position, so every tensor
/// operand and result has one shape.
#[derive(Debug, Doc)]
pub struct Elementwise;

impl OpTrait for Elementwise {}

impl fmt::Display for Elementwise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("elementwise")
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data {
            None => f.write_str("movement"),
            Some(1) => f.write_str("movement of operand 0"),
            Some(count) => write!(f, "movement of operands 0 to {}", count - 1),
        }
    }
}

impl OpTrait for Movement {
    fn fold(&self, op: &OpRef<'_>) -> Option<FoldResult> {
        let [result] = op.results else {
            return None;
        };
        let result = result.as_tensor()?;
        let dims = result.static_dims()?;
        let data = self.data(op.operands);
        let operand_elements: usize = data
            .iter()
            .map(|&v| op.module.constant(v).map(Dense::element_count))
            .sum::<Option<usize>>()?;
        if dims.iter().product::<usize>() > operand_elements {
            return None;
        }
        let (first, rest) = data.split_first()?;
        let element = op.module.constant(*first)?.splat_bytes()?;
        let same = |&v| {
            op.module
                .constant(v)
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("terminator")
    }
}

/// Its regions read no value defined outside them.
#[derive(Debug, Doc)]
pub struct IsolatedFromAbove;

impl OpTrait for IsolatedFromAbove {}

impl fmt::Display for IsolatedFromAbove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("isolated from above")
    }
}
