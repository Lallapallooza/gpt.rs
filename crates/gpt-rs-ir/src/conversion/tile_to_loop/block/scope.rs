//! Where the code being built stands in the loops of a block, and what it knows there. Entering a
//! loop or a fold's state makes a new [`Scope`]. A scope never changes once made.

use std::rc::Rc;

use super::fold::FoldState;
use super::placed::Piece;
use super::stage::Staged;
use crate::dialects::tile::Clips;
use crate::ir::{Value, MAX_ITERS};

/// The value of each iterator in scope, by position, none past the block's iterators.
pub(super) type Point = [Option<Value>; MAX_ITERS];

/// The points of an iterator that the loops around narrow it to: from `start` up to `end`, and
/// how many there are when that is known before the run.
#[derive(Debug, Clone, Copy)]
pub(super) struct Span {
    pub(super) start: Value,
    pub(super) end: Value,
    pub(super) points: Option<usize>,
}

/// The span of each iterator that the loops around have narrowed it to, or none for its whole
/// extent.
pub(super) type Ranges = [Option<Span>; MAX_ITERS];

/// What the code built at one place of a block's loops knows.
#[derive(Clone)]
pub(super) struct Scope {
    /// The loops of the block that skip points where padded reads fall outside their operands, and
    /// the reads they keep in range.
    pub(super) clips: Rc<Clips>,
    /// The span of each iterator that the loops around narrow it to.
    pub(super) ranges: Ranges,
    /// The value of each iterator that a loop around walks.
    pub(super) point: Point,
    /// The copy that the reads of each operand take their elements from, where a holding made one.
    pub(super) staged: Vec<Option<Rc<Staged>>>,
    /// The tile of each loop of the nest that the code stands in.
    pub(super) tiles: Vec<Option<Value>>,
    /// The index that each loop of the nest stands at, given by a `loop.forall` around that runs
    /// it on the workers of its level together with the loops around it: such a loop builds that
    /// tile or point alone.
    pub(super) entered: Vec<Option<Value>>,
    /// The start of the innermost tile of each iterator, inside the loops over its tiles.
    pub(super) starts: Point,
    /// The state that the points built here fold into.
    pub(super) state: Option<Rc<FoldState>>,
    /// The sweep of the fold that the points built here make.
    pub(super) sweep: usize,
    /// The pieces that the placed blocks which read across one take their elements from.
    pub(super) pieces: Rc<Vec<Piece>>,
}

impl Scope {
    /// The scope of a block with a nest of `nest` loops, before any loop: with the copies
    /// `staged`, and the clips `clips`.
    pub(super) fn outside(nest: usize, staged: Vec<Option<Rc<Staged>>>, clips: Rc<Clips>) -> Scope {
        Scope {
            clips,
            ranges: [None; MAX_ITERS],
            point: [None; MAX_ITERS],
            staged,
            tiles: vec![None; nest],
            entered: vec![None; nest],
            starts: [None; MAX_ITERS],
            state: None,
            sweep: 0,
            pieces: Rc::default(),
        }
    }

    /// This scope with iterator `position` at `value`.
    pub(super) fn at(&self, position: usize, value: Value) -> Scope {
        let mut inner = self.clone();
        inner.point[position] = Some(value);
        inner
    }
}
