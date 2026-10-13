//! `tile-to-loop`: each `tile.dispatch` as a `loop.func`. `dispatch.rs` holds the pattern of
//! `tile.dispatch`, and `block.rs` and `combine.rs` the lowering of the blocks it holds and of their
//! combine kinds.

mod block;
mod combine;
mod dispatch;
mod utils;

use anyhow::Result;

use self::dispatch::*;
use super::{apply_conversion, ConversionTarget, ConvertOp, Mode, OpForm};
use crate::dialects::tile::{self, TileOp};
use crate::dialects::Dialect;
use crate::ir::Module;
use crate::passes::{pass, Pass};
use crate::rewrite::{PatternSet, Rewrites};

/// Lowers each `tile.dispatch` to a `loop.func` of the same name that computes the same values, bit
/// for bit, with the loops that its schedule record asks for. A dispatch whose schedule record has
/// no nest gets the reference schedule: one sequential loop per iterator, in the order of the
/// block.
///
/// - The function takes each argument of the dispatch in an `in` buffer, and gives each result in
///   an `out` buffer. A block result that no result gives lives in `private` scratch, and its
///   extents must be static.
/// - Each fold of a block is a loop nest of its own, which runs the reads and the body at each
///   point. Without a nest, a fold into a new result walks the parallel iterators, then the
///   reduction iterators inside them. It folds into private scratch as the combine kind's reference
///   does. A fold into an operand copies the operand, then walks every point in order.
/// - `tile.block` and `tile.group` outside a dispatch are errors: `seal-dispatches` outlines them.
#[pass(name = "tile-to-loop", stats(lowered = "dispatches lowered"))]
pub struct TileToLoop;

impl TileToLoop {
    /// The target and the patterns of the conversion: every dialect but `tile` is legal, and each
    /// `tile` op is as its [`ConvertOp`] decision says.
    pub fn conversion() -> (ConversionTarget, PatternSet) {
        let mut target = ConversionTarget::of_dialects(&[
            Dialect::Func,
            Dialect::Tensor,
            Dialect::Arith,
            Dialect::Host,
            Dialect::Loop,
        ]);
        let mut patterns = PatternSet::new();
        TileOp::convert::<TileToLoop>(&mut target, &mut patterns);
        (target, patterns)
    }
}

impl Pass for TileToLoop {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            folds: false,
            patterns: Self::conversion().1,
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let (target, patterns) = Self::conversion();
        let lowered = apply_conversion(module, &target, &patterns, Mode::Full)?;
        TileToLoop::LOWERED.add(lowered);
        Ok(lowered > 0)
    }
}

/// The decision of `tile-to-loop` for each `tile` op.
macro_rules! decide {
    ($($op:ident => $form:expr,)*) => {$(
        impl ConvertOp<TileToLoop> for tile::$op {
            fn form() -> OpForm {
                $form
            }
        }
    )*};
}

decide! {
    Dispatch => OpForm::rewrite(DispatchToLoop),
    Block => OpForm::Illegal("a block lowers in a dispatch; seal-dispatches outlines it into one"),
    Group => OpForm::Illegal("a group lowers as a dispatch; seal-dispatches outlines it into one"),
    Index => OpForm::Illegal("tile.index lowers with the body of its block"),
    Yield => OpForm::Illegal("tile.yield lowers with the body of its block"),
}
