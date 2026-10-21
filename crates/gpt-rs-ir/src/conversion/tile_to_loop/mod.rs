//! `tile-to-loop`: each `tile.dispatch` as a `loop.func`. `dispatch.rs` holds the pattern of
//! `tile.dispatch`; `block/` and `combine.rs` the lowering of the blocks and of their combine
//! kinds; `utils.rs` the builders of `loop` ops. The `tile` dialect holds the walk of a block's
//! modes and the rules that follow from it ([`crate::dialects::tile::Walk`]). Among those rules
//! are the clips of a block's loops and the piece that its placed blocks read. The verifier checks
//! a block's schedule with the same walk. The dialect also says which results may take an
//! argument's buffer ([`crate::dialects::tile::Dispatch::updated_args`]).

mod block;
mod combine;
mod dispatch;
mod utils;

use anyhow::Result;

use self::dispatch::*;
use super::{apply_conversion, ConversionTarget, ConvertOp, Mode, OpForm};
use crate::dialects::tile::{self, TileOps};
use crate::dialects::Dialect;
use crate::passes::{pass, Pass};
use crate::rewrite::{PatternSet, Rewrites, Rewriting};

/// Lowers each `tile.dispatch` to a `loop.func` of the same name that computes the same values, bit
/// for bit but for the rounding changes its `numerics` names, with the loops that its blocks'
/// modes, holdings, clips and placements say. A block without modes walks the points of every
/// iterator one at a time, in the order of its reference walk
/// ([`Block::reference_modes`](crate::dialects::tile::Block::reference_modes)): the walk order of
/// the `tile.block` entry, or the order of the block when a fold updates an operand where its
/// points may collide. It decides nothing: each loop follows from the IR by one rule, and every
/// block that the verifier accepts lowers.
///
/// - The function takes each argument of the dispatch in an `in` buffer, and gives each result in
///   an `out` buffer. A result that a block writes as an updated copy of an argument that nothing
///   else in the dispatch reads updates the argument's buffer instead, an `inout` parameter. A
///   result that a map writes at each point may share the buffer of an argument that only the
///   map reads, and only at that point (`out<over = ..>`). A block result that no result gives
///   lives in `private` scratch, and its extents must be static.
/// - The blocks lower in runs: a block, and the blocks after it that a read places in its loops
///   (`at`). Each walk of the points of the run's first block is a loop nest of its own, which runs
///   the reads and the body at each point. Consecutive `none` folds into new results share one walk
///   ([`Block::fold_walks`](crate::dialects::tile::Block::fold_walks)), and every other fold walks
///   alone. Each of the block's modes but those of its register tile is a loop over the tiles of
///   its iterator, of the mode's size, outermost first: one after another, or one tile per worker
///   of its level (`workers`). Worker modes of one level, each right inside the one before it,
///   over another iterator and not clipped, are one `loop.forall`. It walks the tuples of their
///   tiles, which a worker takes in turn. A mode of one point walks the points of its iterator's
///   innermost tile, so every iterator that the register tile leaves out ends in such a loop, in
///   the order of the modes. A fold into an operand copies the operand, unless it updates the
///   operand's buffer, then walks every point in order.
/// - The `reg` and `lanes` modes are the register tile, innermost: one `loop.atom` over the
///   innermost tile of its iterators, in vectors of the lanes. A loop over the tiles of an iterator
///   of the register tile peels a tile that does not hold a whole tile's points: its whole tiles
///   and its last tile each have a body of their own. So where the points of a tile are known
///   before the run, the atom's extents are constants, and its tile holds those points, rounded up
///   to whole vectors along the lanes. A loop over the tiles of any other iterator masks such a
///   tile: one body serves every tile.
/// - A fold's state starts before the loop of the mode its `held scratch` names, or without a
///   holding before its first loop over a reduced iterator, one element per point of the tiles
///   the loops inside walk, and finishes after the loops inside. Without a holding, it lives in a
///   `reg` buffer, the register tile's points, or in one element of private scratch for a block
///   without a register tile. The scratch of a holding is private. Where the loops inside end with
///   a run that walks one register tile and reduced iterators alone, that run folds into a copy, in
///   a `reg` buffer, of the part of the scratch that the tile covers. The part is copied in before
///   the run and back after. Otherwise each point folds into the scratch. A prefix fold walks the
///   points of its prefix up to the element's innermost.
/// - A block placed `at reg` computes on each element that the block before it writes, where that
///   block writes it. A block placed at a mode of the first block's loops computes on the elements
///   of the piece that a tile of that mode's loop finishes: at the point where the block before it
///   finishes an element, when it reads that element alone; else once the piece is finished,
///   reading the piece's elements from private scratch.
/// - A read held in a copy reads from a copy in the buffer space that the dispatch's target gives
///   the memory it is held for. The loop of its mode makes the copy in each of its tiles, or the
///   function makes it before the nest. The copy holds the elements that the loops inside read, in
///   the order they walk them ([`Walk::copy_axes`](crate::dialects::tile::Walk::copy_axes)). A
///   loop inside over the tiles of an iterator that the read takes gives an axis for those tiles,
///   but none when the block clips the iterator. A loop of one point over such an iterator gives an
///   axis for the iterator's points. The iterators of the register tile that the read takes give
///   the last axes, one for the points of each. A read held in a prefetch asks its cache in each
///   tile of the loop of its mode for the element at the same place of the next tile.
/// - A loop over an iterator the block clips, around which the other iterators of a padded read's
///   coordinate stand fixed, narrows to the points where the coordinate stays in range, when a
///   point outside contributes nothing the fold keeps: the body yields the fold's identity at the
///   pad, or products of a zero pad into a sum under `skip_pad`. A loop over tiles of an iterator
///   of the register tile keeps its tiles and cuts each to its points in range instead, building
///   the tile once for each count of points a cut tile can hold, under a check of the count, so
///   that the register tile's shape stays known before the run. When the loops narrow every
///   coordinate of a read this way, the read reads without its pad. The other reads keep their pad.
/// - The `loop.func` names the dispatch's target, and says whether the dispatch contracts, which
///   `contract` then realizes.
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
        TileOps::convert::<TileToLoop>(&mut target, &mut patterns);
        (target, patterns)
    }
}

impl Pass for TileToLoop {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            patterns: Self::conversion().1,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
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
