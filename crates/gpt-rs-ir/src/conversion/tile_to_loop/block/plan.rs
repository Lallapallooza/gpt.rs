//! How each fold of a block runs as loops, from its walk before any `loop` op is built: a list of
//! plain values, outermost first, each wrapping the ones after it, which the emitters build
//! without choosing anything.

use super::fold::StateTiles;
use super::{BlockLowering, Part};
use crate::conversion::tile_to_loop::combine::Folding;
use crate::dialects::tile::{IterKind, Placement, StatePlace, Step};
use crate::ir::{DType, Dim, Space};

/// One node of a fold's loops, which wraps the nodes after it.
#[derive(Clone)]
pub(super) enum Node<'p> {
    /// A step of the walk: a loop over the tiles of a loop of the nest, or the register tile as
    /// one `loop.atom`.
    Step(Step),
    /// The state of a fold into new results, one element per point of the tiles of the nodes
    /// after it: started, folded by them in each sweep, and finished.
    State(Box<StatePlan<'p>>),
    /// A prefix's points, up to and including the element's, of the iterator at this position.
    Prefix(usize),
    /// A copy in registers of the part of the state in memory that the register tile covers,
    /// which the nodes after it fold into.
    Copies(StateFold<'p>),
    /// A piece of the run's loops, with the tile of each iterator there. The blocks placed after
    /// the first that read across the piece take its elements from scratch once it is finished.
    Piece(Vec<Dim>),
    /// The body at one point.
    Body(Leaf<'p>),
}

/// The fold that a state belongs to: the block's fold, how its kind folds, and the dtypes of the
/// values it takes.
#[derive(Clone, Copy)]
pub(super) struct StateFold<'p> {
    pub(super) part: &'p Part<'p>,
    pub(super) folding: Folding<'p>,
    pub(super) dtypes: &'p [DType],
}

/// The state of `fold`, over the tiles `tiles`, each an iterator with the extent of its tile, in
/// `space`.
#[derive(Clone)]
pub(super) struct StatePlan<'p> {
    pub(super) fold: StateFold<'p>,
    pub(super) tiles: StateTiles,
    pub(super) space: Space,
}

/// What the body at one point does with what it yields.
#[derive(Clone, Copy)]
pub(super) enum Leaf<'p> {
    /// Writes its element of each of the `none` folds into new results `parts`.
    Maps(&'p [Part<'p>]),
    /// Writes its value into a copy of an operand, or folds it into the element it names there,
    /// for the fold `part`, whose yielded values are of `dtypes`.
    Into {
        part: &'p Part<'p>,
        dtypes: &'p [DType],
    },
    /// Folds into the element of the state of `fold` that the scope holds.
    Push(StateFold<'p>),
}

impl BlockLowering<'_> {
    /// The loops `steps` of the `none` folds `parts` into new results, which every point writes.
    pub(super) fn plan_maps<'p>(&self, parts: &'p [Part<'p>], steps: &[Step]) -> Vec<Node<'p>> {
        let nodes = [steps_of(steps), vec![Node::Body(Leaf::Maps(parts))]].concat();
        self.with_piece(nodes)
    }

    /// The loops `steps` of the fold `part` into a copy of an operand, whose yielded values are of
    /// `dtypes`: every point in order.
    pub(super) fn plan_into<'p>(
        &self,
        part: &'p Part<'p>,
        dtypes: &'p [DType],
        steps: &[Step],
    ) -> Vec<Node<'p>> {
        let leaf = Node::Body(Leaf::Into { part, dtypes });
        [steps_of(steps), vec![leaf]].concat()
    }

    /// The loops `steps` of the fold `part` into new results, whose yielded values are of
    /// `dtypes`. The loops outside where its state starts
    /// ([`crate::dialects::tile::Walk::fold_start`]) walk the elements. The state of each element
    /// of the tiles inside lives where its holding says
    /// ([`crate::dialects::tile::Walk::state_place`]). A prefix fold walks the points of its prefix
    /// innermost.
    pub(super) fn plan_fold<'p>(
        &self,
        part: &'p Part<'p>,
        dtypes: &'p [DType],
        steps: &[Step],
    ) -> Vec<Node<'p>> {
        let walk = self.walk;
        let start = walk.fold_start(self.block, steps, part.fold);
        let mut tiles = StateTiles::new();
        for (position, extent) in walk.state_tiles(self.block, steps, start) {
            let points = extent
                .as_static()
                .expect("the verifier keeps the tiles of a fold's state static");
            tiles.push((position, points));
        }
        let fold = StateFold {
            part,
            folding: part.fold.combine.fold(),
            dtypes: &dtypes[part.yields.clone()],
        };
        let (inside, space) = match walk.state_place(self.block, steps, start, part.fold) {
            StatePlace::Reg => (steps_of(&steps[start..]), Space::Reg),
            StatePlace::Private => (steps_of(&steps[start..]), Space::Private),
            StatePlace::RegCopies(run) => (
                [
                    steps_of(&steps[start..run]),
                    vec![Node::Copies(fold)],
                    steps_of(&steps[run..]),
                ]
                .concat(),
                Space::Private,
            ),
        };
        let mut nodes = [
            steps_of(&steps[..start]),
            vec![Node::State(Box::new(StatePlan { fold, tiles, space }))],
            inside,
        ]
        .concat();
        let prefix = self
            .block
            .iters
            .iter()
            .position(|iter| iter.kind == IterKind::Prefix);
        if let Some(position) = prefix {
            nodes.push(Node::Prefix(position));
        }
        nodes.push(Node::Body(Leaf::Push(fold)));
        self.with_piece(nodes)
    }

    /// `nodes`, the loops of the first block of a run, with the piece that the blocks placed
    /// after it read across, when one does: around the steps from the one where the piece of
    /// their placement starts ([`crate::dialects::tile::Walk::piece`]), and the nodes after them.
    fn with_piece<'p>(&self, mut nodes: Vec<Node<'p>>) -> Vec<Node<'p>> {
        if self.placed.iter().all(|placed| placed.at_point) {
            return nodes;
        }
        let placement = self.placement().expect("a placed block has a placement");
        let (start, tiles) = self.walk.piece(self.block, placement);
        let mut steps = 0;
        let mut at = nodes.len();
        for (index, node) in nodes.iter().enumerate() {
            if steps == start {
                at = index;
                break;
            }
            steps += usize::from(matches!(node, Node::Step(_)));
        }
        nodes.insert(at, Node::Piece(tiles));
        nodes
    }

    /// Where the blocks placed after this one run, as their reads say.
    fn placement(&self) -> Option<Placement> {
        let placed = self.placed.first()?;
        placed.block.reads.iter().find_map(|read| read.at)
    }
}

/// The node of each of `steps`, in order.
fn steps_of<'p>(steps: &[Step]) -> Vec<Node<'p>> {
    let mut nodes = Vec::with_capacity(steps.len());
    for &step in steps {
        nodes.push(Node::Step(step));
    }
    nodes
}
