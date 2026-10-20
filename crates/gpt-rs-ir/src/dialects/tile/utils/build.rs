//! Building `tile.block`s: their attributes, and their bodies through [`ScalarBuilder`], which
//! gains `index` here.

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::dialects::arith::ScalarBuilder;
use crate::dialects::tile::ir::{Block, Fold, Index, IterList, Iters, Yield};
use crate::dialects::tile::{Iter, IterKind, Read};
use crate::ir::{AccessExpr, Constraint, DType, Dim, IndexExpr, Module, Region, Value, MAX_ITERS};
use crate::rewrite::{Build, Builder};
use crate::traits::Terminator;

impl Block {
    /// A block over `iters` with these reads and folds, and no constraints.
    pub fn new(iters: Iters, reads: Vec<Read>, folds: Vec<Fold>) -> Self {
        Self {
            iters,
            reads: reads.into(),
            folds: folds.into(),
            constraints: SmallVec::new(),
            loops: Vec::new(),
            clip: IterList::default(),
        }
    }

    /// This block restricted to the points where `constraints` hold.
    pub fn restrict(self, constraints: Vec<Constraint>) -> Self {
        Self {
            constraints: constraints.into(),
            ..self
        }
    }

    /// Creates the block on `ins` through `builder`, a rewriter or a builder. `body(body, args)`
    /// builds the body on the read values `args` and returns the values to yield. Returns the
    /// block's results.
    pub fn build(
        self,
        builder: &mut impl Build,
        ins: Vec<Value>,
        body: impl FnOnce(&mut ScalarBuilder<Builder>, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let mut unbuilt = Some(body);
        builder.create_with_regions(self, ins, |_, body_builder, args| {
            let build_body = unbuilt.take().expect("a block has one region");
            let yields = build_body(
                &mut ScalarBuilder {
                    builder: body_builder,
                },
                args,
            );
            body_builder.create(Yield, yields);
        })
    }
}

/// The operands of a block and its reads of them, each value once among the operands.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub values: Vec<Value>,
    pub reads: Vec<Read>,
}

impl Inputs {
    /// Inputs over the operands `values`, with no reads yet.
    pub fn new(values: Vec<Value>) -> Self {
        Self {
            values,
            reads: Vec::new(),
        }
    }

    /// The operand that holds `value`, added at the end when it is new.
    pub fn operand(&mut self, value: Value) -> usize {
        match self.values.iter().position(|&held| held == value) {
            Some(index) => index,
            None => {
                self.values.push(value);
                self.values.len() - 1
            }
        }
    }

    /// Reads `value` at `coords`, and returns the index of the read.
    pub fn read(
        &mut self,
        value: Value,
        coords: impl IntoIterator<Item = impl Into<AccessExpr>>,
    ) -> usize {
        let operand = self.operand(value);
        self.push(Read::new(operand, coords))
    }

    /// Adds `read`, whose operands are already among these, and returns its index.
    pub fn push(&mut self, read: Read) -> usize {
        self.reads.push(read);
        self.reads.len() - 1
    }
}

/// One parallel iterator over each of `dims`, the first ones of a block, with the expressions
/// that name them.
#[track_caller]
pub fn par_iters(dims: &[Dim]) -> (Iters, Vec<IndexExpr>) {
    let mut iters = Iters::new();
    for dim in dims {
        push_iter(&mut iters, IterKind::Par, dim.clone());
    }
    let mut coords = Vec::with_capacity(dims.len());
    for position in 0..dims.len() {
        coords.push(IndexExpr::iter(position));
    }
    (iters, coords)
}

/// Adds an iterator of `kind` over `extent` after `iters`, and gives the expression that names
/// it.
#[track_caller]
pub fn push_iter(iters: &mut Iters, kind: IterKind, extent: Dim) -> IndexExpr {
    assert!(
        iters.len() < MAX_ITERS,
        "a block has at most {MAX_ITERS} iterators"
    );
    iters.push(Iter::new(kind, extent));
    IndexExpr::iter(iters.len() - 1)
}

impl<B: Build> ScalarBuilder<'_, B> {
    /// The value of `expr` at the current point of the block, as a `dtype`.
    #[track_caller]
    pub fn index(&mut self, expr: IndexExpr, dtype: DType) -> Value {
        self.builder.create(Index { expr, dtype }, ())
    }
}

/// Creates copies of the ops of `body`, a block body, with `builder`, each operand the value that
/// `values` maps it to, and gives the values that its `tile.yield` yields. An op whose results
/// `values` maps already is left out: those values stand for it.
pub fn copy_body_ops(
    module: &Module,
    body: Region,
    builder: &mut Builder,
    values: &mut HashMap<Value, Value>,
) -> Vec<Value> {
    for inst in module.insts(body) {
        if module.op(inst).has_trait::<Terminator>() {
            let mut yielded = Vec::new();
            for value in module.operands(inst) {
                yielded.push(values[value]);
            }
            return yielded;
        }
        let results = module.results(inst);
        if !results.is_empty() && results.iter().all(|result| values.contains_key(result)) {
            continue;
        }
        builder
            .create_copy(module, inst, values)
            .expect("the ops of a verified body");
    }
    unreachable!("a block body ends with tile.yield")
}
