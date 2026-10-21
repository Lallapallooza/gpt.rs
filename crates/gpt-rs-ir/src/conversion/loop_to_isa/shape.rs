//! How the vector code of a `loop.atom` computes each op of its body, from how the values of the
//! body vary over the points of its tile, which `loop-to-isa` writes the atom's vector code from.

use std::collections::HashMap;

use anyhow::{anyhow, ensure, Result};
use tinyvec::ArrayVec;

use crate::dialects::isa;
use crate::dialects::loops::{
    AffineForms, AtomOp, Index, IndexOp, Load, LoadOp, Prefetch, PrefetchOp, Store, StoreOp,
};
use crate::ir::{
    Dim, IndexExpr, Inst, Module, PerGroupAxis, Ranked, Value, ValueRef, MAX_GROUP_RANK, MAX_ITERS,
};
use crate::traits::Terminator;
use crate::Op;

/// A value for each entry of a point's coordinates: one for each axis of the tile but the last,
/// then two for the last, its vector and its lane (the coordinate `p` along it is
/// `lanes * vector + lane`).
pub(super) type PerEntry<T> = ArrayVec<[T; MAX_GROUP_RANK + 1]>;

/// How a value of a `loop.atom`'s body varies over the points of its tile, as its vector code
/// takes it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Varies {
    /// The same at every point: a scalar computed once, before the atom.
    Uniform,
    /// An index, a base plus `coefficients[entry]` times each entry of the point's coordinates.
    Affine(PerEntry<i64>),
    /// Along the axes of the tile that it marks, in the lanes of a group of registers.
    Lanes(PerGroupAxis<bool>),
}

/// How an access of memory in the body of a `loop.atom` moves with the point.
#[derive(Debug, Clone, Copy)]
pub(super) struct Moves {
    /// How far each index of the access moves with each entry of the point's coordinates,
    /// `coefficients[index][entry]` ([`PerEntry`]).
    pub(super) coefficients: Ranked<PerEntry<i64>>,
    /// The tile axes along which the access moves: the last with its vectors or its lanes.
    pub(super) along: PerGroupAxis<bool>,
    /// The tile axes along which the access's mask varies: those it moves along whose extent may
    /// stop short of the tile. An access of a `reg` buffer, which holds the whole tile, has none.
    pub(super) masked: PerGroupAxis<bool>,
}

/// How the vector code computes one op of the body of a `loop.atom`.
pub(super) enum InVectors<'m> {
    /// A `loop.index`: its base before the atom, plus `coefficients[entry]` times each entry of the
    /// point's coordinates, all 0 for an index the same at every point.
    Index(IndexOp<'m>, PerEntry<i64>),
    /// A load at the same element for every point: one `loop.load` before the atom.
    UniformLoad(LoadOp<'m>),
    /// A load whose last index moves with the lanes by 1, and no other index: `isa.load`.
    VectorLoad(LoadOp<'m>, Moves),
    /// A load whose indices do not move with the lanes: `isa.broadcast`.
    Broadcast(LoadOp<'m>, Moves),
    /// A store along the lanes: `isa.store`.
    Store(StoreOp<'m>, Moves),
    /// A prefetch at the same element for every point: one `loop.prefetch` before the atom.
    UniformPrefetch(PrefetchOp<'m>),
    /// A prefetch that moves with the point: `isa.prefetch`.
    Prefetch(PrefetchOp<'m>, Moves),
    /// An op on values the same at every point: the op itself, before the atom.
    Uniform,
    /// The lane-wise `isa` op `op` on the registers of the op's operands, whose value varies along
    /// the tile axes `along`.
    Lanewise { op: Op, along: PerGroupAxis<bool> },
}

/// How the vector code computes the body of a `loop.atom`, from how each value of the body varies
/// over the points of its tile.
pub(super) struct AtomShape<'m> {
    /// The extents of the vectors that hold the whole tile ([`isa::tile_vectors`]), which are
    /// also the points of each entry of a point's coordinates ([`PerEntry`]).
    pub(super) vectors: Ranked<usize>,
    /// Each op of the body but its terminator, in order, and how the vector code computes it.
    pub(super) ops: Vec<(Inst, InVectors<'m>)>,
    /// Whether the extent of each tile axis may stop short of the tile.
    partial: PerGroupAxis<bool>,
    values: HashMap<Value, Varies>,
    /// Each index of the body as an expression of the entries of the point's coordinates, `d0`
    /// to `d{tile}` as in [`PerEntry`], and of the indices outside the atom that the affine forms
    /// of its operands name, each at a position of `outside` after them.
    forms: HashMap<Value, IndexExpr>,
    outside: Vec<Value>,
}

impl<'m> AtomShape<'m> {
    /// How the vector code computes the body of `atom`, or the rule the body breaks. Each op is an
    /// index, a sum of multiples of the coordinates and of values the same at every point; a load,
    /// a store or a prefetch at such indices; an op on values the same at every point that runs
    /// once before the atom ([`computes_once`]); or an `arith` op with a lane-wise `isa` form, on
    /// values none of which is an index that moves with the point. A load moves along the lanes
    /// by 1 on the last index of its buffer and on no other, or not along the lanes. A store
    /// moves along the lanes by 1 on the last index alone, and along every axis its value varies
    /// on. A buffer of the body is the same at every point: an op that would give another at each
    /// point, such as a `loop.pick` by a coordinate, has no vector form.
    ///
    /// An index moves with the point as the canonical form of its expression says, its operands
    /// written as their forms: a coordinate as its entries, the last as `lanes * vector + lane`,
    /// and an index outside the atom as its affine form over the loops around. Take an index that
    /// is a base plus the coordinate along the lanes, where the base is a multiple of k. When k is
    /// the lanes, or is at least the tile's points along them, its `floordiv k` and `mod k` are a
    /// base plus a vector and a lane. This is the case for a tile walked in tiles of k.
    pub(super) fn of(atom: &AtomOp<'m>) -> Result<AtomShape<'m>> {
        let module = atom.module();
        let tile = atom.tile;
        let lanes = atom.lanes;
        let mut affine_forms = AffineForms::new(module);
        let mut partial = PerGroupAxis::new();
        for (&extent, &points) in atom.extents().iter().zip(&tile) {
            partial.push(affine_forms.of(extent.id()).as_constant() != Some(points as i64));
        }
        let mut shape = AtomShape {
            vectors: isa::tile_vectors(&tile, lanes),
            ops: Vec::new(),
            partial,
            values: HashMap::new(),
            forms: HashMap::new(),
            outside: Vec::new(),
        };
        for (axis, &coordinate) in module.args(atom.body().id()).iter().enumerate() {
            let mut coefficients =
                PerEntry::from_array_len([0; MAX_GROUP_RANK + 1], tile.len() + 1);
            let form = match axis + 1 == tile.len() {
                true => {
                    coefficients[axis] = lanes as i64;
                    coefficients[axis + 1] = 1;
                    IndexExpr::constant(lanes as i64) * IndexExpr::iter(axis)
                        + IndexExpr::iter(axis + 1)
                }
                false => {
                    coefficients[axis] = 1;
                    IndexExpr::iter(axis)
                }
            };
            shape
                .values
                .insert(coordinate, Varies::Affine(coefficients));
            shape.forms.insert(coordinate, form);
        }
        for inst in atom.body().insts() {
            if module.op(inst).has_trait::<Terminator>() {
                break;
            }
            let computed = shape.computed(module, &mut affine_forms, inst)?;
            shape.ops.push((inst, computed));
        }
        Ok(shape)
    }

    /// The rank of the tile.
    pub(super) fn rank(&self) -> usize {
        self.vectors.len() - 1
    }

    /// How the vector code computes the op of `inst`, or the rule it breaks. The shape gains how
    /// the op's value varies.
    fn computed(
        &mut self,
        module: &'m Module,
        affine_forms: &mut AffineForms,
        inst: Inst,
    ) -> Result<InVectors<'m>> {
        if let Some(index) = module.op_as::<Index>(inst) {
            let coefficients = self.index(affine_forms, &index)?;
            return Ok(InVectors::Index(index, coefficients));
        }
        if let Some(load) = module.op_as::<Load>(inst) {
            return self.load(module, load);
        }
        if let Some(store) = module.op_as::<Store>(inst) {
            let moves = self.store(module, &store)?;
            return Ok(InVectors::Store(store, moves));
        }
        if let Some(prefetch) = module.op_as::<Prefetch>(inst) {
            let moves = self.moves(module, prefetch.buffer(), prefetch.indices())?;
            return Ok(match moves.along.contains(&true) {
                true => InVectors::Prefetch(prefetch, moves),
                false => InVectors::UniformPrefetch(prefetch),
            });
        }
        let op = module.op(inst);
        let operands = module.operands(inst);
        if self.uniform(operands) && computes_once(module, inst) {
            self.values.insert(module.results(inst)[0], Varies::Uniform);
            return Ok(InVectors::Uniform);
        }
        let lanewise = isa::lanewise(op)
            .ok_or_else(|| anyhow!("{} in an atom has no vector form", op.name()))?;
        let along = self.lanewise(operands);
        self.values
            .insert(module.results(inst)[0], Varies::Lanes(along));
        Ok(InVectors::Lanewise {
            op: lanewise,
            along,
        })
    }

    /// How `value` varies: the same at every point when it comes from outside the atom.
    fn varies(&self, value: Value) -> &Varies {
        self.values.get(&value).unwrap_or(&Varies::Uniform)
    }

    /// The multiple of each entry of the point's coordinates that the index `value` moves by:
    /// none for one the same at every point.
    fn coefficients(&self, value: Value) -> PerEntry<i64> {
        match self.varies(value) {
            Varies::Uniform => PerEntry::from_array_len([0; MAX_GROUP_RANK + 1], self.rank() + 1),
            Varies::Affine(coefficients) => *coefficients,
            Varies::Lanes(_) => {
                panic!("an index varies by lanes, which only scalars of loads and lane-wise ops do")
            }
        }
    }

    /// The tile axes along which an access whose indices move by `coefficients` varies: the last
    /// with its vectors or its lanes.
    fn along(&self, coefficients: &[PerEntry<i64>]) -> PerGroupAxis<bool> {
        let rank = self.rank();
        let mut along = PerGroupAxis::new();
        for axis in 0..rank {
            let entries = match axis + 1 == rank {
                true => axis..axis + 2,
                false => axis..axis + 1,
            };
            along.push(coefficients.iter().any(|moves| {
                moves[entries.clone()]
                    .iter()
                    .any(|&coefficient| coefficient != 0)
            }));
        }
        along
    }

    /// Whether an access whose indices move by `coefficients` runs along the last index of its
    /// buffer by 1 as the lanes do, and along no other index.
    fn runs_along_lanes(&self, coefficients: &[PerEntry<i64>]) -> bool {
        let lane = self.rank();
        let Some((last, outer)) = coefficients.split_last() else {
            return false;
        };
        last[lane] == 1 && outer.iter().all(|moves| moves[lane] == 0)
    }

    /// Whether each of `operands` is the same at every point.
    fn uniform(&self, operands: &[Value]) -> bool {
        operands
            .iter()
            .all(|&operand| *self.varies(operand) == Varies::Uniform)
    }

    /// `index`: the canonical form of its expression on the forms of its operands, and how it
    /// moves with each entry of the point's coordinates.
    fn index(&mut self, affine_forms: &mut AffineForms, index: &IndexOp) -> Result<PerEntry<i64>> {
        let mut operand_forms = Vec::with_capacity(index.ins().len());
        for operand in index.ins() {
            operand_forms.push(self.form(affine_forms, operand.id())?);
        }
        let substituted = index
            .expr
            .substitute(&|position| operand_forms.get(position).cloned());
        let form = substituted.canonical(&|position| self.entry_extent(position));
        let mut coefficients = PerEntry::new();
        for entry in 0..=self.rank() {
            coefficients.push(form.coefficient(entry).ok_or_else(|| {
                anyhow!(
                    "the index {} is not a multiple of the point's coordinates plus a base",
                    index.expr
                )
            })?);
        }
        let result = index.output().id();
        self.forms.insert(result, form);
        let varies = match coefficients.iter().all(|&coefficient| coefficient == 0) {
            true => Varies::Uniform,
            false => Varies::Affine(coefficients),
        };
        self.values.insert(result, varies);
        Ok(coefficients)
    }

    /// The form of the index `value`: of an index of the body, or the affine form of one outside
    /// it, over the indices that the form names, each given a position after the entries. Fails
    /// when an expression would name more iterators than [`MAX_ITERS`].
    fn form(&mut self, affine_forms: &mut AffineForms, value: Value) -> Result<IndexExpr> {
        if let Some(form) = self.forms.get(&value) {
            return Ok(form.clone());
        }
        let affine = affine_forms.of(value);
        let mut form = IndexExpr::constant(affine.constant);
        let entries = self.rank() + 1;
        for (term, coefficient) in affine.terms {
            let position = match self.outside.iter().position(|&known| known == term) {
                Some(index) => index,
                None => {
                    self.outside.push(term);
                    self.outside.len() - 1
                }
            };
            ensure!(
                entries + position < MAX_ITERS,
                "the indices of the atom name more than {} values from outside it: an index \
                 expression names at most {MAX_ITERS} iterators, {entries} of them the entries of \
                 the point",
                MAX_ITERS - entries
            );
            form = form + IndexExpr::constant(coefficient) * IndexExpr::iter(entries + position);
        }
        Ok(form)
    }

    /// The points of the entry at `position` of the point's coordinates: of a tile axis but the
    /// last, the vectors along the last, or its lanes; none for an index outside the atom.
    fn entry_extent(&self, position: usize) -> Option<Dim> {
        self.vectors
            .get(position)
            .map(|&points| Dim::Static(points))
    }

    /// How an access of `buffer` at `indices` moves with the point.
    fn moves(&self, module: &Module, buffer: ValueRef, indices: &[ValueRef]) -> Result<Moves> {
        let mut coefficients = Ranked::new();
        for index in indices {
            coefficients.push(self.coefficients(index.id()));
        }
        let along = self.along(&coefficients);
        let in_registers = isa::in_registers(module, buffer.id());
        let mut masked = PerGroupAxis::new();
        for (&moves_along, &partial) in along.iter().zip(&self.partial) {
            masked.push(moves_along && partial && !in_registers);
        }
        Ok(Moves {
            coefficients,
            along,
            masked,
        })
    }

    /// `load`: at the same element for every point, or along the axes its position moves on,
    /// by a vector load along the lanes or a broadcast into them.
    fn load(&mut self, module: &Module, load: LoadOp<'m>) -> Result<InVectors<'m>> {
        let moves = self.moves(module, load.buffer(), load.indices())?;
        let result = load.output().id();
        if !moves.along.contains(&true) {
            self.values.insert(result, Varies::Uniform);
            return Ok(InVectors::UniformLoad(load));
        }
        let lane = self.rank();
        let moves_with_lanes = moves
            .coefficients
            .iter()
            .any(|index_moves| index_moves[lane] != 0);
        let along_lanes = self.runs_along_lanes(&moves.coefficients);
        ensure!(
            along_lanes || !moves_with_lanes,
            "an atom loads from {} along its lanes with a stride: neither a vector load nor a \
             broadcast",
            load.buffer().value_type()
        );
        self.values.insert(result, Varies::Lanes(moves.along));
        Ok(match along_lanes {
            true => InVectors::VectorLoad(load, moves),
            false => InVectors::Broadcast(load, moves),
        })
    }

    /// How `store` moves, along the lanes and along every axis its value varies on.
    fn store(&self, module: &Module, store: &StoreOp) -> Result<Moves> {
        let moves = self.moves(module, store.buffer(), store.indices())?;
        ensure!(
            self.runs_along_lanes(&moves.coefficients),
            "an atom stores into {} other than along its lanes: its points would collide or \
             scatter",
            store.buffer().value_type()
        );
        let value_along = match self.varies(store.value().id()) {
            Varies::Uniform => PerGroupAxis::from_array_len([false; MAX_GROUP_RANK], self.rank()),
            Varies::Lanes(value_along) => *value_along,
            Varies::Affine(_) => panic!("a stored value is an index, which the type rules forbid"),
        };
        for (axis, (&value_moves, &position_moves)) in
            value_along.iter().zip(&moves.along).enumerate()
        {
            ensure!(
                !value_moves || position_moves || self.vectors[axis] == 1,
                "an atom stores a value that varies over points whose position does not"
            );
        }
        Ok(moves)
    }

    /// A lane-wise op on `operands`: it varies along every axis that one of them does.
    fn lanewise(&self, operands: &[Value]) -> PerGroupAxis<bool> {
        let mut along = PerGroupAxis::from_array_len([false; MAX_GROUP_RANK], self.rank());
        for &operand in operands {
            match self.varies(operand) {
                Varies::Uniform => {}
                Varies::Lanes(operand_along) => {
                    for (joined, &moves) in along.iter_mut().zip(operand_along) {
                        *joined |= moves;
                    }
                }
                Varies::Affine(_) => {
                    panic!("an operand of a lane-wise op is an index, which the type rules forbid")
                }
            }
        }
        along
    }
}

/// Whether the op of `inst`, on operands that are the same at every point, computes once before
/// an atom: an op with no effects and no regions, which gives one value. An atom runs at least one
/// point, so the op computes there as the body would.
fn computes_once(module: &Module, inst: Inst) -> bool {
    module.op(inst).effects().is_pure()
        && module.regions(inst).is_empty()
        && module.results(inst).len() == 1
}
