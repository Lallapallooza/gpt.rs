//! `loop.atom` as vector code: each value of its body as the lanes of a group of registers.

use std::collections::HashMap;

use super::shape::{AtomShape, InVectors, Moves, PerEntry};
use crate::dialects::isa;
use crate::dialects::loops::{
    self, Affine, Atom, AtomOp, IndexOp, IndexRanges, LoadOp, PrefetchOp, StoreOp,
};
use crate::ir::{
    IndexExpr, Inst, Module, PerGroupAxis, Ranked, Region, Value, ValueRef, Values, F64,
    MAX_GROUP_RANK, MAX_ITERS,
};
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};
use crate::Op;

/// `loop.atom` becomes vector code before it, which computes every point of its tile at once.
#[pattern]
pub(super) struct AtomToIsa;

impl Pattern for AtomToIsa {
    type Root = Atom;

    fn rewrite(&self, atom: AtomOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = atom.module();
        let shape =
            AtomShape::of(&atom).map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        let mut realization = Realization {
            module,
            shape: &shape,
            extents: module.operands(atom.inst()),
            masks: HashMap::new(),
            splats: HashMap::new(),
            values: HashMap::new(),
            ranges: IndexRanges::new(module),
            base_ranges: HashMap::new(),
            base_forms: HashMap::new(),
        };
        realization.realize(rewriter, atom.body().id());
        rewriter.erase(&atom);
        Ok(())
    }
}

/// How a value of an atom's body varies over the points of its tile.
#[derive(Clone)]
enum Lanes {
    /// The same at every point: a scalar computed once, before the atom.
    Uniform(Value),
    /// An index that moves with the point, as the atom's shape says, from its base computed
    /// before the atom.
    Affine { base: Value },
    /// The lanes of a group of registers of the extents `dims`, which holds the value at each
    /// point: extent 1 along each axis of the tile that the value does not vary on.
    Vector { value: Value, dims: Ranked<usize> },
}

/// The vector code of one atom, as it is written before the atom.
struct Realization<'m, 's> {
    module: &'m Module,
    /// How the vector code computes each op of the body.
    shape: &'s AtomShape<'m>,
    /// The extents of the points the atom runs, one per tile axis.
    extents: &'m [Value],
    /// The masks written so far, by their extents and the extents they are repeated to.
    masks: HashMap<(Ranked<usize>, Ranked<usize>), Value>,
    /// The splats written so far, by the scalar in their lanes.
    splats: HashMap<Value, Value>,
    /// What each value of the body is, as vector code computes it.
    values: HashMap<Value, Lanes>,
    /// The ranges of the indices before the atom.
    ranges: IndexRanges<'m>,
    /// The ranges of the bases that the vector code computes before the atom.
    base_ranges: HashMap<Value, Option<(i64, i64)>>,
    /// The affine forms of those bases, over the indices outside the atom, where they are affine.
    base_forms: HashMap<Value, Option<Affine>>,
}

/// How far the position of an access that moves as `moves` says goes from one register of its
/// group to the next, per axis of the group: one coordinate of the point, or one vector of lanes
/// along the last.
fn steps(moves: &Moves) -> PerGroupAxis<Ranked<i64>> {
    let mut steps = PerGroupAxis::new();
    for axis in 0..moves.along.len() {
        let mut step = Ranked::new();
        for coefficients in &moves.coefficients {
            step.push(coefficients[axis]);
        }
        steps.push(step);
    }
    steps
}

impl Realization<'_, '_> {
    /// Writes the vector code of the body `body`, op by op, as the shape computes each.
    fn realize(&mut self, rewriter: &mut Rewriter, body: Region) {
        let zero = rewriter.create(
            loops::Index {
                expr: IndexExpr::constant(0),
            },
            Vec::<Value>::new(),
        );
        self.base_ranges.insert(zero, Some((0, 0)));
        self.base_forms.insert(zero, Some(Affine::constant(0)));
        for &coordinate in self.module.args(body) {
            self.values.insert(coordinate, Lanes::Affine { base: zero });
        }
        let shape = self.shape;
        for (inst, computed) in &shape.ops {
            // The vector code of a body op comes from it.
            rewriter.set_origin(*inst);
            match computed {
                InVectors::Index(index, coefficients) => self.index(rewriter, index, coefficients),
                InVectors::UniformLoad(load) => self.uniform_load(rewriter, load),
                InVectors::VectorLoad(load, moves) => self.load(rewriter, load, moves, true),
                InVectors::Broadcast(load, moves) => self.load(rewriter, load, moves, false),
                InVectors::Store(store, moves) => self.store(rewriter, store, moves),
                InVectors::UniformPrefetch(prefetch) => self.uniform_prefetch(rewriter, prefetch),
                InVectors::Prefetch(prefetch, moves) => self.prefetch(rewriter, prefetch, moves),
                InVectors::Uniform => self.uniform_op(rewriter, *inst),
                InVectors::Lanewise { op, along } => self.lanewise(rewriter, *inst, op, along),
            }
        }
    }

    /// The scalars before the atom that `operands` are, each the same at every point.
    fn uniform(&self, operands: &[Value]) -> Vec<Value> {
        let mut scalars = Vec::with_capacity(operands.len());
        for &operand in operands {
            scalars.push(match self.values.get(&operand) {
                None => operand,
                Some(Lanes::Uniform(value)) => *value,
                Some(_) => {
                    panic!("the shape computes once only an op on values the same at every point")
                }
            });
        }
        scalars
    }

    /// The buffer before the atom that `buffer` is: one from outside it, or one the body picks
    /// the same at every point.
    fn buffer(&self, buffer: Value) -> Value {
        match self.values.get(&buffer) {
            Some(Lanes::Uniform(picked)) => *picked,
            _ => buffer,
        }
    }

    /// The base before the atom of the index `value`: the value itself from outside the atom, the
    /// scalar of one the same at every point, or the base of one that moves with the point.
    fn base(&self, value: Value) -> Value {
        match self.values.get(&value) {
            None => value,
            Some(Lanes::Uniform(scalar)) => *scalar,
            Some(Lanes::Affine { base, .. }) => *base,
            Some(Lanes::Vector { .. }) => panic!("the shape of the atom keeps indices affine"),
        }
    }

    /// The bases before the atom of `indices`, the position of an access at the first point.
    fn bases(&self, indices: &[ValueRef]) -> Vec<Value> {
        let mut bases = Vec::with_capacity(indices.len());
        for index in indices {
            bases.push(self.base(index.id()));
        }
        bases
    }

    /// `index`, which moves by `coefficients` with the entries of the point's coordinates: its
    /// base, of the operands' bases, before the atom.
    fn index(&mut self, rewriter: &mut Rewriter, index: &IndexOp, coefficients: &PerEntry<i64>) {
        let expr = &index.expr;
        let bases = self.bases(index.ins());
        let mut ranges = [None; MAX_ITERS];
        for (range, &base) in ranges.iter_mut().zip(&bases) {
            *range = self.range(base);
        }
        let range = expr.bounds(&|position| ranges[position]);
        let mut operand_forms = Vec::with_capacity(bases.len());
        let mut affine = true;
        for &operand_base in &bases {
            match self.base_form(operand_base) {
                Some(form) => operand_forms.push(form),
                None => affine = false,
            }
        }
        let form = match affine {
            true => Affine::of_expr(expr, &operand_forms),
            false => None,
        };
        let base = rewriter.create(loops::Index { expr: expr.clone() }, bases);
        self.base_ranges.insert(base, range);
        self.base_forms.insert(base, form);
        let lanes = match coefficients.iter().all(|&coeff| coeff == 0) {
            true => Lanes::Uniform(base),
            false => Lanes::Affine { base },
        };
        self.values.insert(index.output().id(), lanes);
    }

    /// The extents of a group of registers that varies along the tile axes `along`: those of the
    /// vectors of the whole tile, and 1 along each other axis.
    fn dims(&self, along: &[bool]) -> Ranked<usize> {
        let mut dims = self.shape.vectors;
        let rank = self.shape.rank();
        for (extent, &varies) in dims[..rank].iter_mut().zip(along) {
            if !varies {
                *extent = 1;
            }
        }
        dims
    }

    /// The mask of the points of the tile for an access by registers of extents `dims` that moves
    /// as `moves` says, or none when the access takes none. It varies along the axes that `moves`
    /// masks alone, and repeats along the others.
    fn mask(
        &mut self,
        rewriter: &mut Rewriter,
        dims: &Ranked<usize>,
        moves: &Moves,
    ) -> Option<Value> {
        if !moves.masked.contains(&true) {
            return None;
        }
        let mask_dims = self.dims(&moves.masked);
        Some(self.repeated_mask(rewriter, mask_dims, *dims))
    }

    /// The mask of the points of the tile in registers of extents `mask_dims`, repeated to `dims`:
    /// each written once, where an access first takes it.
    fn repeated_mask(
        &mut self,
        rewriter: &mut Rewriter,
        mask_dims: Ranked<usize>,
        dims: Ranked<usize>,
    ) -> Value {
        if let Some(&mask) = self.masks.get(&(mask_dims, dims)) {
            return mask;
        }
        let mask = match mask_dims == dims {
            true => rewriter.create(isa::TileMask { dims }, self.extents.to_vec()),
            false => {
                let tile_mask = self.repeated_mask(rewriter, mask_dims, mask_dims);
                rewriter.create(isa::Expand { dims }, tile_mask)
            }
        };
        self.masks.insert((mask_dims, dims), mask);
        mask
    }

    /// `load`, the same at every point: one scalar load before the atom. The atom runs at least
    /// one point, so the load reads there what the body reads.
    fn uniform_load(&mut self, rewriter: &mut Rewriter, load: &LoadOp) {
        let buffer = self.buffer(load.buffer().id());
        let indices = self.bases(load.indices());
        let value = rewriter.create(
            loops::Load { pad: load.pad },
            loops::LoadOperands::<Values> { buffer, indices },
        );
        self.values
            .insert(load.output().id(), Lanes::Uniform(value));
    }

    /// `load`, which moves as `moves` says: a vector load along the lanes when `along_lanes`
    /// holds, else a broadcast into them, masked when `moves` masks it.
    fn load(&mut self, rewriter: &mut Rewriter, load: &LoadOp, moves: &Moves, along_lanes: bool) {
        let loaded = load.buffer().id();
        // The buffer itself, picked before the atom when the body picks it.
        let buffer = self.buffer(loaded);
        let base = self.bases(load.indices());
        let dims = self.dims(&moves.along);
        let steps = steps(moves);
        let (checked_pad, checked) = match load.pad {
            Some(pad_value) => self.checked(loaded, (&base, &steps), &dims, along_lanes, pad_value),
            None => (None, Vec::new()),
        };
        let lane_mask = self.mask(rewriter, &dims, moves);
        let read = isa::ReadAttrs {
            dims,
            steps,
            pad: checked_pad,
            checked,
        };
        let value = match (along_lanes, lane_mask) {
            (true, None) => rewriter.create(
                isa::Load(read),
                isa::ReadOperands::<Values> { buffer, base },
            ),
            (true, Some(mask)) => rewriter.create(
                isa::MaskedLoad(read),
                isa::MaskedReadOperands::<Values> { mask, buffer, base },
            ),
            (false, None) => rewriter.create(
                isa::Broadcast(read),
                isa::ReadOperands::<Values> { buffer, base },
            ),
            (false, Some(mask)) => rewriter.create(
                isa::MaskedBroadcast(read),
                isa::MaskedReadOperands::<Values> { mask, buffer, base },
            ),
        };
        self.values
            .insert(load.output().id(), Lanes::Vector { value, dims });
    }

    /// The range of the index `value` before the atom: of a base the vector code computes, of an
    /// index of the loops around it, or none for another value the vector code computes, such as
    /// one it loads.
    fn range(&mut self, value: Value) -> Option<(i64, i64)> {
        if let Some(&range) = self.base_ranges.get(&value) {
            return range;
        }
        if value.staged_index().is_some() {
            return None;
        }
        self.ranges.of(value)
    }

    /// The affine form of the index `value` before the atom, over the indices outside it: of a
    /// base the vector code computes, of an index of the loops around it, or none for another
    /// value the vector code computes.
    fn base_form(&mut self, value: Value) -> Option<Affine> {
        if let Some(form) = self.base_forms.get(&value) {
            return form.clone();
        }
        if value.staged_index().is_some() {
            return None;
        }
        Some(self.ranges.form(value))
    }

    /// Whether the positions from `base` plus `offset` to it plus `offset + count - 1` lie in an
    /// axis of `extent` elements wherever the loops around the atom run, by the bounds of those
    /// loops ([`IndexRanges::least`]).
    fn proves_inside(&mut self, base: Value, (offset, count): (i64, i64), extent: i64) -> bool {
        let Some(form) = self.base_form(base) else {
            return false;
        };
        let lowest = self.ranges.least(&form);
        let highest = self.ranges.least(&form.negated()).map(|least| -least);
        matches!(
            (lowest, highest),
            (Some(low), Some(high)) if low + offset >= 0 && high + offset + count <= extent
        )
    }

    /// The pad and the axes each register checks of a read of `buffer` padded with `pad`, from
    /// `base` by `steps`, in registers of extents `dims`, along the lanes when `along` holds: the
    /// axes on which the positions of a register's lanes may fall outside the buffer, from the
    /// ranges of the base. The read pads with none when no register checks one.
    fn checked(
        &mut self,
        buffer: Value,
        (base, steps): (&[Value], &[Ranked<i64>]),
        dims: &[usize],
        along: bool,
        pad: F64,
    ) -> (Option<F64>, Vec<Ranked<usize>>) {
        let ty = self
            .module
            .ty(buffer)
            .as_buffer()
            .expect("a read reads a buffer")
            .clone();
        let mut ranges: Ranked<Option<(i64, i64)>> = Ranked::new();
        for &index in base {
            ranges.push(self.range(index));
        }
        let (lanes, group) = dims.split_last().expect("registers have lanes");
        let last = base.len() - 1;
        let mut checked = Vec::new();
        let mut checks_any = false;
        for coords in isa::row_major(group) {
            let mut axes = Ranked::new();
            for (axis, (range, dim)) in ranges.iter().zip(&ty.dims).enumerate() {
                let mut offset = 0;
                for (&coord, step) in coords.iter().zip(steps) {
                    offset += coord as i64 * step[axis];
                }
                let count = match along && axis == last {
                    true => *lanes as i64,
                    false => 1,
                };
                let Some(extent) = dim.as_static().map(|extent| extent as i64) else {
                    axes.push(axis);
                    continue;
                };
                let ranged = match range {
                    Some((lowest, highest)) => {
                        lowest + offset >= 0 && highest + offset + count <= extent
                    }
                    None => false,
                };
                if !ranged && !self.proves_inside(base[axis], (offset, count), extent) {
                    axes.push(axis);
                }
            }
            checks_any |= !axes.is_empty();
            checked.push(axes);
        }
        match checks_any {
            true => (Some(pad), checked),
            false => (None, Vec::new()),
        }
    }

    /// `prefetch`, at the same element for every point: one before the atom.
    fn uniform_prefetch(&mut self, rewriter: &mut Rewriter, prefetch: &PrefetchOp) {
        let buffer = self.buffer(prefetch.buffer().id());
        let indices = self.bases(prefetch.indices());
        rewriter.create(
            loops::Prefetch {
                level: prefetch.level.clone(),
            },
            loops::PrefetchOperands::<Values> { buffer, indices },
        );
    }

    /// `prefetch`, which moves as `moves` says: one for the first lane of each register that a
    /// load at the same indices reads.
    fn prefetch(&mut self, rewriter: &mut Rewriter, prefetch: &PrefetchOp, moves: &Moves) {
        let buffer = self.buffer(prefetch.buffer().id());
        let base = self.bases(prefetch.indices());
        let dims = self.dims(&moves.along);
        rewriter.create(
            isa::Prefetch {
                dims,
                steps: steps(moves),
                level: prefetch.level.clone(),
            },
            isa::PrefetchOperands::<Values> { buffer, base },
        );
    }

    /// `store`, which moves as `moves` says: a vector store along the lanes of its value's
    /// registers, repeated to the store's, masked when `moves` masks it.
    fn store(&mut self, rewriter: &mut Rewriter, store: &StoreOp, moves: &Moves) {
        let buffer = self.buffer(store.buffer().id());
        let base = self.bases(store.indices());
        let dims = self.dims(&moves.along);
        let (stored_value, value_dims) = self.vector(rewriter, store.value().id());
        let value = self.expand(rewriter, stored_value, &value_dims, &dims);
        let steps = steps(moves);
        match self.mask(rewriter, &dims, moves) {
            Some(mask) => rewriter.create(
                isa::MaskedStore { steps },
                isa::MaskedStoreOperands::<Values> {
                    mask,
                    value,
                    buffer,
                    base,
                },
            ),
            None => rewriter.create(
                isa::Store { steps },
                isa::StoreOperands::<Values> {
                    value,
                    buffer,
                    base,
                },
            ),
        };
    }

    /// The op of `inst`, on values the same at every point: the op itself before the atom, on
    /// their scalars.
    fn uniform_op(&mut self, rewriter: &mut Rewriter, inst: Inst) {
        let module = self.module;
        let scalars = self.uniform(module.operands(inst));
        let value = rewriter.create_op(module.op(inst).clone(), scalars)[0];
        self.values
            .insert(module.results(inst)[0], Lanes::Uniform(value));
    }

    /// The op of `inst` as its lane-wise `isa` op `op`, on the registers of its operands repeated
    /// to those of its value, which varies along the tile axes `along`.
    fn lanewise(&mut self, rewriter: &mut Rewriter, inst: Inst, op: &Op, along: &[bool]) {
        let module = self.module;
        let dims = self.dims(along);
        let operands = module.operands(inst);
        let mut vectors = Vec::with_capacity(operands.len());
        for &operand in operands {
            vectors.push(self.vector(rewriter, operand));
        }
        let mut expanded = Vec::with_capacity(vectors.len());
        for (value, operand_dims) in vectors {
            expanded.push(self.expand(rewriter, value, &operand_dims, &dims));
        }
        let value = rewriter.create_op(op.clone(), expanded)[0];
        self.values
            .insert(module.results(inst)[0], Lanes::Vector { value, dims });
    }

    /// `value` in registers, and their extents: a value the same at every point in each lane of
    /// one register.
    fn vector(&mut self, rewriter: &mut Rewriter, value: Value) -> (Value, Ranked<usize>) {
        let scalar = match self.values.get(&value) {
            Some(Lanes::Vector {
                value: vector,
                dims,
            }) => return (*vector, *dims),
            Some(Lanes::Affine { .. }) => {
                panic!("the shape of the atom keeps indices out of its vector code")
            }
            Some(Lanes::Uniform(scalar)) => *scalar,
            None => value,
        };
        let dims = self.dims(&[false; MAX_GROUP_RANK]);
        if let Some(&splat) = self.splats.get(&scalar) {
            return (splat, dims);
        }
        let splat = rewriter.create(isa::Splat { dims }, scalar);
        self.splats.insert(scalar, splat);
        (splat, dims)
    }

    /// `value`, registers of extents `from`, repeated along its axes of extent 1 to `to`.
    fn expand(
        &mut self,
        rewriter: &mut Rewriter,
        value: Value,
        from: &Ranked<usize>,
        to: &Ranked<usize>,
    ) -> Value {
        if from == to {
            return value;
        }
        rewriter.create(isa::Expand { dims: *to }, value)
    }
}
