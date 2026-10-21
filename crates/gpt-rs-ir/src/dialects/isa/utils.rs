//! Helpers of the `isa` dialect: the lane-wise form of each scalar op; the accesses of the memory
//! ops; what a target's code for an op needs; and the registers of a register tile's code: the
//! order its code computes them in, and how many it holds at once.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::dialects::isa::ir::*;
use crate::dialects::loops::AffineForms;
use crate::dialects::{arith, Dialect};
use crate::ir::{
    DType, InOp, Inst, Module, PerGroupAxis, Ranked, Region, Space, Value, ValueRef, VectorType,
    F64, MAX_RANK,
};
use crate::targets::{Feature, Isa};
use crate::traits::Terminator;
use crate::Op;

/// The `isa` op that applies the scalar op `op` to every lane, when the dialect has one: the
/// lane-wise form of an `arith` op in the body of a register tile.
pub fn lanewise(op: &Op) -> Option<Op> {
    if let Some(cast) = op.get::<arith::Cast>() {
        return Some(Cast { to: cast.to }.into());
    }
    if let Some(bitcast) = op.get::<arith::Bitcast>() {
        return Some(Bitcast { to: bitcast.to }.into());
    }
    if op.get::<arith::Select>().is_some() {
        return Some(Select.into());
    }
    if op.get::<arith::Fma>().is_some() {
        return Some(Fma.into());
    }
    FORMS.iter().find_map(|form| (form.lanewise)(op))
}

/// Whether `op` is a lane-wise `isa` op, which applies an `arith` op to each lane: one that
/// [`lanewise`] gives.
pub fn is_lanewise(op: &Op) -> bool {
    op.get::<Cast>().is_some()
        || op.get::<Bitcast>().is_some()
        || op.get::<Select>().is_some()
        || op.get::<Fma>().is_some()
        || FORMS.iter().any(|form| (form.is_lanewise)(op))
}

/// A lane-wise `isa` op of a family, and the `arith` op it applies to each lane: each recognizes
/// its op.
struct Form {
    /// The lane-wise op of `op`, when `op` is the `arith` op.
    lanewise: fn(&Op) -> Option<Op>,
    /// Whether `op` is the lane-wise op.
    is_lanewise: fn(&Op) -> bool,
}

/// The form of the lane-wise op `T`, which applies `Scalar` to each lane.
const fn form<T: InOp + Default, Scalar: InOp>() -> Form {
    Form {
        lanewise: |op| op.get::<Scalar>().map(|_| T::default().into()),
        is_lanewise: |op| op.get::<T>().is_some(),
    }
}

/// Every [`Unary`], [`Binary`] and [`Compare`] op.
const FORMS: &[Form] = &[
    form::<Neg, <Neg as Unary>::Scalar>(),
    form::<Abs, <Abs as Unary>::Scalar>(),
    form::<Exp, <Exp as Unary>::Scalar>(),
    form::<Log, <Log as Unary>::Scalar>(),
    form::<Tanh, <Tanh as Unary>::Scalar>(),
    form::<Erf, <Erf as Unary>::Scalar>(),
    form::<Rsqrt, <Rsqrt as Unary>::Scalar>(),
    form::<Reciprocal, <Reciprocal as Unary>::Scalar>(),
    form::<Sin, <Sin as Unary>::Scalar>(),
    form::<Cos, <Cos as Unary>::Scalar>(),
    form::<Not, <Not as Unary>::Scalar>(),
    form::<Add, <Add as Binary>::Scalar>(),
    form::<Sub, <Sub as Binary>::Scalar>(),
    form::<Mul, <Mul as Binary>::Scalar>(),
    form::<Div, <Div as Binary>::Scalar>(),
    form::<Maximum, <Maximum as Binary>::Scalar>(),
    form::<Minimum, <Minimum as Binary>::Scalar>(),
    form::<And, <And as Binary>::Scalar>(),
    form::<Or, <Or as Binary>::Scalar>(),
    form::<Xor, <Xor as Binary>::Scalar>(),
    form::<ShiftLeft, <ShiftLeft as Binary>::Scalar>(),
    form::<ShiftRightLogical, <ShiftRightLogical as Binary>::Scalar>(),
    form::<ShiftRightArithmetic, <ShiftRightArithmetic as Binary>::Scalar>(),
    form::<Less, <Less as Compare>::Scalar>(),
    form::<LessEqual, <LessEqual as Compare>::Scalar>(),
    form::<Equal, <Equal as Compare>::Scalar>(),
    form::<GreaterEqual, <GreaterEqual as Compare>::Scalar>(),
    form::<Greater, <Greater as Compare>::Scalar>(),
    form::<NotEqual, <NotEqual as Compare>::Scalar>(),
];

// ------------------------------------------------------------------------------------------------
// Memory ops.

/// The operands and attributes of an `isa` memory op: a read, a store or a prefetch.
#[derive(Debug, Clone, PartialEq)]
pub struct Access {
    /// The buffer it accesses.
    pub buffer: Value,
    /// The position of its first register, one index per axis of the buffer.
    pub base: Vec<Value>,
    /// For each axis of the group, how far the position moves on each axis of the buffer from one
    /// register to the next.
    pub steps: PerGroupAxis<Ranked<i64>>,
    /// The mask of the lanes it accesses, for a masked op.
    pub mask: Option<Value>,
    /// What a read gives where it falls outside the buffer.
    pub pad: Option<F64>,
    /// The axes on which each register's lanes may fall outside the buffer: every axis when empty.
    pub checked: Vec<Ranked<usize>>,
    /// The stored registers, for a store.
    pub value: Option<Value>,
    /// The type of the registers it moves, or of those a prefetch asks for.
    pub vector: VectorType,
    /// Whether lane `l` of a register reaches the element `l` further along the last axis of the
    /// buffer than the register's position, as a load and a store do, rather than every lane
    /// reaching the element there, as a broadcast does and a prefetch asks for.
    pub along: bool,
}

impl Access {
    /// The access of the op of `inst`, when it is a memory op.
    pub fn of(module: &Module, inst: Inst) -> Option<Self> {
        Self::read::<Load>(module, inst)
            .or_else(|| Self::read::<Broadcast>(module, inst))
            .or_else(|| Self::masked_read::<MaskedLoad>(module, inst))
            .or_else(|| Self::masked_read::<MaskedBroadcast>(module, inst))
            .or_else(|| Self::store(module, inst))
            .or_else(|| Self::prefetch(module, inst))
    }

    /// The access of the op of `inst`, when it is the [`Read`] op `T`.
    fn read<T: Read>(module: &Module, inst: Inst) -> Option<Self> {
        let read = module.op_as::<T>(inst)?;
        let vector = read.output().value_type().as_vector()?.clone();
        Some(Self {
            pad: read.pad,
            checked: read.checked.clone(),
            ..Self::new(read.buffer(), read.base(), read.steps, vector, T::ALONG)
        })
    }

    /// The access of the op of `inst`, when it is the [`MaskedRead`] op `T`.
    fn masked_read<T: MaskedRead>(module: &Module, inst: Inst) -> Option<Self> {
        let read = module.op_as::<T>(inst)?;
        let vector = read.output().value_type().as_vector()?.clone();
        Some(Self {
            mask: Some(read.mask().id()),
            pad: read.pad,
            checked: read.checked.clone(),
            ..Self::new(read.buffer(), read.base(), read.steps, vector, T::ALONG)
        })
    }

    /// The access of the op of `inst`, when it is `isa.store` or `isa.masked_store`.
    fn store(module: &Module, inst: Inst) -> Option<Self> {
        if let Some(store) = module.op_as::<MaskedStore>(inst) {
            let vector = store.value().value_type().as_vector()?.clone();
            return Some(Self {
                mask: Some(store.mask().id()),
                value: Some(store.value().id()),
                ..Self::new(store.buffer(), store.base(), store.steps, vector, true)
            });
        }
        let store = module.op_as::<Store>(inst)?;
        let vector = store.value().value_type().as_vector()?.clone();
        Some(Self {
            value: Some(store.value().id()),
            ..Self::new(store.buffer(), store.base(), store.steps, vector, true)
        })
    }

    /// The access of the op of `inst`, when it is `isa.prefetch`.
    fn prefetch(module: &Module, inst: Inst) -> Option<Self> {
        let prefetch = module.op_as::<Prefetch>(inst)?;
        let dtype = prefetch.buffer().value_type().as_buffer()?.dtype;
        let vector = VectorType::new(dtype, &prefetch.dims);
        Some(Self::new(
            prefetch.buffer(),
            prefetch.base(),
            prefetch.steps,
            vector,
            false,
        ))
    }

    /// An access of `buffer` from `base` by registers of type `vector`, which moves by `steps`
    /// from one register to the next: with no mask, no pad and no stored value.
    fn new(
        buffer: ValueRef,
        base: &[ValueRef],
        steps: PerGroupAxis<Ranked<i64>>,
        vector: VectorType,
        along: bool,
    ) -> Self {
        let mut indices = Vec::with_capacity(base.len());
        for index in base {
            indices.push(index.id());
        }
        Self {
            buffer: buffer.id(),
            base: indices,
            steps,
            mask: None,
            pad: None,
            checked: Vec::new(),
            value: None,
            vector,
            along,
        }
    }

    /// The position of the first lane of each register, in row-major order of the group, from
    /// the base position `base`.
    pub fn positions(&self, base: &[i64]) -> Vec<Ranked<i64>> {
        let mut positions = Vec::new();
        for coords in row_major(self.vector.group()) {
            positions.push(register_position(base, &self.steps, &coords));
        }
        positions
    }
}

/// The position of the first lane of the register at `coords` in a group, from the base position
/// `base`, which moves by `steps` from one register to the next: `base + coords[0] * steps[0] +
/// ...`.
pub(super) fn register_position(
    base: &[i64],
    steps: &[Ranked<i64>],
    coords: &[usize],
) -> Ranked<i64> {
    let mut position = Ranked::new();
    position.extend_from_slice(base);
    for (&coord, step) in coords.iter().zip(steps) {
        for (axis, &offset) in step.iter().enumerate() {
            position[axis] += coord as i64 * offset;
        }
    }
    position
}

// ------------------------------------------------------------------------------------------------
// What a target's code needs.

/// What a target's code for one `isa` op needs besides its operands and its result.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Needs {
    /// The extensions of the instruction set that its instructions need.
    pub features: &'static [Feature],
    /// The mask registers it holds besides its operands and its result.
    pub scratch_masks: usize,
    /// The vector registers it holds besides its operands and its result.
    pub scratch_vectors: usize,
    /// The operands, by position, that it can read from memory, one of them at a time: a
    /// constant in every lane that it reads there holds no register for the read.
    pub memory_operands: &'static [usize],
}

/// How a target computes the ops of the `isa` dialect: for each op, whether the target has code for
/// it on the registers it computes, and what that code needs ([`Needs`]). Every check of vector
/// code against a target, and the count of the registers that code holds, asks these rules.
pub trait Rules: Sync {
    /// What the target's code for the `isa` op of `inst` needs, or why the target has none.
    fn needs(&self, module: &Module, inst: Inst) -> Result<Needs>;

    /// What the target's code for the lane-wise `isa` op `op` needs on registers of `lanes` lanes
    /// of `dtype`, the dtype of its first operand, or of its values for `isa.select`, or why the
    /// target has none.
    fn lanewise(&self, op: &Op, dtype: DType, lanes: usize) -> Result<Needs>;

    /// What the target's registers of `lanes` lanes of `dtype` need, which its loads and stores
    /// move, or why the target holds no such register.
    fn holds(&self, dtype: DType, lanes: usize) -> Result<Needs>;
}

// ------------------------------------------------------------------------------------------------
// The registers of a register tile's code.

/// The extents of the vectors that hold a register tile of extents `tile` in vectors of `lanes`
/// lanes along its last axis: an axis of registers per tile axis, the last counting the vectors
/// of lanes along the tile's last axis, then the lanes.
pub fn tile_vectors(tile: &[usize], lanes: usize) -> Ranked<usize> {
    let (&last, outer) = tile.split_last().expect("a register tile has an axis");
    let mut dims = Ranked::new();
    dims.extend_from_slice(outer);
    dims.push(last / lanes);
    dims.push(lanes);
    dims
}

/// The lanes of the registers that hold the `reg` buffer `buffer`: those of the first `isa` op
/// that reaches it, or none when none does.
pub fn reg_lanes(module: &Module, buffer: Value) -> Option<usize> {
    for used in module.uses(buffer) {
        if let Some(access) = Access::of(module, used.inst) {
            return Some(access.vector.lanes());
        }
    }
    None
}

/// The registers of a `reg` buffer that `access`, an `isa` op that reaches it, moves: each by its
/// place among the buffer's registers in row-major order, in the order of [`Access::positions`].
/// Or the rule of a `reg` buffer that the access breaks. Vector code reaches a `reg` buffer in
/// whole registers at positions fixed before the run, so that a backend holds the buffer in vector
/// variables: by loads without a pad and stores of the buffer's dtype, in registers of the lanes of
/// [`reg_lanes`], which divide its last extent, each register inside the buffer and starting at a
/// multiple of the lanes.
pub fn reg_registers(
    module: &Module,
    access: &Access,
    forms: &mut AffineForms,
) -> Result<Vec<usize>, &'static str> {
    let ty = module
        .ty(access.buffer)
        .as_buffer()
        .expect("an access reaches a buffer");
    if !access.along || access.vector.dtype != ty.dtype {
        return Err("which only loads and stores of its dtype reach");
    }
    if access.pad.is_some() {
        return Err("which holds every element a read takes, with no pad");
    }
    let mut shape: Ranked<usize> = Ranked::new();
    for dim in &ty.dims {
        shape.push(
            dim.as_static()
                .expect("`loop.alloc` gives a reg buffer static extents"),
        );
    }
    let (&last, rows) = shape
        .split_last()
        .expect("a load or a store moves along an axis");
    let lanes = reg_lanes(module, access.buffer).expect("the access reaches the buffer");
    if access.vector.lanes() != lanes || last % lanes != 0 {
        return Err("whose registers hold one lane count that divides its last extent");
    }
    let mut base: Ranked<i64> = Ranked::new();
    for &index in &access.base {
        let Some(position) = forms.of(index).as_constant() else {
            return Err("at a position fixed only during the run");
        };
        base.push(position);
    }
    let per_row = (last / lanes) as i64;
    let mut registers = Vec::new();
    for position in access.positions(&base) {
        let (&column, coords) = position.split_last().expect("a position per axis");
        let mut inside = (0..last as i64).contains(&column) && column % lanes as i64 == 0;
        let mut row = 0;
        for (&coord, &extent) in coords.iter().zip(rows) {
            inside &= (0..extent as i64).contains(&coord);
            row = row * extent as i64 + coord;
        }
        if !inside {
            return Err("by a vector that straddles two of its registers or lies outside it");
        }
        registers.push((row * per_row + column / lanes as i64) as usize);
    }
    Ok(registers)
}

/// The extents of the group of registers that the `isa` op of `inst` computes or stores, or that
/// a prefetch asks for, its lanes left out.
fn group_of(module: &Module, inst: Inst) -> Option<PerGroupAxis<usize>> {
    let vector = match Access::of(module, inst) {
        Some(access) => access.vector,
        None => module
            .ty(*module.results(inst).first()?)
            .as_vector()?
            .clone(),
    };
    Some(vector.group().iter().copied().collect())
}

/// The position, in a group of extents `dims`, of the register that the register at `coords` of a
/// larger group reads: the same coordinates, or 0 along an axis of extent 1, which repeats.
pub fn repeated_position(coords: &[usize], dims: &[usize]) -> usize {
    let mut position = 0;
    for (&coord, &extent) in coords.iter().zip(dims) {
        position = position * extent + if extent == 1 { 0 } else { coord };
    }
    position
}

/// The coordinates of the element at `position`, in row-major order, of extents `dims`.
fn coords_of(position: usize, dims: &[usize]) -> Ranked<usize> {
    let mut coords = Ranked::from_array_len([0; MAX_RANK], dims.len());
    let mut rest = position;
    for axis in (0..dims.len()).rev() {
        coords[axis] = rest % dims[axis];
        rest /= dims[axis];
    }
    coords
}

/// Every coordinate of `dims`, in row-major order: one, the empty one, for no axes.
pub fn row_major(dims: &[usize]) -> Vec<Ranked<usize>> {
    let count: usize = dims.iter().product();
    let mut all = Vec::with_capacity(count);
    for position in 0..count {
        all.push(coords_of(position, dims));
    }
    all
}

/// The buffers that the op of `inst` reads, and those it writes: its operands of buffer type, as
/// its effects say.
fn buffers_of(module: &Module, inst: Inst) -> (Vec<Value>, Vec<Value>) {
    let effects = module.op(inst).effects();
    let mut buffers = Vec::new();
    for &operand in module.operands(inst) {
        if module.ty(operand).as_buffer().is_some() {
            buffers.push(operand);
        }
    }
    let read = match effects.reads {
        0 => Vec::new(),
        _ => buffers.clone(),
    };
    let written = match effects.writes {
        0 => Vec::new(),
        _ => buffers,
    };
    (read, written)
}

/// Whether `buffer` is a `reg` buffer, whose elements a register tile holds in registers: the
/// vector code reaches it only in whole registers, and a backend holds it in vector variables.
pub fn in_registers(module: &Module, buffer: Value) -> bool {
    module
        .ty(buffer)
        .as_buffer()
        .is_some_and(|ty| ty.space == Space::Reg)
}

/// The runs of `region` whose `isa` code computes register by register: the longest sequences of
/// its ops that hold `isa` ops and scalar ops computed before them, in order. A scalar op joins a
/// run when it has no region, is no terminator, writes no buffer, uses no register, and reads no
/// buffer that an `isa` op of the run writes. The points of a run are those of one register tile,
/// so its registers read only registers at their own positions. Two kinds of `isa` op end the run
/// before them. The first shares a buffer, other than a `reg` buffer, with an earlier op of the
/// run, and one of the two writes it. The second belongs to another register tile. Its group has
/// another rank, or along some axis both it and the run span more than one register, but not the
/// same number. An `isa` op without a group counts as a group of rank 0.
pub fn vector_runs(module: &Module, region: Region) -> Vec<Vec<Inst>> {
    let mut runs = Vec::new();
    let mut run: Vec<Inst> = Vec::new();
    let mut accessed: HashSet<Value> = HashSet::new();
    let mut written: HashSet<Value> = HashSet::new();
    // The extents that the groups of the run span, from its first `isa` op on.
    let mut tile: Option<PerGroupAxis<usize>> = None;
    for inst in module.insts(region) {
        let op = module.op(inst);
        let vector = op.dialect() == Dialect::Isa;
        let (reads, writes) = buffers_of(module, inst);
        let mut buffers = reads.clone();
        buffers.extend(writes.iter().copied());
        let uses_registers = module
            .operands(inst)
            .iter()
            .any(|&operand| module.ty(operand).as_vector().is_some());
        let scalar = !vector
            && module.regions(inst).is_empty()
            && !op.has_trait::<Terminator>()
            && writes.is_empty()
            && !uses_registers
            && !reads.iter().any(|buffer| written.contains(buffer));
        let conflicts = vector
            && buffers.iter().any(|&buffer| {
                !in_registers(module, buffer)
                    && (written.contains(&buffer)
                        || (!writes.is_empty() && accessed.contains(&buffer)))
            });
        let group = match vector {
            true => group_of(module, inst).unwrap_or_default(),
            false => PerGroupAxis::new(),
        };
        let other_tile = vector && tile.is_some_and(|whole| !within_tile(&group, &whole));
        if !(vector || scalar) || conflicts || other_tile {
            push_run(module, &mut runs, std::mem::take(&mut run));
            accessed.clear();
            written.clear();
            tile = None;
        }
        if !(vector || scalar) {
            continue;
        }
        if vector {
            tile = Some(widened(tile, &group));
        }
        run.push(inst);
        accessed.extend(buffers);
        written.extend(
            writes
                .into_iter()
                .filter(|&buffer| !in_registers(module, buffer)),
        );
    }
    push_run(module, &mut runs, run);
    runs
}

/// Whether a group of extents `group` is of the register tile whose groups so far span `tile`:
/// of its rank, and along each axis one register, or the tile's extent where that is more.
fn within_tile(group: &[usize], tile: &[usize]) -> bool {
    if group.len() != tile.len() {
        return false;
    }
    for (&extent, &whole) in group.iter().zip(tile) {
        if extent != 1 && whole != 1 && extent != whole {
            return false;
        }
    }
    true
}

/// The extents that the groups so far and `group` span together. `tile` holds the extents of the
/// groups so far, and [`within_tile`] accepts `group`. With no groups so far, it is `group`
/// itself.
fn widened(tile: Option<PerGroupAxis<usize>>, group: &[usize]) -> PerGroupAxis<usize> {
    let Some(mut whole) = tile else {
        return group.iter().copied().collect();
    };
    for (extent, &other) in whole.iter_mut().zip(group) {
        *extent = (*extent).max(other);
    }
    whole
}

/// Adds `run` to `runs` when it holds an `isa` op.
fn push_run(module: &Module, runs: &mut Vec<Vec<Inst>>, run: Vec<Inst>) {
    if run
        .iter()
        .any(|&inst| module.op(inst).dialect() == Dialect::Isa)
    {
        runs.push(run);
    }
}

/// What an op of a register tile's code computes, as the register model counts registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Computes {
    /// The registers of its group, of vectors or of masks.
    Registers { masks: bool },
    /// The registers of its one operand, repeated along its axes of extent 1: `isa.expand`.
    Repeats,
    /// One register, which every register of its group is: a splat of a value computed before.
    OneRegister,
    /// No register: a splat of a constant, which the code makes again where an op reads it.
    Constant,
    /// The registers of a `reg` buffer, which hold through the code: a load from it.
    Held,
    /// No register: a store or a prefetch.
    Nothing,
}

/// An operand of an op of a register tile's code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operand {
    /// The registers of the op at this position of the code.
    Op(usize),
    /// The registers of the value at this position of the values computed before the code.
    Before(usize),
}

/// One op of a register tile's code, as the register model sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RegisterOp {
    /// The extents of the group of registers it computes or stores, its lanes left out.
    group: PerGroupAxis<usize>,
    /// The registers it reads.
    operands: Vec<Operand>,
    computes: Computes,
    /// The registers of `reg` buffers that it reads or writes: each by the buffer's number and its
    /// place among the buffer's registers ([`reg_registers`]).
    buffer_registers: Vec<(usize, usize)>,
    /// The mask registers it holds besides its operands and its result: those of its code, and
    /// one for each constant mask that it reads.
    scratch_masks: usize,
    /// The vector registers it holds besides its operands and its result: those of its code, and
    /// one for each constant that it reads in a register.
    scratch_vectors: usize,
    /// Whether code after this code reads its registers.
    read_after: bool,
}

/// A value of registers computed before a register tile's code, which the code reads.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BeforeValue {
    group: PerGroupAxis<usize>,
    masks: bool,
}

/// A register tile's code, as the register model sees it: its ops, in order, and the values
/// computed before it that they read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterCode {
    ops: Vec<RegisterOp>,
    before: Vec<BeforeValue>,
}

/// One register of a register tile's code: the op that computes or stores it, by its position in
/// the code, and its position in the op's group, in row-major order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register {
    pub op: usize,
    pub position: usize,
}

/// What holds a register that the code reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Holder {
    /// A register that an op of the code computes.
    Own(Register),
    /// A register of a value computed before the code, live through it.
    Before(usize, usize),
    /// A register of a `reg` buffer, live through the code.
    Buffer,
    /// A constant in every lane, which holds no register between the ops.
    Constant,
}

/// What the code reads of the registers, computed in [`RegisterCode::order`].
struct Reads {
    /// The step of the order at which each register that the code computes is read for the last
    /// time, the end of the order for one read after the code.
    last_read: HashMap<Register, usize>,
    /// The registers of values computed before the code that it reads.
    before: HashSet<(usize, usize)>,
}

/// The most registers that a register tile's code holds at once when it computes them in
/// [`RegisterCode::order`], and an op, by its position in the code, where it holds them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pressure {
    pub vectors: usize,
    pub vectors_at: Option<usize>,
    pub masks: usize,
    pub masks_at: Option<usize>,
}

impl Pressure {
    /// The vector registers and the mask registers that the code holds on the target of `isa`: a
    /// target without mask registers holds masks in vector registers.
    pub fn held(&self, isa: &Isa) -> (usize, usize) {
        match isa.write_masks {
            0 => (self.vectors + self.masks, 0),
            _ => (self.vectors, self.masks),
        }
    }
}

/// What the `isa` op of `inst` computes, as the register model counts registers: whether it
/// reads a `reg` buffer, `reads_registers`, decides a load.
fn computes(module: &Module, inst: Inst, reads_registers: bool) -> Computes {
    let op = module.op(inst);
    if op.get::<Expand>().is_some() {
        return Computes::Repeats;
    }
    if op.get::<Splat>().is_some() {
        return match module.constant(module.operands(inst)[0]) {
            Some(_) => Computes::Constant,
            None => Computes::OneRegister,
        };
    }
    if reads_registers {
        return Computes::Held;
    }
    let Some(&result) = module.results(inst).first() else {
        return Computes::Nothing;
    };
    let masks = module
        .ty(result)
        .as_vector()
        .is_some_and(|ty| ty.dtype == DType::I1);
    Computes::Registers { masks }
}

/// The bytes of the constant in every lane of the registers `value`, when they are a splat of a
/// constant, repeated or not.
fn constant_lanes(module: &Module, value: Value) -> Option<&[u8]> {
    let inst = module.def_inst(value)?;
    let op = module.op(inst);
    if op.get::<Expand>().is_some() {
        return constant_lanes(module, module.operands(inst)[0]);
    }
    op.get::<Splat>()?;
    module.constant(module.operands(inst)[0])?.splat_bytes()
}

impl RegisterCode {
    /// The register model of `insts`, the `isa` ops of a run of [`vector_runs`], with what `rules`
    /// say the code of each op needs: the registers it holds besides its operands and its result,
    /// and the operands it reads from memory. An op reads a constant other than zero from memory
    /// in the first operand that can read it there, and every other constant in a register; code
    /// makes a zero in a register. Fails where the rules have no code for an op, or where an op
    /// reaches a `reg` buffer other than as [`reg_registers`] says.
    pub fn of_isa(module: &Module, insts: &[Inst], rules: &dyn Rules) -> Result<RegisterCode> {
        let mut positions: HashMap<Inst, usize> = HashMap::new();
        for (position, &inst) in insts.iter().enumerate() {
            positions.insert(inst, position);
        }
        let mut forms = AffineForms::new(module);
        let mut code = RegisterCode::default();
        let mut before: HashMap<Value, usize> = HashMap::new();
        let mut buffers: HashMap<Value, usize> = HashMap::new();
        for &inst in insts {
            let needs = rules.needs(module, inst)?;
            let mut operands = Vec::new();
            let mut scratch_masks = needs.scratch_masks;
            let mut scratch_vectors = needs.scratch_vectors;
            let mut reads_memory = false;
            for (slot, &operand) in module.operands(inst).iter().enumerate() {
                let Some(ty) = module.ty(operand).as_vector() else {
                    continue;
                };
                let constant = constant_lanes(module, operand);
                let from_memory = !reads_memory
                    && needs.memory_operands.contains(&slot)
                    && constant.is_some_and(|bytes| bytes.iter().any(|&byte| byte != 0));
                reads_memory |= from_memory;
                if constant.is_some() && !from_memory {
                    match ty.dtype == DType::I1 {
                        true => scratch_masks += 1,
                        false => scratch_vectors += 1,
                    }
                }
                let computed_in = module.def_inst(operand).and_then(|def| positions.get(&def));
                if let Some(&position) = computed_in {
                    operands.push(Operand::Op(position));
                    continue;
                }
                let next = code.before.len();
                let position = *before.entry(operand).or_insert(next);
                if position == next {
                    code.before.push(BeforeValue {
                        group: ty.group().iter().copied().collect(),
                        masks: ty.dtype == DType::I1,
                    });
                }
                operands.push(Operand::Before(position));
            }
            let held =
                Access::of(module, inst).filter(|access| in_registers(module, access.buffer));
            let mut buffer_registers = Vec::new();
            if let Some(access) = &held {
                let registers = reg_registers(module, access, &mut forms).map_err(|why| {
                    anyhow!(
                        "{} reaches a reg buffer of {}, {why}",
                        module.op(inst).name(),
                        module.ty(access.buffer)
                    )
                })?;
                let next = buffers.len();
                let buffer = *buffers.entry(access.buffer).or_insert(next);
                for register in registers {
                    buffer_registers.push((buffer, register));
                }
            }
            let reads_registers = held.is_some_and(|access| access.value.is_none());
            let read_after = module.results(inst).first().is_some_and(|&produced| {
                module
                    .uses(produced)
                    .iter()
                    .any(|used| !positions.contains_key(&used.inst))
            });
            code.ops.push(RegisterOp {
                group: group_of(module, inst).unwrap_or_default(),
                operands,
                computes: computes(module, inst, reads_registers),
                buffer_registers,
                scratch_masks,
                scratch_vectors,
                read_after,
            });
        }
        Ok(code)
    }

    /// The order in which every backend's code MUST compute the registers of the code. The code
    /// walks the tile that its groups span register by register, in row-major order. At each
    /// register it runs the ops in turn. An op of a smaller group, which repeats along an axis,
    /// computes each of its registers at the first register of the tile that reads it. Each op then
    /// reads only registers computed before. In a register tile, the live registers are the
    /// accumulators and one row of the registers that repeat down the rows. The operands of the op
    /// that computes the current register are live too. Groups that do not span one tile, whose
    /// extent along some axis is neither the tile's nor 1, compute op by op.
    pub fn order(&self) -> Vec<Register> {
        let mut spanned: Option<PerGroupAxis<usize>> = None;
        let mut spans_tile = true;
        for op in &self.ops {
            spans_tile &= spanned.is_none_or(|whole| within_tile(&op.group, &whole));
            spanned = Some(widened(spanned, &op.group));
        }
        let mut order = Vec::new();
        let Some(tile) = spanned.filter(|_| spans_tile) else {
            for (index, op) in self.ops.iter().enumerate() {
                let count: usize = op.group.iter().product();
                for position in 0..count {
                    order.push(Register {
                        op: index,
                        position,
                    });
                }
            }
            return order;
        };
        let mut done: Vec<Vec<bool>> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            done.push(vec![false; op.group.iter().product()]);
        }
        for coords in row_major(&tile) {
            for (index, op) in self.ops.iter().enumerate() {
                let position = repeated_position(&coords, &op.group);
                if !done[index][position] {
                    done[index][position] = true;
                    order.push(Register {
                        op: index,
                        position,
                    });
                }
            }
        }
        order
    }

    /// The extents of the group of registers of `operand`, its lanes left out.
    fn group(&self, operand: Operand) -> &[usize] {
        match operand {
            Operand::Op(index) => &self.ops[index].group,
            Operand::Before(index) => &self.before[index].group,
        }
    }

    /// What holds register `position` of `operand`.
    fn holder(&self, operand: Operand, position: usize) -> Holder {
        let index = match operand {
            Operand::Op(index) => index,
            Operand::Before(value) => return Holder::Before(value, position),
        };
        let op = &self.ops[index];
        match op.computes {
            Computes::Repeats => {
                let input = op.operands[0];
                let coords = coords_of(position, &op.group);
                self.holder(input, repeated_position(&coords, self.group(input)))
            }
            Computes::OneRegister => Holder::Own(Register {
                op: index,
                position: 0,
            }),
            Computes::Held => Holder::Buffer,
            Computes::Constant => Holder::Constant,
            Computes::Registers { .. } | Computes::Nothing => Holder::Own(Register {
                op: index,
                position,
            }),
        }
    }

    /// The most registers that the code holds at once, of vectors and of masks, when it computes
    /// them in [`RegisterCode::order`]. The rules of liveness:
    /// - A register is live from the op that computes it to the last op that reads it. The
    ///   registers an op reads for the last time free their places before it computes its own.
    /// - An op that repeats registers computes none. A splat computes one, which every register of
    ///   its group is.
    /// - A constant holds no register between the ops: the code makes it again where an op reads
    ///   it.
    /// - The code holds a `reg` buffer in registers. Each register of it that the code reads or
    ///   writes is live through the code. A load from it computes none, and a store into it ends
    ///   the stored register.
    /// - A register computed before the code and read in it is live through the code. One computed
    ///   in it and read after it is live to its end.
    /// - An op holds its scratch masks and scratch vectors while it computes or stores a register.
    ///   It holds one more register for each constant that it reads in a register.
    pub fn pressure(&self) -> Pressure {
        let order = self.order();
        let Reads { last_read, before } = self.reads(&order);
        let mut held: HashSet<(usize, usize)> = HashSet::new();
        for op in &self.ops {
            held.extend(op.buffer_registers.iter().copied());
        }
        let mut dies_at: HashMap<usize, Vec<Register>> = HashMap::new();
        for (&register, &step) in &last_read {
            dies_at.entry(step).or_default().push(register);
        }
        let before_masks = before
            .iter()
            .filter(|&&(value, _)| self.before[value].masks)
            .count();
        let before_vectors = before.len() - before_masks;
        let mut live_vectors: HashSet<Register> = HashSet::new();
        let mut live_masks: HashSet<Register> = HashSet::new();
        let mut pressure = Pressure {
            vectors: held.len() + before_vectors,
            masks: before_masks,
            ..Pressure::default()
        };
        for (step, register) in order.iter().enumerate() {
            for dead in dies_at.get(&step).into_iter().flatten() {
                live_vectors.remove(dead);
                live_masks.remove(dead);
            }
            let op = &self.ops[register.op];
            let computed =
                self.holder(Operand::Op(register.op), register.position) == Holder::Own(*register);
            if computed && last_read.contains_key(register) {
                match op.computes {
                    Computes::Registers { masks: true } => live_masks.insert(*register),
                    _ => live_vectors.insert(*register),
                };
            }
            let vectors = held.len() + before_vectors + live_vectors.len() + op.scratch_vectors;
            if vectors > pressure.vectors || pressure.vectors_at.is_none() {
                pressure.vectors = pressure.vectors.max(vectors);
                pressure.vectors_at = Some(register.op);
            }
            let masks = before_masks + live_masks.len() + op.scratch_masks;
            if masks > pressure.masks {
                pressure.masks = masks;
                pressure.masks_at = Some(register.op);
            }
        }
        pressure
    }

    /// What the code reads of the registers when it computes them in `order`.
    fn reads(&self, order: &[Register]) -> Reads {
        let mut last_read: HashMap<Register, usize> = HashMap::new();
        let mut before: HashSet<(usize, usize)> = HashSet::new();
        for (step, register) in order.iter().enumerate() {
            let op = &self.ops[register.op];
            let coords = coords_of(register.position, &op.group);
            for &operand in &op.operands {
                let position = repeated_position(&coords, self.group(operand));
                match self.holder(operand, position) {
                    Holder::Own(read) => {
                        last_read.insert(read, step);
                    }
                    Holder::Before(value, at) => {
                        before.insert((value, at));
                    }
                    Holder::Buffer | Holder::Constant => {}
                }
            }
        }
        // Registers read after the code live to its end.
        for (index, op) in self.ops.iter().enumerate() {
            if !op.read_after {
                continue;
            }
            let count: usize = op.group.iter().product();
            for position in 0..count {
                if let Holder::Own(read) = self.holder(Operand::Op(index), position) {
                    last_read.insert(read, order.len());
                }
            }
        }
        Reads { last_read, before }
    }
}
