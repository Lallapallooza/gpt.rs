//! The values the interpreter computes with: index values, scalars, tensors, `loop` buffers and
//! `isa` vectors, with the units of work that access a buffer.

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{anyhow, bail, ensure, Result};

use crate::host::{HostScalar, HostTensor};
use crate::ir::{BufferType, Space, Type, VectorType};

/// Where an access to a buffer runs: the unit of work of each `loop.forall` and `loop.atom`
/// around it, outermost first, as the run of the op and the unit's number in it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Unit(Vec<(u64, i64)>);

impl Unit {
    /// The unit outside every forall.
    pub const NONE: Unit = Unit(Vec::new());

    /// The unit of iteration `index` of the run `run` of a forall, inside this one.
    pub fn enter(&self, run: u64, index: i64) -> Unit {
        let mut path = self.0.clone();
        path.push((run, index));
        Unit(path)
    }

    /// Whether `self` and `other` may run at the same time: they run in different iterations of
    /// one run of a forall.
    pub fn concurrent(&self, other: &Unit) -> bool {
        for (&(run, index), &(other_run, other_index)) in self.0.iter().zip(&other.0) {
            if run != other_run {
                return false;
            }
            if index != other_index {
                return true;
            }
        }
        false
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let indices: Vec<String> = self.0.iter().map(|(_, index)| index.to_string()).collect();
        write!(f, "[{}]", indices.join(", "))
    }
}

/// A buffer of the `loop` dialect as the interpreter holds it: its type, with static extents, and
/// one element per position in row-major order, unset until a store writes it. Clones share the
/// elements, as the values of one buffer do.
///
/// Each element remembers the unit of its last write and the units that read it since, so that
/// an access that races with one of them fails.
///
/// The buffers of the `host` buffer form also know their memory. A slice of an arena keeps its
/// elements in cells of its own. The arena remembers which slice wrote each of its bytes last. A
/// read through a slice fails when another slice wrote its bytes since, because then the memory
/// plan let two live values share bytes.
///
/// A launch whose `out<over = k>` parameter may write the elements of its `in` parameter `k` gets
/// a handle of each that knows which elements the `out` parameter wrote, so that a read through
/// the `in` parameter of one of them fails.
#[derive(Clone)]
pub struct HostBuffer {
    cells: Arc<Mutex<BufferCells>>,
    over: Option<Over>,
}

/// The handle of an `in` parameter `k`, or of the `out<over = k>` parameter that may write its
/// elements, in one launch.
#[derive(Clone)]
struct Over {
    /// For each element, whether the `out` parameter wrote it.
    written: Arc<Mutex<Vec<bool>>>,
    /// Whether the handle is the `out` parameter's, whose stores mark what they write, rather than
    /// the `in` parameter's, which reads only what is not marked.
    writes: bool,
}

struct BufferCells {
    /// The buffer's own number, which no other buffer of the process takes, even once this one
    /// is gone.
    identity: usize,
    ty: BufferType,
    dims: Vec<usize>,
    elements: Vec<Option<HostScalar>>,
    accesses: Vec<Accesses>,
    memory: Memory,
}

/// The memory of a buffer, as the `host` buffer form sees it.
#[derive(Clone)]
enum Memory {
    /// Memory of its own.
    Own,
    /// Memory of its own that no launch and no copy writes: a buffer of `host.to_buffer`.
    ReadOnly,
    /// An arena, whose slices hold the elements.
    Arena(Arc<Mutex<Holders>>),
    /// The bytes `bytes` of an arena. `written` is true once the slice writes them.
    Slice {
        arena: Arc<Mutex<Holders>>,
        bytes: Range<u64>,
        written: bool,
    },
}

/// The number of the next buffer the process makes.
static NEXT_BUFFER: AtomicUsize = AtomicUsize::new(0);

/// Which slice wrote each byte of an arena last: disjoint byte ranges, each with the identity of
/// the slice.
#[derive(Default)]
struct Holders(Vec<(Range<u64>, usize)>);

impl Holders {
    /// Gives `bytes` to the slice `holder`.
    fn hold(&mut self, bytes: Range<u64>, holder: usize) {
        let mut kept = Vec::with_capacity(self.0.len() + 2);
        for (range, other) in self.0.drain(..) {
            if range.end <= bytes.start || bytes.end <= range.start {
                kept.push((range, other));
                continue;
            }
            if range.start < bytes.start {
                kept.push((range.start..bytes.start, other));
            }
            if bytes.end < range.end {
                kept.push((bytes.end..range.end, other));
            }
        }
        kept.push((bytes, holder));
        self.0 = kept;
    }

    /// Whether the slice `holder` wrote all of `bytes` last.
    fn holds(&self, bytes: &Range<u64>, holder: usize) -> bool {
        self.0
            .iter()
            .any(|(range, other)| *other == holder && range == bytes)
    }
}

/// The accesses to one element that a later access may race with.
#[derive(Clone, Default)]
struct Accesses {
    write: Option<Unit>,
    reads: Vec<Unit>,
}

impl BufferCells {
    /// The row-major position of `coords`, or `None` when one is out of range.
    fn position(&self, coords: &[i64]) -> Option<usize> {
        if coords.len() != self.dims.len() {
            return None;
        }
        let mut position = 0usize;
        for (&coord, &extent) in coords.iter().zip(&self.dims) {
            let index = usize::try_from(coord)
                .ok()
                .filter(|&index| index < extent)?;
            position = position * extent + index;
        }
        Some(position)
    }

    fn checked_position(&self, coords: &[i64]) -> Result<usize> {
        ensure!(
            !matches!(self.memory, Memory::Arena(_)),
            "an arena holds no elements of its own; its slices do"
        );
        self.position(coords).ok_or_else(|| {
            anyhow!(
                "{coords:?} is outside {}, whose extents are {:?}",
                self.ty,
                self.dims
            )
        })
    }

    /// Fails when an access at `position` by `unit` races with an earlier one: two units of one
    /// forall access the element, and one of them writes it.
    fn check_race(&self, position: usize, coords: &[i64], unit: &Unit, writes: bool) -> Result<()> {
        let accesses = &self.accesses[position];
        let mut earlier: Vec<&Unit> = accesses.write.iter().collect();
        if writes {
            earlier.extend(&accesses.reads);
        }
        if let Some(other) = earlier.into_iter().find(|other| other.concurrent(unit)) {
            bail!(
                "units {other} and {unit} of a loop.forall race on element {coords:?} of {}",
                self.ty
            );
        }
        Ok(())
    }
}

impl HostBuffer {
    /// A buffer of type `ty`, whose extents must be static, with no element written.
    pub fn new(ty: BufferType) -> Result<Self> {
        let dims = static_dims(&ty)?;
        let count = dims.iter().product();
        Ok(Self::of_cells(BufferCells {
            identity: NEXT_BUFFER.fetch_add(1, Ordering::Relaxed),
            ty,
            dims,
            elements: vec![None; count],
            accesses: vec![Accesses::default(); count],
            memory: Memory::Own,
        }))
    }

    /// An arena of type `ty`, the bytes of `host.arena`, whose slices hold the elements.
    pub fn arena(ty: BufferType) -> Result<Self> {
        let dims = static_dims(&ty)?;
        Ok(Self::of_cells(BufferCells {
            identity: NEXT_BUFFER.fetch_add(1, Ordering::Relaxed),
            ty,
            dims,
            elements: Vec::new(),
            accesses: Vec::new(),
            memory: Memory::Arena(Arc::default()),
        }))
    }

    fn of_cells(cells: BufferCells) -> Self {
        Self {
            cells: Arc::new(Mutex::new(cells)),
            over: None,
        }
    }

    /// The handles of `input`, the `in` parameter `k` of a launch, and of `output`, its
    /// `out<over = k>` parameter, which may be one buffer: a store through the output's handle
    /// marks the element, a load through the input's handle of a marked element fails, and so does
    /// one through the output's handle of an element it has not marked.
    pub fn written_over(input: &HostBuffer, output: &HostBuffer) -> Result<(Self, Self)> {
        let count = output.cells().elements.len();
        ensure!(
            input.cells().elements.len() == count,
            "an `out<over = k>` parameter has other extents than parameter k"
        );
        let written = Arc::new(Mutex::new(vec![false; count]));
        let handle = |buffer: &HostBuffer, writes: bool| Self {
            cells: Arc::clone(&buffer.cells),
            over: Some(Over {
                written: Arc::clone(&written),
                writes,
            }),
        };
        Ok((handle(input, false), handle(output, true)))
    }

    /// The buffer of type `ty` at byte `offset` of this arena, with no element written.
    pub fn slice(&self, offset: u64, ty: BufferType) -> Result<Self> {
        let Memory::Arena(arena_holders) = &self.cells().memory else {
            bail!("a slice is of an arena, not of {}", self.ty());
        };
        let arena = Arc::clone(arena_holders);
        let buffer = Self::new(ty)?;
        let bytes = {
            let cells = buffer.cells();
            let count = cells.elements.len();
            let size = cells
                .ty
                .dtype
                .bytes_for(count)
                .ok_or_else(|| anyhow!("{} takes more bytes than fit", cells.ty))?;
            offset..offset + size as u64
        };
        buffer.cells().memory = Memory::Slice {
            arena,
            bytes,
            written: false,
        };
        Ok(buffer)
    }

    /// A buffer that holds the elements of `tensor` and that no launch and no copy writes.
    pub fn read_only(tensor: &HostTensor) -> Result<Self> {
        let buffer = Self::of_tensor(tensor, Space::ReadOnly)?;
        buffer.cells().memory = Memory::ReadOnly;
        Ok(buffer)
    }

    /// Takes the buffer's bytes for the elements that a launch or a copy is about to write. Fails
    /// for a buffer that nothing may write.
    pub fn hold(&self) -> Result<()> {
        let mut guard = self.cells();
        let cells = &mut *guard;
        match &mut cells.memory {
            Memory::Own => Ok(()),
            Memory::ReadOnly => bail!(
                "it writes {}, which holds the elements of a tensor that nothing may write",
                cells.ty
            ),
            Memory::Arena(_) => bail!("it writes an arena, whose slices hold the elements"),
            Memory::Slice {
                arena,
                bytes,
                written,
            } => {
                let mut holders = arena.lock().unwrap_or_else(PoisonError::into_inner);
                holders.hold(bytes.clone(), cells.identity);
                *written = true;
                Ok(())
            }
        }
    }

    /// Fails when the buffer is an arena, when it is a slice that no launch and no copy wrote, or
    /// when another slice wrote one of its bytes after it wrote its elements.
    pub fn check_held(&self) -> Result<()> {
        let cells = self.cells();
        match &cells.memory {
            Memory::Own | Memory::ReadOnly => Ok(()),
            Memory::Arena(_) => bail!("it reads an arena, whose slices hold the elements"),
            Memory::Slice {
                arena,
                bytes,
                written,
            } => {
                ensure!(
                    *written,
                    "it reads {} at bytes {}..{} of an arena, which no launch and no copy wrote",
                    cells.ty,
                    bytes.start,
                    bytes.end
                );
                let holders = arena.lock().unwrap_or_else(PoisonError::into_inner);
                ensure!(
                    holders.holds(bytes, cells.identity),
                    "it reads {} at bytes {}..{} of an arena, which another value wrote since",
                    cells.ty,
                    bytes.start,
                    bytes.end
                );
                Ok(())
            }
        }
    }

    /// Writes the elements of `from`, a buffer of the same extents and dtype, into this one. A copy
    /// runs between launches, after every access before it and before every access after it, so
    /// it leaves no access that a later one may race with.
    pub fn copy_from(&self, from: &HostBuffer) -> Result<()> {
        let elements = from.cells().elements.clone();
        let mut cells = self.cells();
        ensure!(
            cells.elements.len() == elements.len(),
            "it copies {} elements into {}",
            elements.len(),
            cells.ty
        );
        cells.elements = elements;
        let count = cells.elements.len();
        cells.accesses = vec![Accesses::default(); count];
        Ok(())
    }

    /// A buffer of `space` that holds the elements of `tensor`.
    pub fn of_tensor(tensor: &HostTensor, space: Space) -> Result<Self> {
        let buffer = Self::new(BufferType::of(&tensor.tensor_type(), space))?;
        let decoded = crate::host::Decoded::new(&tensor.data);
        let mut cells = buffer.cells();
        for (index, element) in cells.elements.iter_mut().enumerate() {
            *element = Some(decoded.get(index));
        }
        drop(cells);
        Ok(buffer)
    }

    fn cells(&self) -> MutexGuard<'_, BufferCells> {
        self.cells.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The buffer's type, with its extents.
    pub fn ty(&self) -> BufferType {
        self.cells().ty.clone()
    }

    /// Whether `coords` is a position of the buffer.
    pub fn contains(&self, coords: &[i64]) -> bool {
        self.cells().position(coords).is_some()
    }

    /// The element at `coords`, read by `unit`. Fails when it is out of range, when no store wrote
    /// it, when the read races with a write, or when the handle may not read it: an `in`
    /// parameter's element that the `out<over = k>` parameter wrote.
    pub fn load(&self, coords: &[i64], unit: &Unit) -> Result<HostScalar> {
        let mut cells = self.cells();
        let position = cells.checked_position(coords)?;
        cells.check_race(position, coords, unit, false)?;
        if let Some(over) = &self.over {
            let written = over.written.lock().unwrap_or_else(PoisonError::into_inner)[position];
            match (over.writes, written) {
                (false, true) => bail!(
                    "it reads element {coords:?} of an `in` parameter that the `out<over = k>` \
                     parameter wrote already, and a launch reads parameter k only at positions \
                     that it has not written yet"
                ),
                (true, false) => bail!("no store wrote element {coords:?} of {}", cells.ty),
                _ => {}
            }
        }
        let value = cells.elements[position]
            .ok_or_else(|| anyhow!("no store wrote element {coords:?} of {}", cells.ty))?;
        let reads = &mut cells.accesses[position].reads;
        if reads.last() != Some(unit) {
            reads.push(unit.clone());
        }
        Ok(value)
    }

    /// Writes `value`, an element of the buffer's dtype, at `coords`, by `unit`. Fails when it is
    /// out of range, or when the write races with another access.
    pub fn store(&self, coords: &[i64], value: HostScalar, unit: &Unit) -> Result<()> {
        let mut cells = self.cells();
        ensure!(
            value.dtype == cells.ty.dtype,
            "a {} value stored into {}",
            value.dtype,
            cells.ty
        );
        let position = cells.checked_position(coords)?;
        cells.check_race(position, coords, unit, true)?;
        if let Some(over) = self.over.as_ref().filter(|over| over.writes) {
            over.written.lock().unwrap_or_else(PoisonError::into_inner)[position] = true;
        }
        cells.elements[position] = Some(value);
        cells.accesses[position] = Accesses {
            write: Some(unit.clone()),
            reads: Vec::new(),
        };
        Ok(())
    }

    /// The tensor of the buffer's elements. Fails when no store wrote one of them.
    pub fn to_tensor(&self) -> Result<HostTensor> {
        let cells = self.cells();
        let mut elements = Vec::with_capacity(cells.elements.len());
        for (position, element) in cells.elements.iter().enumerate() {
            elements.push(
                element
                    .ok_or_else(|| anyhow!("no store wrote element {position} of {}", cells.ty))?,
            );
        }
        HostTensor::new(&cells.dims, crate::host::encode(cells.ty.dtype, &elements)?)
    }
}

impl fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HostBuffer({})", self.cells().ty)
    }
}

/// Two buffers are equal when they are one buffer, through any handles.
impl PartialEq for HostBuffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.cells, &other.cells)
    }
}

/// The static extents of `ty`.
fn static_dims(ty: &BufferType) -> Result<Vec<usize>> {
    let dims = ty
        .tensor_type()
        .static_dims()
        .ok_or_else(|| anyhow!("{ty} has extents that are not static"))?;
    Ok(dims.to_vec())
}

/// Any run-time value.
#[derive(Debug, Clone, PartialEq)]
pub enum HostValue {
    Index(i64),
    Scalar(HostScalar),
    Tensor(HostTensor),
    Buffer(HostBuffer),
    /// The lanes of a group of `isa` vector registers, as a tensor of the vector type's extents.
    Vector(HostTensor),
}

impl HostValue {
    /// The value's type, with its extents as static dimensions.
    pub fn ty(&self) -> Type {
        match self {
            HostValue::Tensor(tensor) => Type::Tensor(tensor.tensor_type()),
            HostValue::Buffer(buffer) => Type::Buffer(buffer.ty()),
            HostValue::Vector(lanes) => Type::Vector(VectorType {
                dtype: lanes.dtype(),
                dims: lanes.dims,
            }),
            HostValue::Index(_) => Type::Index,
            HostValue::Scalar(scalar) => Type::Scalar(scalar.dtype()),
        }
    }

    pub fn as_tensor(&self) -> Option<&HostTensor> {
        match self {
            HostValue::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    /// The tensor this value holds.
    pub fn into_tensor(self) -> Result<HostTensor> {
        match self {
            HostValue::Tensor(tensor) => Ok(tensor),
            other => Err(anyhow!("expected a tensor, got {}", other.ty())),
        }
    }

    /// The tensors `values` hold, such as the values a region gives.
    pub fn into_tensors(values: Vec<HostValue>) -> Result<Vec<HostTensor>> {
        values.into_iter().map(HostValue::into_tensor).collect()
    }

    /// `tensors` as values, such as the arguments of a region.
    pub fn of_tensors(tensors: &[&HostTensor]) -> Vec<HostValue> {
        tensors
            .iter()
            .map(|&tensor| HostValue::Tensor(tensor.clone()))
            .collect()
    }
}
