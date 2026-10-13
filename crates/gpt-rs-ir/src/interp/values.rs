//! The values the interpreter computes with: index values, scalars, tensors and `loop` buffers,
//! with the units of work that access a buffer.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use anyhow::{anyhow, bail, ensure, Result};

use crate::host::{HostScalar, HostTensor};
use crate::ir::{BufferType, Space, Type};

/// Where an access to a buffer runs: the iteration of each `loop.forall` around it, outermost
/// first, as the run of the forall and its index.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
#[derive(Clone)]
pub struct HostBuffer(Arc<Mutex<BufferCells>>);

struct BufferCells {
    ty: BufferType,
    dims: Vec<usize>,
    elements: Vec<Option<HostScalar>>,
    accesses: Vec<Accesses>,
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
        let dims = ty
            .tensor_type()
            .static_dims()
            .ok_or_else(|| anyhow!("{ty} has extents that are not static"))?
            .to_vec();
        let count = dims.iter().product();
        Ok(Self(Arc::new(Mutex::new(BufferCells {
            ty,
            dims,
            elements: vec![None; count],
            accesses: vec![Accesses::default(); count],
        }))))
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
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
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
    /// it, or when the read races with a write.
    pub fn load(&self, coords: &[i64], unit: &Unit) -> Result<HostScalar> {
        let mut cells = self.cells();
        let position = cells.checked_position(coords)?;
        cells.check_race(position, coords, unit, false)?;
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HostBuffer({})", self.cells().ty)
    }
}

/// Two buffers are equal when they are one buffer.
impl PartialEq for HostBuffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Any run-time value.
#[derive(Debug, Clone, PartialEq)]
pub enum HostValue {
    Index(i64),
    Scalar(HostScalar),
    Tensor(HostTensor),
    Buffer(HostBuffer),
}

impl HostValue {
    /// The value's type, with its extents as static dimensions.
    pub fn ty(&self) -> Type {
        match self {
            HostValue::Tensor(t) => Type::Tensor(t.tensor_type()),
            HostValue::Buffer(b) => Type::Buffer(b.ty()),
            HostValue::Index(_) => Type::Index,
            HostValue::Scalar(s) => Type::Scalar(s.dtype()),
        }
    }

    pub fn as_tensor(&self) -> Option<&HostTensor> {
        match self {
            HostValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// The tensor this value holds.
    pub fn into_tensor(self) -> Result<HostTensor> {
        match self {
            HostValue::Tensor(t) => Ok(t),
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
            .map(|&t| HostValue::Tensor(t.clone()))
            .collect()
    }
}
