//! Tensors in host memory, and `Buffer`, the element storage that the CPU backends share.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

use crate::ir::{ranked, DType, Ranked, TensorType, F8E4M3};

/// Read-only little-endian bytes that another allocation owns, such as a memory-mapped checkpoint.
#[derive(Clone)]
pub struct ExternalBytes {
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    offset: usize,
    len: usize,
}

impl ExternalBytes {
    /// A view of `owner[offset..offset + len]`. Fails when the range does not fit in `owner`.
    pub fn new(
        owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
        offset: usize,
        len: usize,
    ) -> Result<Self> {
        let total = (*owner).as_ref().len();
        let end = offset
            .checked_add(len)
            .ok_or_else(|| anyhow!("external byte range overflow"))?;
        ensure!(
            end <= total,
            "external byte range {offset}..{end} exceeds owner length {total}"
        );
        Ok(Self { owner, offset, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        &(*self.owner).as_ref()[self.offset..self.offset + self.len]
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for ExternalBytes {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl fmt::Debug for ExternalBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExternalBytes")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

/// One cache line of an owned buffer. Owned elements start on a 64-byte boundary, so a vector load
/// of up to 64 bytes at a multiple of 64 bytes stays in one cache line.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([u8; 64]);

/// Elements of a host tensor: an owned allocation that starts on a cache line, or a view into
/// bytes that another allocation owns, such as a memory-mapped checkpoint. Clones share the
/// elements.
#[derive(Clone)]
pub struct Buffer<T>(Storage<T>);

#[derive(Clone)]
enum Storage<T> {
    /// The first `len` elements in the bytes of `lines`.
    Owned {
        lines: Arc<[Line]>,
        len: usize,
        _elements: PhantomData<T>,
    },
    /// `len` elements at `ptr`, inside the bytes that `_owner` holds alive.
    View {
        _owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
        ptr: NonNull<T>,
        len: usize,
    },
}

// SAFETY: `Owned` is `Send` and `Sync`: its lines are plain bytes behind an `Arc`. A view shares
// `T`s, which `T: Sync` allows, and its owner is `Send + Sync`.
unsafe impl<T: Send + Sync> Send for Buffer<T> {}

// SAFETY: as for `Send`.
unsafe impl<T: Send + Sync> Sync for Buffer<T> {}

const _: () = assert!(
    cfg!(target_endian = "little"),
    "tensor bytes are little-endian"
);

/// The number of `T`s in `bytes` bytes. Fails unless they divide into whole elements.
fn element_count<T>(bytes: usize) -> Result<usize> {
    let size = size_of::<T>();
    ensure!(
        bytes.is_multiple_of(size),
        "{bytes} bytes do not divide into {size}-byte elements"
    );
    Ok(bytes / size)
}

impl<T: Pod> Buffer<T> {
    /// `len` elements whose bytes are all zero, in a new allocation.
    pub fn zeroed(len: usize) -> Self {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("a buffer size that fits in usize");
        // SAFETY: a `Line` is plain bytes, so zeroed memory is a valid `Line`.
        let lines = unsafe { Arc::<[Line]>::new_zeroed_slice(bytes.div_ceil(64)).assume_init() };
        Self(Storage::Owned {
            lines,
            len,
            _elements: PhantomData,
        })
    }

    /// The elements of little-endian `bytes`, which must be aligned for `T`. Never copies: a view
    /// of no bytes is an empty buffer.
    pub fn view(bytes: ExternalBytes) -> Result<Self> {
        element_count::<T>(bytes.len)?;
        if bytes.is_empty() {
            return Ok(Self::zeroed(0));
        }
        let Ok(elements) = bytemuck::try_cast_slice::<u8, T>(bytes.as_slice()) else {
            bail!(
                "{} bytes at offset {} are not aligned for {}-byte elements",
                bytes.len,
                bytes.offset,
                align_of::<T>()
            );
        };
        let (ptr, len) = (NonNull::from(elements).cast(), elements.len());
        Ok(Self(Storage::View {
            _owner: bytes.owner,
            ptr,
            len,
        }))
    }

    /// Reads the bytes of `owner` as little-endian elements. Keeps a view of them when they are
    /// aligned for `T`, and copies them otherwise.
    pub fn from_bytes(owner: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Result<Self> {
        let len = (*owner).as_ref().len();
        let bytes = ExternalBytes::new(owner, 0, len)?;
        if bytes.as_slice().as_ptr().cast::<T>().is_aligned() {
            return Self::view(bytes);
        }
        let mut buffer = Self::zeroed(element_count::<T>(len)?);
        bytemuck::cast_slice_mut::<T, u8>(buffer.make_mut()).copy_from_slice(bytes.as_slice());
        Ok(buffer)
    }

    /// A copy of `values` in a new allocation.
    fn copy_of(values: &[T]) -> Self {
        let mut buffer = Self::zeroed(values.len());
        buffer.make_mut().copy_from_slice(values);
        buffer
    }

    /// The elements for writing. Copies them first when they are a view or shared with a clone.
    pub fn make_mut(&mut self) -> &mut [T] {
        if matches!(self.0, Storage::View { .. }) {
            *self = Self::copy_of(self);
        }
        let Storage::Owned { lines, len, .. } = &mut self.0 else {
            unreachable!("views are copied above");
        };
        let bytes = bytemuck::cast_slice_mut::<Line, u8>(Arc::make_mut(lines));
        bytemuck::cast_slice_mut(&mut bytes[..*len * size_of::<T>()])
    }
}

impl<T> Buffer<T> {
    /// Whether no other handle can read the elements: they are owned, and no clone shares them.
    /// False for a view.
    pub fn is_unique(&self) -> bool {
        match &self.0 {
            Storage::Owned { lines, .. } => Arc::strong_count(lines) == 1,
            Storage::View { .. } => false,
        }
    }
}

impl<T: Pod> Deref for Buffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.0 {
            Storage::Owned { lines, len, .. } => {
                let bytes = bytemuck::cast_slice::<Line, u8>(lines);
                bytemuck::cast_slice(&bytes[..len * size_of::<T>()])
            }
            // SAFETY: `view` took `ptr` and `len` from a `&[T]` that bytemuck cast from the bytes
            // of `_owner`, so the elements are aligned, in bounds and valid. `_owner` keeps those
            // bytes alive, and shared access to it cannot change them.
            Storage::View { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(ptr.as_ptr(), *len)
            },
        }
    }
}

impl<T: Pod> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T: Pod> From<Vec<T>> for Buffer<T> {
    fn from(values: Vec<T>) -> Self {
        Self::copy_of(&values)
    }
}

impl<T: Pod> FromIterator<T> for Buffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::copy_of(&iter.into_iter().collect::<Vec<T>>())
    }
}

impl<T: Pod + fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Pod + PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

/// What a buffer offers whatever its element type.
trait Elements {
    fn bytes(&self) -> &[u8];
    fn bytes_mut(&mut self) -> &mut [u8];
    fn len(&self) -> usize;
    fn is_unique(&self) -> bool;
}

impl<T: Pod> Elements for Buffer<T> {
    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice::<T, u8>(self)
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::cast_slice_mut::<T, u8>(self.make_mut())
    }

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn is_unique(&self) -> bool {
        Buffer::is_unique(self)
    }
}

/// How a new buffer gets its elements.
enum Fill {
    /// That many zero elements.
    Zeros(usize),
    /// A view of the bytes, which must be aligned.
    View(ExternalBytes),
    /// A view of the owner's bytes when they are aligned, and a copy otherwise.
    Read(Arc<dyn AsRef<[u8]> + Send + Sync>),
}

impl Fill {
    fn buffer<T: Pod>(self) -> Result<Buffer<T>> {
        match self {
            Fill::Zeros(len) => Ok(Buffer::zeroed(len)),
            Fill::View(bytes) => Buffer::view(bytes),
            Fill::Read(owner) => Buffer::from_bytes(owner),
        }
    }
}

/// Elements narrower than a byte, packed densely in row-major order from the least significant
/// bits of each byte, as [`DType::bits`] gives their width.
#[derive(Debug, Clone, PartialEq)]
pub struct Packed {
    dtype: DType,
    len: usize,
    bytes: Buffer<u8>,
}

impl Packed {
    fn new(dtype: DType, len: usize, bytes: Buffer<u8>) -> Result<Self> {
        let need = dtype.bytes_for(len);
        ensure!(
            need == Some(bytes.len()),
            "{len} {dtype} elements take {need:?} bytes, got {}",
            bytes.len()
        );
        Ok(Self { dtype, len, bytes })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn mask(&self) -> u16 {
        (1 << self.dtype.bits()) - 1
    }

    /// The bits of element `index`.
    pub fn get(&self, index: usize) -> u8 {
        let bit = index * self.dtype.bits();
        let (byte, shift) = (bit / 8, bit % 8);
        let low = u16::from(self.bytes[byte]);
        let high = self.bytes.get(byte + 1).map_or(0, |&b| u16::from(b));
        (((high << 8 | low) >> shift) & self.mask()) as u8
    }

    /// Sets element `index` to the low bits of `code`.
    pub fn set(&mut self, index: usize, code: u8) {
        let bit = index * self.dtype.bits();
        let (byte, shift) = (bit / 8, bit % 8);
        let mask = self.mask() << shift;
        let value = (u16::from(code) << shift) & mask;
        let bytes = self.bytes.make_mut();
        bytes[byte] = (bytes[byte] & !(mask as u8)) | value as u8;
        if let Some(next) = bytes.get_mut(byte + 1) {
            *next = (*next & !((mask >> 8) as u8)) | (value >> 8) as u8;
        }
    }
}

impl Elements for Packed {
    fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.bytes.make_mut()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_unique(&self) -> bool {
        self.bytes.is_unique()
    }
}

/// Element storage of a host tensor. `I1` stores one byte per element, 0 or 1. The 8-bit float
/// formats other than `F8E4M3`, and `E8M0`, store each element's code.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F64(Buffer<f64>),
    F32(Buffer<f32>),
    F16(Buffer<f16>),
    Bf16(Buffer<bf16>),
    F8E4M3(Buffer<F8E4M3>),
    F8E5M2(Buffer<u8>),
    F8E4M3Fnuz(Buffer<u8>),
    F8E5M2Fnuz(Buffer<u8>),
    E8M0(Buffer<u8>),
    Si8(Buffer<i8>),
    Ui8(Buffer<u8>),
    Si16(Buffer<i16>),
    Ui16(Buffer<u16>),
    Si32(Buffer<i32>),
    Ui32(Buffer<u32>),
    Si64(Buffer<i64>),
    Ui64(Buffer<u64>),
    I1(Buffer<u8>),
    Packed(Packed),
}

impl TensorData {
    /// The `count` elements of `dtype` that `fill` gives.
    fn filled(dtype: DType, count: usize, fill: Fill) -> Result<Self> {
        if dtype.size_in_bytes().is_none() {
            let bytes = dtype
                .bytes_for(count)
                .ok_or_else(|| anyhow!("{count} {dtype} elements take more bytes than fit"))?;
            let fill = match fill {
                Fill::Zeros(_) => Fill::Zeros(bytes),
                other => other,
            };
            return Ok(TensorData::Packed(Packed::new(
                dtype,
                count,
                fill.buffer()?,
            )?));
        }
        Ok(match dtype {
            DType::F64 => TensorData::F64(fill.buffer()?),
            DType::F32 => TensorData::F32(fill.buffer()?),
            DType::F16 => TensorData::F16(fill.buffer()?),
            DType::Bf16 => TensorData::Bf16(fill.buffer()?),
            DType::F8E4M3 => TensorData::F8E4M3(fill.buffer()?),
            DType::F8E5M2 => TensorData::F8E5M2(fill.buffer()?),
            DType::F8E4M3Fnuz => TensorData::F8E4M3Fnuz(fill.buffer()?),
            DType::F8E5M2Fnuz => TensorData::F8E5M2Fnuz(fill.buffer()?),
            DType::E8M0 => TensorData::E8M0(fill.buffer()?),
            DType::Si8 => TensorData::Si8(fill.buffer()?),
            DType::Ui8 => TensorData::Ui8(fill.buffer()?),
            DType::Si16 => TensorData::Si16(fill.buffer()?),
            DType::Ui16 => TensorData::Ui16(fill.buffer()?),
            DType::Si32 => TensorData::Si32(fill.buffer()?),
            DType::Ui32 => TensorData::Ui32(fill.buffer()?),
            DType::Si64 => TensorData::Si64(fill.buffer()?),
            DType::Ui64 => TensorData::Ui64(fill.buffer()?),
            DType::I1 => TensorData::I1(fill.buffer()?),
            DType::Si4 | DType::Ui4 | DType::F4E2M1 | DType::F6E2M3 | DType::F6E3M2 => {
                unreachable!("sub-byte dtypes are packed above")
            }
        })
    }

    /// `len` elements of `dtype` whose bytes are all zero: `+0.0` for floats.
    pub fn zeroed(dtype: DType, len: usize) -> Result<Self> {
        Self::filled(dtype, len, Fill::Zeros(len))
    }

    /// The `count` elements of `dtype` in the little-endian bytes of `owner`: a view of them when
    /// they are aligned, and a copy otherwise.
    pub(crate) fn read(
        dtype: DType,
        count: usize,
        owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    ) -> Result<Self> {
        Self::filled(dtype, count, Fill::Read(owner))
    }

    pub fn dtype(&self) -> DType {
        match self {
            TensorData::F64(_) => DType::F64,
            TensorData::F32(_) => DType::F32,
            TensorData::F16(_) => DType::F16,
            TensorData::Bf16(_) => DType::Bf16,
            TensorData::F8E4M3(_) => DType::F8E4M3,
            TensorData::F8E5M2(_) => DType::F8E5M2,
            TensorData::F8E4M3Fnuz(_) => DType::F8E4M3Fnuz,
            TensorData::F8E5M2Fnuz(_) => DType::F8E5M2Fnuz,
            TensorData::E8M0(_) => DType::E8M0,
            TensorData::Si8(_) => DType::Si8,
            TensorData::Ui8(_) => DType::Ui8,
            TensorData::Si16(_) => DType::Si16,
            TensorData::Ui16(_) => DType::Ui16,
            TensorData::Si32(_) => DType::Si32,
            TensorData::Ui32(_) => DType::Ui32,
            TensorData::Si64(_) => DType::Si64,
            TensorData::Ui64(_) => DType::Ui64,
            TensorData::I1(_) => DType::I1,
            TensorData::Packed(packed) => packed.dtype,
        }
    }

    fn elements(&self) -> &dyn Elements {
        match self {
            TensorData::F64(v) => v,
            TensorData::F32(v) => v,
            TensorData::F16(v) => v,
            TensorData::Bf16(v) => v,
            TensorData::F8E4M3(v) => v,
            TensorData::F8E5M2(v)
            | TensorData::F8E4M3Fnuz(v)
            | TensorData::F8E5M2Fnuz(v)
            | TensorData::E8M0(v)
            | TensorData::Ui8(v)
            | TensorData::I1(v) => v,
            TensorData::Si8(v) => v,
            TensorData::Si16(v) => v,
            TensorData::Ui16(v) => v,
            TensorData::Si32(v) => v,
            TensorData::Ui32(v) => v,
            TensorData::Si64(v) => v,
            TensorData::Ui64(v) => v,
            TensorData::Packed(v) => v,
        }
    }

    fn elements_mut(&mut self) -> &mut dyn Elements {
        match self {
            TensorData::F64(v) => v,
            TensorData::F32(v) => v,
            TensorData::F16(v) => v,
            TensorData::Bf16(v) => v,
            TensorData::F8E4M3(v) => v,
            TensorData::F8E5M2(v)
            | TensorData::F8E4M3Fnuz(v)
            | TensorData::F8E5M2Fnuz(v)
            | TensorData::E8M0(v)
            | TensorData::Ui8(v)
            | TensorData::I1(v) => v,
            TensorData::Si8(v) => v,
            TensorData::Si16(v) => v,
            TensorData::Ui16(v) => v,
            TensorData::Si32(v) => v,
            TensorData::Ui32(v) => v,
            TensorData::Si64(v) => v,
            TensorData::Ui64(v) => v,
            TensorData::Packed(v) => v,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.elements().len()
    }

    /// The elements as little-endian bytes.
    pub fn bytes(&self) -> &[u8] {
        self.elements().bytes()
    }

    /// The elements as little-endian bytes for writing. Copies them first when they are a view or
    /// shared with a clone.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.elements_mut().bytes_mut()
    }

    /// Whether no other handle can read the elements, as [`Buffer::is_unique`] says.
    pub fn is_unique(&self) -> bool {
        self.elements().is_unique()
    }
}

/// The extents of `ty` and the bytes its elements take. Fails unless every extent is static.
fn layout(ty: &TensorType) -> Result<(Ranked<usize>, usize)> {
    let dims = ty
        .static_dims()
        .ok_or_else(|| anyhow!("a host tensor has static extents, not {ty}"))?;
    let bytes = dims
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .and_then(|count| ty.dtype.bytes_for(count))
        .ok_or_else(|| anyhow!("{ty} takes more bytes than fit in usize"))?;
    Ok((dims, bytes))
}

/// A dense row-major tensor with static extents.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub dims: Ranked<usize>,
    pub data: TensorData,
}

impl HostTensor {
    pub fn new(dims: impl AsRef<[usize]>, data: TensorData) -> Result<Self> {
        let dims = ranked(dims.as_ref().iter().copied())?;
        let count = dims
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent));
        ensure!(
            count == Some(data.len()),
            "a tensor of shape {dims:?} does not hold {} elements",
            data.len()
        );
        Ok(Self { dims, data })
    }

    pub fn f32(dims: impl AsRef<[usize]>, values: Vec<f32>) -> Result<Self> {
        Self::new(dims, TensorData::F32(values.into()))
    }

    pub fn si32(dims: impl AsRef<[usize]>, values: Vec<i32>) -> Result<Self> {
        Self::new(dims, TensorData::Si32(values.into()))
    }

    /// A tensor of type `ty` whose every element is zero: `+0.0` for floats.
    pub fn zeros(ty: &TensorType) -> Result<Self> {
        let (dims, _) = layout(ty)?;
        let count = dims.iter().product();
        Self::new(dims, TensorData::zeroed(ty.dtype, count)?)
    }

    /// A tensor of type `ty` with a copy of the little-endian `bytes`.
    pub fn from_le_bytes(ty: &TensorType, bytes: &[u8]) -> Result<Self> {
        let (_, need) = layout(ty)?;
        ensure!(
            bytes.len() == need,
            "{ty} takes {need} bytes, got {}",
            bytes.len()
        );
        let mut tensor = Self::zeros(ty)?;
        tensor.bytes_mut().copy_from_slice(bytes);
        Ok(tensor)
    }

    /// A tensor of type `ty` over the little-endian `bytes`, which must be aligned for its dtype.
    /// Never copies.
    pub fn view(ty: &TensorType, bytes: ExternalBytes) -> Result<Self> {
        let (dims, need) = layout(ty)?;
        ensure!(
            bytes.len() == need,
            "{ty} takes {need} bytes, got {}",
            bytes.len()
        );
        let count = dims.iter().product();
        Self::new(
            dims,
            TensorData::filled(ty.dtype, count, Fill::View(bytes))?,
        )
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype(), self.dims.iter().copied())
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as little-endian bytes.
    pub fn to_le_bytes(&self) -> &[u8] {
        self.data.bytes()
    }

    /// The elements as little-endian bytes for writing. Copies them first when they are a view or
    /// shared with a clone.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.data.bytes_mut()
    }

    /// Whether no other handle can read the elements, as [`Buffer::is_unique`] says.
    pub fn is_unique(&self) -> bool {
        self.data.is_unique()
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.data {
            TensorData::F32(v) => Ok(v),
            other => bail!("expected f32 data, got {}", other.dtype()),
        }
    }

    pub fn as_si32(&self) -> Result<&[i32]> {
        match &self.data {
            TensorData::Si32(v) => Ok(v),
            other => bail!("expected si32 data, got {}", other.dtype()),
        }
    }

    /// The elements of an si32 or si64 tensor, as indices and offsets are.
    pub fn as_indices(&self) -> Result<std::borrow::Cow<'_, [i64]>> {
        match &self.data {
            TensorData::Si32(v) => Ok(v.iter().map(|&i| i64::from(i)).collect()),
            TensorData::Si64(v) => Ok(std::borrow::Cow::Borrowed(v)),
            other => bail!("expected si32 or si64 indices, got {}", other.dtype()),
        }
    }

    pub fn as_i1(&self) -> Result<&[u8]> {
        match &self.data {
            TensorData::I1(v) => Ok(v),
            other => bail!("expected i1 data, got {}", other.dtype()),
        }
    }
}
