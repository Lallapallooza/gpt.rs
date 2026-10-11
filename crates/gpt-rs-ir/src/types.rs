//! Value types shared by every dialect.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use tinyvec::ArrayVec;

use crate::attr::Mnemonic;
use crate::minifloat::Format;

/// Scalar element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    I1,
    Si4,
    Ui4,
    Si8,
    Ui8,
    Si16,
    Ui16,
    Si32,
    Ui32,
    Si64,
    Ui64,
    F4E2M1,
    F6E2M3,
    F6E3M2,
    F8E4M3,
    F8E5M2,
    F8E4M3Fnuz,
    F8E5M2Fnuz,
    Bf16,
    F16,
    F32,
    F64,
    /// A power of two, `2^(e - 127)`: the scale of a microscaling block.
    E8M0,
}

impl Mnemonic for DType {
    const ALL: &'static [DType] = &[
        DType::I1,
        DType::Si4,
        DType::Ui4,
        DType::Si8,
        DType::Ui8,
        DType::Si16,
        DType::Ui16,
        DType::Si32,
        DType::Ui32,
        DType::Si64,
        DType::Ui64,
        DType::F4E2M1,
        DType::F6E2M3,
        DType::F6E3M2,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::F8E4M3Fnuz,
        DType::F8E5M2Fnuz,
        DType::Bf16,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::E8M0,
    ];

    fn mnemonic(self) -> &'static str {
        match self {
            DType::I1 => "i1",
            DType::Si4 => "si4",
            DType::Ui4 => "ui4",
            DType::Si8 => "si8",
            DType::Ui8 => "ui8",
            DType::Si16 => "si16",
            DType::Ui16 => "ui16",
            DType::Si32 => "si32",
            DType::Ui32 => "ui32",
            DType::Si64 => "si64",
            DType::Ui64 => "ui64",
            DType::F4E2M1 => "f4e2m1",
            DType::F6E2M3 => "f6e2m3",
            DType::F6E3M2 => "f6e3m2",
            DType::F8E4M3 => "f8e4m3",
            DType::F8E5M2 => "f8e5m2",
            DType::F8E4M3Fnuz => "f8e4m3fnuz",
            DType::F8E5M2Fnuz => "f8e5m2fnuz",
            DType::Bf16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::E8M0 => "e8m0",
        }
    }
}

impl DType {
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DType::F4E2M1
                | DType::F6E2M3
                | DType::F6E3M2
                | DType::F8E4M3
                | DType::F8E5M2
                | DType::F8E4M3Fnuz
                | DType::F8E5M2Fnuz
                | DType::Bf16
                | DType::F16
                | DType::F32
                | DType::F64
        )
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Si4
                | DType::Ui4
                | DType::Si8
                | DType::Ui8
                | DType::Si16
                | DType::Ui16
                | DType::Si32
                | DType::Ui32
                | DType::Si64
                | DType::Ui64
        )
    }

    /// Bytes per element, for dtypes stored in whole bytes. `i1` is stored as one byte.
    pub fn size_in_bytes(self) -> Option<usize> {
        match self {
            DType::I1
            | DType::Si8
            | DType::Ui8
            | DType::F8E4M3
            | DType::F8E5M2
            | DType::F8E4M3Fnuz
            | DType::F8E5M2Fnuz
            | DType::E8M0 => Some(1),
            DType::Si16 | DType::Ui16 | DType::Bf16 | DType::F16 => Some(2),
            DType::Si32 | DType::Ui32 | DType::F32 => Some(4),
            DType::Si64 | DType::Ui64 | DType::F64 => Some(8),
            DType::Si4 | DType::Ui4 | DType::F4E2M1 | DType::F6E2M3 | DType::F6E3M2 => None,
        }
    }

    /// The accumulator of sums and products of `self` elements unless an op names one: f32 for
    /// floats narrower than f64, and the dtype itself otherwise.
    pub fn default_accumulator(self) -> DType {
        if self.is_float() && self != DType::F64 {
            DType::F32
        } else {
            self
        }
    }

    /// The significant bits of a float dtype, its implicit leading bit included.
    pub fn significand_bits(self) -> Option<u32> {
        match self {
            DType::F64 => Some(53),
            DType::F32 => Some(24),
            other => crate::minifloat::Format::of(other).map(|format| format.man_bits + 1),
        }
    }

    /// Whether `value` is a value of the dtype: any value, rounded, for a float dtype, a power of
    /// two in its range for `e8m0`, and a number in range for an integer dtype.
    pub fn represents(self, value: f64) -> bool {
        match (self, self.int_format()) {
            (DType::E8M0, _) => {
                let exponent = value.log2();
                exponent.fract() == 0.0 && (-127.0..=127.0).contains(&exponent)
            }
            // `as` saturates, so a value beyond i128 falls outside the range too.
            (_, Some(int)) => {
                value.fract() == 0.0 && (int.min()..=int.max()).contains(&(value as i128))
            }
            (_, None) => true,
        }
    }

    /// The unsigned integer dtype of `bits` bits.
    pub fn unsigned(bits: usize) -> Result<DType> {
        Ok(match bits {
            4 => DType::Ui4,
            8 => DType::Ui8,
            16 => DType::Ui16,
            32 => DType::Ui32,
            64 => DType::Ui64,
            other => anyhow::bail!("no unsigned integer dtype has {other} bits"),
        })
    }

    /// The width and signedness of an integer dtype or `i1`.
    pub fn int_format(self) -> Option<IntFormat> {
        let (bits, signed) = match self {
            DType::I1 => (1, false),
            DType::Si4 => (4, true),
            DType::Ui4 => (4, false),
            DType::Si8 => (8, true),
            DType::Ui8 => (8, false),
            DType::Si16 => (16, true),
            DType::Ui16 => (16, false),
            DType::Si32 => (32, true),
            DType::Ui32 => (32, false),
            DType::Si64 => (64, true),
            DType::Ui64 => (64, false),
            _ => return None,
        };
        Some(IntFormat { bits, signed })
    }

    /// The elements of a quantized dtype: a minifloat format or an integer range.
    pub fn elements(self) -> Result<Elements> {
        match (Format::of(self), self.int_format()) {
            (Some(format), _) => Ok(Elements::Float(format)),
            (None, Some(int)) => Ok(Elements::Int(int)),
            _ => bail!("{self} is not a quantized dtype"),
        }
    }

    /// Bits per element. Elements narrower than a byte pack densely in row-major order, from the
    /// least significant bits of each byte.
    pub fn bits(self) -> usize {
        match self {
            DType::Si4 | DType::Ui4 | DType::F4E2M1 => 4,
            DType::F6E2M3 | DType::F6E3M2 => 6,
            other => 8 * other.size_in_bytes().expect("a whole-byte dtype"),
        }
    }

    /// Bytes that `count` elements take, the last byte padded.
    pub fn bytes_for(self, count: usize) -> Option<usize> {
        count.checked_mul(self.bits()).map(|bits| bits.div_ceil(8))
    }
}

/// The width and signedness of an integer dtype, with `i1` as one unsigned bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntFormat {
    pub bits: u32,
    pub signed: bool,
}

impl IntFormat {
    pub fn min(self) -> i128 {
        if self.signed {
            -(1 << (self.bits - 1))
        } else {
            0
        }
    }

    pub fn max(self) -> i128 {
        if self.signed {
            (1 << (self.bits - 1)) - 1
        } else {
            (1 << self.bits) - 1
        }
    }

    /// The bits of `value` as an unsigned number: `value` modulo `2^bits`.
    pub fn pattern(self, value: i128) -> i128 {
        value & ((1 << self.bits) - 1)
    }

    /// `value` modulo `2^bits`, read as this format reads its bits.
    pub fn wrap(self, value: i128) -> i128 {
        let pattern = self.pattern(value);
        if self.signed && pattern > self.max() {
            pattern - (1 << self.bits)
        } else {
            pattern
        }
    }

    pub fn saturate(self, value: i128) -> i128 {
        value.clamp(self.min(), self.max())
    }

    /// The little-endian bytes of `value`, an integer of this format, in `size` bytes.
    pub fn to_le_bytes(self, value: i128, size: usize) -> Vec<u8> {
        value.to_le_bytes()[..size].to_vec()
    }
}

/// The elements a quantized dtype holds: a float format or an integer range.
#[derive(Debug, Clone, Copy)]
pub enum Elements {
    Float(Format),
    Int(IntFormat),
}

impl Elements {
    /// The largest finite element.
    pub fn largest(self) -> f32 {
        match self {
            Elements::Float(format) => format.max() as f32,
            Elements::Int(int) => int.max() as f32,
        }
    }

    /// The smallest finite element.
    pub fn smallest(self) -> f32 {
        match self {
            Elements::Float(format) => -format.max() as f32,
            Elements::Int(int) => int.min() as f32,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mnemonic())
    }
}

/// Parses a mnemonic as the text form writes it (`bf16`), ignoring ASCII case and surrounding
/// whitespace.
impl FromStr for DType {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> anyhow::Result<Self> {
        let name = text.trim();
        DType::ALL
            .iter()
            .copied()
            .find(|dtype| dtype.mnemonic().eq_ignore_ascii_case(name))
            .ok_or_else(|| anyhow::anyhow!("unknown dtype `{name}`"))
    }
}

/// Name of a dimension whose extent is known only at run time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(Arc<str>);

impl Sym {
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One tensor axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Static(usize),
    Dynamic(Sym),
}

/// An extent of 0, which fills the unused entries of a [`Ranked`] list.
impl Default for Dim {
    fn default() -> Self {
        Dim::Static(0)
    }
}

/// The largest rank of a tensor type.
pub const MAX_RANK: usize = 8;

/// A list indexed by axis, such as the extents of a tensor type or the axes of an attribute: at
/// most [`MAX_RANK`] entries, inline.
pub type Ranked<T> = ArrayVec<[T; MAX_RANK]>;

/// `items` as a [`Ranked`] list. Fails when there are more than [`MAX_RANK`].
pub fn ranked<T: Default>(items: impl IntoIterator<Item = T>) -> Result<Ranked<T>> {
    let mut list = Ranked::new();
    let mut rank = 0;
    for item in items {
        rank += 1;
        if rank <= MAX_RANK {
            list.push(item);
        }
    }
    ensure!(
        rank <= MAX_RANK,
        "rank {rank} is above the largest rank, {MAX_RANK}"
    );
    Ok(list)
}

impl Dim {
    pub fn as_static(&self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic(_) => None,
        }
    }

    /// Whether the axis has extent 1, which broadcasting repeats and reshapes add or drop freely.
    pub fn is_one(&self) -> bool {
        *self == Dim::Static(1)
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Static(n)
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Static(n) => write!(f, "{n}"),
            Dim::Dynamic(sym) => write!(f, "{{{}}}", sym.as_str()),
        }
    }
}

/// Dense row-major tensor type, of rank at most [`MAX_RANK`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub dims: Ranked<Dim>,
}

impl TensorType {
    /// The type of `dims`. Panics, at the caller, when there are more than [`MAX_RANK`].
    #[track_caller]
    pub fn new(dtype: DType, dims: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        let dims = match ranked(dims.into_iter().map(Into::into)) {
            Ok(dims) => dims,
            Err(err) => panic!("tensor type: {err}"),
        };
        Self { dtype, dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The extents, when every dimension is static.
    pub fn static_dims(&self) -> Option<Ranked<usize>> {
        let mut dims = Ranked::new();
        for dim in &self.dims {
            dims.push(dim.as_static()?);
        }
        Some(dims)
    }

    /// The extent of `axis`, when it is static.
    pub fn static_dim(&self, axis: usize) -> Option<usize> {
        self.dims[axis].as_static()
    }

    /// The element count, when every dimension is static and the count fits in `usize`.
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim.as_static()?))
    }

    pub fn with_dims(&self, dims: Ranked<Dim>) -> Self {
        Self {
            dtype: self.dtype,
            dims,
        }
    }

    pub fn with_dtype(&self, dtype: DType) -> Self {
        Self {
            dtype,
            dims: self.dims.clone(),
        }
    }

    /// The type with axes reordered by `perm`, a permutation of `0..rank`.
    pub fn permuted(&self, perm: &Perm) -> Self {
        self.with_dims(perm.apply(&self.dims))
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor<")?;
        for dim in &self.dims {
            write!(f, "{dim}x")?;
        }
        write!(f, "{}>", self.dtype)
    }
}

/// A permutation of axes: result axis `i` is input axis `self[i]`, as `tensor.transpose` reads
/// its input. It may name axes out of range or twice until [`Perm::is_permutation_of`] checks it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Perm(Ranked<usize>);

impl Perm {
    /// The permutation that keeps each of `rank` axes in place.
    pub fn identity(rank: usize) -> Self {
        Self((0..rank).collect())
    }

    /// Whether every result axis reads the input axis of the same index.
    pub fn is_identity(&self) -> bool {
        self.0.iter().copied().eq(0..self.0.len())
    }

    /// Whether it names every axis below `rank` once.
    pub fn is_permutation_of(&self, rank: usize) -> bool {
        let mut seen = [false; MAX_RANK];
        self.0.len() == rank
            && self
                .0
                .iter()
                .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true))
    }

    /// The permutation that undoes this one: transposing by `self` and then by the inverse gives
    /// the input back.
    pub fn inverse(&self) -> Self {
        let mut inverse = self.0;
        for (result_axis, &input_axis) in self.0.iter().enumerate() {
            inverse[input_axis] = result_axis;
        }
        Self(inverse)
    }

    /// The single permutation that transposing by `first` and then by `self` amounts to.
    pub fn after(&self, first: &Perm) -> Self {
        Self(self.apply(&first.0))
    }

    /// `items` in the order this permutation reads them: `items[self[i]]` for each result axis
    /// `i`, such as the extents of the transposed type.
    pub fn apply<T: Clone + Default>(&self, items: &[T]) -> Ranked<T> {
        let mut applied = Ranked::new();
        for &axis in &self.0 {
            applied.push(items[axis].clone());
        }
        applied
    }

    /// Whether the result axes `axes`, listed in increasing order, read input axes in increasing
    /// order: the transpose keeps their order.
    pub fn keeps_order(&self, axes: &[usize]) -> bool {
        let read = self.read_axes(axes);
        read.windows(2).all(|pair| pair[0] < pair[1])
    }

    /// The input axes that the result axes `axes` read.
    pub fn read_axes(&self, axes: &[usize]) -> Ranked<usize> {
        let mut read = Ranked::new();
        for &axis in axes {
            read.push(self.0[axis]);
        }
        read
    }
}

impl std::ops::Deref for Perm {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a Perm {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Panics when there are more than [`MAX_RANK`] axes.
impl FromIterator<usize> for Perm {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = usize>>(axes: I) -> Self {
        match ranked(axes) {
            Ok(axes) => Self(axes),
            Err(err) => panic!("permutation: {err}"),
        }
    }
}

/// The type of an SSA value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Index,
    Scalar(DType),
    Tensor(TensorType),
}

impl Type {
    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    /// The element dtype of a tensor, or the dtype of a scalar.
    pub fn dtype(&self) -> Option<DType> {
        match self {
            Type::Tensor(tensor) => Some(tensor.dtype),
            Type::Scalar(dtype) => Some(*dtype),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<DType> {
        match self {
            Type::Scalar(dtype) => Some(*dtype),
            _ => None,
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, Type::Index)
    }
}

impl From<TensorType> for Type {
    fn from(tensor: TensorType) -> Self {
        Type::Tensor(tensor)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Index => f.write_str("index"),
            Type::Scalar(dtype) => write!(f, "{dtype}"),
            Type::Tensor(tensor) => write!(f, "{tensor}"),
        }
    }
}

/// `types` in parentheses, as the text form and error messages show them.
pub(crate) fn type_list(types: &[&Type]) -> String {
    let items: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    format!("({})", items.join(", "))
}

/// A Rust scalar that stores one element of `DTYPE`.
pub trait Element: Copy {
    const DTYPE: DType;
    type Bytes: AsRef<[u8]>;
    /// The element as stored in a tensor: little-endian, and one byte for `i1`.
    fn to_le(self) -> Self::Bytes;
}

impl Element for bool {
    const DTYPE: DType = DType::I1;
    type Bytes = [u8; 1];
    fn to_le(self) -> [u8; 1] {
        [u8::from(self)]
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    type Bytes = [u8; 4];
    fn to_le(self) -> [u8; 4] {
        self.to_le_bytes()
    }
}

/// Dimension symbols bound to extents: what a call binds when its arguments meet the argument
/// types of its callee.
#[derive(Debug, Clone, Default)]
pub struct SymBindings(HashMap<Sym, Dim>);

impl SymBindings {
    /// Binds the symbols of `param` to the extents of `arg`. Fails when the types differ in
    /// dtype, rank or a static extent, or when a symbol is already bound to another extent.
    pub fn bind(&mut self, param: &TensorType, arg: &TensorType) -> Result<()> {
        ensure!(
            param.dtype == arg.dtype && param.rank() == arg.rank(),
            "{arg} does not match {param}"
        );
        for (p, a) in param.dims.iter().zip(&arg.dims) {
            match p {
                Dim::Static(_) => ensure!(p == a, "{arg} does not match {param}"),
                Dim::Dynamic(sym) => {
                    let previous = self.0.entry(sym.clone()).or_insert_with(|| a.clone());
                    ensure!(
                        previous == a,
                        "{{{}}} is bound to both {previous} and {a}",
                        sym.as_str()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, sym: &Sym) -> Option<&Dim> {
        self.0.get(sym)
    }

    /// `ty` with each bound symbol replaced by its extent.
    pub fn resolve(&self, ty: &TensorType) -> TensorType {
        let dims = ty
            .dims
            .iter()
            .map(|dim| match dim {
                Dim::Dynamic(sym) => self.0.get(sym).cloned().unwrap_or_else(|| dim.clone()),
                Dim::Static(_) => dim.clone(),
            })
            .collect();
        ty.with_dims(dims)
    }
}
