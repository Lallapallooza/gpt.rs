//! Value types shared by every dialect.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::attr::Mnemonic;

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

impl Dim {
    pub fn as_static(&self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic(_) => None,
        }
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

/// Dense row-major tensor type. Its dims are inline up to rank 4.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub dims: SmallVec<[Dim; 4]>,
}

impl TensorType {
    pub fn new(dtype: DType, dims: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        Self {
            dtype,
            dims: dims.into_iter().map(Into::into).collect(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The extents, when every dimension is static.
    pub fn static_dims(&self) -> Option<SmallVec<[usize; 4]>> {
        self.dims.iter().map(Dim::as_static).collect()
    }

    /// The element count, when every dimension is static and the count fits in `usize`.
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim.as_static()?))
    }

    pub fn with_dims(&self, dims: SmallVec<[Dim; 4]>) -> Self {
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

    /// The type with axes reordered by `perm`. `perm` must be a permutation of `0..rank`.
    pub fn permuted(&self, perm: &[usize]) -> Self {
        self.with_dims(perm.iter().map(|&axis| self.dims[axis].clone()).collect())
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
