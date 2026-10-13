//! Element types: the dtypes, their integer formats and the Rust types that store them.

use std::fmt;
use std::str::FromStr;

use anyhow::{bail, Result};

use super::float::Format;
use crate::ir::Mnemonic;

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
            other => crate::ir::Format::of(other).map(|format| format.man_bits + 1),
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
