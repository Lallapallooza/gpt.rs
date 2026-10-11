//! Binary float formats narrower than f64, with their exact values and their round-to-nearest-even
//! encoding. They cover the elements of MX, NVFP4 and FP8 quantization, bf16, f16 and the `e8m0`
//! scales.

use crate::types::DType;

/// `floor(log2(x))` of a finite `x > 0` whose f64 is normal, exactly: its exponent.
pub fn floor_log2(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// What the special codes of a [`Format`] encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specials {
    /// Finite values only.
    None,
    /// NaN at the all-ones code of each sign, as `float8_e4m3fn`.
    Nan,
    /// IEEE 754: infinities at the largest exponent with mantissa 0, NaN at the largest exponent
    /// otherwise.
    Ieee,
    /// NaN at the sign bit alone, which would be `-0.0`, as `float8_e4m3fnuz`. No infinities, and
    /// the largest exponent holds finite values.
    Fnuz,
}

/// What a value beyond the largest finite one encodes as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    /// The largest finite value of that sign.
    Saturate,
    /// An infinity, in a format that has one, else the largest finite value.
    Infinity,
}

/// A float format with a sign bit, `exp_bits` exponent bits of bias `bias` and `man_bits`
/// mantissa bits, whose zero exponent holds the subnormals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub exp_bits: u32,
    pub man_bits: u32,
    pub bias: i32,
    pub specials: Specials,
}

impl Format {
    pub const F4E2M1: Format = Format {
        exp_bits: 2,
        man_bits: 1,
        bias: 1,
        specials: Specials::None,
    };
    pub const F6E2M3: Format = Format {
        exp_bits: 2,
        man_bits: 3,
        bias: 1,
        specials: Specials::None,
    };
    pub const F6E3M2: Format = Format {
        exp_bits: 3,
        man_bits: 2,
        bias: 3,
        specials: Specials::None,
    };
    pub const F8E4M3: Format = Format {
        exp_bits: 4,
        man_bits: 3,
        bias: 7,
        specials: Specials::Nan,
    };
    pub const F8E5M2: Format = Format {
        exp_bits: 5,
        man_bits: 2,
        bias: 15,
        specials: Specials::Ieee,
    };
    pub const F8E4M3FNUZ: Format = Format {
        exp_bits: 4,
        man_bits: 3,
        bias: 8,
        specials: Specials::Fnuz,
    };
    pub const F8E5M2FNUZ: Format = Format {
        exp_bits: 5,
        man_bits: 2,
        bias: 16,
        specials: Specials::Fnuz,
    };
    /// IEEE binary64, whose bit fields the block builders read; [`Format::of`] leaves f32 and f64
    /// out, as no quantized dtype is one.
    pub const F64: Format = Format {
        exp_bits: 11,
        man_bits: 52,
        bias: 1023,
        specials: Specials::Ieee,
    };
    pub const F16: Format = Format {
        exp_bits: 5,
        man_bits: 10,
        bias: 15,
        specials: Specials::Ieee,
    };
    pub const BF16: Format = Format {
        exp_bits: 8,
        man_bits: 7,
        bias: 127,
        specials: Specials::Ieee,
    };

    /// The format of a float dtype narrower than f32.
    pub fn of(dtype: DType) -> Option<Format> {
        Some(match dtype {
            DType::F4E2M1 => Format::F4E2M1,
            DType::F6E2M3 => Format::F6E2M3,
            DType::F6E3M2 => Format::F6E3M2,
            DType::F8E4M3 => Format::F8E4M3,
            DType::F8E5M2 => Format::F8E5M2,
            DType::F8E4M3Fnuz => Format::F8E4M3FNUZ,
            DType::F8E5M2Fnuz => Format::F8E5M2FNUZ,
            DType::F16 => Format::F16,
            DType::Bf16 => Format::BF16,
            _ => return None,
        })
    }

    /// Bits per element.
    pub fn bits(self) -> u32 {
        1 + self.exp_bits + self.man_bits
    }

    fn sign_bit(self) -> u32 {
        1 << (self.exp_bits + self.man_bits)
    }

    /// The largest exponent field.
    fn top(self) -> u32 {
        (1 << self.exp_bits) - 1
    }

    /// The code of the largest finite value.
    fn max_code(self) -> u32 {
        let all = self.sign_bit() - 1;
        match self.specials {
            Specials::None | Specials::Fnuz => all,
            Specials::Nan => all - 1,
            Specials::Ieee => all - (1 << self.man_bits),
        }
    }

    /// The code of a NaN.
    fn nan_code(self) -> Option<u32> {
        match self.specials {
            Specials::None => None,
            Specials::Nan => Some(self.sign_bit() - 1),
            Specials::Ieee => Some(self.top() << self.man_bits | 1 << (self.man_bits - 1)),
            Specials::Fnuz => Some(self.sign_bit()),
        }
    }

    /// The value of `code`, exactly: every value of these formats is an f32.
    pub fn decode(self, code: u32) -> f64 {
        let negative = code & self.sign_bit() != 0;
        let magnitude = code & (self.sign_bit() - 1);
        let exponent = magnitude >> self.man_bits;
        let mantissa = magnitude & ((1 << self.man_bits) - 1);
        let nan = match self.specials {
            Specials::None => false,
            Specials::Nan => magnitude == self.sign_bit() - 1,
            Specials::Ieee => exponent == self.top() && mantissa != 0,
            Specials::Fnuz => negative && magnitude == 0,
        };
        if nan {
            return f64::NAN;
        }
        let value = if self.specials == Specials::Ieee && exponent == self.top() {
            f64::INFINITY
        } else if exponent == 0 {
            f64::from(mantissa) * 2f64.powi(1 - self.bias - self.man_bits as i32)
        } else {
            f64::from(mantissa | 1 << self.man_bits)
                * 2f64.powi(exponent as i32 - self.bias - self.man_bits as i32)
        };
        if negative {
            -value
        } else {
            value
        }
    }

    /// The largest finite value.
    pub fn max(self) -> f64 {
        self.decode(self.max_code())
    }

    /// The code of `x` rounded to nearest, ties to even. A value beyond the largest finite one,
    /// infinities included, encodes as `overflow` says. NaN gives the format's NaN, or 0 in a format
    /// without one. `-0.0` keeps its sign, except in an `fnuz` format, which has no `-0.0`.
    pub fn encode(self, x: f64, overflow: Overflow) -> u32 {
        if x.is_nan() {
            return self.nan_code().unwrap_or(0);
        }
        let sign = if x.is_sign_negative() {
            self.sign_bit()
        } else {
            0
        };
        let min_normal = 1 - self.bias;
        let man = self.man_bits as i32;
        let magnitude = x.abs();
        // The spacing of the values next to `magnitude`: that of its binade, and that of the
        // subnormals below the smallest normal exponent.
        let exponent = if magnitude.is_infinite() {
            min_normal
        } else {
            binade(magnitude).max(min_normal)
        };
        let quantum = 2f64.powi(exponent - man);
        let rounded = (magnitude / quantum).round_ties_even() * quantum;
        let code = if rounded > self.max() {
            match (overflow, self.specials) {
                (Overflow::Infinity, Specials::Ieee) => self.top() << self.man_bits,
                _ => self.max_code(),
            }
        } else if rounded < 2f64.powi(min_normal) {
            (rounded / 2f64.powi(min_normal - man)) as u32
        } else {
            let exponent = binade(rounded);
            let mantissa = (rounded / 2f64.powi(exponent - man)) as u32 - (1 << self.man_bits);
            ((exponent + self.bias) as u32) << self.man_bits | mantissa
        };
        if code == 0 && self.specials == Specials::Fnuz {
            return 0;
        }
        sign | code
    }
}

/// `floor(log2(x))` of a finite `x >= 0`, exactly, read from its bits. Zero and the subnormals
/// give -1023, below the smallest normal exponent of every format.
fn binade(x: f64) -> i32 {
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    biased.max(0) - 1023
}

/// The value of the `e8m0` code `code`: `2^(code - 127)`, or NaN for `0xFF`.
pub fn e8m0_decode(code: u8) -> f64 {
    if code == 0xFF {
        return f64::NAN;
    }
    2f64.powi(i32::from(code) - 127)
}

/// The `e8m0` code of `2^exponent`, clamped to the codes of finite scales.
pub fn e8m0_encode(exponent: i32) -> u8 {
    (exponent + 127).clamp(0, 254) as u8
}
