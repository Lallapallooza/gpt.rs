//! Index arithmetic, element movement and element conversions shared by the tensor ops' `eval`
//! implementations.
//!
//! The interpreter computes on three types, into which [`values`] decodes every dtype:
//!
//! - `i128` for the integers and `i1`, which the result then wraps or saturates to;
//! - f64 for f64;
//! - f32 for every other float and for `e8m0`, which f32 holds exactly.
//!
//! f32 arithmetic rounded to bf16 or f16 rounds once, since f32 has at least twice their precision
//! plus two bits.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use half::{bf16, f16};

use super::elementwise::Arith;
use crate::host::TensorData;
use crate::ir::{e8m0_decode, DType, Format, Overflow, F8E4M3};

/// Row-major strides of `dims`.
pub fn strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * dims[axis + 1];
    }
    strides
}

/// The coordinates of row-major element `index` in a tensor of shape `dims`.
pub fn unravel(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coord = vec![0usize; dims.len()];
    for axis in (0..dims.len()).rev() {
        coord[axis] = index % dims[axis];
        index /= dims[axis];
    }
    coord
}

/// The element offset of every point of a loop nest, in row-major order: loop `i` runs
/// `extents[i]` times and steps `steps[i]` elements.
pub fn offsets(extents: &[usize], steps: &[usize]) -> Vec<usize> {
    let count: usize = extents.iter().product();
    (0..count)
        .map(|i| {
            unravel(i, extents)
                .iter()
                .zip(steps)
                .map(|(coord, step)| coord * step)
                .sum()
        })
        .collect()
}

/// The elements of `data` at `start` plus each of `offsets(extents, steps)`, in order: a strided
/// view of the elements, such as a transpose, broadcast or slice reads.
pub fn strided(
    data: &TensorData,
    start: usize,
    extents: &[usize],
    steps: &[usize],
) -> Result<TensorData> {
    let at = offsets(extents, steps);
    gather_elements(data, at.len(), |i| start + at[i])
}

/// The elements of a tensor as equal-size chunks, for the ops that only move elements: the
/// little-endian bytes of a whole-byte dtype, and one byte per element, its code, for a dtype
/// narrower than a byte.
pub struct Chunks<'a> {
    dtype: DType,
    size: usize,
    bytes: Cow<'a, [u8]>,
}

impl<'a> Chunks<'a> {
    pub fn of(data: &'a TensorData) -> Self {
        let bytes = match data {
            TensorData::Packed(packed) => (0..packed.len()).map(|i| packed.get(i)).collect(),
            data => Cow::Borrowed(data.bytes()),
        };
        Self {
            dtype: data.dtype(),
            size: data.dtype().size_in_bytes().unwrap_or(1),
            bytes,
        }
    }

    /// `count` elements of `dtype` whose bytes are zero.
    pub fn zeroed(dtype: DType, count: usize) -> Chunks<'static> {
        let size = dtype.size_in_bytes().unwrap_or(1);
        Chunks {
            dtype,
            size,
            bytes: Cow::Owned(vec![0; count * size]),
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len() / self.size
    }

    /// Element `index`.
    pub fn get(&self, index: usize) -> &[u8] {
        self.run(index, 1)
    }

    /// The `len` elements from element `start` on.
    pub fn run(&self, start: usize, len: usize) -> &[u8] {
        &self.bytes[start * self.size..][..len * self.size]
    }

    /// Sets element `index` to `element`, a chunk of a tensor of the same dtype.
    pub fn set(&mut self, index: usize, element: &[u8]) {
        self.set_run(index, element);
    }

    /// Sets the elements from element `start` on to `elements`, chunks of a tensor of the same
    /// dtype.
    pub fn set_run(&mut self, start: usize, elements: &[u8]) {
        let at = start * self.size;
        self.bytes.to_mut()[at..at + elements.len()].copy_from_slice(elements);
    }

    pub fn into_data(self) -> Result<TensorData> {
        let count = self.len();
        if self.dtype.size_in_bytes().is_none() {
            return packed(self.dtype, &self.bytes);
        }
        TensorData::read(self.dtype, count, Arc::new(self.bytes.into_owned()))
    }
}

/// The elements of `data` at `source(i)` for `i` in `0..count`, in a new tensor of the same dtype.
pub fn gather_elements(
    data: &TensorData,
    count: usize,
    source: impl Fn(usize) -> usize,
) -> Result<TensorData> {
    let input = Chunks::of(data);
    let mut out = Chunks::zeroed(data.dtype(), count);
    for i in 0..count {
        out.set(i, input.get(source(i)));
    }
    out.into_data()
}

/// The elements of a tensor in the type the interpreter computes with.
pub enum Values<'a> {
    /// The integer dtypes and `i1`.
    Int(Vec<i128>),
    /// The floats narrower than f64, and `e8m0`.
    F32(Cow<'a, [f32]>),
    F64(Cow<'a, [f64]>),
}

impl Values<'_> {
    pub fn len(&self) -> usize {
        match self {
            Values::Int(v) => v.len(),
            Values::F32(v) => v.len(),
            Values::F64(v) => v.len(),
        }
    }

    /// The values, owned.
    pub fn into_owned(self) -> Values<'static> {
        match self {
            Values::Int(v) => Values::Int(v),
            Values::F32(v) => Values::F32(Cow::Owned(v.into_owned())),
            Values::F64(v) => Values::F64(Cow::Owned(v.into_owned())),
        }
    }
}

/// The elements of `data` in the type the interpreter computes with. Every dtype decodes exactly.
pub fn values(data: &TensorData) -> Values<'_> {
    let ints = |v: &mut dyn Iterator<Item = i128>| Values::Int(v.collect());
    let codes = |format: Format, v: &[u8]| {
        Values::F32(
            v.iter()
                .map(|&c| format.decode(u32::from(c)) as f32)
                .collect(),
        )
    };
    match data {
        TensorData::F64(v) => Values::F64(Cow::Borrowed(v)),
        TensorData::F32(v) => Values::F32(Cow::Borrowed(v)),
        TensorData::F16(v) => Values::F32(v.iter().map(|x| x.to_f32()).collect()),
        TensorData::Bf16(v) => Values::F32(v.iter().map(|x| x.to_f32()).collect()),
        TensorData::F8E4M3(v) => Values::F32(v.iter().map(|x| x.to_f32()).collect()),
        TensorData::F8E5M2(v) => codes(Format::F8E5M2, v),
        TensorData::F8E4M3Fnuz(v) => codes(Format::F8E4M3FNUZ, v),
        TensorData::F8E5M2Fnuz(v) => codes(Format::F8E5M2FNUZ, v),
        TensorData::E8M0(v) => Values::F32(v.iter().map(|&c| e8m0_decode(c) as f32).collect()),
        TensorData::Si8(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Ui8(v) | TensorData::I1(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Si16(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Ui16(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Si32(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Ui32(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Si64(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Ui64(v) => ints(&mut v.iter().map(|&x| i128::from(x))),
        TensorData::Packed(packed) => {
            let dtype = packed.dtype();
            let code = |i| u32::from(packed.get(i));
            match (Format::of(dtype), dtype.int_format()) {
                (Some(format), _) => Values::F32(
                    (0..packed.len())
                        .map(|i| format.decode(code(i)) as f32)
                        .collect(),
                ),
                (None, Some(int)) => {
                    ints(&mut (0..packed.len()).map(|i| int.wrap(i128::from(code(i)))))
                }
                (None, None) => unreachable!("every packed dtype is a float or an integer"),
            }
        }
    }
}

/// `values`, integers in the range of the integer dtype or `i1` `dtype`, in a tensor of it.
pub fn from_ints(dtype: DType, values: impl Iterator<Item = i128>) -> Result<TensorData> {
    // The low bits of each value are its code.
    Ok(match dtype {
        DType::I1 => TensorData::I1(values.map(|x| x as u8).collect()),
        DType::Si8 => TensorData::Si8(values.map(|x| x as i8).collect()),
        DType::Ui8 => TensorData::Ui8(values.map(|x| x as u8).collect()),
        DType::Si16 => TensorData::Si16(values.map(|x| x as i16).collect()),
        DType::Ui16 => TensorData::Ui16(values.map(|x| x as u16).collect()),
        DType::Si32 => TensorData::Si32(values.map(|x| x as i32).collect()),
        DType::Ui32 => TensorData::Ui32(values.map(|x| x as u32).collect()),
        DType::Si64 => TensorData::Si64(values.map(|x| x as i64).collect()),
        DType::Ui64 => TensorData::Ui64(values.map(|x| x as u64).collect()),
        DType::Si4 | DType::Ui4 => {
            let codes: Vec<u8> = values.map(|x| x as u8 & 0x0f).collect();
            packed(dtype, &codes)?
        }
        other => bail!("{other} is not an integer dtype"),
    })
}

/// `values` converted to the float dtype `dtype`: rounded to nearest, ties to even. bf16, f16 and
/// f32 overflow to infinity, and the narrower floats saturate to their largest finite value.
pub fn from_f32(dtype: DType, values: impl Iterator<Item = f32>) -> Result<TensorData> {
    Ok(match dtype {
        DType::F64 => TensorData::F64(values.map(f64::from).collect()),
        DType::F32 => TensorData::F32(values.collect()),
        DType::F16 => TensorData::F16(values.map(f16::from_f32).collect()),
        DType::Bf16 => TensorData::Bf16(values.map(bf16::from_f32).collect()),
        other => narrow(other, values.map(f64::from), Overflow::Saturate)?,
    })
}

/// `values`, each a value of the float dtype `dtype`, as its data. Unlike [`from_f32`], which
/// converts, it keeps every value, infinities among them.
pub fn encode_f32(dtype: DType, values: impl Iterator<Item = f32>) -> Result<TensorData> {
    match dtype {
        DType::F32 | DType::F16 | DType::Bf16 => from_f32(dtype, values),
        other => narrow(other, values.map(f64::from), Overflow::Infinity),
    }
}

/// `values` converted to the float dtype `dtype`, as [`from_f32`] converts.
pub fn from_f64(dtype: DType, values: impl Iterator<Item = f64>) -> Result<TensorData> {
    let ieee = |format: Format, x: f64| format.encode(x, Overflow::Infinity) as u16;
    Ok(match dtype {
        DType::F64 => TensorData::F64(values.collect()),
        DType::F32 => TensorData::F32(values.map(|x| x as f32).collect()),
        DType::F16 => TensorData::F16(
            values
                .map(|x| f16::from_bits(ieee(Format::F16, x)))
                .collect(),
        ),
        DType::Bf16 => TensorData::Bf16(
            values
                .map(|x| bf16::from_bits(ieee(Format::BF16, x)))
                .collect(),
        ),
        other => narrow(other, values, Overflow::Saturate)?,
    })
}

/// `values` rounded into a float dtype narrower than 16 bits, with `overflow` beyond its range.
fn narrow(
    dtype: DType,
    values: impl Iterator<Item = f64>,
    overflow: Overflow,
) -> Result<TensorData> {
    let Some(format) = Format::of(dtype) else {
        bail!("{dtype} is not a float dtype");
    };
    let codes: Vec<u8> = values.map(|x| format.encode(x, overflow) as u8).collect();
    Ok(match dtype {
        DType::F8E4M3 => TensorData::F8E4M3(codes.into_iter().map(F8E4M3).collect()),
        DType::F8E5M2 => TensorData::F8E5M2(codes.into()),
        DType::F8E4M3Fnuz => TensorData::F8E4M3Fnuz(codes.into()),
        DType::F8E5M2Fnuz => TensorData::F8E5M2Fnuz(codes.into()),
        _ => packed(dtype, &codes)?,
    })
}

/// `values`, computed in the type the interpreter computes `dtype` in, in a tensor of `dtype`.
/// Integers must lie in the dtype's range.
pub fn from_values(dtype: DType, values: Values<'_>) -> Result<TensorData> {
    match values {
        Values::Int(v) => from_ints(dtype, v.into_iter()),
        Values::F32(v) => from_f32(dtype, v.iter().copied()),
        Values::F64(v) => from_f64(dtype, v.iter().copied()),
    }
}

/// The value of an f32 `x` after rounding to the float dtype `dtype`, bf16, f16 or f32: what an
/// accumulator of `dtype` keeps of each step.
pub fn rounder(dtype: DType) -> fn(f32) -> f32 {
    match dtype {
        DType::Bf16 => |x| bf16::from_f32(x).to_f32(),
        DType::F16 => |x| f16::from_f32(x).to_f32(),
        _ => |x| x,
    }
}

/// `x` rounded to nearest, ties to even, to `bits` significant bits. The result has at most
/// `bits + 1` significant bits, so it is an f64 exactly.
fn round_integer(x: i128, bits: u32) -> f64 {
    let magnitude = x.unsigned_abs();
    let length = 128 - magnitude.leading_zeros();
    let Some(shift) = length.checked_sub(bits).filter(|&shift| shift > 0) else {
        return x as f64;
    };
    let (kept, dropped) = (magnitude >> shift, magnitude & ((1 << shift) - 1));
    let half = 1 << (shift - 1);
    let up = dropped > half || (dropped == half && kept & 1 == 1);
    let rounded = (kept + u128::from(up)) as f64 * 2f64.powi(shift as i32);
    if x < 0 {
        -rounded
    } else {
        rounded
    }
}

/// The elements of `data` converted to `to`, as `tensor.cast` converts them:
///
/// - a float to a float rounds to nearest, ties to even;
/// - a float to an integer truncates toward zero and saturates, and NaN gives 0;
/// - an integer to an integer saturates;
/// - an integer to a float rounds once to nearest;
/// - any value to `i1` gives 1 unless the value is zero.
pub fn cast(data: &TensorData, to: DType) -> Result<TensorData> {
    if data.dtype() == to {
        return Ok(data.clone());
    }
    if to == DType::I1 {
        let nonzero: Vec<u8> = match values(data) {
            Values::Int(v) => v.iter().map(|&x| u8::from(x != 0)).collect(),
            Values::F32(v) => v.iter().map(|&x| u8::from(x != 0.0)).collect(),
            Values::F64(v) => v.iter().map(|&x| u8::from(x != 0.0)).collect(),
        };
        return Ok(TensorData::I1(nonzero.into()));
    }
    if let Some(int) = to.int_format() {
        let saturate = |x: i128| int.saturate(x);
        return match values(data) {
            Values::Int(v) => from_ints(to, v.into_iter().map(saturate)),
            // `as` truncates toward zero, saturates and gives 0 for NaN.
            Values::F32(v) => from_ints(to, v.iter().map(|&x| saturate(x as i128))),
            Values::F64(v) => from_ints(to, v.iter().map(|&x| saturate(x as i128))),
        };
    }
    let Some(bits) = to.significand_bits() else {
        bail!("the interpreter casts nothing into {to}");
    };
    match values(data) {
        Values::Int(v) => from_f64(to, v.into_iter().map(|x| round_integer(x, bits))),
        Values::F32(v) => from_f32(to, v.iter().copied()),
        Values::F64(v) => from_f64(to, v.iter().copied()),
    }
}

/// A tensor of the sub-byte dtype `dtype` whose elements have the codes `codes`.
pub fn packed(dtype: DType, codes: &[u8]) -> Result<TensorData> {
    let mut data = TensorData::zeroed(dtype, codes.len())?;
    let TensorData::Packed(packed) = &mut data else {
        bail!("{dtype} elements are not packed");
    };
    for (i, &code) in codes.iter().enumerate() {
        packed.set(i, code);
    }
    Ok(data)
}

/// A computation that accumulates: from its inputs as elements of the accumulator's computation
/// type, it gives the result elements, rounding every step with `round`.
pub(super) trait Accumulation {
    fn run<T: Arith>(
        &self,
        inputs: &[&[T]],
        accum: DType,
        round: impl Fn(T) -> T,
    ) -> Result<Vec<T>>;
}

/// `kernel` on `inputs` cast to `accum`, with its result cast to `out`: how the contractions,
/// reductions and scatters compute.
pub(super) fn accumulate(
    inputs: &[&TensorData],
    accum: DType,
    out: DType,
    kernel: &impl Accumulation,
) -> Result<TensorData> {
    let data: Vec<TensorData> = inputs
        .iter()
        .map(|&data| cast(data, accum))
        .collect::<Result<_>>()?;
    let decoded: Vec<Values<'_>> = data.iter().map(values).collect();
    let result = match decoded.first() {
        Some(Values::Int(_)) => run::<i128>(kernel, &decoded, accum)?,
        Some(Values::F32(_)) => run::<f32>(kernel, &decoded, accum)?,
        Some(Values::F64(_)) => run::<f64>(kernel, &decoded, accum)?,
        None => bail!("an accumulation reads a tensor"),
    };
    cast(&from_values(accum, result)?, out)
}

fn run<T: Arith>(
    kernel: &impl Accumulation,
    decoded: &[Values<'_>],
    accum: DType,
) -> Result<Values<'static>> {
    let inputs = decoded
        .iter()
        .map(|values| T::of(values).ok_or_else(|| anyhow!("the inputs of {accum} differ")))
        .collect::<Result<Vec<_>>>()?;
    Ok(T::into_values(kernel.run(
        &inputs,
        accum,
        T::rounder(accum)?,
    )?))
}

/// The positions of the `k` best of `count` candidates, best first: the largest by `order` with
/// `largest`, else the smallest. Of candidates that order equal, the one at the smaller position
/// comes first.
pub(crate) fn best_k(
    count: usize,
    k: usize,
    largest: bool,
    order: impl Fn(usize, usize) -> Ordering,
) -> Vec<usize> {
    let mut ranking: Vec<usize> = (0..count).collect();
    // A stable sort keeps equal candidates in position order.
    ranking.sort_by(|&first, &second| {
        let ordering = order(first, second);
        if largest {
            ordering.reverse()
        } else {
            ordering
        }
    });
    ranking.truncate(k);
    ranking
}
