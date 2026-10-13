//! Semantics of the quantization ops, computed in f32.

use anyhow::{bail, Result};

use super::kernels::{from_f32, from_ints, strides, unravel, values, Values};
use crate::dialects::arith;
use crate::dialects::arith::math::BinaryMath;
use crate::dialects::tensor::ir::*;
use crate::host::{HostTensor, TensorData};
use crate::ir::{
    e8m0_encode, floor_log2, DType, Elements, Eval, EvalCx, Host, NoRegions, Overflow, OwnedHost,
};

/// The value of every element of `data` as an f32.
fn reals(data: &TensorData) -> Vec<f32> {
    match values(data) {
        Values::F32(v) => v.into_owned(),
        Values::F64(v) => v.iter().map(|&x| x as f32).collect(),
        Values::Int(v) => v.iter().map(|&x| x as f32).collect(),
    }
}

/// `ceil(log2(x))` of a finite `x > 0` whose f64 is normal, exactly.
fn ceil_log2(x: f64) -> i32 {
    let floor = floor_log2(x);
    if x == 2f64.powi(floor) {
        floor
    } else {
        floor + 1
    }
}

impl Elements {
    /// For a `magnitude >= 0`, the largest value at or below it and the spacing to the next one.
    fn bracket(self, magnitude: f64) -> (f64, f64) {
        let spacing = match self {
            Elements::Int(_) => 1.0,
            Elements::Float(format) => {
                let min_normal = 1 - format.bias;
                let binade = if magnitude > 0.0 {
                    floor_log2(magnitude).max(min_normal)
                } else {
                    min_normal
                };
                2f64.powi(binade - format.man_bits as i32)
            }
        };
        ((magnitude / spacing).floor() * spacing, spacing)
    }

    /// What a non-finite `value` quantizes to: NaN stays NaN for a float, which the encoding maps,
    /// and gives 0 for an integer. Infinities saturate.
    fn non_finite(self, value: f64) -> f64 {
        match self {
            Elements::Int(_) if value.is_nan() => 0.0,
            Elements::Int(int) if value > 0.0 => int.max() as f64,
            Elements::Int(int) => int.min() as f64,
            Elements::Float(_) => value,
        }
    }

    /// `rounded` saturated to the range of an integer dtype. A float's encoding saturates it.
    fn saturate(self, rounded: f64) -> f64 {
        match self {
            Elements::Int(int) => rounded.clamp(int.min() as f64, int.max() as f64),
            Elements::Float(_) => rounded,
        }
    }

    /// `y` rounded to the nearest value, the even one on a tie, then saturated.
    fn round_nearest(self, y: f32) -> f64 {
        let value = f64::from(y);
        if !value.is_finite() {
            return self.non_finite(value);
        }
        let rounded = match self {
            Elements::Int(_) => value.round_ties_even(),
            Elements::Float(format) => format.decode(format.encode(value, Overflow::Saturate)),
        };
        self.saturate(rounded)
    }

    /// `y` rounded up or down as the draw `draw` decides, then saturated.
    fn round_stochastic(self, y: f32, draw: u32) -> f64 {
        let value = f64::from(y);
        if !value.is_finite() {
            return self.non_finite(value);
        }
        // Up with probability `(|y| - lo) / spacing`, decided by the draw's 32 bits.
        let (lo, spacing) = self.bracket(value.abs());
        let threshold = ((value.abs() - lo) / spacing * 2f64.powi(32)).floor();
        let up = f64::from(draw) < threshold;
        self.saturate((lo + if up { spacing } else { 0.0 }).copysign(value))
    }

    /// `values`, each a value of `dtype` or, for a float, beyond its range, in a tensor of it.
    fn encode(self, dtype: DType, values: Vec<f64>) -> Result<TensorData> {
        match self {
            Elements::Int(_) => from_ints(dtype, values.into_iter().map(|x| x as i128)),
            Elements::Float(_) => from_f32(dtype, values.into_iter().map(|x| x as f32)),
        }
    }
}

/// The extents of the scales of a tensor of `dims` with `block`.
fn scale_extents(dims: &[usize], block: &[usize]) -> Vec<usize> {
    let mut extents = Vec::with_capacity(dims.len());
    for (&extent, &size) in dims.iter().zip(block) {
        extents.push(extent.div_ceil(size));
    }
    extents
}

/// For each element of a tensor of `dims`, the index of its scale with `block`.
fn scale_indices(dims: &[usize], block: &[usize]) -> Vec<usize> {
    let scale_strides = strides(&scale_extents(dims, block));
    let count: usize = dims.iter().product();
    let mut indices = Vec::with_capacity(count);
    for flat in 0..count {
        let mut index = 0;
        for ((coord, size), stride) in unravel(flat, dims).iter().zip(block).zip(&scale_strides) {
            index += coord / size * stride;
        }
        indices.push(index);
    }
    indices
}

/// `y = (x - offset) / scale` for each element of `x`, in f32, with its block's scale and offset;
/// 0 where the scale is 0.
fn quotients(
    x: &HostTensor,
    scale: &HostTensor,
    offset: Option<&HostTensor>,
    block: &[usize],
) -> Vec<f32> {
    let scales = reals(&scale.data);
    let offsets = offset.map(|offset| reals(&offset.data));
    let elements = reals(&x.data);
    let indices = scale_indices(&x.dims, block);
    let mut quotients = Vec::with_capacity(elements.len());
    for (&element, &at) in elements.iter().zip(&indices) {
        let shift = offsets.as_ref().map_or(0.0, |offsets| offsets[at]);
        quotients.push(match scales[at] {
            0.0 => 0.0,
            divisor => (element - shift) / divisor,
        });
    }
    quotients
}

/// The index of the value of `table` nearest to `y`, the lowest index on a tie, or 0 for NaN.
fn nearest(table: &[f32], y: f32) -> usize {
    let distance = |k: usize| (table[k] - y).abs();
    (0..table.len())
        .reduce(|best, k| {
            if distance(k) < distance(best) {
                k
            } else {
                best
            }
        })
        .unwrap_or(0)
}

impl Eval for Dequantize {
    fn eval(
        &self,
        ins: DequantizeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DequantizeResults<OwnedHost>> {
        let scales = reals(&ins.scale.data);
        let offsets = ins.offset.map(|offset| reals(&offset.data));
        let codes = reals(&ins.q.data);
        let indices = scale_indices(&ins.q.dims, &self.block);
        let mut real = Vec::with_capacity(codes.len());
        for (&code, &at) in codes.iter().zip(&indices) {
            // Without an offset the sum adds nothing: `-0.0 + 0.0` would be `0.0`.
            real.push(match &offsets {
                Some(offsets) => code * scales[at] + offsets[at],
                None => code * scales[at],
            });
        }
        Ok(DequantizeResults {
            output: HostTensor::new(ins.q.dims, from_f32(self.to, real.into_iter())?)?,
        })
    }
}

impl Eval for Quantize {
    fn eval(
        &self,
        ins: QuantizeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<QuantizeResults<OwnedHost>> {
        let elements = self.to.elements()?;
        let mut rounded = Vec::new();
        for y in quotients(ins.x, ins.scale, ins.offset, &self.block) {
            rounded.push(elements.round_nearest(y));
        }
        Ok(QuantizeResults {
            output: HostTensor::new(ins.x.dims, elements.encode(self.to, rounded)?)?,
        })
    }
}

impl Eval for QuantizeStochastic {
    fn eval(
        &self,
        ins: QuantizeStochasticOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<QuantizeStochasticResults<OwnedHost>> {
        let TensorData::Ui32(draws) = &ins.random.data else {
            bail!("random bits must be ui32, got {}", ins.random.dtype());
        };
        let elements = self.to.elements()?;
        let mut rounded = Vec::new();
        let quotients = quotients(ins.x, ins.scale, ins.offset, &self.block);
        for (y, &draw) in quotients.into_iter().zip(draws.iter()) {
            rounded.push(elements.round_stochastic(y, draw));
        }
        Ok(QuantizeStochasticResults {
            output: HostTensor::new(ins.x.dims, elements.encode(self.to, rounded)?)?,
        })
    }
}

impl Eval for QuantizeTable {
    fn eval(
        &self,
        ins: QuantizeTableOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<QuantizeTableResults<OwnedHost>> {
        let table = reals(&ins.table.data);
        let mut codes = Vec::new();
        for y in quotients(ins.x, ins.scale, ins.offset, &self.block) {
            codes.push(nearest(&table, y) as i128);
        }
        Ok(QuantizeTableResults {
            output: HostTensor::new(ins.x.dims, from_ints(self.to, codes.into_iter())?)?,
        })
    }
}

/// The range of each block of a tensor: the smallest and the largest of its non-NaN elements and
/// 0, with `-0.0` below `0.0` as `minimum` and `maximum` order them.
struct BlockRanges {
    low: Vec<f32>,
    high: Vec<f32>,
    /// The extents of the blocks, which the scales have.
    dims: Vec<usize>,
}

impl BlockRanges {
    fn of(x: &HostTensor, block: &[usize]) -> Self {
        let dims = scale_extents(&x.dims, block);
        let count: usize = dims.iter().product();
        let minimum = arith::Minimum::float::<f32>().expect("floats have minimum");
        let maximum = arith::Maximum::float::<f32>().expect("floats have maximum");
        let (mut low, mut high) = (vec![0.0f32; count], vec![0.0f32; count]);
        let indices = scale_indices(&x.dims, block);
        for (&element, &at) in reals(&x.data).iter().zip(&indices) {
            if !element.is_nan() {
                low[at] = minimum(low[at], element);
                high[at] = maximum(high[at], element);
            }
        }
        BlockRanges { low, high, dims }
    }
}

/// `2^exponent(amax)`, the exponent clamped to `[-127, 127]`, or `2^-127` for a block of zeros.
fn power_of_two(amax: f64, exponent: impl Fn(f64) -> i32) -> f64 {
    let clamped = if amax == 0.0 {
        -127
    } else {
        exponent(amax).clamp(-127, 127)
    };
    2f64.powi(clamped)
}

impl ScaleFormula for BlockScaleAmax {
    fn scale(amax: f64, largest: f64) -> f64 {
        amax / largest
    }
}

impl ScaleFormula for BlockScalePow2Floor {
    fn scale(amax: f64, largest: f64) -> f64 {
        power_of_two(amax, |amax| floor_log2(amax) - floor_log2(largest))
    }
}

impl ScaleFormula for BlockScalePow2Ceil {
    fn scale(amax: f64, largest: f64) -> f64 {
        power_of_two(amax, |amax| ceil_log2(amax / largest))
    }
}

/// The semantics of every [`BlockScale`] op.
pub(crate) fn block_scale<T: BlockScale>(
    op: &T,
    ins: BlockScaleOperands<Host<'_>>,
    _cx: &mut EvalCx<'_, NoRegions>,
) -> Result<BlockScaleResults<OwnedHost>> {
    let ranges = BlockRanges::of(ins.x, &op.block);
    let largest = f64::from(op.to.elements()?.largest());
    let maximum = arith::Maximum::float::<f32>().expect("floats have maximum");
    let mut scales = Vec::with_capacity(ranges.low.len());
    for (&low, &high) in ranges.low.iter().zip(&ranges.high) {
        let amax = maximum(high, -low);
        scales.push(T::scale(f64::from(amax), largest));
    }
    let data = if op.scale_dtype == DType::E8M0 {
        // A power-of-two scale is stored as the code of its exponent.
        let codes: Vec<u8> = scales
            .iter()
            .map(|&scale| e8m0_encode(floor_log2(scale)))
            .collect();
        TensorData::E8M0(codes.into())
    } else {
        from_f32(op.scale_dtype, scales.into_iter().map(|scale| scale as f32))?
    };
    Ok(BlockScaleResults {
        scale: HostTensor::new(ranges.dims, data)?,
    })
}

impl Eval for BlockScaleRange {
    fn eval(
        &self,
        ins: BlockScaleRangeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<BlockScaleRangeResults<OwnedHost>> {
        let ranges = BlockRanges::of(ins.x, &self.block);
        let elements = self.to.elements()?;
        let (smallest, largest) = (elements.smallest(), elements.largest());
        let span = largest - smallest;
        let mut widths = Vec::with_capacity(ranges.low.len());
        for (&low, &high) in ranges.low.iter().zip(&ranges.high) {
            widths.push((high - low) / span);
        }
        let scale = from_f32(self.scale_dtype, widths.into_iter())?;
        // The zero point, the code of 0, is an integer in `to`'s range, so 0 quantizes exactly. The
        // offset is `-zero_point * scale`, with the scale as stored.
        let mut offsets = Vec::with_capacity(ranges.low.len());
        for (&low, stored) in ranges.low.iter().zip(reals(&scale)) {
            let zero_point = match stored {
                0.0 => 0.0,
                divisor => (smallest - low / divisor)
                    .round_ties_even()
                    .clamp(smallest, largest),
            };
            offsets.push(-zero_point * stored);
        }
        Ok(BlockScaleRangeResults {
            scale: HostTensor::new(ranges.dims.clone(), scale)?,
            offset: HostTensor::new(
                ranges.dims,
                from_f32(self.scale_dtype, offsets.into_iter())?,
            )?,
        })
    }
}
