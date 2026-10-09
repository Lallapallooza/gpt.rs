//! Semantics of the ops in `ir/quant.rs`, computed in f32.

use anyhow::{anyhow, bail, ensure, Result};

use super::kernels::{from_f32, from_ints, strides, unravel, values, Values};
use crate::dialects::tensor::ir::{
    Dequantize, DequantizeOperands, DequantizeResults, Quantize, QuantizeDynamic,
    QuantizeDynamicOperands, QuantizeDynamicResults, QuantizeOperands, QuantizeResults, Rounding,
    ScaleRule,
};
use crate::host::{HostTensor, TensorData};
use crate::minifloat::{e8m0_encode, Format, Overflow};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::{DType, IntFormat};

/// The value of every element of `data` as an f32.
fn reals(data: &TensorData) -> Vec<f32> {
    match values(data) {
        Values::F32(v) => v.into_owned(),
        Values::F64(v) => v.iter().map(|&x| x as f32).collect(),
        Values::Int(v) => v.iter().map(|&x| x as f32).collect(),
    }
}

/// `floor(log2(x))` of a finite `x > 0` whose f64 is normal, exactly.
fn floor_log2(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
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

/// The elements a quantized dtype holds: a float format or an integer range.
#[derive(Clone, Copy)]
enum Elements {
    Float(Format),
    Int(IntFormat),
}

impl Elements {
    fn of(dtype: DType) -> Result<Self> {
        match (Format::of(dtype), dtype.int_format()) {
            (Some(format), _) => Ok(Elements::Float(format)),
            (None, Some(int)) => Ok(Elements::Int(int)),
            _ => bail!("{dtype} is not a quantized dtype"),
        }
    }

    fn largest(self) -> f32 {
        match self {
            Elements::Float(format) => format.max() as f32,
            Elements::Int(int) => int.max() as f32,
        }
    }

    fn smallest(self) -> f32 {
        match self {
            Elements::Float(format) => -format.max() as f32,
            Elements::Int(int) => int.min() as f32,
        }
    }

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

    /// `y` rounded as `rounding` says, with the draw `r` for stochastic rounding, then saturated.
    /// NaN stays NaN for a float, which the encoding maps, and gives 0 for an integer.
    fn round(self, y: f32, rounding: Rounding, r: u32) -> f64 {
        let y = f64::from(y);
        if !y.is_finite() {
            return match self {
                Elements::Int(_) if y.is_nan() => 0.0,
                Elements::Int(int) if y > 0.0 => int.max() as f64,
                Elements::Int(int) => int.min() as f64,
                Elements::Float(_) => y,
            };
        }
        let rounded = match (rounding, self) {
            (Rounding::NearestEven, Elements::Int(_)) => y.round_ties_even(),
            (Rounding::NearestEven, Elements::Float(format)) => {
                format.decode(format.encode(y, Overflow::Saturate))
            }
            (Rounding::Stochastic, _) => {
                // Up with probability `(|y| - lo) / spacing`, decided by the draw's 32 bits.
                let (lo, spacing) = self.bracket(y.abs());
                let threshold = ((y.abs() - lo) / spacing * 2f64.powi(32)).floor();
                let up = f64::from(r) < threshold;
                (lo + if up { spacing } else { 0.0 }).copysign(y)
            }
        };
        match self {
            Elements::Int(int) => rounded.clamp(int.min() as f64, int.max() as f64),
            Elements::Float(_) => rounded,
        }
    }

    /// `values`, each a value of `dtype` or, for a float, beyond its range, in a tensor of it.
    fn encode(self, dtype: DType, values: Vec<f64>) -> Result<TensorData> {
        match self {
            Elements::Int(_) => from_ints(dtype, values.into_iter().map(|x| x as i128)),
            Elements::Float(_) => from_f32(dtype, values.into_iter().map(|x| x as f32)),
        }
    }
}

/// For each element of a tensor of `dims`, the index of its scale with `block`.
fn scale_indices(dims: &[usize], block: &[usize]) -> Vec<usize> {
    let scale_dims: Vec<usize> = dims
        .iter()
        .zip(block)
        .map(|(&n, &b)| n.div_ceil(b))
        .collect();
    let scale_strides = strides(&scale_dims);
    let count: usize = dims.iter().product();
    (0..count)
        .map(|i| {
            unravel(i, dims)
                .iter()
                .zip(block)
                .zip(&scale_strides)
                .map(|((coord, b), stride)| coord / b * stride)
                .sum()
        })
        .collect()
}

/// What quantizing elements reads besides them.
struct Scheme<'a> {
    scales: &'a [f32],
    offsets: Option<&'a [f32]>,
    table: Option<&'a [f32]>,
    rounding: Rounding,
    random: Option<&'a [u32]>,
}

/// `x` quantized into `to`, element `i` with scale and offset `index[i]`.
fn quantize(x: &[f32], index: &[usize], to: DType, scheme: &Scheme<'_>) -> Result<TensorData> {
    let elements = Elements::of(to)?;
    let quotient = |i: usize| -> f32 {
        let s = index[i];
        let offset = scheme.offsets.map_or(0.0, |o| o[s]);
        match scheme.scales[s] {
            0.0 => 0.0,
            scale => (x[i] - offset) / scale,
        }
    };
    if let Some(table) = scheme.table {
        let codes = (0..x.len()).map(|i| nearest(table, quotient(i)) as i128);
        return from_ints(to, codes);
    }
    let rounded = (0..x.len())
        .map(|i| {
            let draw = scheme.random.map_or(0, |r| r[i]);
            elements.round(quotient(i), scheme.rounding, draw)
        })
        .collect();
    elements.encode(to, rounded)
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

/// The draws of a `random` operand.
fn draws(random: Option<&HostTensor>) -> Result<Option<Vec<u32>>> {
    let Some(random) = random else {
        return Ok(None);
    };
    match &random.data {
        TensorData::Ui32(v) => Ok(Some(v.to_vec())),
        other => bail!("random bits must be ui32, got {}", other.dtype()),
    }
}

impl Eval for Dequantize {
    fn eval(
        &self,
        ins: DequantizeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<DequantizeResults<OwnedHost>> {
        let scales = reals(&ins.scale.data);
        let offsets = ins.offset.map(|o| reals(&o.data));
        let q = match ins.table {
            Some(table) => {
                let table = reals(&table.data);
                let Values::Int(codes) = values(&ins.q.data) else {
                    bail!("table codes must be integers");
                };
                codes
                    .iter()
                    .map(|&code| {
                        usize::try_from(code)
                            .ok()
                            .and_then(|k| table.get(k).copied())
                            .ok_or_else(|| {
                                anyhow!("code {code} is out of range for {} values", table.len())
                            })
                    })
                    .collect::<Result<Vec<f32>>>()?
            }
            None => reals(&ins.q.data),
        };
        let index = scale_indices(&ins.q.dims, &self.block);
        let real = q.iter().zip(&index).map(|(&q, &s)| {
            let offset = offsets.as_ref().map_or(0.0, |o| o[s]);
            q * scales[s] + offset
        });
        Ok(DequantizeResults {
            output: HostTensor::new(ins.q.dims.clone(), from_f32(self.to, real)?)?,
        })
    }
}

impl Eval for Quantize {
    fn eval(
        &self,
        ins: QuantizeOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<QuantizeResults<OwnedHost>> {
        let x = reals(&ins.x.data);
        let scales = reals(&ins.scale.data);
        let offsets = ins.offset.map(|o| reals(&o.data));
        let table = ins.table.map(|t| reals(&t.data));
        let random = draws(ins.random)?;
        let scheme = Scheme {
            scales: &scales,
            offsets: offsets.as_deref(),
            table: table.as_deref(),
            rounding: self.rounding,
            random: random.as_deref(),
        };
        let index = scale_indices(&ins.x.dims, &self.block);
        Ok(QuantizeResults {
            output: HostTensor::new(ins.x.dims.clone(), quantize(&x, &index, self.to, &scheme)?)?,
        })
    }
}

impl Eval for QuantizeDynamic {
    fn eval(
        &self,
        ins: QuantizeDynamicOperands<Host<'_>>,
        _cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<QuantizeDynamicResults<OwnedHost>> {
        let dims = &ins.x.dims;
        let x = reals(&ins.x.data);
        let index = scale_indices(dims, &self.block);
        let scale_dims: Vec<usize> = dims
            .iter()
            .zip(&self.block)
            .map(|(&n, &b)| n.div_ceil(b))
            .collect();
        let blocks: usize = scale_dims.iter().product();
        let elements = Elements::of(self.to)?;
        // Each block's range, widened to hold 0. `min` and `max` skip NaN.
        let (mut lo, mut hi) = (vec![0.0f32; blocks], vec![0.0f32; blocks]);
        for (&x, &s) in x.iter().zip(&index) {
            lo[s] = lo[s].min(x);
            hi[s] = hi[s].max(x);
        }
        let (scale, offset) = if self.asymmetric {
            let span = elements.largest() - elements.smallest();
            let scale = from_f32(
                self.scale_dtype,
                lo.iter().zip(&hi).map(|(&lo, &hi)| (hi - lo) / span),
            )?;
            // The zero point, the code of 0, is an integer in `to`'s range, so 0 quantizes
            // exactly. The offset is `-zero_point * scale`, with the scale as stored.
            let (smallest, largest) = (elements.smallest(), elements.largest());
            let offset = from_f32(
                self.scale_dtype,
                lo.iter().zip(reals(&scale)).map(|(&lo, s)| {
                    let zero_point = match s {
                        0.0 => 0.0,
                        s => (smallest - lo / s)
                            .round_ties_even()
                            .clamp(smallest, largest),
                    };
                    -zero_point * s
                }),
            )?;
            (scale, Some(offset))
        } else {
            let amax: Vec<f32> = lo.iter().zip(&hi).map(|(&lo, &hi)| hi.max(-lo)).collect();
            (self.symmetric_scales(&amax, elements)?, None)
        };
        let scales = reals(&scale);
        let offsets = offset.as_ref().map(reals);
        let random = draws(ins.random)?;
        let scheme = Scheme {
            scales: &scales,
            offsets: offsets.as_deref(),
            table: None,
            rounding: self.rounding,
            random: random.as_deref(),
        };
        let q = quantize(&x, &index, self.to, &scheme)?;
        Ok(QuantizeDynamicResults {
            q: HostTensor::new(dims.clone(), q)?,
            offset: offset
                .map(|o| HostTensor::new(scale_dims.clone(), o))
                .transpose()?,
            scale: HostTensor::new(scale_dims, scale)?,
        })
    }
}

impl QuantizeDynamic {
    /// The scales of blocks whose largest magnitudes are `amax`, as `scale_rule` computes them.
    fn symmetric_scales(&self, amax: &[f32], elements: Elements) -> Result<TensorData> {
        let largest = f64::from(elements.largest());
        let exponent: fn(f64, f64) -> i32 = match self.scale_rule {
            ScaleRule::Amax => {
                ensure!(
                    self.scale_dtype != DType::E8M0,
                    "e8m0 scales need a power-of-two scale rule"
                );
                let scales = amax.iter().map(|&a| (f64::from(a) / largest) as f32);
                return from_f32(self.scale_dtype, scales);
            }
            ScaleRule::Pow2Floor => |a, largest| floor_log2(a) - floor_log2(largest),
            ScaleRule::Pow2Ceil => |a, largest| ceil_log2(a / largest),
        };
        // A block of zeros gets the smallest scale.
        let exponents: Vec<i32> = amax
            .iter()
            .map(|&a| match f64::from(a) {
                0.0 => -127,
                a => exponent(a, largest).clamp(-127, 127),
            })
            .collect();
        if self.scale_dtype == DType::E8M0 {
            let codes: Vec<u8> = exponents.into_iter().map(e8m0_encode).collect();
            return Ok(TensorData::E8M0(codes.into()));
        }
        from_f32(
            self.scale_dtype,
            exponents.into_iter().map(|e| 2f64.powi(e) as f32),
        )
    }
}
