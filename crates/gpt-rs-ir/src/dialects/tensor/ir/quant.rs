//! Quantization: elements of a narrow dtype with scales, and offsets, that map them back to real
//! values. "Quantization" in the dialect documentation gives the scale layout the ops share.

use anyhow::{bail, ensure, Result};
use smallvec::SmallVec;

use super::class::{ensure_class, ensure_dtype_class, Class};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{DType, Dim, TensorType};
use crate::Attr;

/// The classes of quantized elements: integers and narrow floats.
const ELEMENTS: &[Class] = &[Class::Int, Class::Narrow];
/// The classes of scales.
const SCALES: &[Class] = &[Class::Float, Class::Narrow, Class::Scale];

/// Whether `dtype` is a real-valued input or output of quantization.
fn is_real(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::Bf16 | DType::F16)
}

/// How a value `y` between two quantized values picks one.
#[derive(Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// The nearest value of the quantized dtype, the even one on a tie.
    #[default]
    NearestEven,
    /// Rounds `abs(y)` to one of its two neighbors `lo` and `lo + q` in the quantized dtype, where
    /// `q` is their spacing. It picks `lo + q` when the element's draw `r` from the `random`
    /// operand satisfies `r < floor((abs(y) - lo) / q * 2^32)`, else `lo`, and keeps the sign of
    /// `y`. So rounding picks `lo + q` with probability `(abs(y) - lo) / q`. FP8 and FP4 training
    /// round gradients this way.
    Stochastic,
}

/// How `tensor.quantize_dynamic` computes a block's symmetric scale from the block's largest
/// magnitude `amax`, where `max(to)` is the largest finite value of the quantized dtype `to`. A
/// power-of-two scale has its exponent clamped to `[-127, 127]`, and a block whose `amax` is 0
/// gets the smallest one, `2^-127`. `e8m0` scales need a power-of-two rule.
#[derive(Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ScaleRule {
    /// `scale_dtype(amax / max(to))`, rounded to nearest-even, so `amax` maps to the largest value
    /// of `to`; a block whose `amax` is 0 gets the scale 0. NVFP4 takes its `f8e4m3` block scales
    /// this way, and per-tensor, per-channel and per-token scales are `f32` or `bf16`.
    #[default]
    Amax,
    /// `2^(floor(log2(amax)) - emax)`, the microscaling (MX) rule, where
    /// `emax = floor(log2(max(to)))` is the exponent of the largest power of two `to` holds
    /// (`f4e2m1`: 2, `f6e2m3`: 2, `f6e3m2`: 4, `f8e4m3`: 8, `f8e5m2`: 15). Values above `max(to)`
    /// times the scale saturate.
    Pow2Floor,
    /// `2^ceil(log2(amax / max(to)))`, the smallest power of two that maps `amax` into `to`, as
    /// DeepSeek's UE8M0 scales are.
    Pow2Ceil,
}

/// The scale extents of a tensor of `dims` with `block`, `ceil(extent / block)` on each axis.
fn scale_dims(dims: &[Dim], block: &[usize]) -> Result<SmallVec<[Dim; 4]>> {
    ensure!(
        block.len() == dims.len(),
        "block {block:?} must have one extent per axis of rank {}",
        dims.len()
    );
    dims.iter()
        .zip(block)
        .map(|(dim, &b)| {
            ensure!(b > 0, "block {block:?} must be positive");
            Ok(match dim {
                Dim::Static(n) => Dim::Static(n.div_ceil(b)),
                Dim::Dynamic(_) if b == 1 => dim.clone(),
                Dim::Dynamic(_) => bail!("a block of {b} needs a static extent, got {dim}"),
            })
        })
        .collect()
}

/// Checks that `scale`, and `offset` when there is one, fit a tensor of `dims` with `block`.
fn ensure_scales(
    dims: &[Dim],
    block: &[usize],
    scale: &TensorType,
    offset: Option<&TensorType>,
) -> Result<()> {
    ensure_class("the scale", scale, SCALES)?;
    let expected = scale_dims(dims, block)?;
    let extents: Vec<String> = expected.iter().map(ToString::to_string).collect();
    ensure!(
        scale.dims == expected,
        "the scale must have the extents [{}] for block {block:?}, got {scale}",
        extents.join(", ")
    );
    if let Some(offset) = offset {
        ensure_class("the offset", offset, &[Class::Float])?;
        ensure!(
            offset.dims == scale.dims,
            "the offset must have the scale's extents, got {offset} and {scale}"
        );
    }
    Ok(())
}

/// Checks that `codes` is an integer dtype and `table` is a rank-1 float tensor.
fn ensure_table(codes: DType, table: &TensorType) -> Result<()> {
    ensure_dtype_class("codes with a table", codes, &[Class::Int])?;
    ensure_class("the table", table, &[Class::Float, Class::Narrow])?;
    ensure!(table.rank() == 1, "the table must have rank 1, got {table}");
    Ok(())
}

/// Checks that `random` is present exactly when `rounding` reads it, as ui32 draws of `x`'s shape.
fn ensure_random(rounding: Rounding, x: &TensorType, random: Option<&TensorType>) -> Result<()> {
    match (rounding, random) {
        (Rounding::NearestEven, None) => Ok(()),
        (Rounding::NearestEven, Some(_)) => bail!("only stochastic rounding reads random bits"),
        (Rounding::Stochastic, None) => bail!("stochastic rounding needs a random operand"),
        (Rounding::Stochastic, Some(random)) => {
            let expected = x.with_dtype(DType::Ui32);
            ensure!(
                *random == expected,
                "the random bits must be {expected}, got {random}"
            );
            Ok(())
        }
    }
}

/// `tensor.dequantize`: `out[i] = to(value(q[i]) * scale[i / block] + offset[i / block])`,
/// computed in f32, then rounded to nearest-even in `to`.
///
/// - `to` is `f32`, `bf16` or `f16`. The scale and the offset follow the scale layout of
///   "Quantization" in the dialect documentation.
/// - `value(q)` is `table[q]` with a table, whose codes `q` are integers in `[0, n)`, and the value
///   of `q` otherwise. Without an offset, the sum adds nothing.
/// - The offset carries asymmetric quantization: the zero points of GPTQ and AWQ as
///   `offset = -zero_point * scale`, computed once at load time, and the block minimums of GGUF.
/// - A backend SHOULD read `q` and `scale` in place inside the op that consumes the result, such
///   as a `dot_general` with a quantized weight.
/// - Autodiff MUST differentiate it as multiplication by the scale, so that
///   `dequantize(quantize(x))` passes gradients through unchanged.
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.dequantize",
    operands(
        q: Tensor,
        scale: Tensor,
        offset: Optional<Tensor>,
        table: Optional<Tensor>
    ),
    results(output: Tensor)
)]
pub struct Dequantize {
    pub block: SmallVec<[usize; 4]>,
    pub to: DType,
}

impl OpDef for Dequantize {
    fn result_types(
        &self,
        ins: DequantizeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<DequantizeResults<OwnedTypes>> {
        ensure_class("q", ins.q, ELEMENTS)?;
        if let Some(table) = ins.table {
            ensure_table(ins.q.dtype, table)?;
        }
        ensure!(is_real(self.to), "cannot dequantize to {}", self.to);
        ensure_scales(&ins.q.dims, &self.block, ins.scale, ins.offset)?;
        Ok(DequantizeResults {
            output: ins.q.with_dtype(self.to),
        })
    }
}

/// `tensor.quantize`: quantizes `x` into `to` with given scales and offsets, or into the indices
/// of a table.
///
/// - `x` is `f32`, `bf16` or `f16`, and `scale` holds static scales, for example parameters. The
///   scale and the offset follow the scale layout of "Quantization" in the dialect documentation.
/// - `y[i] = (x[i] - offset[i / block]) / scale[i / block]`, computed in f32. A zero scale gives
///   `y[i] = 0`, so without a table `q[i] = 0`.
/// - Without a table, `q[i] = to(y[i])`, rounded as `rounding` says and saturated to the largest
///   finite value of `to`. Infinities saturate. NaN maps to NaN in a dtype that has one, and to 0
///   otherwise.
/// - With a table, `q[i]` is the index of the table value nearest to `y[i]`, the lowest index on
///   a tie, and 0 for NaN. `to` MUST be an integer dtype and `rounding` `nearest_even`. NF4 weights
///   and 8-bit optimizer states quantize this way.
/// - `random` is present exactly when `rounding` is `stochastic`, and holds one `ui32` draw per
///   element of `x`. The draws come from `tensor.rng_uniform` with `dtype = ui32`, so a state
///   reproduces the rounding.
/// - Autodiff MUST differentiate it as division by the scale, ignoring the rounding: the
///   straight-through estimator of quantization-aware training.
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.quantize",
    operands(
        x: Tensor,
        scale: Tensor,
        offset: Optional<Tensor>,
        table: Optional<Tensor>,
        random: Optional<Tensor>
    ),
    results(output: Tensor)
)]
pub struct Quantize {
    pub block: SmallVec<[usize; 4]>,
    pub to: DType,
    #[attr(default)]
    pub rounding: Rounding,
}

impl OpDef for Quantize {
    fn result_types(
        &self,
        ins: QuantizeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<QuantizeResults<OwnedTypes>> {
        ensure!(is_real(ins.x.dtype), "cannot quantize {}", ins.x);
        ensure_dtype_class("to", self.to, ELEMENTS)?;
        if let Some(table) = ins.table {
            ensure_table(self.to, table)?;
            ensure!(
                self.rounding == Rounding::NearestEven,
                "a table quantizes to the nearest value"
            );
        }
        ensure_scales(&ins.x.dims, &self.block, ins.scale, ins.offset)?;
        ensure_random(self.rounding, ins.x, ins.random)?;
        Ok(QuantizeResults {
            output: ins.x.with_dtype(self.to),
        })
    }
}

/// `tensor.quantize_dynamic`: computes each block's scale, and with `asymmetric` its offset, from
/// the block's values, then quantizes as `tensor.quantize` does.
///
/// - `x` is `f32`, `bf16` or `f16`. The blocks, and the shape of the scales and offsets, follow
///   the scale layout of "Quantization" in the dialect documentation.
/// - It quantizes `x` as `tensor.quantize` does with the scales and offsets as stored, in
///   `scale_dtype`. The scales and offsets ignore NaN elements.
/// - A symmetric scale comes from the block's largest magnitude `amax`, as `scale_rule` says.
/// - An asymmetric scale maps the block's range `[lo, hi]`, widened to hold 0, onto the range of
///   `to`: `scale = (hi - lo) / (max(to) - min(to))`. The integer zero point is
///   `zero_point = round(min(to) - lo / scale)`, rounded to nearest-even and clamped into the range
///   of `to`, so that 0 quantizes exactly. The offset is `offset = -zero_point * scale`. `max(to)`
///   and `min(to)` are the largest and smallest finite values of `to`. `scale_dtype` is a float,
///   and `scale_rule` is `amax`.
/// - Its results are the quantized tensor `q`, of dtype `to`, the scales, and with `asymmetric`
///   the offsets, both of dtype `scale_dtype`.
/// - `rounding` and `random` are those of `tensor.quantize`.
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.quantize_dynamic",
    operands(x: Tensor, random: Optional<Tensor>),
    results(q: Tensor, scale: Tensor, offset: Optional<Tensor>)
)]
pub struct QuantizeDynamic {
    pub block: SmallVec<[usize; 4]>,
    pub to: DType,
    pub scale_dtype: DType,
    #[attr(default)]
    pub scale_rule: ScaleRule,
    #[attr(default)]
    pub asymmetric: bool,
    #[attr(default)]
    pub rounding: Rounding,
}

impl OpDef for QuantizeDynamic {
    fn result_types(
        &self,
        ins: QuantizeDynamicOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<QuantizeDynamicResults<OwnedTypes>> {
        ensure!(is_real(ins.x.dtype), "cannot quantize {}", ins.x);
        ensure_dtype_class("to", self.to, ELEMENTS)?;
        ensure_dtype_class("scale_dtype", self.scale_dtype, SCALES)?;
        ensure!(
            self.scale_dtype != DType::E8M0 || self.scale_rule != ScaleRule::Amax,
            "e8m0 scales are powers of two: they need scale_rule pow2_floor or pow2_ceil"
        );
        if self.asymmetric {
            ensure_dtype_class(
                "an asymmetric scale_dtype",
                self.scale_dtype,
                &[Class::Float],
            )?;
            ensure!(
                self.scale_rule == ScaleRule::Amax,
                "an asymmetric scale maps the block's range, not a power of two"
            );
        }
        ensure_random(self.rounding, ins.x, ins.random)?;
        let dims = scale_dims(&ins.x.dims, &self.block)?;
        let scale = TensorType {
            dtype: self.scale_dtype,
            dims,
        };
        Ok(QuantizeDynamicResults {
            q: ins.x.with_dtype(self.to),
            offset: self.asymmetric.then(|| scale.clone()),
            scale,
        })
    }
}
