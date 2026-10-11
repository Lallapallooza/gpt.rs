use std::marker::PhantomData;

use super::utils::elementwise;
use crate::dialects::arith;
use crate::dialects::tensor::{
    BlockScale, BlockScaleAmax, BlockScalePow2Ceil, BlockScalePow2Floor, BlockScaleRange,
    Dequantize, Quantize, QuantizeStochastic, QuantizeTable,
};
use crate::dialects::tile::{par_iters, push_iter, Block, Body, CombineKind, Fold, Inputs};
use crate::index::{IndexExpr, IterKind, Read, Write};
use crate::minifloat::floor_log2;
use crate::module::{Value, ValueRef};
use crate::op::Matched;
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};
use crate::types::{DType, Dim, Elements};

/// `tensor.dequantize` is a parallel block that multiplies each value by its block's scale and
/// adds the offset, in f32.
#[pattern]
pub(super) struct DequantizeToTile;

impl Pattern for DequantizeToTile {
    type Root = Dequantize;

    fn rewrite(
        &self,
        dequantize: Matched<'_, Dequantize>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let codes = dequantize.q();
        let block = &dequantize.block;
        let (iters, coords) = par_iters(&codes.tensor_type().dims);
        let mut inputs = Inputs::new(Vec::new());
        let code = inputs.read(codes.id(), coords.clone());
        let scale = read_block(&mut inputs, dequantize.scale().id(), &coords, block);
        let offset = dequantize
            .offset()
            .map(|offset| read_block(&mut inputs, offset.id(), &coords, block));
        let to = dequantize.to;
        let block = Block::new(iters, inputs.reads, vec![Fold::plain(Write::new(coords))]);
        let result = block.build(rewriter, inputs.values, |body, args| {
            let value = body.cast(args[code], DType::F32);
            let scale = body.cast(args[scale], DType::F32);
            let product = body.create(arith::Mul, (value, scale));
            let real = match offset {
                Some(offset) => {
                    let offset = body.cast(args[offset], DType::F32);
                    body.create(arith::Add, (product, offset))
                }
                None => product,
            };
            vec![body.cast(real, to)]
        })[0];
        rewriter.replace_op(&dequantize, result);
        Ok(())
    }
}

/// `tensor.quantize` is a parallel block that rounds `(x - offset) / scale` to the nearest value
/// of the quantized dtype in `arith`.
#[pattern]
pub(super) struct QuantizeToTile;

impl Pattern for QuantizeToTile {
    type Root = Quantize;

    fn rewrite(
        &self,
        quantize: Matched<'_, Quantize>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let to = quantize.to;
        require!(let Ok(elements) = to.elements(), "{to} is not a quantized dtype");
        let operands = QuantizeOperands {
            input: quantize.x(),
            scale: quantize.scale().id(),
            offset: quantize.offset().map(ValueRef::id),
        };
        let result = quantize_elements(rewriter, &operands, &quantize.block, None, |body, y, _| {
            round_nearest(body, y, to, elements)
        });
        rewriter.replace_op(&quantize, result);
        Ok(())
    }
}

/// `tensor.quantize_stochastic` is the block of `tensor.quantize` that rounds up when the
/// element's draw is below the fraction of the way to the next value, compared in f64.
#[pattern]
pub(super) struct QuantizeStochasticToTile;

impl Pattern for QuantizeStochasticToTile {
    type Root = QuantizeStochastic;

    fn rewrite(
        &self,
        quantize: Matched<'_, QuantizeStochastic>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let to = quantize.to;
        require!(let Ok(elements) = to.elements(), "{to} is not a quantized dtype");
        let operands = QuantizeOperands {
            input: quantize.x(),
            scale: quantize.scale().id(),
            offset: quantize.offset().map(ValueRef::id),
        };
        let random = Some(quantize.random().id());
        let result = quantize_elements(
            rewriter,
            &operands,
            &quantize.block,
            random,
            |body, y, draw| {
                let draw = draw.expect("a stochastic quantization reads a draw for each element");
                round_stochastic(body, y, to, elements, draw)
            },
        );
        rewriter.replace_op(&quantize, result);
        Ok(())
    }
}

/// `tensor.quantize_table` is a block that folds the table with `argmin` over the distances of its
/// values to `(x - offset) / scale`, then a block that stores each index as a code.
#[pattern]
pub(super) struct QuantizeTableToTile;

impl Pattern for QuantizeTableToTile {
    type Root = QuantizeTable;

    fn rewrite(
        &self,
        quantize: Matched<'_, QuantizeTable>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let operands = QuantizeOperands {
            input: quantize.x(),
            scale: quantize.scale().id(),
            offset: quantize.offset().map(ValueRef::id),
        };
        let input_type = operands.input.tensor_type();
        let (mut iters, coords) = par_iters(&input_type.dims);
        let mut inputs = Inputs::new(Vec::new());
        let element = inputs.read(operands.input.id(), coords.clone());
        let scale = read_block(&mut inputs, operands.scale, &coords, &quantize.block);
        let offset = operands
            .offset
            .map(|offset| read_block(&mut inputs, offset, &coords, &quantize.block));
        // The index of the nearest table value, the lowest on a tie. When the first distance is
        // NaN, as for a NaN element, nothing is nearer, so the index is 0.
        let table = quantize.table();
        let entries = table.tensor_type().dims[0].clone();
        let entry_iter = push_iter(&mut iters, IterKind::Red, entries);
        let entry = inputs.read(table.id(), [entry_iter.clone()]);
        let first_entry = inputs.read(table.id(), [IndexExpr::constant(0)]);
        let write = Write::new(coords);
        let nearest_entry = Fold::new(CombineKind::Argmin, vec![write.clone(), write]);
        let block = Block::new(iters, inputs.reads, vec![nearest_entry]);
        let nearest = block.build(rewriter, inputs.values, |body, args| {
            let quotient = divide_by_scale(body, args, element, scale, offset);
            let mut distance = |entry: Value| {
                let entry = body.cast(entry, DType::F32);
                let difference = body.create(arith::Sub, (entry, quotient));
                body.create(arith::Abs, difference)
            };
            let entry_distance = distance(args[entry]);
            let first_distance = distance(args[first_entry]);
            let nan = body.is_nan(first_distance);
            let entry_index = body.index(entry_iter, DType::Si64);
            let zero = body.int(DType::Si64, 0);
            let is_first = body.create(arith::Equal, (entry_index, zero));
            let below_all = body.float(DType::F32, f64::NEG_INFINITY);
            let above_all = body.float(DType::F32, f64::INFINITY);
            let forced = body.create(arith::Select, (is_first, below_all, above_all));
            let key = body.create(arith::Select, (nan, forced, entry_distance));
            vec![key, entry_index]
        });
        let to = quantize.to;
        let codes = input_type.with_dtype(to);
        let result = elementwise(rewriter, &nearest[1..], &codes, |body, args| {
            wrap_int(body, args[0], to)
        });
        rewriter.replace_op(&quantize, result);
        Ok(())
    }
}

/// The tile form of the [`ScaleFormula`](crate::dialects::tensor::ScaleFormula) of a
/// `tensor.block_scale_*` op.
pub(super) trait ScaleToTile {
    /// The scale, an f64, of a block whose largest magnitude is `amax`, an f64, for a quantized
    /// dtype whose largest finite value is `largest`.
    fn scale(body: &mut Body<'_, '_>, amax: Value, largest: f64) -> Value;
}

impl ScaleToTile for BlockScaleAmax {
    fn scale(body: &mut Body<'_, '_>, amax: Value, largest: f64) -> Value {
        let divisor = body.float(DType::F64, largest);
        body.create(arith::Div, (amax, divisor))
    }
}

impl ScaleToTile for BlockScalePow2Floor {
    fn scale(body: &mut Body<'_, '_>, amax: Value, largest: f64) -> Value {
        let amax_exponent = body.exponent_f64(amax);
        let largest_exponent = body.int(DType::Si64, i128::from(floor_log2(largest)));
        let exponent = body.create(arith::Sub, (amax_exponent, largest_exponent));
        power_of_two(body, amax, exponent)
    }
}

impl ScaleToTile for BlockScalePow2Ceil {
    fn scale(body: &mut Body<'_, '_>, amax: Value, largest: f64) -> Value {
        let divisor = body.float(DType::F64, largest);
        let ratio = body.create(arith::Div, (amax, divisor));
        let floor = body.exponent_f64(ratio);
        let power = body.pow2_f64(floor);
        let exact = body.create(arith::Equal, (ratio, power));
        let one = body.int(DType::Si64, 1);
        let ceil = body.create(arith::Add, (floor, one));
        let exponent = body.create(arith::Select, (exact, floor, ceil));
        power_of_two(body, amax, exponent)
    }
}

/// `2^exponent` as an f64, the si64 `exponent` clamped to `[-127, 127]`, or `2^-127` where `amax`
/// is 0.
fn power_of_two(body: &mut Body<'_, '_>, amax: Value, exponent: Value) -> Value {
    let lowest = body.int(DType::Si64, -127);
    let highest = body.int(DType::Si64, 127);
    let raised = body.create(arith::Maximum, (exponent, lowest));
    let clamped = body.create(arith::Minimum, (raised, highest));
    let zero = body.float(DType::F64, 0.0);
    let empty = body.create(arith::Equal, (amax, zero));
    let chosen = body.create(arith::Select, (empty, lowest, clamped));
    body.pow2_f64(chosen)
}

/// A `tensor.block_scale_*` op is a block that folds each block's range, then a block that computes
/// each block's scale from its largest magnitude and stores it.
#[pattern]
pub(super) struct BlockScaleToTile<T>(pub(super) PhantomData<T>);

impl<T: BlockScale + ScaleToTile> Pattern for BlockScaleToTile<T> {
    type Root = T;

    fn rewrite(&self, scaling: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let to = scaling.to;
        require!(let Ok(elements) = to.elements(), "{to} is not a quantized dtype");
        let scale_dtype = scaling.scale_dtype;
        require!(
            scale_dtype != DType::E8M0 || T::POWER_OF_TWO,
            "e8m0 scales need a power-of-two op"
        );
        let (scale_dims, range) = block_range(rewriter, scaling.x(), &scaling.block)?;
        let largest = f64::from(elements.largest());
        let (iters, coords) = par_iters(&scale_dims);
        let reads = vec![Read::new(0, coords.clone()), Read::new(1, coords.clone())];
        let block = Block::new(iters, reads, vec![Fold::plain(Write::new(coords))]);
        let scale = block.build(rewriter, range, |body, args| {
            let amax = block_amax(body, args[0], args[1]);
            let real = <T as ScaleToTile>::scale(body, amax, largest);
            vec![store_scale(body, real, scale_dtype)]
        })[0];
        rewriter.replace_op(&scaling, scale);
        Ok(())
    }
}

/// `tensor.block_scale_range` is a block that folds each block's range, then a block that computes
/// each block's scale and offset from it.
#[pattern]
pub(super) struct BlockScaleRangeToTile;

impl Pattern for BlockScaleRangeToTile {
    type Root = BlockScaleRange;

    fn rewrite(
        &self,
        scaling: Matched<'_, BlockScaleRange>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let to = scaling.to;
        require!(let Ok(elements) = to.elements(), "{to} is not a quantized dtype");
        let scale_dtype = scaling.scale_dtype;
        let (scale_dims, range) = block_range(rewriter, scaling.x(), &scaling.block)?;
        let (iters, coords) = par_iters(&scale_dims);
        let reads = vec![Read::new(0, coords.clone()), Read::new(1, coords.clone())];
        let folds = vec![
            Fold::plain(Write::new(coords.clone())),
            Fold::plain(Write::new(coords)),
        ];
        let block = Block::new(iters, reads, folds);
        let results = block.build(rewriter, range, |body, args| {
            asymmetric_scale(body, args[0], args[1], scale_dtype, elements)
        });
        rewriter.replace_op(&scaling, results);
        Ok(())
    }
}

/// A block that folds the smallest and the largest non-NaN element of each scale block of `input`
/// with `block`, and the extents of the scales. A tail block reads 0 past the end, which the range
/// holds anyway.
fn block_range(
    rewriter: &mut Rewriter<'_>,
    input: ValueRef<'_>,
    block: &[usize],
) -> Result<(Vec<Dim>, Vec<Value>), NoMatch> {
    let mut scale_dims = Vec::new();
    for (extent, &size) in input.tensor_type().dims.iter().zip(block) {
        match extent {
            Dim::Static(extent) => scale_dims.push(Dim::Static(extent.div_ceil(size))),
            Dim::Dynamic(_) => {
                require!(size == 1, "a block of {size} needs a static extent");
                scale_dims.push(extent.clone());
            }
        }
    }
    let (mut iters, block_coords) = par_iters(&scale_dims);
    let mut source = Vec::new();
    for (axis, &size) in block.iter().enumerate() {
        let offset = push_iter(&mut iters, IterKind::Red, Dim::Static(size));
        let start = block_coords[axis].clone() * IndexExpr::constant(size as i64);
        source.push((start + offset).simplify());
    }
    let read = Read {
        pad: Some(crate::attr::F64(0.0)),
        ..Read::new(0, source)
    };
    let folds = vec![
        Fold::new(CombineKind::Min, vec![Write::new(block_coords.clone())]),
        Fold::new(CombineKind::Max, vec![Write::new(block_coords)]),
    ];
    let range_block = Block::new(iters, vec![read], folds);
    let range = range_block.build(rewriter, vec![input.id()], |body, args| {
        let element = body.cast(args[0], DType::F32);
        let nan = body.is_nan(element);
        let zero = body.float(DType::F32, 0.0);
        let defined = body.create(arith::Select, (nan, zero, element));
        vec![defined, defined]
    });
    Ok((scale_dims, range))
}

/// The bounds of a block's range, `[range_min, range_max]` widened to hold 0.
fn widened(body: &mut Body<'_, '_>, range_min: Value, range_max: Value) -> (Value, Value) {
    let zero = body.float(DType::F32, 0.0);
    let low = body.create(arith::Minimum, (zero, range_min));
    let high = body.create(arith::Maximum, (zero, range_max));
    (low, high)
}

/// The largest magnitude of a block whose elements range over `[range_min, range_max]`, an f64.
fn block_amax(body: &mut Body<'_, '_>, range_min: Value, range_max: Value) -> Value {
    let (low, high) = widened(body, range_min, range_max);
    let negated_low = body.create(arith::Neg, low);
    let amax = body.create(arith::Maximum, (high, negated_low));
    body.cast(amax, DType::F64)
}

/// The f64 scale `real` stored as `scale_dtype`: an `e8m0` scale, a power of two, as the code of
/// its exponent.
fn store_scale(body: &mut Body<'_, '_>, real: Value, scale_dtype: DType) -> Value {
    if scale_dtype == DType::E8M0 {
        let exponent = body.exponent_f64(real);
        let bias = body.int(DType::Si64, 127);
        let biased = body.create(arith::Add, (exponent, bias));
        let code = body.cast(biased, DType::Ui8);
        return body.bitcast(code, DType::E8M0);
    }
    let single = body.cast(real, DType::F32);
    body.cast(single, scale_dtype)
}

/// The asymmetric scale and offset of a block whose elements range over
/// `[range_min, range_max]`.
fn asymmetric_scale(
    body: &mut Body<'_, '_>,
    range_min: Value,
    range_max: Value,
    scale_dtype: DType,
    elements: Elements,
) -> Vec<Value> {
    let (largest, smallest) = (elements.largest(), elements.smallest());
    let (low, high) = widened(body, range_min, range_max);
    let code_span = body.float(DType::F32, f64::from(largest - smallest));
    let width = body.create(arith::Sub, (high, low));
    let real = body.create(arith::Div, (width, code_span));
    let scale = body.cast(real, scale_dtype);
    // The zero point, the code of 0, as stored with the scale.
    let stored_scale = body.cast(scale, DType::F32);
    let quotient = body.create(arith::Div, (low, stored_scale));
    let smallest_code = body.float(DType::F32, f64::from(smallest));
    let largest_code = body.float(DType::F32, f64::from(largest));
    let shifted = body.create(arith::Sub, (smallest_code, quotient));
    let rounded = body.round_even(shifted);
    let raised = body.create(arith::Maximum, (rounded, smallest_code));
    let clamped = body.create(arith::Minimum, (raised, largest_code));
    let zero = body.float(DType::F32, 0.0);
    let is_zero = body.create(arith::Equal, (stored_scale, zero));
    let zero_point = body.create(arith::Select, (is_zero, zero, clamped));
    let negated = body.create(arith::Neg, zero_point);
    let offset = body.create(arith::Mul, (negated, stored_scale));
    vec![scale, body.cast(offset, scale_dtype)]
}

/// Reads `value` at the block of `coords`, coordinate `c` divided by its block extent, and returns
/// the index of the read.
fn read_block(inputs: &mut Inputs, value: Value, coords: &[IndexExpr], block: &[usize]) -> usize {
    let mut at = Vec::with_capacity(coords.len());
    for (coord, &extent) in coords.iter().zip(block) {
        at.push(IndexExpr::floordiv(coord.clone(), IndexExpr::constant(extent as i64)).simplify());
    }
    inputs.read(value, at)
}

/// The operands of a quantization: the input, and its scales and offsets as stored.
struct QuantizeOperands<'m> {
    input: ValueRef<'m>,
    scale: Value,
    offset: Option<Value>,
}

/// `(element - offset) / scale` in f32, or 0 where the scale is 0, from the reads of a block.
fn divide_by_scale(
    body: &mut Body<'_, '_>,
    args: &[Value],
    element: usize,
    scale: usize,
    offset: Option<usize>,
) -> Value {
    let value = body.cast(args[element], DType::F32);
    let numerator = match offset {
        Some(offset) => {
            let offset = body.cast(args[offset], DType::F32);
            body.create(arith::Sub, (value, offset))
        }
        None => value,
    };
    let divisor = body.cast(args[scale], DType::F32);
    let zero = body.float(DType::F32, 0.0);
    let is_zero = body.create(arith::Equal, (divisor, zero));
    let quotient = body.create(arith::Div, (numerator, divisor));
    body.create(arith::Select, (is_zero, zero, quotient))
}

/// A parallel block over the input of `operands` that rounds each `(x - offset) / scale` as
/// `round` does, given the element's draw from `random` when there is one.
fn quantize_elements(
    rewriter: &mut Rewriter<'_>,
    operands: &QuantizeOperands<'_>,
    block: &[usize],
    random: Option<Value>,
    round: impl FnOnce(&mut Body<'_, '_>, Value, Option<Value>) -> Value,
) -> Value {
    let input_type = operands.input.tensor_type();
    let (iters, coords) = par_iters(&input_type.dims);
    let mut inputs = Inputs::new(Vec::new());
    let element = inputs.read(operands.input.id(), coords.clone());
    let scale = read_block(&mut inputs, operands.scale, &coords, block);
    let offset = operands
        .offset
        .map(|offset| read_block(&mut inputs, offset, &coords, block));
    let draw = random.map(|random| inputs.read(random, coords.clone()));
    let block = Block::new(iters, inputs.reads, vec![Fold::plain(Write::new(coords))]);
    block.build(rewriter, inputs.values, |body, args| {
        let quotient = divide_by_scale(body, args, element, scale, offset);
        vec![round(body, quotient, draw.map(|draw| args[draw]))]
    })[0]
}

/// `quotient`, an f32, rounded to the nearest value of `to`, the even one on a tie.
fn round_nearest(body: &mut Body<'_, '_>, quotient: Value, to: DType, elements: Elements) -> Value {
    match elements {
        Elements::Float(_) => body.cast(quotient, to),
        Elements::Int(_) => {
            let rounded = body.round_even(quotient);
            body.cast(rounded, to)
        }
    }
}

/// `quotient`, an f32, rounded up from `lower` to `lower + spacing` when the ui32 `draw` is below
/// the fraction of the way, in f64, then stored as `to`.
fn round_stochastic(
    body: &mut Body<'_, '_>,
    quotient: Value,
    to: DType,
    elements: Elements,
    draw: Value,
) -> Value {
    let wide = body.cast(quotient, DType::F64);
    let magnitude = body.create(arith::Abs, wide);
    let spacing = match elements {
        Elements::Int(_) => body.float(DType::F64, 1.0),
        Elements::Float(format) => {
            let min_normal = body.int(DType::Si64, i128::from(1 - format.bias));
            let exponent = body.exponent_f64(magnitude);
            let normal_exponent = body.create(arith::Maximum, (exponent, min_normal));
            let zero = body.float(DType::F64, 0.0);
            let positive = body.create(arith::Greater, (magnitude, zero));
            let binade = body.create(arith::Select, (positive, normal_exponent, min_normal));
            let mantissa_bits = body.int(DType::Si64, i128::from(format.man_bits));
            let spacing_exponent = body.create(arith::Sub, (binade, mantissa_bits));
            body.pow2_f64(spacing_exponent)
        }
    };
    let ratio = body.create(arith::Div, (magnitude, spacing));
    let steps = body.floor_nonnegative(ratio);
    let lower = body.create(arith::Mul, (steps, spacing));
    let rest = body.create(arith::Sub, (magnitude, lower));
    let fraction = body.create(arith::Div, (rest, spacing));
    let draw_range = body.float(DType::F64, 2f64.powi(32));
    let scaled = body.create(arith::Mul, (fraction, draw_range));
    let threshold = body.floor_nonnegative(scaled);
    let wide_draw = body.cast(draw, DType::F64);
    let up = body.create(arith::Less, (wide_draw, threshold));
    let zero = body.float(DType::F64, 0.0);
    let step = body.create(arith::Select, (up, spacing, zero));
    let upward = body.create(arith::Add, (lower, step));
    let negative = body.sign_bit(wide);
    let negated = body.create(arith::Neg, upward);
    let signed = body.create(arith::Select, (negative, negated, upward));
    let rounded = match elements {
        Elements::Int(int) => {
            let lowest = body.float(DType::F64, int.min() as f64);
            let highest = body.float(DType::F64, int.max() as f64);
            let raised = body.create(arith::Maximum, (signed, lowest));
            let saturated = body.create(arith::Minimum, (raised, highest));
            body.cast(saturated, to)
        }
        Elements::Float(_) => {
            let single = body.cast(signed, DType::F32);
            body.cast(single, to)
        }
    };
    // Infinities saturate and NaN is NaN, or 0 for an integer: what the cast gives.
    let infinity = body.float(DType::F64, f64::INFINITY);
    let finite = body.create(arith::Less, (magnitude, infinity));
    let direct = body.cast(quotient, to);
    body.create(arith::Select, (finite, rounded, direct))
}

/// The si64 `index` as `to`, an integer dtype: its low bits, as a stored code.
fn wrap_int(body: &mut Body<'_, '_>, index: Value, to: DType) -> Value {
    let width = to.bits();
    if width == 64 {
        return body.bitcast(index, to);
    }
    let mask = body.int(DType::Si64, (1i128 << width) - 1);
    let low_bits = body.create(arith::And, (index, mask));
    let unsigned = DType::unsigned(width).expect("an integer dtype's width");
    let unsigned_bits = body.cast(low_bits, unsigned);
    body.bitcast(unsigned_bits, to)
}
