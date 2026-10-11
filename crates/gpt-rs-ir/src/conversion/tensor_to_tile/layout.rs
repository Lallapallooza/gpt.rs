use super::utils::{copy, elementwise};
use crate::dialects::arith;
use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tensor::{Bitcast, BroadcastTo, Concat, Reshape, Slice, Transpose};
use crate::dialects::tile::{par_iters, push_iter, Block, CombineKind, Fold};
use crate::index::{Constraint, IndexExpr, IterKind, Read, Relation, Write};
use crate::op::Matched;
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, Rewriter};
use crate::types::{DType, Dim};

/// `tensor.reshape` is a copy block that reads the input at the row-major position of each
/// result element.
#[pattern]
pub(super) struct ReshapeToTile;

impl Pattern for ReshapeToTile {
    type Root = Reshape;

    fn rewrite(
        &self,
        reshape: Matched<'_, Reshape>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = reshape.input().tensor_type();
        let shape = &reshape.shape;
        let (iters, coords) = par_iters(shape);
        let source = IndexExpr::linearize(&coords, shape).delinearize(&input.dims);
        let inputs = vec![reshape.input().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&reshape, result);
        Ok(())
    }
}

/// `tensor.transpose` is a copy block with permuted read coordinates.
#[pattern]
pub(super) struct TransposeToTile;

impl Pattern for TransposeToTile {
    type Root = Transpose;

    fn rewrite(
        &self,
        transpose: Matched<'_, Transpose>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = transpose.input().tensor_type();
        let output = transpose.output().tensor_type();
        let (iters, coords) = par_iters(&output.dims);
        let mut source = vec![IndexExpr::constant(0); input.rank()];
        for (output_axis, &input_axis) in transpose.perm.iter().enumerate() {
            source[input_axis] = coords[output_axis].clone();
        }
        let inputs = vec![transpose.input().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&transpose, result);
        Ok(())
    }
}

/// `tensor.broadcast_to` is a copy block that reads coordinate 0 on each broadcast axis.
#[pattern]
pub(super) struct BroadcastToTile;

impl Pattern for BroadcastToTile {
    type Root = BroadcastTo;

    fn rewrite(
        &self,
        broadcast: Matched<'_, BroadcastTo>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = broadcast.input().tensor_type();
        let shape = &broadcast.shape;
        let (iters, coords) = par_iters(shape);
        let mut source = vec![IndexExpr::constant(0); input.rank()];
        for (output_axis, extent) in shape.iter().enumerate() {
            let Some(input_axis) = broadcast.source_axis(output_axis, input.rank()) else {
                continue;
            };
            let repeated = input.dims[input_axis].is_one() && !extent.is_one();
            if !repeated {
                source[input_axis] = coords[output_axis].clone();
            }
        }
        let inputs = vec![broadcast.input().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&broadcast, result);
        Ok(())
    }
}

/// `tensor.slice` is a copy block that reads `start + k * stride` on each axis.
#[pattern]
pub(super) struct SliceToTile;

impl Pattern for SliceToTile {
    type Root = Slice;

    fn rewrite(
        &self,
        slice: Matched<'_, Slice>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let output = slice.output().tensor_type();
        let (iters, coords) = par_iters(&output.dims);
        let mut source = Vec::new();
        for (axis, coord) in coords.iter().enumerate() {
            let step = coord.clone() * IndexExpr::constant(slice.stride(axis) as i64);
            source.push((IndexExpr::constant(slice.starts[axis] as i64) + step).simplify());
        }
        let inputs = vec![slice.input().id()];
        let result = copy(rewriter, inputs, iters, coords, Read::new(0, source));
        rewriter.replace_op(&slice, result);
        Ok(())
    }
}

/// `tensor.concat` is a block that reads every input shifted by its offset, padded outside it,
/// and selects the input whose range holds the point.
#[pattern]
pub(super) struct ConcatToTile;

impl Pattern for ConcatToTile {
    type Root = Concat;

    fn rewrite(
        &self,
        concat: Matched<'_, Concat>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let output = concat.output().tensor_type();
        let axis = axis_index(concat.axis, output.rank()).expect("a verified axis");
        let (iters, coords) = par_iters(&output.dims);
        let mut starts = Vec::new();
        let mut start = IndexExpr::constant(0);
        let mut reads = Vec::new();
        for (operand, input) in concat.inputs().iter().enumerate() {
            let input_type = input.tensor_type();
            let mut source = coords.clone();
            source[axis] = (coords[axis].clone() - start.clone()).simplify();
            // A pad every dtype holds: 0, or 1 for a dtype without a zero.
            let pad = if input_type.dtype.represents(0.0) {
                0.0
            } else {
                1.0
            };
            reads.push(Read {
                pad: Some(crate::attr::F64(pad)),
                ..Read::new(operand, source)
            });
            starts.push(start.clone());
            start = (start + IndexExpr::dim(&input_type.dims[axis])).simplify();
        }
        let position = coords[axis].clone();
        let block = Block::new(iters, reads, vec![Fold::plain(Write::new(coords))]);
        let inputs: Vec<_> = concat.inputs().iter().map(|input| input.id()).collect();
        let result = block.build(rewriter, inputs, |body, args| {
            let mut value = args[0];
            if starts.len() == 1 {
                return vec![value];
            }
            let at = body.index(position, DType::Si64);
            for (operand, start) in starts.iter().enumerate().skip(1) {
                let start = body.index(start.clone(), DType::Si64);
                let inside = body.create(arith::GreaterEqual, (at, start));
                value = body.create(arith::Select, (inside, args[operand], value));
            }
            vec![value]
        })[0];
        rewriter.replace_op(&concat, result);
        Ok(())
    }
}

/// `tensor.bitcast` of equal widths is a parallel block with `arith.bitcast`. Otherwise each
/// result element ors together, in a `ui64`, the shifted bits of the input elements that overlap
/// it in the row's bit stream, and a second block cuts them to the result's width.
#[pattern]
pub(super) struct BitcastToTile;

impl Pattern for BitcastToTile {
    type Root = Bitcast;

    fn rewrite(
        &self,
        bitcast: Matched<'_, Bitcast>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let input = bitcast.input().tensor_type();
        let output = bitcast.output().tensor_type();
        let to = bitcast.to;
        let (input_width, output_width) = (input.dtype.bits() as i64, to.bits() as i64);
        if input_width == output_width {
            let inputs = [bitcast.input().id()];
            let result = elementwise(rewriter, &inputs, output, |body, args| {
                body.bitcast(args[0], to)
            });
            rewriter.replace_op(&bitcast, result);
            return Ok(());
        }
        require!(
            let Ok(input_bits) = DType::unsigned(input_width as usize),
            "no integer has the width of {} to carry its bits",
            input.dtype
        );
        require!(
            let Ok(output_bits) = DType::unsigned(output_width as usize),
            "no integer has the width of {to} to carry its bits"
        );
        let rank = output.rank();
        let (mut iters, coords) = par_iters(&output.dims);
        let element = coords[rank - 1].clone();
        // Input elements per result element, at most: aligned when one width divides the other.
        let span = if output_width % input_width == 0 {
            output_width / input_width
        } else if input_width % output_width == 0 {
            1
        } else {
            (output_width - 1) / input_width + 2
        };
        let input_offset = push_iter(&mut iters, IterKind::Red, Dim::Static(span as usize));
        let first_input = IndexExpr::floordiv(
            IndexExpr::constant(output_width) * element.clone(),
            IndexExpr::constant(input_width),
        );
        let input_element = (first_input + input_offset).simplify();
        // The bit offset of the input element from the start of the result element.
        let offset = ((IndexExpr::constant(input_width) * input_element.clone())
            - (IndexExpr::constant(output_width) * element))
            .simplify();
        let mut constraints = vec![Constraint::cmp(
            offset.clone(),
            Relation::Less,
            IndexExpr::constant(output_width),
        )];
        if span > 1 {
            let row_length = IndexExpr::dim(&input.dims[rank - 1]);
            constraints.push(Constraint::cmp(
                input_element.clone(),
                Relation::Less,
                row_length,
            ));
        }
        let mut source = coords[..rank - 1].to_vec();
        source.push(input_element);
        let shift_left = IndexExpr::max(offset.clone(), IndexExpr::constant(0));
        let shift_right = IndexExpr::max(-offset, IndexExpr::constant(0));
        let any = Fold::new(CombineKind::Or, vec![Write::new(coords)]);
        let block = Block::new(iters, vec![Read::new(0, source)], vec![any]).restrict(constraints);
        let inputs = vec![bitcast.input().id()];
        let gathered = block.build(rewriter, inputs, |body, args| {
            let bits = body.bitcast(args[0], input_bits);
            let wide = body.cast(bits, DType::Ui64);
            let right = body.index(shift_right, DType::Ui64);
            let wide = body.create(arith::ShiftRightLogical, (wide, right));
            let left = body.index(shift_left, DType::Ui64);
            vec![body.create(arith::ShiftLeft, (wide, left))]
        });
        let result = elementwise(rewriter, &gathered, output, |body, args| {
            let mut bits = args[0];
            if output_width < 64 {
                let mask = body.int(DType::Ui64, (1i128 << output_width) - 1);
                bits = body.create(arith::And, (bits, mask));
            }
            let narrow = body.cast(bits, output_bits);
            body.bitcast(narrow, to)
        });
        rewriter.replace_op(&bitcast, result);
        Ok(())
    }
}
