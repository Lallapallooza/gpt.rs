//! `tensor-to-tile`: every `tensor` op as `tile.block`s. Each file holds the patterns of the ops
//! of the `tensor` file of the same name.

mod contraction;
mod elementwise;
mod indexing;
mod layout;
mod quant;
mod random;
mod utils;
mod window;

use std::marker::PhantomData;

use anyhow::Result;

use self::contraction::*;
use self::elementwise::*;
use self::indexing::*;
use self::layout::*;
use self::quant::*;
use self::random::*;
use self::window::*;
use super::{apply_conversion, ConversionTarget, ConvertOp, Mode, OpForm};
use crate::dialects::tensor::{self, TensorOps};
use crate::dialects::Dialect;
use crate::passes::{pass, Pass};
use crate::rewrite::{PatternSet, Rewrites, Rewriting};

/// Converts the `tensor` ops that compute on elements to `tile.block`s that compute the same
/// values, bit for bit.
///
/// - Elementwise ops become a parallel block with one `arith` op. Contractions, reductions,
///   `argmax`, `top_k` and windows become blocks with reduction iterators, declared in the order
///   the tensor interpreter folds them. An op with an accumulator dtype casts into it in the body
///   and back to its result dtype in a block of its own.
/// - Movement ops (`reshape`, `transpose`, `broadcast_to`, `slice`, `take`, `gather`,
///   `dynamic_slice`, `extract_patches`) become copy blocks. `concat` selects among padded
///   reads. `iota` is `tile.index`.
/// - `dynamic_update_slice` and `scatter_*` write into an operand, and `segment_*` into a tensor
///   filled with the initial value of its fold. `rng_*` draws each element with `arith.rng_*`,
///   and the quantization ops are their compositions in `arith`.
/// - `tensor.cond`, `tensor.while`, `tensor.scan` and `tensor.constant` stay, with the ops of
///   their regions converted, and a `tensor.custom_call` is an error.
#[pass(name = "tensor-to-tile", stats(converted = "tensor ops converted"))]
pub struct TensorToTile;

impl TensorToTile {
    /// The target and the patterns of the conversion: `func`, `arith` and `tile` ops are legal,
    /// and each `tensor` op is as its [`ConvertOp`] decision says.
    pub fn conversion() -> (ConversionTarget, PatternSet) {
        let mut target =
            ConversionTarget::of_dialects(&[Dialect::Func, Dialect::Arith, Dialect::Tile]);
        let mut patterns = PatternSet::new();
        TensorOps::convert::<TensorToTile>(&mut target, &mut patterns);
        (target, patterns)
    }
}

impl Pass for TensorToTile {
    fn rewrites(&self) -> Rewrites {
        Rewrites {
            patterns: Self::conversion().1,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        let (target, patterns) = Self::conversion();
        let converted = apply_conversion(module, &target, &patterns, Mode::Full)?;
        TensorToTile::CONVERTED.add(converted);
        Ok(converted > 0)
    }
}

/// The decision of `tensor-to-tile` for each tensor op of its own.
macro_rules! decide {
    ($($op:ident => $form:expr,)*) => {$(
        impl ConvertOp<TensorToTile> for tensor::$op {
            fn form() -> OpForm {
                $form
            }
        }
    )*};
}

/// The decision of `tensor-to-tile` for the ops of a family: the family's pattern rewrites each.
macro_rules! decide_family {
    ($($pattern:ident: $($op:ident),*;)*) => {$($(
        impl ConvertOp<TensorToTile> for tensor::$op {
            fn form() -> OpForm {
                OpForm::rewrite($pattern::<tensor::$op>(PhantomData))
            }
        }
    )*)*};
}

decide! {
    Constant => OpForm::Legal,
    Yield => OpForm::Legal,
    Cond => OpForm::Legal,
    While => OpForm::Legal,
    Scan => OpForm::Legal,
    CustomCall => OpForm::Illegal("a custom call has no tile form; record core ops instead"),
    StopGradient => OpForm::rewrite(StopGradientToTile),
    Select => OpForm::rewrite(SelectToTile),
    Cast => OpForm::rewrite(CastToTile),
    DotGeneral => OpForm::rewrite(DotToTile),
    ArgMax => OpForm::rewrite(ArgMaxToTile),
    Reshape => OpForm::rewrite(ReshapeToTile),
    Transpose => OpForm::rewrite(TransposeToTile),
    BroadcastTo => OpForm::rewrite(BroadcastToTile),
    Slice => OpForm::rewrite(SliceToTile),
    Concat => OpForm::rewrite(ConcatToTile),
    Bitcast => OpForm::rewrite(BitcastToTile),
    Iota => OpForm::rewrite(IotaToTile),
    Take => OpForm::rewrite(TakeToTile),
    Gather => OpForm::rewrite(GatherToTile),
    DynamicSlice => OpForm::rewrite(DynamicSliceToTile),
    DynamicUpdateSlice => OpForm::rewrite(DynamicUpdateSliceToTile),
    ScatterReplace => OpForm::rewrite(ScatterReplaceToTile),
    ExtractPatches => OpForm::rewrite(ExtractPatchesToTile),
    Dequantize => OpForm::rewrite(DequantizeToTile),
    Quantize => OpForm::rewrite(QuantizeToTile),
    QuantizeStochastic => OpForm::rewrite(QuantizeStochasticToTile),
    QuantizeTable => OpForm::rewrite(QuantizeTableToTile),
    BlockScaleRange => OpForm::rewrite(BlockScaleRangeToTile),
}

decide_family! {
    UnaryToTile: Neg, Abs, Exp, Log, Tanh, Erf, Rsqrt, Reciprocal, Sin, Cos, Not;
    BinaryToTile: Add, Sub, Mul, Div, Maximum, Minimum, And, Or, Xor, ShiftLeft,
        ShiftRightLogical, ShiftRightArithmetic;
    CompareToTile: Less, LessEqual, Equal, GreaterEqual, Greater, NotEqual;
    ReduceToTile: ReduceSum, ReduceMax, ReduceMin, ReduceAnd, ReduceOr;
    SegmentToTile: SegmentSum, SegmentMax, SegmentMin, SegmentAnd, SegmentOr;
    ReduceWindowToTile: ReduceWindowSum, ReduceWindowMax, ReduceWindowMin, ReduceWindowAnd,
        ReduceWindowOr;
    ScatterToTile: ScatterAdd, ScatterMax, ScatterMin;
    RngToTile: RngUniform, RngNormal;
    SelectKToTile: TopK, BottomK;
    BlockScaleToTile: BlockScaleAmax, BlockScalePow2Floor, BlockScalePow2Ceil;
}
