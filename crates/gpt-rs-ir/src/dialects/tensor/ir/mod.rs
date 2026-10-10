//! The `tensor` dialect's ops: their attributes, groups and type rules.

pub(crate) mod class;
mod constant;
mod contraction;
mod control;
mod custom_call;
mod elementwise;
mod indexing;
mod layout;
mod quant;
mod random;
pub(crate) mod shape;
mod window;

pub use self::class::Class;
pub use self::constant::{Constant, ConstantOperands, ConstantResults};
pub use self::contraction::{
    ArgMax, ArgMaxOperands, ArgMaxResults, DotAxes, DotGeneral, DotGeneralOperands,
    DotGeneralResults, Reduce, ReduceKind, ReduceOperands, ReduceResults, Segment, SegmentOperands,
    SegmentResults, TopK, TopKOperands, TopKResults,
};
pub use self::control::{
    Cond, CondOperands, CondRegions, CondResults, Scan, ScanOperands, ScanRegions, ScanResults,
    While, WhileOperands, WhileRegions, WhileResults, Yield, YieldOperands, YieldResults,
};
pub use self::custom_call::{CustomAttr, CustomCall, CustomCallOperands, CustomCallResults};
pub use self::elementwise::{
    Binary, BinaryKind, BinaryOperands, BinaryResults, Cast, CastOperands, CastResults, Compare,
    CompareKind, CompareOperands, CompareResults, Select, SelectOperands, SelectResults,
    StopGradient, StopGradientOperands, StopGradientResults, Unary, UnaryKind, UnaryOperands,
    UnaryResults,
};
pub use self::indexing::{
    DynamicSlice, DynamicSliceOperands, DynamicSliceResults, DynamicUpdateSlice,
    DynamicUpdateSliceOperands, DynamicUpdateSliceResults, Gather, GatherOperands, GatherResults,
    Iota, IotaOperands, IotaResults, Scatter, ScatterKind, ScatterOperands, ScatterResults, Take,
    TakeOperands, TakeResults,
};
pub use self::layout::{
    Bitcast, BitcastOperands, BitcastResults, BroadcastTo, BroadcastToOperands, BroadcastToResults,
    Concat, ConcatOperands, ConcatResults, Reshape, ReshapeOperands, ReshapeResults, Slice,
    SliceOperands, SliceResults, Transpose, TransposeOperands, TransposeResults,
};
pub use self::quant::{
    Dequantize, DequantizeOperands, DequantizeResults, Quantize, QuantizeDynamic,
    QuantizeDynamicOperands, QuantizeDynamicResults, QuantizeOperands, QuantizeResults, Rounding,
    ScaleRule,
};
pub use self::random::{state_type, Distribution, Rng, RngOperands, RngResults};
pub use self::shape::axis_index;
pub use self::window::{
    ExtractPatches, ExtractPatchesOperands, ExtractPatchesResults, ReduceWindow,
    ReduceWindowOperands, ReduceWindowResults,
};

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "tensor", op = Tensor)]
pub enum TensorOp {
    Constant(Constant),
    Unary(Unary),
    Binary(Binary),
    Compare(Compare),
    Select(Select),
    Cast(Cast),
    DotGeneral(DotGeneral),
    Reduce(Reduce),
    Reshape(Reshape),
    Transpose(Transpose),
    BroadcastTo(BroadcastTo),
    Slice(Slice),
    Bitcast(Bitcast),
    Concat(Concat),
    Iota(Iota),
    Take(Take),
    DynamicSlice(DynamicSlice),
    DynamicUpdateSlice(DynamicUpdateSlice),
    ExtractPatches(ExtractPatches),
    ReduceWindow(ReduceWindow),
    CustomCall(CustomCall),
    StopGradient(StopGradient),
    ArgMax(ArgMax),
    Segment(Segment),
    TopK(TopK),
    Gather(Gather),
    Scatter(Scatter),
    Rng(Rng),
    Yield(Yield),
    Cond(Cond),
    While(While),
    Scan(Scan),
    Dequantize(Dequantize),
    Quantize(Quantize),
    QuantizeDynamic(QuantizeDynamic),
}
