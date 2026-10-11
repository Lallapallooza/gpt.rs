//! The canonical patterns of the `tensor` ops: rewrites that keep every bit, hold on every backend
//! and never add work.

use std::marker::PhantomData;

use crate::dialects::tensor::utils::shape::axis_index;
use crate::dialects::tensor::{
    Accumulates, ArgMax, BottomK, BroadcastTo, Compare, Concat, DotAxes, DotGeneral, Equal, Gather,
    Greater, GreaterEqual, Iota, Less, LessEqual, NotEqual, Reduce, ReduceAnd, ReduceAttrs,
    ReduceMax, ReduceMin, ReduceOr, ReduceSum, ReduceWindowAnd, ReduceWindowMax, ReduceWindowMin,
    ReduceWindowOr, ReduceWindowSum, Reshape, ScatterAdd, ScatterMax, ScatterMin, ScatterReplace,
    Side, Slice, TopK, Transpose,
};
use crate::host::HostScalar;
use crate::module::{Value, ValueRef};
use crate::op::{InOp, Matched, OpKind};
use crate::require;
use crate::rewrite::{pattern, Build, Has, NoMatch, Pattern, Rewriter, WithTrait};
use crate::scalar::{Combiner, CompareMath};
use crate::traits::{Associative, Commutative, Elementwise, Idempotent, Traits};
use crate::types::{DType, Dim, Ranked, TensorType};

/// A movement op where two in a row make one op of the same kind.
trait Compose: InOp {
    /// The op that does `first`, then `self`.
    fn after(&self, first: &Self) -> Self;
}

impl Compose for Reshape {
    /// The last shape: both keep the row-major order.
    fn after(&self, _first: &Self) -> Self {
        self.clone()
    }
}

impl Compose for BroadcastTo {
    /// The last shape: every axis that broadcasts to the first shape broadcasts to the last.
    fn after(&self, _first: &Self) -> Self {
        self.clone()
    }
}

impl Compose for Slice {
    /// Element `k` of the second slice is element `s2.starts + k * s2.strides` of the first,
    /// which is element `s1.starts + (s2.starts + k * s2.strides) * s1.strides` of the input.
    fn after(&self, first: &Self) -> Self {
        let rank = self.sizes.len();
        let mut starts = Ranked::new();
        let mut strides = Ranked::new();
        for axis in 0..rank {
            starts.push(first.starts[axis] + self.starts[axis] * first.stride(axis));
            strides.push(first.stride(axis) * self.stride(axis));
        }
        if strides.iter().all(|&stride| stride == 1) {
            strides.clear();
        }
        Slice {
            starts,
            sizes: self.sizes,
            strides,
        }
    }
}

impl Compose for Transpose {
    /// `q[i] = p1[p2[i]]`.
    fn after(&self, first: &Self) -> Self {
        Transpose {
            perm: self.perm.after(&first.perm),
        }
    }
}

/// `t(t(x))` is one `t` of `x` for a movement op `t`: a reshape or `broadcast_to` to the last
/// shape, the slice whose starts are `s1.starts + s2.starts * s1.strides` and whose strides are
/// `s1.strides * s2.strides`, and the transpose `q[i] = p1[p2[i]]`.
#[pattern(canonical(Reshape, BroadcastTo, Slice, Transpose))]
struct CollapseChain<T>(PhantomData<T>);

impl<T: Compose> Pattern for CollapseChain<T> {
    type Root = T;

    fn rewrite(&self, second: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let first = second
            .operand(0)
            .defining_op::<T>()
            .ok_or("the input is another op")?;
        let combined = second.after(first.attrs());
        let result = rewriter.create_op(combined.into(), first.operand(0));
        rewriter.replace_op(&second, result);
        Ok(())
    }
}

/// Steps of 1 on every axis are written as no strides.
#[pattern(canonical)]
struct UnitStrides;

impl Pattern for UnitStrides {
    type Root = Slice;

    fn rewrite(
        &self,
        slice: Matched<'_, Slice>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        require!(!slice.strides.is_empty(), "the slice has no strides");
        require!(
            slice.strides.iter().all(|&stride| stride == 1),
            "a stride is not 1"
        );
        rewriter.set_op(&slice, Slice::new(slice.starts, slice.sizes));
        Ok(())
    }
}

/// An op with one `axis` attribute, an axis of its first operand, or of its result when it has
/// no operands.
trait OneAxis: InOp {
    fn axis_mut(&mut self) -> &mut i64;
}

/// A negative `axis` is written as the axis it counts back to.
#[pattern(canonical(
    ArgMax,
    TopK,
    BottomK,
    Concat,
    Iota,
    Gather,
    ScatterAdd,
    ScatterMax,
    ScatterMin,
    ScatterReplace
))]
struct NonNegativeAxis<T>(PhantomData<T>);

impl<T: OneAxis> Pattern for NonNegativeAxis<T> {
    type Root = T;

    fn rewrite(&self, root: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let mut op = root.attrs().clone();
        let axis = *op.axis_mut();
        require!(axis < 0, "the axis is not negative");
        let has_operands = !root.module().operands(root.inst()).is_empty();
        let counted = if has_operands {
            root.operand(0)
        } else {
            root.result(0)
        };
        *op.axis_mut() =
            axis_index(axis, counted.tensor_type().rank()).expect("a verified axis") as i64;
        rewriter.set_op(&root, op);
        Ok(())
    }
}

macro_rules! one_axis {
    ($($op:ty),*) => {$(
        impl OneAxis for $op {
            fn axis_mut(&mut self) -> &mut i64 {
                &mut self.axis
            }
        }
    )*};
}

one_axis!(
    ArgMax,
    TopK,
    BottomK,
    Concat,
    Iota,
    Gather,
    ScatterAdd,
    ScatterMax,
    ScatterMin,
    ScatterReplace
);

/// Negative batch and contract axes are written as the axes they count back to.
#[pattern(canonical)]
struct NonNegativeDotAxes;

impl Pattern for NonNegativeDotAxes {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let lists = [
            &dot.batch_lhs,
            &dot.batch_rhs,
            &dot.contract_lhs,
            &dot.contract_rhs,
        ];
        let negative = lists.iter().any(|axes| axes.iter().any(|&axis| axis < 0));
        require!(negative, "no axis is negative");
        let axes = dot.axes();
        let signed = |axes: Ranked<usize>| axes.into_iter().map(|axis| axis as i64).collect();
        let normalized = DotGeneral {
            batch_lhs: signed(axes.batch_lhs),
            batch_rhs: signed(axes.batch_rhs),
            contract_lhs: signed(axes.contract_lhs),
            contract_rhs: signed(axes.contract_rhs),
            ..dot.attrs().clone()
        };
        rewriter.set_op(&dot, normalized);
        Ok(())
    }
}

/// An `out` equal to the input dtype, or an `accum` equal to its default accumulator, is left out.
#[pattern(canonical(
    DotGeneral,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceAnd,
    ReduceOr,
    ReduceWindowSum,
    ReduceWindowMax,
    ReduceWindowMin,
    ReduceWindowAnd,
    ReduceWindowOr
))]
struct DropDefaultDtypes<T>(PhantomData<T>);

impl<T: InOp + Accumulates> Pattern for DropDefaultDtypes<T> {
    type Root = T;

    fn rewrite(&self, root: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let input_dtype = root.operand(0).tensor_type().dtype;
        let mut op = root.attrs().clone();
        let (accum, out) = op.dtypes_mut();
        let default_out = *out == Some(input_dtype);
        let default_accum = *accum == Some(input_dtype.default_accumulator());
        require!(default_out || default_accum, "no dtype is the default");
        if default_out {
            *out = None;
        }
        if default_accum {
            *accum = None;
        }
        rewriter.set_op(&root, op);
        Ok(())
    }
}

/// The reduced axes are written non-negative and in increasing order.
#[pattern(canonical(ReduceSum, ReduceMax, ReduceMin, ReduceAnd, ReduceOr))]
struct SortedReduceAxes<T>(PhantomData<T>);

impl<T: Reduce> Pattern for SortedReduceAxes<T> {
    type Root = T;

    fn rewrite(&self, reduce: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let sorted = reduce.reduced_axes();
        let axes: Ranked<i64> = sorted.into_iter().map(|axis| axis as i64).collect();
        require!(axes != reduce.axes, "the axes are sorted already");
        let attrs = ReduceAttrs {
            axes,
            ..ReduceAttrs::clone(reduce.attrs())
        };
        rewriter.set_op(&reduce, T::from(attrs));
        Ok(())
    }
}

/// `f(broadcast_to(a), broadcast_to(b))` is `broadcast_to(f(a, b))` when `a` and `b` have the
/// same shape, so `f` computes on fewer elements. The pattern splats a splat constant operand to
/// that shape.
#[pattern(canonical)]
struct HoistBroadcast;

/// What an operand of the elementwise op becomes in the smaller shape.
enum Narrowed<'m> {
    /// The input of the `broadcast_to` that defines the operand.
    Input(ValueRef<'m>),
    /// The element of a splat constant operand.
    Splat(HostScalar),
}

fn narrowed(operand: ValueRef<'_>) -> Option<Narrowed<'_>> {
    if let Some(broadcast) = operand.defining_op::<BroadcastTo>() {
        return Some(Narrowed::Input(broadcast.input()));
    }
    Some(Narrowed::Splat(operand.splat()?))
}

/// The shape of the broadcast inputs, when there is at least one and they all have it.
fn input_dims<'m>(operands: &[Narrowed<'m>]) -> Option<&'m Ranked<Dim>> {
    let mut shapes = operands.iter().filter_map(|operand| match operand {
        Narrowed::Input(input) => Some(&input.tensor_type().dims),
        Narrowed::Splat(_) => None,
    });
    let first = shapes.next()?;
    shapes.all(|dims| dims == first).then_some(first)
}

impl Pattern for HoistBroadcast {
    type Root = Has<Elementwise>;

    fn rewrite(
        &self,
        root: WithTrait<'_, Elementwise>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let narrow: Option<Vec<_>> = root
            .operands
            .iter()
            .map(|&operand| narrowed(operand))
            .collect();
        require!(let Some(narrow) = narrow, "an operand is neither a broadcast nor a splat");
        require!(let Some(dims) = input_dims(&narrow), "the broadcast inputs differ in shape");
        let wide = root.results[0].tensor_type();
        require!(wide.dims != *dims, "the broadcasts add nothing");
        let mut narrow_operands = Vec::with_capacity(narrow.len());
        for narrowed in narrow {
            let value = match narrowed {
                Narrowed::Input(input) => input.id(),
                Narrowed::Splat(element) => rewriter.splat(element, dims),
            };
            narrow_operands.push(value);
        }
        let narrow_result = rewriter.create_op(root.op.clone(), narrow_operands)[0];
        let broadcast = BroadcastTo {
            shape: wide.dims.clone(),
        };
        let result = rewriter.create(broadcast, narrow_result);
        rewriter.replace_op(&root, result);
        Ok(())
    }
}

/// Whether the elements that `C` folds may combine in any order and grouping, in `dtype`.
fn combines_in_any_order<C: Combiner>(dtype: DType) -> bool {
    let traits = Traits::new(C::INFO.traits);
    let associative = traits
        .get::<Associative>()
        .is_some_and(|fact| fact.covers(dtype));
    associative && traits.has::<Commutative>()
}

/// The input type of a reduction that accumulates and returns in its input dtype.
fn plain_input<'m, T: Reduce>(reduce: &Matched<'m, T>) -> Option<&'m TensorType> {
    let input = reduce.input().tensor_type();
    (reduce.accum_dtype(input.dtype) == input.dtype && reduce.out.is_none()).then_some(input)
}

/// `reduce(reduce(x, a), b)` of one fold is one reduction of `x` over the axes of both, reshaped
/// to the result, when the elements may combine in any order.
#[pattern(canonical(ReduceSum, ReduceMax, ReduceMin, ReduceAnd, ReduceOr))]
struct MergeReductions<T>(PhantomData<T>);

impl<T: Reduce> Pattern for MergeReductions<T> {
    type Root = T;

    fn rewrite(&self, outer: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let inner = outer
            .input()
            .defining_op::<T>()
            .ok_or("the input is not a reduction of the same fold")?;
        require!(
            plain_input(&outer).is_some(),
            "the outer reduction changes dtype"
        );
        require!(let Some(input) = plain_input(&inner), "the inner reduction changes dtype");
        require!(
            combines_in_any_order::<T::Combiner>(input.dtype),
            "the elements combine in one order"
        );
        require!(
            outer.input().has_one_use(),
            "the inner reduction has other users"
        );
        let mut axes: Vec<usize> = inner.reduced_axes().to_vec();
        for axis in outer.reduced_axes() {
            axes.push(inner.input_axis(axis));
        }
        axes.sort_unstable();
        axes.dedup();
        let merged = T::from(ReduceAttrs {
            axes: axes.into_iter().map(|axis| axis as i64).collect(),
            keepdims: inner.keepdims && outer.keepdims,
            accum: None,
            out: None,
        });
        let merged = rewriter.create(merged, inner.input());
        let shape = outer.output().tensor_type().dims.clone();
        let result = rewriter.on(merged).reshape(shape).value();
        rewriter.replace_op(&outer, result);
        Ok(())
    }
}

/// A reduction of `broadcast_to(x)` is `x` reshaped to the result when its fold is idempotent,
/// the reduction covers every axis the broadcast repeats `x` along, and its other axes have
/// extent 1. Each repeated axis must have a static extent above zero, since a reduction over no
/// elements gives its initial value.
#[pattern(canonical(ReduceSum, ReduceMax, ReduceMin, ReduceAnd, ReduceOr))]
struct ReduceOfBroadcast<T>(PhantomData<T>);

impl<T: Reduce> Pattern for ReduceOfBroadcast<T> {
    type Root = T;

    fn rewrite(&self, reduce: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let broadcast = reduce
            .input()
            .defining_op::<BroadcastTo>()
            .ok_or("the input is not a broadcast")?;
        require!(
            plain_input(&reduce).is_some(),
            "the reduction changes dtype"
        );
        require!(
            Traits::new(T::Combiner::INFO.traits).has::<Idempotent>(),
            "the fold is not idempotent"
        );
        let source = broadcast.input().tensor_type();
        let shape = &broadcast.shape;
        let reduced = reduce.reduced_axes();
        for (axis, extent) in shape.iter().enumerate() {
            let source_axis = broadcast.source_axis(axis, source.rank());
            let repeated = source_axis.is_none_or(|source_axis| source.dims[source_axis].is_one())
                && !extent.is_one();
            let nonempty = matches!(extent, Dim::Static(extent) if *extent > 0);
            match (repeated, reduced.contains(&axis)) {
                (true, true) => require!(nonempty, "a repeated axis may be empty"),
                (true, false) => return Err("the reduction keeps a repeated axis".into()),
                (false, true) => require!(extent.is_one(), "the reduction folds a source axis"),
                (false, false) => {}
            }
        }
        let output = reduce.output().tensor_type().dims.clone();
        let result = rewriter.on(broadcast.input()).reshape(output).value();
        rewriter.replace_op(&reduce, result);
        Ok(())
    }
}

/// A reduction of `concat(x1, x2, ...)` that reduces the concatenated axis is its fold applied to
/// the reductions of the parts, when the elements may combine in any order.
#[pattern(canonical(ReduceSum, ReduceMax, ReduceMin, ReduceAnd, ReduceOr))]
struct ReduceOfConcat<T>(PhantomData<T>);

impl<T: Reduce> Pattern for ReduceOfConcat<T> {
    type Root = T;

    fn rewrite(&self, reduce: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let concat = reduce
            .input()
            .defining_op::<Concat>()
            .ok_or("the input is not a concat")?;
        require!(let Some(input) = plain_input(&reduce), "the reduction changes dtype");
        let rank = input.rank();
        let concat_axis = axis_index(concat.axis, rank).expect("a verified axis");
        let reduced = reduce.reduced_axes();
        require!(concat.inputs().len() > 1, "the concat has one input");
        require!(
            reduced.contains(&concat_axis),
            "the reduction keeps the concatenated axis"
        );
        require!(
            combines_in_any_order::<T::Combiner>(input.dtype),
            "the elements combine in one order"
        );
        require!(reduce.input().has_one_use(), "the concat has other users");
        let mut combined = rewriter.create(reduce.attrs().clone(), concat.inputs()[0]);
        for &part in &concat.inputs()[1..] {
            let partial = rewriter.create(reduce.attrs().clone(), part);
            combined = rewriter.create(T::Combiner::default(), (combined, partial));
        }
        rewriter.replace_op(&reduce, combined);
        Ok(())
    }
}

/// A transpose that moves only axes of extent 1 keeps the row-major order of the elements, so it
/// is a reshape to its result shape.
#[pattern(canonical)]
struct TransposeOfUnitAxes;

impl Pattern for TransposeOfUnitAxes {
    type Root = Transpose;

    fn rewrite(
        &self,
        transpose: Matched<'_, Transpose>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let output = transpose.output().tensor_type();
        let mut moved = Vec::new();
        for (axis, extent) in output.dims.iter().enumerate() {
            if !extent.is_one() {
                moved.push(axis);
            }
        }
        require!(
            transpose.perm.keeps_order(&moved),
            "the transpose reorders axes of more than one element"
        );
        let shape = output.dims.clone();
        let reshaped = rewriter.create(Reshape::new(shape), transpose.input());
        rewriter.replace_op(&transpose, reshaped);
        Ok(())
    }
}

/// A slice of a concat that reads one input along the concatenated axis is a slice of that input.
/// A slice that spans several inputs stays: as a concat of slices, it would copy what it reads
/// twice.
#[pattern(canonical)]
struct SliceOfConcat;

impl Pattern for SliceOfConcat {
    type Root = Slice;

    fn rewrite(
        &self,
        slice: Matched<'_, Slice>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let concat = slice
            .input()
            .defining_op::<Concat>()
            .ok_or("the input is not a concat")?;
        let rank = slice.input().tensor_type().rank();
        let axis = axis_index(concat.axis, rank).expect("a verified axis");
        let first = slice.starts[axis];
        let last = first + slice.span(axis);
        let mut offset = 0;
        for input in concat.inputs() {
            require!(
                let Some(extent) = input.tensor_type().static_dim(axis),
                "an input has a dynamic extent on the concatenated axis"
            );
            if first >= offset && last <= offset + extent {
                let mut starts = slice.starts;
                starts[axis] -= offset;
                let piece = Slice {
                    starts,
                    ..slice.attrs().clone()
                };
                let result = rewriter.create(piece, input.id());
                rewriter.replace_op(&slice, result);
                return Ok(());
            }
            offset += extent;
        }
        Err("the slice spans several inputs".into())
    }
}

/// A concat of a concat on the same axis is one concat of the inner concat's inputs.
#[pattern(canonical)]
struct ConcatOfConcat;

impl Pattern for ConcatOfConcat {
    type Root = Concat;

    fn rewrite(
        &self,
        concat: Matched<'_, Concat>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let rank = concat.output().tensor_type().rank();
        let axis = axis_index(concat.axis, rank).expect("a verified axis");
        let mut inputs = Vec::new();
        let mut flattened = false;
        for input in concat.inputs() {
            let inner = input
                .defining_op::<Concat>()
                .filter(|inner| axis_index(inner.axis, rank).ok() == Some(axis));
            match inner {
                Some(inner) => {
                    inputs.extend(inner.inputs().iter().map(|part| part.id()));
                    flattened = true;
                }
                None => inputs.push(input.id()),
            }
        }
        require!(flattened, "no input is a concat on the same axis");
        let merged = rewriter.create(Concat { axis: axis as i64 }, inputs);
        rewriter.replace_op(&concat, merged);
        Ok(())
    }
}

/// A `broadcast_to` of a reshape that only adds leading axes of extent 1 is the `broadcast_to` of
/// the reshape's input: broadcasting aligns trailing axes already.
#[pattern(canonical)]
struct BroadcastOfLeadingUnitAxes;

impl Pattern for BroadcastOfLeadingUnitAxes {
    type Root = BroadcastTo;

    fn rewrite(
        &self,
        broadcast: Matched<'_, BroadcastTo>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let reshape = broadcast
            .input()
            .defining_op::<Reshape>()
            .ok_or("the input is not a reshape")?;
        let source = &reshape.input().tensor_type().dims;
        let reshaped = &reshape.output().tensor_type().dims;
        let added = reshaped
            .len()
            .checked_sub(source.len())
            .ok_or("the reshape drops axes")?;
        let leading_units = reshaped[..added].iter().all(Dim::is_one);
        require!(
            leading_units && reshaped[added..] == source[..],
            "the reshape does more than add leading axes of extent 1"
        );
        let direct = rewriter.create(BroadcastTo::new(broadcast.shape.clone()), reshape.input());
        rewriter.replace_op(&broadcast, direct);
        Ok(())
    }
}

/// A `tensor.iota` along an axis of extent 1 counts no further than 0: it is 0, broadcast.
#[pattern(canonical)]
struct IotaOfUnitAxis;

impl Pattern for IotaOfUnitAxis {
    type Root = Iota;

    fn rewrite(&self, iota: Matched<'_, Iota>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        let axis = axis_index(iota.axis, iota.shape.len()).expect("a verified axis");
        require!(
            iota.shape[axis].is_one(),
            "the axis has more than one element"
        );
        require!(let Ok(zero) = HostScalar::zero(iota.dtype), "the dtype has no 0");
        let zeros = rewriter.splat(zero, &iota.shape);
        rewriter.replace_op(&iota, zeros);
        Ok(())
    }
}

/// A comparison of an integer or `i1` value with itself holds everywhere or nowhere, as its
/// relation does on equal operands. Equal constants are one value once the driver merges them.
/// Floats are left alone: NaN is not equal to itself.
#[pattern(canonical(Less, LessEqual, Equal, GreaterEqual, Greater, NotEqual))]
struct CompareWithItself<T>(PhantomData<T>);

impl<T: Compare> Pattern for CompareWithItself<T> {
    type Root = T;

    fn rewrite(&self, compare: Matched<'_, T>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
        require!(compare.lhs() == compare.rhs(), "the operands may differ");
        let dtype = compare.lhs().tensor_type().dtype;
        require!(
            dtype.int_format().is_some(),
            "a float can differ from itself"
        );
        let holds = HostScalar::int(DType::I1, i128::from(T::Scalar::holds(0, 0)))
            .expect("i1 holds 0 and 1");
        let shape = compare.output().tensor_type().dims.clone();
        let result = rewriter.splat(holds, &shape);
        rewriter.replace_op(&compare, result);
        Ok(())
    }
}

/// A dot whose contracted axes hold one element sums one product. It is the product of the
/// operands, laid out along the result's axes and cast to the accumulator, plus 0: the sum starts
/// from 0, so a product of `-0.0` gives `0.0`. The result is cast to `out`.
#[pattern(canonical)]
struct DotOfOneProduct;

/// `operand`, the dot's operand on `side`, laid out along the result: its batch and free axes
/// where the result holds them and extent 1 at the other side's free axes, broadcast to `result`
/// and cast to `accum`.
fn along_result(
    rewriter: &mut Rewriter<'_>,
    operand: ValueRef<'_>,
    (axes, side): (&DotAxes, Side),
    result: &TensorType,
    accum: DType,
) -> Value {
    let dims = &operand.tensor_type().dims;
    let landing = axes.result_axes(side);
    // The axes in the order of the result axes they land on, then the contracted ones, which
    // hold one element each.
    let mut order: Vec<usize> = (0..dims.len()).collect();
    order.sort_by_key(|&axis| landing[axis].unwrap_or(usize::MAX));
    let mut shape = vec![Dim::Static(1); result.rank()];
    for (axis, result_axis) in landing.iter().enumerate() {
        if let Some(result_axis) = result_axis {
            shape[*result_axis] = dims[axis].clone();
        }
    }
    rewriter
        .on(operand.id())
        .transpose(order)
        .reshape(shape)
        .broadcast_to(result.dims.clone())
        .cast(accum)
        .value()
}

impl Pattern for DotOfOneProduct {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let axes = dot.axes();
        let lhs_type = dot.lhs().tensor_type();
        let one_product = axes
            .contract(Side::Lhs)
            .iter()
            .all(|&axis| lhs_type.dims[axis].is_one());
        require!(one_product, "the dot sums more than one product");
        let result = dot.output().tensor_type();
        let accum = dot.accum_dtype(lhs_type.dtype);
        require!(let Ok(zero) = HostScalar::zero(accum), "the accumulator has no 0");
        let lhs = along_result(rewriter, dot.lhs(), (&axes, Side::Lhs), result, accum);
        let rhs = along_result(rewriter, dot.rhs(), (&axes, Side::Rhs), result, accum);
        let start = rewriter.splat(zero, &result.dims);
        let output = (rewriter.on(lhs) * rhs + start).cast(result.dtype).value();
        rewriter.replace_op(&dot, output);
        Ok(())
    }
}

/// A transpose that feeds a dot folds into the dot's axes when it keeps the order of the
/// operand's free axes: the dot reads the transpose's input and names its batch and contracted
/// axes there. Each sum keeps its order, which follows the listed contracted axes.
#[pattern(canonical)]
struct TransposeIntoDot;

impl Pattern for TransposeIntoDot {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let mut axes = dot.axes();
        let mut operands = [dot.lhs(), dot.rhs()];
        let mut folded = false;
        for (operand, side) in operands.iter_mut().zip([Side::Lhs, Side::Rhs]) {
            let Some(transpose) = operand.defining_op::<Transpose>() else {
                continue;
            };
            if !transpose.perm.keeps_order(&axes.free(side)) {
                continue;
            }
            axes = axes.through(side, &transpose.perm);
            *operand = transpose.input();
            folded = true;
        }
        require!(
            folded,
            "no operand is a transpose that keeps the order of its free axes"
        );
        let [lhs, rhs] = operands;
        let op = dot.attrs().clone().with_axes(&axes);
        let result = rewriter.create(op, (lhs.id(), rhs.id()));
        rewriter.replace_op(&dot, result);
        Ok(())
    }
}
