//! The ops of [`DeviceTensor`]: each method records one op of the `tensor` dialect
//! (`gpt_rs_ir::dialects::tensor`) into the lazy graph of its operands and returns its result.
//!
//! The graph is the current default arena, else the graph of the first operand that has one, else
//! a new graph. An operand from another graph joins it as an input. The op's type rule gives the
//! result type. When the operands do not fit the op, recording panics with the caller's location.
//! Binary ops broadcast their operands as Torch does, through explicit `broadcast_to` ops, and take
//! an `f32` for a tensor ([`Operand`]). A graph records each such scalar once for each shape it is
//! used at. The operators `+`, `-`, `*`, `/` and unary `-` record the same ops.

use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use gpt_rs_ir::dialects::tensor::{
    self, ArgMax, Binary, BroadcastTo, Cast, Concat, Constant, Dequantize, DotGeneral,
    DynamicSlice, DynamicUpdateSlice, ExtractPatches, Iota, Reduce, ReduceAttrs, ReduceWindow,
    ReduceWindowAttrs, Reshape, RngAttrs, RngUniform, Select, SelectKAttrs, Take, TopK, Transpose,
    Unary,
};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::{ranked, DType, Dense, Dim, InOp, Location, OpDyn, Ranked, F64};
use half::{bf16, f16};
use smallvec::SmallVec;

use super::slicing::{self, AxisIndex};
use super::{DeviceTensor, Shape};
use crate::backend::PortableBackend;
use crate::nn::capture;
use crate::ops::graph::{context, GraphArena, GraphBuilder, NodeToken, ValueId};
use crate::ops::trace;

/// What a binary op or `select` takes besides a tensor: another tensor, or an `f32` scalar, which
/// becomes a constant.
pub trait Operand<B: PortableBackend + 'static> {
    /// The extents of this operand, or `None` for a scalar, which broadcasts to any.
    fn dims(&self) -> Option<&[usize]>;

    /// This operand as a tensor. A tensor stays as it is. A scalar becomes a constant of `dtype`
    /// and extents `dims` in the graph of `like`.
    fn tensor<'a>(
        self,
        like: &DeviceTensor<B>,
        dtype: DType,
        dims: &[usize],
    ) -> Cow<'a, DeviceTensor<B>>
    where
        Self: 'a;
}

impl<B: PortableBackend + 'static> Operand<B> for &DeviceTensor<B> {
    fn dims(&self) -> Option<&[usize]> {
        Some(self.shape().dims())
    }

    fn tensor<'a>(self, _: &DeviceTensor<B>, _: DType, _: &[usize]) -> Cow<'a, DeviceTensor<B>>
    where
        Self: 'a,
    {
        Cow::Borrowed(self)
    }
}

impl<B: PortableBackend + 'static> Operand<B> for DeviceTensor<B> {
    fn dims(&self) -> Option<&[usize]> {
        Some(self.shape().dims())
    }

    fn tensor<'a>(self, _: &DeviceTensor<B>, _: DType, _: &[usize]) -> Cow<'a, DeviceTensor<B>>
    where
        Self: 'a,
    {
        Cow::Owned(self)
    }
}

impl<B: PortableBackend + 'static> Operand<B> for f32 {
    fn dims(&self) -> Option<&[usize]> {
        None
    }

    #[track_caller]
    fn tensor<'a>(
        self,
        like: &DeviceTensor<B>,
        dtype: DType,
        dims: &[usize],
    ) -> Cow<'a, DeviceTensor<B>>
    where
        Self: 'a,
    {
        let mut bytes = [0u8; 8];
        let size = match dtype {
            DType::F32 => put(&mut bytes, self.to_le_bytes()),
            DType::F16 => put(&mut bytes, f16::from_f32(self).to_le_bytes()),
            DType::Bf16 => put(&mut bytes, bf16::from_f32(self).to_le_bytes()),
            DType::F64 => put(&mut bytes, f64::from(self).to_le_bytes()),
            other => panic!("a scalar operand needs a float tensor, got {other}"),
        };
        let graph = graph_of(&like.backend, &[like]);
        let recorded = DeviceTensor::capture(&graph, &[], |builder, _| {
            let value = builder.scalar(dtype, &bytes[..size], dims)?;
            recorded(builder, value)
        });
        Cow::Owned(DeviceTensor::lazy(&graph, recorded))
    }
}

/// Writes `element` at the start of `bytes` and returns its size.
fn put<const N: usize>(bytes: &mut [u8; 8], element: [u8; N]) -> usize {
    bytes[..N].copy_from_slice(&element);
    N
}

/// The graph that an op over `operands` records into.
fn graph_of<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    operands: &[&DeviceTensor<B>],
) -> Arc<GraphArena<B>> {
    context::current_arena::<B>()
        .filter(|arena| arena.runs_on(backend))
        .or_else(|| operands.iter().find_map(|tensor| tensor.graph()))
        .unwrap_or_else(|| GraphArena::new(Arc::clone(backend)))
}

/// `items` as an attribute indexed by axis. Panics, at the caller, when `items` has more than
/// `MAX_RANK` entries, the most axes a tensor can have.
#[track_caller]
fn axis_list<T: Copy + Default>(items: impl AsRef<[T]>) -> Ranked<T> {
    match ranked(items.as_ref().iter().copied()) {
        Ok(list) => list,
        Err(err) => panic!("{err}"),
    }
}

#[track_caller]
fn static_dims(dims: impl AsRef<[usize]>) -> Ranked<Dim> {
    let extents = axis_list(dims);
    let mut shape = Ranked::new();
    for &extent in &extents {
        shape.push(Dim::Static(extent));
    }
    shape
}

/// The value `value` of `builder`'s graph with the static shape of its type, and the token that
/// its tensors hold.
fn recorded<B: PortableBackend + 'static>(
    builder: &mut GraphBuilder<B>,
    value: ValueId,
) -> anyhow::Result<Recorded> {
    let ty = builder.type_of(value)?;
    let mut dims = Ranked::<usize>::new();
    for dim in &ty.dims {
        let Some(extent) = dim.as_static() else {
            anyhow::bail!("{value:?} has the dynamic type {ty}");
        };
        dims.push(extent);
    }
    let dtype = ty.dtype;
    Ok((value, Shape::new(dims), dtype, builder.node_token(value)))
}

/// Where the op being recorded comes from, while a trace sink keeps locations: the code that
/// called the `#[track_caller]` chain of op methods and functionals, in the module path of the
/// enclosing layers.
#[track_caller]
fn recording_location() -> Option<Location> {
    // Not `bool::then`: a closure would end the `#[track_caller]` chain at itself.
    if trace::sink_installed() {
        Some(Location::caller(capture::path()))
    } else {
        None
    }
}

/// The extents that `lhs` and `rhs` broadcast to, aligned from the last axis.
#[track_caller]
fn broadcast_dims(lhs: &[usize], rhs: &[usize]) -> Shape {
    let rank = lhs.len().max(rhs.len());
    let extent = |dims: &[usize], axis: usize| {
        (axis + dims.len())
            .checked_sub(rank)
            .map_or(1, |index| dims[index])
    };
    (0..rank)
        .map(|axis| {
            let (lhs_extent, rhs_extent) = (extent(lhs, axis), extent(rhs, axis));
            assert!(
                lhs_extent == rhs_extent || lhs_extent == 1 || rhs_extent == 1,
                "shapes {lhs:?} and {rhs:?} do not broadcast"
            );
            lhs_extent.max(rhs_extent)
        })
        .collect()
}

/// A recorded value with the static shape and dtype of its type, and the token that its tensors
/// hold.
type Recorded = (ValueId, Shape, DType, Arc<NodeToken>);

impl<B: PortableBackend + 'static> DeviceTensor<B> {
    /// Records `op` on `operands` in the graph of `operands` and returns its result.
    #[track_caller]
    fn record<T: InOp + OpDyn>(op: T, operands: &[&Self]) -> Self {
        let graph = graph_of(&operands[0].backend, operands);
        Self::record_in(&graph, op, operands)
    }

    #[track_caller]
    fn record_in<T: InOp + OpDyn>(graph: &Arc<GraphArena<B>>, op: T, operands: &[&Self]) -> Self {
        let location = recording_location();
        let recorded = Self::capture(graph, operands, |builder, operand_values| {
            builder.set_location(location);
            let results = builder.emit(op, operand_values)?;
            anyhow::ensure!(results.len() == 1, "the op gives {} results", results.len());
            recorded(builder, ValueId(results.start))
        });
        Self::lazy(graph, recorded)
    }

    /// Records `op` on `operands` in `graph` and returns all of its results.
    #[track_caller]
    fn record_results<T: InOp + OpDyn>(
        graph: &Arc<GraphArena<B>>,
        op: T,
        operands: &[&Self],
    ) -> SmallVec<[Self; 2]> {
        let location = recording_location();
        let recorded: SmallVec<[Recorded; 2]> =
            Self::capture(graph, operands, |builder, operand_values| {
                builder.set_location(location);
                builder
                    .emit(op, operand_values)?
                    .map(|value| recorded(builder, ValueId(value)))
                    .collect()
            });
        recorded
            .into_iter()
            .map(|result| Self::lazy(graph, result))
            .collect()
    }

    /// Runs `record` on the values of `operands` in `graph`, which it imports, and panics with
    /// its error.
    #[track_caller]
    fn capture<R>(
        graph: &Arc<GraphArena<B>>,
        operands: &[&Self],
        record: impl FnOnce(&mut GraphBuilder<B>, &[ValueId]) -> anyhow::Result<R>,
    ) -> R {
        if let Some(other) = operands
            .iter()
            .find(|tensor| !graph.runs_on(&tensor.backend))
        {
            panic!(
                "a {} tensor of another backend instance cannot join this graph",
                other.backend.backend_name()
            );
        }
        let captured = graph.capture(|builder| {
            let mut values = SmallVec::<[ValueId; 4]>::new();
            for tensor in operands {
                values.push(builder.import(tensor)?);
            }
            record(builder, &values)
        });
        match captured {
            Ok(result) => result,
            Err(err) => panic!("{err:#}"),
        }
    }

    /// The lazy tensor of a recorded value.
    #[track_caller]
    fn lazy(graph: &Arc<GraphArena<B>>, (value, shape, dtype, token): Recorded) -> Self {
        match Self::from_lazy(Arc::clone(graph), shape, dtype, value, token) {
            Ok(tensor) => tensor,
            Err(err) => panic!("{err:#}"),
        }
    }

    /// This tensor broadcast to `dims`, or itself when it has them.
    #[track_caller]
    fn broadcast_if(&self, dims: &[usize]) -> Cow<'_, Self> {
        if self.shape().dims() == dims {
            return Cow::Borrowed(self);
        }
        Cow::Owned(self.broadcast_to(dims))
    }

    /// `op` on this tensor and `rhs` broadcast to one shape.
    #[track_caller]
    fn broadcast_binary<T: InOp + OpDyn>(&self, rhs: &Self, op: T) -> Self {
        let dims = broadcast_dims(self.shape().dims(), rhs.shape().dims());
        Self::record(
            op,
            &[
                &*self.broadcast_if(dims.dims()),
                &*rhs.broadcast_if(dims.dims()),
            ],
        )
    }

    #[track_caller]
    fn binary(&self, rhs: impl Operand<B>, op: impl Binary + OpDyn) -> Self {
        let dims = match rhs.dims() {
            Some(dims) => broadcast_dims(self.shape().dims(), dims),
            None => self.shape().clone(),
        };
        let rhs_tensor = rhs.tensor(self, self.dtype(), dims.dims());
        Self::record(
            op,
            &[
                &*self.broadcast_if(dims.dims()),
                &*rhs_tensor.broadcast_if(dims.dims()),
            ],
        )
    }

    #[track_caller]
    fn unary(&self, op: impl Unary + OpDyn) -> Self {
        Self::record(op, &[self])
    }

    /// A tensor of `dims` whose elements count up along `axis`, in the graph of `like`.
    #[track_caller]
    pub fn iota(like: &Self, dims: impl AsRef<[usize]>, axis: i64, dtype: DType) -> Self {
        let op = Iota {
            shape: static_dims(dims),
            dtype,
            axis,
        };
        Self::record_in(&graph_of(&like.backend, &[like]), op, &[])
    }

    /// The elements of `value` as a constant in the graph of `like`. The program holds them, so
    /// they suit small tables that every run reads, not run-time inputs.
    #[track_caller]
    pub fn constant(like: &Self, value: &HostTensor) -> Self {
        let dense = Dense {
            dtype: value.dtype(),
            dims: value.dims.to_vec(),
            bytes: value.to_le_bytes().into(),
        };
        let op = Constant { value: dense };
        Self::record_in(&graph_of(&like.backend, &[like]), op, &[])
    }

    /// Keeps this tensor as a result of the graph although nothing reads it, as for an updated
    /// cache.
    pub fn export(&self) {
        if let Some((graph, value)) = self.graph_value() {
            graph.export(value);
        }
    }

    #[track_caller]
    pub fn add(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Add)
    }

    #[track_caller]
    pub fn sub(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Sub)
    }

    #[track_caller]
    pub fn mul(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Mul)
    }

    #[track_caller]
    pub fn div(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Div)
    }

    #[track_caller]
    pub fn maximum(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Maximum)
    }

    #[track_caller]
    pub fn minimum(&self, rhs: impl Operand<B>) -> Self {
        self.binary(rhs, tensor::Minimum)
    }

    #[track_caller]
    pub fn neg(&self) -> Self {
        self.unary(tensor::Neg)
    }

    #[track_caller]
    pub fn abs(&self) -> Self {
        self.unary(tensor::Abs)
    }

    #[track_caller]
    pub fn exp(&self) -> Self {
        self.unary(tensor::Exp)
    }

    #[track_caller]
    pub fn log(&self) -> Self {
        self.unary(tensor::Log)
    }

    #[track_caller]
    pub fn tanh(&self) -> Self {
        self.unary(tensor::Tanh)
    }

    #[track_caller]
    pub fn erf(&self) -> Self {
        self.unary(tensor::Erf)
    }

    #[track_caller]
    pub fn sin(&self) -> Self {
        self.unary(tensor::Sin)
    }

    #[track_caller]
    pub fn cos(&self) -> Self {
        self.unary(tensor::Cos)
    }

    #[track_caller]
    pub fn rsqrt(&self) -> Self {
        self.unary(tensor::Rsqrt)
    }

    #[track_caller]
    pub fn reciprocal(&self) -> Self {
        self.unary(tensor::Reciprocal)
    }

    /// `1` where this tensor is at least `rhs`, `0` elsewhere.
    #[track_caller]
    pub fn greater_equal(&self, rhs: &Self) -> Self {
        self.broadcast_binary(rhs, tensor::GreaterEqual)
    }

    /// `when_true` where this predicate holds and `when_false` elsewhere, all three broadcast to
    /// one shape. A scalar branch is an f32.
    #[track_caller]
    pub fn select(&self, when_true: impl Operand<B>, when_false: impl Operand<B>) -> Self {
        let mut dims = self.shape().clone();
        for branch in [when_true.dims(), when_false.dims()].into_iter().flatten() {
            dims = broadcast_dims(dims.dims(), branch);
        }
        let true_branch = when_true.tensor(self, DType::F32, dims.dims());
        let false_branch = when_false.tensor(self, DType::F32, dims.dims());
        Self::record(
            Select,
            &[
                &*self.broadcast_if(dims.dims()),
                &*true_branch.broadcast_if(dims.dims()),
                &*false_branch.broadcast_if(dims.dims()),
            ],
        )
    }

    /// These codes dequantized to `to` (`tensor.dequantize`): each code times the scale of its
    /// block of `block` elements, plus the block's `offset` where there is one.
    #[track_caller]
    pub fn dequantize(
        &self,
        scale: &Self,
        offset: Option<&Self>,
        block: impl AsRef<[usize]>,
        to: DType,
    ) -> Self {
        let attrs = Dequantize {
            block: block.as_ref().iter().copied().collect(),
            to,
        };
        match offset {
            Some(block_offsets) => Self::record(attrs, &[self, scale, block_offsets]),
            None => Self::record(attrs, &[self, scale]),
        }
    }

    /// This tensor converted to `dtype`, or itself when it has `dtype`. Float-to-float casts round
    /// to nearest even.
    #[track_caller]
    pub fn cast(&self, dtype: DType) -> Self {
        if self.dtype() == dtype {
            return self.clone();
        }
        Self::record(Cast { to: dtype }, &[self])
    }

    #[track_caller]
    pub fn reshape(&self, dims: impl AsRef<[usize]>) -> Self {
        let shape = static_dims(dims);
        Self::record(Reshape { shape }, &[self])
    }

    #[track_caller]
    pub fn transpose(&self, perm: impl AsRef<[usize]>) -> Self {
        Self::record(Transpose::new(perm.as_ref().iter().copied()), &[self])
    }

    /// This tensor broadcast to `dims`, aligned from the last axis.
    #[track_caller]
    pub fn broadcast_to(&self, dims: impl AsRef<[usize]>) -> Self {
        let shape = static_dims(dims);
        Self::record(BroadcastTo { shape }, &[self])
    }

    /// The elements that a Python-style index takes: `x.slice(s![.., 1..3])` is `x[:, 1:3]`, as
    /// [`s!`](crate::s) describes.
    #[track_caller]
    pub fn slice(&self, index: impl AsRef<[AxisIndex]>) -> Self {
        let dims = self.shape().dims();
        let (op, out) = slicing::resolve(dims, index.as_ref());
        let whole = op.strides.is_empty()
            && op.starts.iter().all(|&start| start == 0)
            && op.sizes[..] == *dims;
        let taken = if whole {
            self.clone()
        } else {
            Self::record(op, &[self])
        };
        if taken.shape().dims() == &out[..] {
            taken
        } else {
            taken.reshape(out)
        }
    }

    /// The block of extents `sizes` at the offsets `starts` (`si32 [rank]`).
    #[track_caller]
    pub fn dynamic_slice(&self, starts: &Self, sizes: impl AsRef<[usize]>) -> Self {
        let op = DynamicSlice {
            sizes: axis_list(sizes),
        };
        Self::record(op, &[self, starts])
    }

    /// This tensor with the block at the offsets `starts` (`si32 [rank]`) replaced by `patch`.
    #[track_caller]
    pub fn dynamic_update_slice(&self, patch: &Self, starts: &Self) -> Self {
        Self::record(DynamicUpdateSlice, &[self, patch, starts])
    }

    /// The tensors joined along `axis`.
    #[track_caller]
    pub fn concat(axis: i64, tensors: &[&Self]) -> Self {
        assert!(!tensors.is_empty(), "concat needs at least one tensor");
        Self::record(Concat { axis }, tensors)
    }

    #[track_caller]
    fn reduce<T: Reduce + OpDyn>(&self, axes: &[i64], keepdims: bool) -> Self {
        let op = T::from(ReduceAttrs {
            axes: axis_list(axes),
            keepdims,
            accum: None,
            out: None,
        });
        Self::record(op, &[self])
    }

    #[track_caller]
    pub fn reduce_sum(&self, axes: impl AsRef<[i64]>, keepdims: bool) -> Self {
        self.reduce::<tensor::ReduceSum>(axes.as_ref(), keepdims)
    }

    #[track_caller]
    pub fn reduce_max(&self, axes: impl AsRef<[i64]>, keepdims: bool) -> Self {
        self.reduce::<tensor::ReduceMax>(axes.as_ref(), keepdims)
    }

    /// The windows of `window` extents that slide over the spatial axes of an NHWC tensor, as
    /// `[N, OH, OW, KH * KW * C]`.
    #[track_caller]
    pub fn extract_patches(
        &self,
        window: impl AsRef<[usize]>,
        strides: impl AsRef<[usize]>,
        dilation: impl AsRef<[usize]>,
        padding: impl AsRef<[(usize, usize)]>,
        pad_value: f64,
    ) -> Self {
        let op = ExtractPatches {
            window: axis_list(window),
            strides: axis_list(strides),
            dilation: axis_list(dilation),
            padding: axis_list(padding),
            pad_value: F64(pad_value),
        };
        Self::record(op, &[self])
    }

    /// The reduction `T` of each window of `window` extents that slides by `strides` over this
    /// tensor, padded by `padding` (`(before, after)` per axis), which adds nothing to a window.
    #[track_caller]
    pub fn reduce_window<T: ReduceWindow + OpDyn>(
        &self,
        window: impl AsRef<[usize]>,
        strides: impl AsRef<[usize]>,
        padding: impl AsRef<[(usize, usize)]>,
    ) -> Self {
        let window_dims = axis_list(window);
        let mut ones = Ranked::new();
        ones.resize(window_dims.len(), 1);
        let op = T::from(ReduceWindowAttrs {
            window_dims,
            strides: axis_list(strides),
            padding: axis_list(padding),
            base_dilation: ones,
            window_dilation: ones,
            accum: None,
            out: None,
        });
        Self::record(op, &[self])
    }

    /// The contraction of this tensor with `rhs` that `dims` describes, for example
    /// `DotGeneral::contract([1], [0])` for a matrix product.
    #[track_caller]
    pub fn dot_general(&self, rhs: &Self, dims: DotGeneral) -> Self {
        Self::record(dims, &[self, rhs])
    }

    /// The rows of this table at `indices`.
    #[track_caller]
    pub fn take(&self, indices: &Self) -> Self {
        Self::record(Take, &[self, indices])
    }

    /// The `si32` index of the largest element along `axis`, which the result drops. When several
    /// tie, it is the first one.
    #[track_caller]
    pub fn argmax(&self, axis: i64) -> Self {
        let op = ArgMax {
            axis,
            keepdims: false,
            out: DType::Si32,
        };
        Self::record(op, &[self])
    }

    /// The `count` largest elements along `axis`, largest first, and their `si32` indices.
    #[track_caller]
    pub fn top_k(&self, count: usize, axis: i64) -> (Self, Self) {
        let op = TopK(SelectKAttrs {
            k: count,
            axis,
            indices: DType::Si32,
        });
        let graph = graph_of(&self.backend, &[self]);
        let Ok([values, indices]) = Self::record_results(&graph, op, &[self]).into_inner() else {
            unreachable!("top_k gives two results")
        };
        (values, indices)
    }

    /// A tensor of `dims` whose `dtype` elements are uniform on `[0, 1)`, drawn from the generator
    /// state this tensor holds (`ui64 [2]`), and the state after them.
    #[track_caller]
    pub fn rng_uniform(&self, dims: impl AsRef<[usize]>, dtype: DType) -> (Self, Self) {
        let op = RngUniform(RngAttrs {
            shape: static_dims(dims),
            dtype,
        });
        let graph = graph_of(&self.backend, &[self]);
        let Ok([state, values]) = Self::record_results(&graph, op, &[self]).into_inner() else {
            unreachable!("rng_uniform gives two results")
        };
        (values, state)
    }
}

macro_rules! binary_operator {
    ($trait:ident, $method:ident) => {
        impl<B: PortableBackend + 'static, R: Operand<B>> $trait<R> for &DeviceTensor<B> {
            type Output = DeviceTensor<B>;
            #[track_caller]
            fn $method(self, rhs: R) -> DeviceTensor<B> {
                DeviceTensor::$method(self, rhs)
            }
        }

        impl<B: PortableBackend + 'static, R: Operand<B>> $trait<R> for DeviceTensor<B> {
            type Output = DeviceTensor<B>;
            #[track_caller]
            fn $method(self, rhs: R) -> DeviceTensor<B> {
                DeviceTensor::$method(&self, rhs)
            }
        }

        impl<B: PortableBackend + 'static> $trait<&DeviceTensor<B>> for f32 {
            type Output = DeviceTensor<B>;
            #[track_caller]
            fn $method(self, rhs: &DeviceTensor<B>) -> DeviceTensor<B> {
                let dims = rhs.shape().dims();
                DeviceTensor::$method(&self.tensor(rhs, rhs.dtype(), dims), rhs)
            }
        }

        impl<B: PortableBackend + 'static> $trait<DeviceTensor<B>> for f32 {
            type Output = DeviceTensor<B>;
            #[track_caller]
            fn $method(self, rhs: DeviceTensor<B>) -> DeviceTensor<B> {
                let dims = rhs.shape().dims();
                DeviceTensor::$method(&self.tensor(&rhs, rhs.dtype(), dims), &rhs)
            }
        }
    };
}

binary_operator!(Add, add);
binary_operator!(Sub, sub);
binary_operator!(Mul, mul);
binary_operator!(Div, div);

impl<B: PortableBackend + 'static> Neg for &DeviceTensor<B> {
    type Output = DeviceTensor<B>;
    #[track_caller]
    fn neg(self) -> DeviceTensor<B> {
        DeviceTensor::neg(self)
    }
}

impl<B: PortableBackend + 'static> Neg for DeviceTensor<B> {
    type Output = DeviceTensor<B>;
    #[track_caller]
    fn neg(self) -> DeviceTensor<B> {
        DeviceTensor::neg(&self)
    }
}
