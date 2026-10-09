//! Ops that move elements to positions their attributes fix.

use std::fmt;

use anyhow::{anyhow, bail, ensure, Result};
use smallvec::SmallVec;

use super::shape::{axis_index, check_window, is_permutation};
use crate::attr::Attr;
use crate::op::{
    fold, Fold, FoldCx, FoldIn, FoldOut, Folded, Folding, NoRegions, OpDef, OwnedTypes, TypeCx,
    Types,
};
use crate::traits::Movement;
use crate::types::{DType, Dim, Sym, TensorType};

/// `tensor.reshape`: the same elements in row-major order, with a new shape.
///
/// - `shape` gives every extent explicitly, and the element count MUST be preserved.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.reshape",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Movement::ALL)
)]
pub struct Reshape {
    pub shape: SmallVec<[Dim; 4]>,
}

impl OpDef for Reshape {
    fn result_types(
        &self,
        ins: ReshapeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ReshapeResults<OwnedTypes>> {
        let input = ins.input;
        let have = Count::of(&input.dims)
            .ok_or_else(|| anyhow!("the element count of {input} overflows"))?;
        let want = Count::of(&self.shape).ok_or_else(|| {
            anyhow!(
                "the element count of shape {} overflows",
                attr_text(&self.shape)
            )
        })?;
        ensure!(
            have == want,
            "shape {} has {want} elements, but {input} has {have}",
            attr_text(&self.shape)
        );
        Ok(ReshapeResults {
            output: input.with_dims(self.shape.clone()),
        })
    }
}

/// `tensor.transpose`: permutes the axes. Output axis `i` is input axis `perm[i]`.
///
/// - `perm` MUST be a permutation of `[0, rank)`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.transpose",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Movement::ALL)
)]
pub struct Transpose {
    pub perm: SmallVec<[usize; 4]>,
}

impl OpDef for Transpose {
    fn result_types(
        &self,
        ins: TransposeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<TransposeResults<OwnedTypes>> {
        ensure!(
            is_permutation(&self.perm, ins.input.rank()),
            "perm {:?} is not a permutation of rank {}",
            self.perm,
            ins.input.rank()
        );
        Ok(TransposeResults {
            output: ins.input.permuted(&self.perm),
        })
    }
}

/// `tensor.broadcast_to`: aligns the input with the trailing axes of `shape` and repeats it along
/// every axis where the input has extent 1 or no axis.
///
/// - The result has the extents of `shape`.
/// - Compatibility follows the right-aligned rule of "Broadcasting policy" in the dialect
///   documentation: `shape` has at least the input's rank, and each input extent is 1 or the
///   extent of `shape` on its axis. Backends MUST reject shapes that are not broadcast-compatible
///   as a spec violation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.broadcast_to",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Movement::ALL)
)]
pub struct BroadcastTo {
    pub shape: SmallVec<[Dim; 4]>,
}

impl OpDef for BroadcastTo {
    fn result_types(
        &self,
        ins: BroadcastToOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<BroadcastToResults<OwnedTypes>> {
        let input = ins.input;
        ensure!(
            self.shape.len() >= input.rank(),
            "shape {} has a lower rank than {input}",
            attr_text(&self.shape)
        );
        let rank_diff = self.shape.len() - input.rank();
        for (axis, dim) in input.dims.iter().enumerate() {
            let out_dim = &self.shape[rank_diff + axis];
            ensure!(
                *dim == Dim::Static(1) || dim == out_dim,
                "input axis {axis} has extent {dim}, which does not broadcast to {out_dim}"
            );
        }
        Ok(BroadcastToResults {
            output: input.with_dims(self.shape.clone()),
        })
    }
}

/// `tensor.slice`: the block of extents `sizes` starting at `starts`, taking every `strides[i]`-th
/// element along axis `i`.
///
/// - Along axis `i`, result element `k` is input element `starts[i] + k * strides[i]`. Without
///   `strides`, the window is contiguous: unit stride.
/// - `starts` and `sizes` have one entry per axis of the input, and `strides` is empty or has one
///   positive entry per axis.
/// - The window MUST lie in the input: on every axis `i`, `starts[i] + span <= in_shape[i]`, where
///   `span` is `(sizes[i] - 1) * strides[i] + 1`, or 0 when `sizes[i]` is 0. With unit stride,
///   that is `starts[i] + sizes[i] <= in_shape[i]`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.slice",
    operands(input: Tensor),
    results(output: Tensor),
    traits(Movement::ALL)
)]
pub struct Slice {
    pub starts: SmallVec<[usize; 4]>,
    pub sizes: SmallVec<[usize; 4]>,
    /// One step per axis, or none for steps of 1.
    #[attr(default)]
    pub strides: SmallVec<[usize; 4]>,
}

impl Slice {
    /// A slice with steps of 1.
    pub fn new(starts: impl AsRef<[usize]>, sizes: impl AsRef<[usize]>) -> Self {
        Self {
            starts: SmallVec::from_slice(starts.as_ref()),
            sizes: SmallVec::from_slice(sizes.as_ref()),
            strides: SmallVec::new(),
        }
    }

    /// The step along `axis`.
    pub fn stride(&self, axis: usize) -> usize {
        self.strides.get(axis).copied().unwrap_or(1)
    }

    /// The extent of input that the slice spans along `axis`: from its first element to its last.
    pub fn span(&self, axis: usize) -> usize {
        match self.sizes[axis] {
            0 => 0,
            size => (size - 1) * self.stride(axis) + 1,
        }
    }
}

impl OpDef for Slice {
    fn result_types(
        &self,
        ins: SliceOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<SliceResults<OwnedTypes>> {
        let input = ins.input;
        let rank = input.rank();
        ensure!(
            self.starts.len() == rank && self.sizes.len() == rank,
            "starts {:?} and sizes {:?} do not match rank {rank}",
            self.starts,
            self.sizes
        );
        ensure!(
            self.strides.is_empty() || self.strides.len() == rank,
            "strides {:?} do not match rank {rank}",
            self.strides
        );
        ensure!(
            self.strides.iter().all(|&stride| stride > 0),
            "strides {:?} must all be positive",
            self.strides
        );
        for (axis, dim) in input.dims.iter().enumerate() {
            if let Dim::Static(extent) = dim {
                check_window(axis, self.starts[axis], self.span(axis), *extent)?;
            }
        }
        Ok(SliceResults {
            output: TensorType::new(input.dtype, self.sizes.iter().copied()),
        })
    }
}

/// `tensor.concat`: joins the inputs along `axis`.
///
/// - Concatenates one or more inputs along `axis`, a negative axis counting from the end. The
///   result's extent on `axis` is the sum of the inputs' extents.
/// - All dtypes MUST match, and all shapes MUST match on the axes other than `axis`.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.concat",
    operands(inputs: Variadic<Tensor>),
    results(output: Tensor),
    traits(Movement::ALL)
)]
pub struct Concat {
    pub axis: i64,
}

impl OpDef for Concat {
    fn result_types(
        &self,
        ins: ConcatOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<ConcatResults<OwnedTypes>> {
        let Some(&first) = ins.inputs.first() else {
            bail!("needs at least one input");
        };
        let axis = axis_index(self.axis, first.rank())?;
        for &input in &ins.inputs[1..] {
            ensure!(
                input.dtype == first.dtype,
                "inputs must have equal dtypes, got {first} and {input}"
            );
            ensure!(
                input.rank() == first.rank(),
                "inputs must have equal ranks, got {first} and {input}"
            );
            let off_axis_equal = (0..first.rank())
                .filter(|&a| a != axis)
                .all(|a| input.dims[a] == first.dims[a]);
            ensure!(
                off_axis_equal,
                "inputs must have equal extents off axis {axis}, got {first} and {input}"
            );
        }
        let mut fixed = 0usize;
        let mut dynamic: Vec<&Sym> = Vec::new();
        for input in &ins.inputs {
            match &input.dims[axis] {
                Dim::Static(n) => {
                    fixed = fixed
                        .checked_add(*n)
                        .ok_or_else(|| anyhow!("the joined extent on axis {axis} overflows"))?;
                }
                Dim::Dynamic(sym) => dynamic.push(sym),
            }
        }
        let joined = match dynamic.as_slice() {
            [] => Dim::Static(fixed),
            [sym] if fixed == 0 => Dim::Dynamic((*sym).clone()),
            _ => {
                let terms: Vec<String> = ins
                    .inputs
                    .iter()
                    .map(|t| t.dims[axis].to_string())
                    .collect();
                bail!(
                    "axis {axis} joins to {}, which is not one dimension",
                    terms.join(" + ")
                );
            }
        };
        let mut dims = first.dims.clone();
        dims[axis] = joined;
        Ok(ConcatResults {
            output: first.with_dims(dims),
        })
    }
}

/// An attribute in its text form, for error messages.
fn attr_text(attr: &impl Attr) -> String {
    let mut out = String::new();
    attr.print(&mut out);
    out
}

/// `tensor.bitcast`: the bits of the input read as elements of `to`.
///
/// - Each row of the last axis is one stream of bits: its elements in order, each from its least
///   significant bit up, as tensors store them. The last axis holds the same bits, so its extent
///   scales by `bits(input) / bits(to)`, which MUST give a whole number; the other axes stay.
///   Dtypes of equal width keep the shape.
/// - `i1` takes no part: its elements are bytes holding 0 or 1, not bits.
/// - It reads published packings in place: GPTQ and AWQ pack eight 4-bit codes into each int32,
///   and gpt-oss stores MXFP4 blocks and their e8m0 scales as bytes.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.bitcast",
    operands(input: Tensor),
    results(output: Tensor)
)]
pub struct Bitcast {
    pub to: DType,
}

impl OpDef for Bitcast {
    fn result_types(
        &self,
        ins: BitcastOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<BitcastResults<OwnedTypes>> {
        let input = ins.input;
        // An i1 element is one byte, 0 or 1, not a bit of a stream.
        ensure!(
            input.dtype != DType::I1 && self.to != DType::I1,
            "i1 elements have no bits to reinterpret"
        );
        let (from_bits, to_bits) = (input.dtype.bits(), self.to.bits());
        if from_bits == to_bits {
            return Ok(BitcastResults {
                output: input.with_dtype(self.to),
            });
        }
        let Some(Dim::Static(last)) = input.dims.last() else {
            bail!(
                "reinterpreting {input} as {} needs a static last axis",
                self.to
            );
        };
        let bits = last * from_bits;
        ensure!(
            bits % to_bits == 0,
            "the {bits} bits of each row of {input} are no whole number of {} elements",
            self.to
        );
        let mut dims = input.dims.clone();
        *dims.last_mut().expect("a last axis") = Dim::Static(bits / to_bits);
        Ok(BitcastResults {
            output: TensorType {
                dtype: self.to,
                dims,
            },
        })
    }
}

/// A bitcast to the input's own type is the input.
#[fold]
impl Fold for Bitcast {
    fn fold(
        &self,
        ins: BitcastOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<BitcastResults<FoldOut>> {
        Some(BitcastResults {
            output: same_type(ins.input, cx)?,
        })
    }
}

/// An element count as a product: a static factor times dynamic extents.
#[derive(Debug, PartialEq, Eq)]
struct Count {
    factor: usize,
    /// Sorted, and empty when `factor` is 0, so equal products compare equal.
    dynamic: Vec<Sym>,
}

impl Count {
    /// `None` when the static factor overflows.
    fn of<'a>(dims: impl IntoIterator<Item = &'a Dim>) -> Option<Count> {
        let mut factor = 1usize;
        let mut dynamic = Vec::new();
        for dim in dims {
            match dim {
                Dim::Static(n) => factor = factor.checked_mul(*n)?,
                Dim::Dynamic(sym) => dynamic.push(sym.clone()),
            }
        }
        if factor == 0 {
            dynamic.clear();
        }
        dynamic.sort();
        Some(Count { factor, dynamic })
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms: Vec<String> = Vec::new();
        if self.factor != 1 || self.dynamic.is_empty() {
            terms.push(self.factor.to_string());
        }
        terms.extend(
            self.dynamic
                .iter()
                .map(|sym| Dim::Dynamic(sym.clone()).to_string()),
        );
        f.write_str(&terms.join("*"))
    }
}

/// The input itself when the op does not change its type.
fn same_type(input: FoldIn<'_>, cx: &FoldCx<'_>) -> Option<Folded> {
    (input.ty == cx.result_type(0)).then_some(Folded::Value(input.value))
}

/// A reshape to the input's own type is the input, and a reshape of a constant is that constant
/// with the new extents, its elements in the same row-major order.
#[fold]
impl Fold for Reshape {
    fn fold(
        &self,
        ins: ReshapeOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<ReshapeResults<FoldOut>> {
        let input = ins.input;
        let output = match same_type(input, cx) {
            Some(output) => output,
            None => Folded::Constant(input.constant?.reshaped(cx.result_dims(0)?)),
        };
        Some(ReshapeResults { output })
    }
}

/// A transpose by the identity `perm` is the input.
#[fold]
impl Fold for Transpose {
    fn fold(
        &self,
        ins: TransposeOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<TransposeResults<FoldOut>> {
        self.perm
            .iter()
            .copied()
            .eq(0..self.perm.len())
            .then_some(TransposeResults {
                output: Folded::Value(ins.input.value),
            })
    }
}

/// A broadcast to the input's own type is the input.
#[fold]
impl Fold for BroadcastTo {
    fn fold(
        &self,
        ins: BroadcastToOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<BroadcastToResults<FoldOut>> {
        Some(BroadcastToResults {
            output: same_type(ins.input, cx)?,
        })
    }
}

/// A slice with the input's extents covers the whole input, so it is the input.
#[fold]
impl Fold for Slice {
    fn fold(
        &self,
        ins: SliceOperands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<SliceResults<FoldOut>> {
        Some(SliceResults {
            output: same_type(ins.input, cx)?,
        })
    }
}

/// A concat of one input is that input.
#[fold]
impl Fold for Concat {
    fn fold(
        &self,
        ins: ConcatOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
    ) -> Option<ConcatResults<FoldOut>> {
        let [input] = ins.inputs.as_slice() else {
            return None;
        };
        Some(ConcatResults {
            output: Folded::Value(input.value),
        })
    }
}
