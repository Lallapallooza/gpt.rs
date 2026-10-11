//! The type rules of the layout ops: the types each op gives, and what it rejects.

use std::fmt;

use anyhow::{anyhow, bail, ensure, Result};

use crate::attr::Attr;
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::shape::{axis_index, check_window};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{DType, Dim, Sym, TensorType};

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

impl OpDef for Transpose {
    fn result_types(
        &self,
        ins: TransposeOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<TransposeResults<OwnedTypes>> {
        ensure!(
            self.perm.is_permutation_of(ins.input.rank()),
            "perm {:?} is not a permutation of rank {}",
            &self.perm[..],
            ins.input.rank()
        );
        Ok(TransposeResults {
            output: ins.input.permuted(&self.perm),
        })
    }
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
        for (out_axis, out_dim) in self.shape.iter().enumerate() {
            let Some(axis) = self.source_axis(out_axis, input.rank()) else {
                continue;
            };
            let dim = &input.dims[axis];
            ensure!(
                dim.is_one() || dim == out_dim,
                "input axis {axis} has extent {dim}, which does not broadcast to {out_dim}"
            );
        }
        Ok(BroadcastToResults {
            output: input.with_dims(self.shape.clone()),
        })
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

/// An attribute in its text form, for error messages.
fn attr_text(attr: &impl Attr) -> String {
    let mut out = String::new();
    attr.print(&mut out);
    out
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
