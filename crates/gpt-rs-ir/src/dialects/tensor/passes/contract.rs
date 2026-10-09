//! Merging the contracted axes of a `tensor.dot_general` that its lhs splits out of one axis.

use anyhow::Result;
use smallvec::{smallvec, SmallVec};

use crate::dialects::tensor::ir::shape::free_axes;
use crate::dialects::tensor::{DotGeneral, Reshape, Transpose};
use crate::module::{Module, Value};
use crate::op::Matched;
use crate::passes::{pass, Greedy, Pass, Rewrites};
use crate::rewrite::{pattern, Pattern, PatternSet, Rewriter};
use crate::types::Dim;

/// `dot_general(reshape(x), w)` becomes `dot_general(x, w')` when the reshape only splits one
/// axis `a` of `x` into the contracted axes of the lhs, which are adjacent and in order, and `w`
/// is static. `w'` is `w` with its batch axes first, then its contracted axes in the order the lhs
/// contracts them, merged into one, then its free axes. The new dot contracts `a` with that axis.
///
/// The rewrite puts the weight of a convolution, `[C_out, C_in, KH, KW]`, in the layout
/// `[KH * KW * C_in, C_out]` that a matrix product of the patches reads. `w'` is static, so
/// `outline-static` computes it once at load time. Both dots sum the same products in the same
/// order, row-major over the contracted axes, so they agree bit for bit.
#[pattern]
struct MergeDotAxes;

/// The axis of `x` that `reshape` splits into `split.len()` axes starting at the same position,
/// when the reshape does nothing else.
fn split_axis(x: &[usize], reshaped: &[usize], split: &[usize]) -> Option<usize> {
    let &a = split.first()?;
    let s = split.len();
    let adjacent = split.iter().enumerate().all(|(i, &axis)| axis == a + i);
    let kept = x.len() + s - 1 == reshaped.len()
        && x[..a] == reshaped[..a]
        && x.get(a + 1..)? == reshaped.get(a + s..)?;
    let merged = x.get(a) == Some(&reshaped[a..a + s].iter().product());
    (s > 1 && adjacent && kept && merged).then_some(a)
}

fn static_dims(module: &Module, value: Value) -> Option<Vec<usize>> {
    module
        .ty(value)
        .as_tensor()?
        .static_dims()
        .map(SmallVec::into_vec)
}

impl Pattern for MergeDotAxes {
    type Root = DotGeneral;
    fn rewrite(&self, dot: Matched<DotGeneral>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let module = &*rw.module;
        let (lhs, w) = (dot.operands.lhs, dot.operands.rhs);
        let Some(reshape) = module.def_as::<Reshape>(lhs) else {
            return Ok(false);
        };
        let x = reshape.operands.input;
        let (Some(x_dims), Some(lhs_dims), Some(w_dims)) = (
            static_dims(module, x),
            static_dims(module, lhs),
            static_dims(module, w),
        ) else {
            return Ok(false);
        };
        let attrs = dot.attrs.axes(lhs_dims.len(), w_dims.len())?;
        let Some(a) = split_axis(&x_dims, &lhs_dims, &attrs.contract_lhs) else {
            return Ok(false);
        };
        if !module.is_static(w) {
            return Ok(false);
        }
        // The batch axes of the lhs after `a` shift down by the number of axes the merge removes.
        let merged_away = attrs.contract_lhs.len() - 1;
        let batch_lhs = attrs
            .batch_lhs
            .iter()
            .map(|&axis| if axis > a { axis - merged_away } else { axis } as i64)
            .collect();
        let w_free = free_axes(w_dims.len(), &attrs.batch_rhs, &attrs.contract_rhs);
        let perm: SmallVec<[usize; 4]> = attrs
            .batch_rhs
            .iter()
            .chain(&attrs.contract_rhs)
            .chain(&w_free)
            .copied()
            .collect();
        let k: usize = attrs
            .contract_rhs
            .iter()
            .map(|&axis| w_dims[axis])
            .product();
        let shape = attrs
            .batch_rhs
            .iter()
            .map(|&axis| w_dims[axis])
            .chain([k])
            .chain(w_free.iter().map(|&axis| w_dims[axis]))
            .map(Dim::Static)
            .collect();
        let batch = attrs.batch_rhs.len() as i64;
        let merged = DotGeneral {
            batch_lhs,
            batch_rhs: (0..batch).collect(),
            contract_lhs: smallvec![a as i64],
            contract_rhs: smallvec![batch],
            accum: dot.attrs.accum,
            out: dot.attrs.out,
        };
        let at = dot.inst;
        let transposed = rw.insert_before(at, Transpose { perm }, w)?.output;
        let w = rw.insert_before(at, Reshape { shape }, transposed)?.output;
        let out = rw.insert_before(at, merged, (x, w))?.output;
        rw.replace(at, &[out])?;
        Ok(true)
    }
}

/// Puts static dot operands in the layout the dot reads. When the lhs of a dot splits one axis into
/// its contracted axes, the dot reads that axis whole. The pass transposes and reshapes the static
/// weight to match.
#[pass(
    name = "merge-contracted-axes",
    stats(merged = "dots whose contracted axes merged")
)]
pub struct MergeContractedAxes;

/// Adds the patterns that merge the contracted axes of dots into one.
pub fn populate_merge_dot_axes_patterns(set: &mut PatternSet) {
    set.add(MergeDotAxes);
}

impl Pass for MergeContractedAxes {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_merge_dot_axes_patterns(&mut patterns);
        Rewrites {
            folds: false,
            patterns,
        }
    }

    fn run(&self, module: &mut Module) -> anyhow::Result<bool> {
        let report = Greedy::new(self.rewrites()).erase_dead().run(module)?;
        MergeContractedAxes::MERGED.add(report.rewritten);
        Ok(report.changed())
    }
}
