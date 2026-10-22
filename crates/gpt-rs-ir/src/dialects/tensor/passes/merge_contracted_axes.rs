//! Merging the contracted axes of a `tensor.dot_general` that its lhs splits out of one axis.

use anyhow::Result;

use crate::dialects::tensor::utils::shape::pick;
use crate::dialects::tensor::{DotAxes, DotGeneral, DotGeneralOp, Reshape, Side};
use crate::ir::Ranked;
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, Build, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

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

/// The axis of the input that `reshape` splits into `split.len()` axes starting at the same
/// position, when the reshape does nothing else.
fn split_axis(input_dims: &[usize], split_dims: &[usize], split: &[usize]) -> Option<usize> {
    let &first = split.first()?;
    let count = split.len();
    let adjacent = split
        .iter()
        .enumerate()
        .all(|(offset, &axis)| axis == first + offset);
    let others_kept = input_dims.len() + count - 1 == split_dims.len()
        && input_dims[..first] == split_dims[..first]
        && input_dims.get(first + 1..)? == split_dims.get(first + count..)?;
    let merged = input_dims.get(first) == Some(&split_dims[first..first + count].iter().product());
    (count > 1 && adjacent && others_kept && merged).then_some(first)
}

/// The weight laid out for the merged dot: its batch axes first, then its contracted axes in the
/// order the lhs contracts them, merged into one, then its free axes.
struct WeightLayout {
    perm: Vec<usize>,
    shape: Vec<usize>,
}

impl WeightLayout {
    fn merging(axes: &DotAxes, weight_dims: &[usize]) -> Self {
        let free = axes.free(Side::Rhs);
        // The batch axes, then the contracted axes in the order the lhs contracts them, then the
        // free axes.
        let perm = [&axes.batch_rhs[..], &axes.contract_rhs[..], &free[..]].concat();
        let contracted_extent = pick(weight_dims, &axes.contract_rhs).iter().product();
        let shape = [
            pick(weight_dims, &axes.batch_rhs),
            vec![contracted_extent],
            pick(weight_dims, &free),
        ]
        .concat();
        Self { perm, shape }
    }
}

/// `dot` contracting `merged_axis` of the input with the merged axis of the weight, with the same
/// accumulator and result dtype. The lhs batch axes after `merged_axis` move down by the number of
/// axes the merge removes.
fn merged_dot(dot: &DotGeneral, axes: &DotAxes, merged_axis: usize) -> DotGeneral {
    let removed_axes = axes.contract_lhs.len() - 1;
    let shift = |axis: usize| {
        if axis > merged_axis {
            axis - removed_axes
        } else {
            axis
        }
    };
    let batch_count = axes.batch_rhs.len() as i64;
    DotGeneral {
        batch_lhs: axes
            .batch_lhs
            .iter()
            .map(|&axis| shift(axis) as i64)
            .collect(),
        batch_rhs: (0..batch_count).collect(),
        contract_lhs: Ranked::from_iter([merged_axis as i64]),
        contract_rhs: Ranked::from_iter([batch_count]),
        ..dot.clone()
    }
}

impl Pattern for MergeDotAxes {
    type Root = DotGeneral;

    fn rewrite(&self, dot: DotGeneralOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let reshape = dot
            .lhs()
            .defining_op::<Reshape>()
            .ok_or("the lhs is not a reshape")?;
        let (input, weight) = (reshape.input(), dot.rhs());
        require!(weight.known_before_run(), "the weight changes at run time");
        require!(let Some(input_dims) = input.tensor_type().static_dims(), "the input has a dynamic extent");
        require!(let Some(split_dims) = dot.lhs().tensor_type().static_dims(), "the lhs has a dynamic extent");
        require!(let Some(weight_dims) = weight.tensor_type().static_dims(), "the weight has a dynamic extent");
        let axes = dot.axes();
        require!(
            let Some(merged_axis) = split_axis(&input_dims, &split_dims, &axes.contract_lhs),
            "the reshape does more than split one axis into the contracted axes"
        );

        let layout = WeightLayout::merging(&axes, &weight_dims);
        let merged_weight = rewriter
            .on(weight)
            .transpose(layout.perm)
            .reshape(layout.shape)
            .value();
        let merged = merged_dot(dot.attrs(), &axes, merged_axis);
        let product = rewriter.create(merged, (input, merged_weight));
        rewriter.replace_op(&dot, product);
        Ok(())
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

fn populate_merge_contracted_axes_patterns(set: &mut PatternSet) {
    set.add(MergeDotAxes);
}

impl Pass for MergeContractedAxes {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_merge_contracted_axes_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::MERGED)
    }
}
