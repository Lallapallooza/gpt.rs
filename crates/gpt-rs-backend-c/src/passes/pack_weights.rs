//! `c-pack-weights`: the weights of the dots that C kernels compute, put in the layout each kernel
//! reads.

use std::collections::HashSet;

use anyhow::Result;
use gpt_rs_ir::dialects::tensor::{DotGeneral, ExtractPatches, Reshape, Transpose};
use gpt_rs_ir::ir::{DType, Matched, Module, Value, ValueDef, ValueRef};
use gpt_rs_ir::passes::{pass, run_patterns, Pass};
use gpt_rs_ir::require;
use gpt_rs_ir::rewrite::{
    matched, pattern, Build, NoMatch, Pattern, PatternSet, Rewriter, Rewrites,
};

use crate::kernels::{LINEAR_PACK_MIN_ROWS, PANEL_N};

/// Puts the weights of the dots that C kernels compute in the layout each kernel reads: GEMM
/// weights in the GEMM's panels, and grouped convolution weights tap by tap.
///
/// Each layout is a transpose of the weight, after a reshape for the GEMM's panels. The dot then
/// contracts the weight in that layout, so the IR keeps its meaning. When the weight is static,
/// `outline-static` moves the layout to load time, and the kernel reads the laid-out parameter.
/// GEMM weights that are not static, activations among them, stay as they are, and the GEMM packs
/// the panels it needs on each call.
#[pass(
    name = "c-pack-weights",
    stats(packed = "weights laid out for a kernel")
)]
pub struct PackWeights;

fn populate_pack_weights_patterns(set: &mut PatternSet) {
    set.add(PackGemmWeight).add(GroupedConvWeight);
}

impl Pass for PackWeights {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_pack_weights_patterns)
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::PACKED)
    }
}

/// The extents of `value` when it is an f32 tensor with static extents.
fn f32_dims(value: ValueRef<'_>) -> Option<Vec<usize>> {
    let ty = value.tensor_type();
    let dims = ty.static_dims().filter(|_| ty.dtype == DType::F32)?;
    Some(dims.to_vec())
}

/// Rewrites an f32 `x [.., K] . w`, which codegen sends to the GEMM (a convolution's patches
/// among them), into `reshape(x . packed)`, where `packed` is `w` in panels of `PANEL_N` columns,
/// `[N / PANEL_N, K, PANEL_N]`. `w` is `[K, N]` or `[N, K]`. The packed dot gives
/// `[.., N / PANEL_N, PANEL_N]`, the same elements as `[.., N]`.
///
/// It packs only a static `w` whose packed form takes no more bytes than the parameters it is
/// computed from, so `outline-static` moves the packing to load time. It skips a `w` whose `N` is
/// not a multiple of `PANEL_N`, since the panels would pad it. A run-time `x` with one row, or with
/// fewer than `LINEAR_PACK_MIN_ROWS` rows against an `[N, K]` weight, leaves `w` unpacked: the GEMM
/// or the dot-product kernels read `w` in place.
#[pattern]
struct PackGemmWeight;

impl Pattern for PackGemmWeight {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let (input, weight) = (dot.lhs(), dot.rhs());
        require!(let Some(input_dims) = f32_dims(input), "the input is not f32 with static extents");
        require!(let Some(output_dims) = f32_dims(dot.output()), "the result is not f32 with static extents");
        let axes = dot.axes();
        let weight_is_nk = match axes.contract_rhs.as_slice() {
            [0] => false,
            [1] => true,
            _ => return Err("the weight contracts another axis".into()),
        };
        let last = input_dims.len().saturating_sub(1);
        let plain = axes.batch_lhs.is_empty()
            && axes.batch_rhs.is_empty()
            && input_dims.len() >= 2
            && axes.contract_lhs[..] == [last]
            && matches!(dot.accum, None | Some(DType::F32));
        require!(plain, "the dot is not a plain f32 product of rows");
        let rows: usize = input_dims[..last].iter().product();
        // `dot_kernel` sends fewer than LINEAR_PACK_MIN_ROWS rows of the input against an [N, K]
        // weight to the dot-product kernels, and the GEMM multiplies a single row in place.
        require!(
            rows > 1 && !(weight_is_nk && rows < LINEAR_PACK_MIN_ROWS),
            "the GEMM does not compute it"
        );
        // A dot of two static operands moves to load time whole.
        require!(!input.known_before_run(), "the dot is known before the run");
        require!(let Some(&[weight_rows, weight_cols]) = f32_dims(weight).as_deref(), "the weight is not f32 of rank 2");
        let (columns, depth) = if weight_is_nk {
            (weight_rows, weight_cols)
        } else {
            (weight_cols, weight_rows)
        };
        require!(columns % PANEL_N == 0, "the panels would pad the weight");
        require!(
            fits_parameters(weight),
            "the packed weight would not move to load time"
        );

        // [N, K] -> [N / P, P, K] -> [N / P, K, P], or [K, N] -> [K, N / P, P] -> [N / P, K, P].
        let (split_shape, perm) = if weight_is_nk {
            ([columns / PANEL_N, PANEL_N, depth], [0, 2, 1])
        } else {
            ([depth, columns / PANEL_N, PANEL_N], [1, 0, 2])
        };
        let packed = rewriter
            .on(weight)
            .reshape(split_shape)
            .transpose(perm)
            .value();
        let panels = DotGeneral {
            accum: dot.accum,
            out: dot.out,
            ..DotGeneral::contract([last as i64], [1])
        };
        let product = rewriter.create(panels, (input, packed));
        let result = rewriter.on(product).reshape(output_dims).value();
        rewriter.replace_op(&dot, result);
        Ok(())
    }
}

/// Whether `weight` is known before the run and takes no more bytes than the parameters it is
/// computed from. `outline-static` stores no other value.
fn fits_parameters(weight: ValueRef<'_>) -> bool {
    if !weight.known_before_run() {
        return false;
    }
    let module = weight.module();
    let bytes = |value: Value| {
        let ty = module.value_ref(value).tensor_type();
        ty.element_count()?.checked_mul(ty.dtype.size_in_bytes()?)
    };
    let mut seen = HashSet::new();
    let mut stack = vec![weight.id()];
    let mut parameter_bytes = Some(0usize);
    while let Some(value) = stack.pop() {
        if !seen.insert(value) {
            continue;
        }
        match module.def(value) {
            ValueDef::Arg(..) => {
                parameter_bytes = parameter_bytes
                    .zip(bytes(value))
                    .and_then(|(sum, size)| sum.checked_add(size));
            }
            ValueDef::Result(inst, _) => stack.extend(module.operands(inst)),
        }
    }
    parameter_bytes
        .zip(bytes(weight.id()))
        .is_some_and(|(have, need)| need <= have)
}

/// Lays out the weight of a grouped convolution for the grouped kernel. The dot of the patches
/// `[N, OH, OW, KH, KW, G, C_in / G]` with the weight `[G, C_out / G, C_in / G, KH, KW]`, batched
/// over the groups, becomes the same dot with the weight transposed to
/// `[KH, KW, C_in / G, G, C_out / G]`, the order of `[KH, KW, C_in / G, C_out]` that the kernel
/// reads. The result keeps its axes. `outline-static` moves the transpose of a static weight to
/// load time.
#[pattern]
struct GroupedConvWeight;

impl Pattern for GroupedConvWeight {
    type Root = DotGeneral;

    fn rewrite(
        &self,
        dot: Matched<'_, DotGeneral>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let grouped = dot.batch_lhs[..] == [5]
            && dot.batch_rhs[..] == [0]
            && dot.contract_lhs[..] == [3, 4, 6]
            && dot.contract_rhs[..] == [3, 4, 2];
        require!(grouped, "the dot is not a grouped convolution");
        matched!(let Reshape(ExtractPatches) = dot.lhs() else {
            return Err("the lhs is not reshaped patches".into());
        });
        let weight_rank_5 = f32_dims(dot.rhs()).is_some_and(|dims| dims.len() == 5);
        require!(weight_rank_5, "the weight is not f32 of rank 5");
        let laid_out_weight = rewriter.create(Transpose::new([3, 4, 2, 0, 1]), dot.rhs());
        let laid_out = DotGeneral {
            batch_rhs: [3].into_iter().collect(),
            contract_rhs: [0, 1, 2].into_iter().collect(),
            ..dot.attrs().clone()
        };
        let result = rewriter.create(laid_out, (dot.lhs(), laid_out_weight));
        rewriter.replace_op(&dot, result);
        Ok(())
    }
}
