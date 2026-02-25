use std::collections::BTreeMap;

use gpt_rs::backend::index::InstId;
use gpt_rs::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use gpt_rs::backend::pattern::OperationView;
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{
    CustomCallAttr, CustomCallSpec, DType, Function, Operand, Operation, ReduceKind, TensorSpec,
    ValueId, ValueType,
};
use gpt_rs::ops::functional::normalization::LayerNormPattern;

use super::rewrite_utils::{
    erase_insts_if_dead, operand_value, replace_value_and_results, scalar_f32_from_operand,
    tensor_spec_of,
};
use crate::targets::TARGET_LAYER_NORM_FUSED_F32_V1;

pub struct TritonLayerNormFusionPass;

impl TritonLayerNormFusionPass {
    const NAME: &'static str = "triton-layer-norm-fusion";
}

impl FunctionPass<crate::TritonBackend> for TritonLayerNormFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::TritonBackend>,
    ) -> PassResult {
        let mut rewriter = match ProgramRewriter::new(function) {
            Ok(rewriter) => rewriter,
            Err(_) => return PassResult::default(),
        };

        let mut changed = false;
        let mut rewrites = 0usize;
        let mut erased = 0usize;
        let insts = rewriter.insts_in_order();
        for inst in insts {
            if !rewriter.contains(inst) {
                continue;
            }
            let Some(view) = LayerNormPattern::extract(inst, &rewriter) else {
                continue;
            };
            gpt_rs::profiling::cache_event("triton_layer_norm_pattern_match");
            let Some(rewrite) = match_layer_norm(&rewriter, &view) else {
                continue;
            };

            let mut attrs = BTreeMap::new();
            attrs.insert(
                "axis".to_string(),
                CustomCallAttr::I64(i64::try_from(rewrite.axis).unwrap_or(-1)),
            );
            attrs.insert(
                "eps".to_string(),
                CustomCallAttr::F64(f64::from(rewrite.eps)),
            );

            let op = Operation::CustomCall(CustomCallSpec {
                target: TARGET_LAYER_NORM_FUSED_F32_V1.to_string(),
                attrs,
            });
            let operands = vec![
                Operand::Value(rewrite.x),
                Operand::Value(rewrite.gamma),
                Operand::Value(rewrite.beta),
            ];
            let output_type = ValueType::Tensor(rewrite.out_spec.clone());
            let Ok((_new_inst, new_value)) =
                rewriter.insert_before(inst, op, operands, output_type)
            else {
                continue;
            };

            replace_value_and_results(&mut rewriter, rewrite.output, new_value);
            erased += erase_insts_if_dead(&mut rewriter, rewrite.roots.as_slice());
            gpt_rs::profiling::cache_event("triton_layer_norm_fused");

            changed = true;
            rewrites += 1;
        }

        PassResult {
            changed,
            iterations: 1,
            rewrites_applied: rewrites,
            erased_insts: erased,
        }
    }
}

struct LayerNormRewrite {
    x: ValueId,
    gamma: ValueId,
    beta: ValueId,
    output: ValueId,
    out_spec: TensorSpec,
    axis: usize,
    eps: f32,
    roots: Vec<InstId>,
}

fn match_layer_norm(
    rewriter: &ProgramRewriter,
    view: &LayerNormPattern,
) -> Option<LayerNormRewrite> {
    let output = view.output();
    let out_spec = tensor_spec_of(rewriter, output)
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_out_spec"))?;
    if out_spec.dtype != DType::F32 {
        return reject_layer_norm("triton_layer_norm_reject_out_dtype");
    }
    let out_dims = out_spec
        .shape
        .static_dims()
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_dynamic_output_shape"))?;
    if out_dims.is_empty() {
        return reject_layer_norm("triton_layer_norm_reject_rank0");
    }
    let axis = out_dims.len() - 1;

    if view.sum.spec.kind != ReduceKind::Sum
        || !view.sum.spec.keepdims
        || view.sum.spec.axes.as_slice() != [axis]
    {
        return reject_layer_norm("triton_layer_norm_reject_sum_shape");
    }
    if view.var_sum.spec.kind != ReduceKind::Sum
        || !view.var_sum.spec.keepdims
        || view.var_sum.spec.axes.as_slice() != [axis]
    {
        return reject_layer_norm("triton_layer_norm_reject_var_sum_shape");
    }

    let x = operand_value(view.sum.operands.first())
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_x_operand"))?;
    let x_spec = tensor_spec_of(rewriter, x)
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_x_spec"))?;
    if x_spec != out_spec {
        return reject_layer_norm("triton_layer_norm_reject_x_out_mismatch");
    }

    let gamma = operand_value(view.gamma_broadcast.operands.first())
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_gamma_operand"))?;
    let beta = operand_value(view.beta_broadcast.operands.first())
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_beta_operand"))?;
    let gamma_spec = tensor_spec_of(rewriter, gamma)
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_gamma_spec"))?;
    let beta_spec = tensor_spec_of(rewriter, beta)
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_missing_beta_spec"))?;
    if gamma_spec.dtype != DType::F32 || beta_spec.dtype != DType::F32 {
        return reject_layer_norm("triton_layer_norm_reject_gamma_beta_dtype");
    }
    let feature_dim = out_dims[axis];
    if gamma_spec
        .shape
        .static_dims()
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_gamma_dynamic_shape"))?
        .as_slice()
        != [feature_dim]
        || beta_spec
            .shape
            .static_dims()
            .or_else(|| reject_layer_norm("triton_layer_norm_reject_beta_dynamic_shape"))?
            .as_slice()
            != [feature_dim]
    {
        return reject_layer_norm("triton_layer_norm_reject_gamma_beta_shape");
    }

    let var_mean_value = rewriter.value_of(view.var_mean.root);
    let lhs = view.var_eps.operands.first();
    let rhs = view.var_eps.operands.get(1);
    let eps_operand = match (operand_value(lhs), operand_value(rhs)) {
        (Some(value), _) if value == var_mean_value => rhs,
        (_, Some(value)) if value == var_mean_value => lhs,
        _ => return reject_layer_norm("triton_layer_norm_reject_var_eps_operands"),
    };
    let eps = scalar_f32_from_operand(rewriter, eps_operand)
        .or_else(|| reject_layer_norm("triton_layer_norm_reject_eps_literal"))?;
    if !eps.is_finite() || eps <= 0.0 {
        return reject_layer_norm("triton_layer_norm_reject_eps_value");
    }

    let roots = vec![
        view.output.root,
        view.beta_broadcast.root,
        view.gamma_broadcast.root,
        view.normalized.root,
        view.var_eps.root,
        view.var_mean.root,
        view.var_sum.root,
        view.centered_sq.root,
        view.centered.root,
        view.mean_broadcast.root,
        view.mean.root,
        view.sum.root,
    ];

    Some(LayerNormRewrite {
        x,
        gamma,
        beta,
        output,
        out_spec,
        axis,
        eps,
        roots,
    })
}

fn reject_layer_norm<T>(reason: &'static str) -> Option<T> {
    gpt_rs::profiling::cache_event(reason);
    None
}
