use std::collections::BTreeMap;

use gpt_rs::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    index::InstId,
    optimizer::{FunctionPass, OptimizeContext, PassResult},
    pattern::{OpRewritePattern, PatternSet},
    ptir_utils::tensor_spec_static,
    rewriter::ProgramRewriter,
    spec::{
        CustomCallAttr, CustomCallSpec, DType, Dimension, ElementwiseBinaryOp, ExtractPatchesSpec,
        Function, Literal, Operand, Operation, ReshapeDim, ReshapeSpec, TransposeSpec, ValueId,
        ValueType,
    },
};
use gpt_rs::ops::functional::conv::Conv2dPattern;

use crate::targets::TARGET_CONV2D_NHWC_F32_V1;

use super::utils::{single_user, static_dims, tensor_spec_of};

fn conv2d_attrs_from_extract(
    spec: &ExtractPatchesSpec,
) -> Option<BTreeMap<String, CustomCallAttr>> {
    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return None;
    }

    let mut attrs = BTreeMap::new();
    attrs.insert(
        "window".into(),
        CustomCallAttr::I64Array(spec.window.iter().map(|&v| v as i64).collect()),
    );
    attrs.insert(
        "strides".into(),
        CustomCallAttr::I64Array(spec.strides.iter().map(|&v| v as i64).collect()),
    );
    attrs.insert(
        "dilation".into(),
        CustomCallAttr::I64Array(spec.dilation.iter().map(|&v| v as i64).collect()),
    );

    let pad_top = spec.padding[0].0 as i64;
    let pad_bottom = spec.padding[0].1 as i64;
    let pad_left = spec.padding[1].0 as i64;
    let pad_right = spec.padding[1].1 as i64;
    attrs.insert(
        "padding".into(),
        CustomCallAttr::I64Array(vec![pad_top, pad_bottom, pad_left, pad_right]),
    );

    // Operand metadata used by generic constant-folding passes: the weight operand is static.
    attrs.insert("static_operands".into(), CustomCallAttr::I64Array(vec![1]));

    Some(attrs)
}

fn match_bias_add(
    rewriter: &ProgramRewriter,
    base_value: ValueId,
) -> Option<(InstId, ValueId, ValueId)> {
    let add_inst = single_user(rewriter, base_value)?;
    let (add_op, add_operands) = (rewriter.op(add_inst), rewriter.operands(add_inst));
    let (Operation::ElementwiseBinary(op), [Operand::Value(lhs), Operand::Value(rhs)]) =
        (add_op, add_operands)
    else {
        return None;
    };
    if *op != ElementwiseBinaryOp::Add {
        return None;
    }

    let broadcast_value = if *lhs == base_value {
        *rhs
    } else if *rhs == base_value {
        *lhs
    } else {
        return None;
    };

    if rewriter.users_of(broadcast_value) != [add_inst] {
        return None;
    }

    let broadcast_inst = rewriter.inst_of(broadcast_value)?;
    let (bias_value, _) = match (
        rewriter.op(broadcast_inst),
        rewriter.operands(broadcast_inst),
    ) {
        (Operation::BroadcastTo(spec), [Operand::Value(bias_value)]) => (*bias_value, spec),
        _ => return None,
    };

    Some((add_inst, bias_value, rewriter.value_of(add_inst)))
}

struct LowerCConv2dNhwcF32;

impl OpRewritePattern<Conv2dPattern> for LowerCConv2dNhwcF32 {
    fn match_and_rewrite(&self, view: Conv2dPattern, rewriter: &mut ProgramRewriter) -> bool {
        if view.weight.is_some() || view.out_transpose.is_some() || view.out_reshape.is_some() {
            return false;
        }

        if !view.closure_report(rewriter).is_closed() {
            return false;
        }

        if !matches!(view.patches.spec.pad_value, Literal::Float(v) if v == 0.0) {
            return false;
        }
        let Some(attrs) = conv2d_attrs_from_extract(&view.patches.spec) else {
            return false;
        };

        let [Operand::Value(input_value)] = view.patches.operands.as_slice() else {
            return false;
        };
        let input_spec = match tensor_spec_of(rewriter, *input_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(in_dims) = static_dims(&input_spec) else {
            return false;
        };
        if in_dims.len() != 4 {
            return false;
        }
        let (n, c_in) = (in_dims[0], in_dims[3]);

        let output_patches_spec = match &view.patches.result_type {
            ValueType::Tensor(spec) if spec.dtype == DType::F32 => spec.clone(),
            _ => return false,
        };
        let Some(patch_dims) = static_dims(&output_patches_spec) else {
            return false;
        };
        if patch_dims.len() != 4 {
            return false;
        }

        let (k_h, k_w) = match view.patches.spec.window.as_slice() {
            [k_h, k_w] => (*k_h, *k_w),
            _ => return false,
        };
        let Some(k) = k_h.checked_mul(k_w).and_then(|v| v.checked_mul(c_in)) else {
            return false;
        };
        if patch_dims[3] != k {
            return false;
        }

        if view.patches_reshape.operands.as_slice() != [Operand::Value(view.patches.result)] {
            return false;
        }
        let patches_spec = match tensor_spec_of(rewriter, view.patches_reshape.result) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(patches_dims) = static_dims(&patches_spec) else {
            return false;
        };
        if patches_dims.as_slice() != [n, patch_dims[1], patch_dims[2], k_h, k_w, c_in] {
            return false;
        }

        let [Operand::Value(lhs), Operand::Value(rhs)] = view.out.operands.as_slice() else {
            return false;
        };
        if *lhs != view.patches_reshape.result {
            return false;
        }
        if !view.out.spec.batch_lhs.is_empty()
            || !view.out.spec.batch_rhs.is_empty()
            || view.out.spec.contract_lhs.as_slice() != [3, 4, 5]
            || view.out.spec.contract_rhs.as_slice() != [2, 3, 1]
        {
            return false;
        }
        let weight_value = *rhs;
        let dot_value = view.out.result;

        let weight_spec = match tensor_spec_of(rewriter, weight_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(weight_dims) = static_dims(&weight_spec) else {
            return false;
        };
        let [c_out, w_c_in, w_k_h, w_k_w] = weight_dims.as_slice() else {
            return false;
        };
        let (c_out, w_c_in, w_k_h, w_k_w) = (*c_out, *w_c_in, *w_k_h, *w_k_w);
        if w_c_in != c_in || w_k_h != k_h || w_k_w != k_w {
            return false;
        }

        let out_spec = match tensor_spec_of(rewriter, dot_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(out_dims) = static_dims(&out_spec) else {
            return false;
        };
        if out_dims.as_slice() != [n, patch_dims[1], patch_dims[2], c_out] {
            return false;
        }

        let (output_inst, output_value, bias_value) = if view.out_add.is_some() {
            match match_bias_add(rewriter, dot_value) {
                Some((add_inst, bias_value, add_result)) => {
                    (add_inst, add_result, Some(bias_value))
                }
                None => return false,
            }
        } else {
            (view.out.root, dot_value, None)
        };
        if view.output() != output_value {
            return false;
        }

        let output_ty = match rewriter.type_of(output_value) {
            Some(ty) => ty.clone(),
            None => return false,
        };

        let transpose_dims = [k_h, k_w, c_in, c_out];
        let packed_dims = [k, c_out];
        let transpose_spec = tensor_spec_static(weight_spec.dtype, &transpose_dims);
        let packed_spec = tensor_spec_static(weight_spec.dtype, &packed_dims);

        let transpose_op = Operation::Transpose(TransposeSpec {
            perm: vec![2, 3, 1, 0],
        });
        let Ok((_t_inst, t_value)) = rewriter.insert_before(
            output_inst,
            transpose_op,
            vec![Operand::Value(weight_value)],
            ValueType::Tensor(transpose_spec),
        ) else {
            return false;
        };

        let reshape_op = Operation::Reshape(ReshapeSpec {
            new_shape: vec![
                ReshapeDim::Explicit(Dimension::Static(k)),
                ReshapeDim::Explicit(Dimension::Static(c_out)),
            ],
        });
        let Ok((_r_inst, packed_weight_value)) = rewriter.insert_before(
            output_inst,
            reshape_op,
            vec![Operand::Value(t_value)],
            ValueType::Tensor(packed_spec),
        ) else {
            return false;
        };

        let mut operands = vec![
            Operand::Value(*input_value),
            Operand::Value(packed_weight_value),
        ];
        if let Some(bias_value) = bias_value {
            operands.push(Operand::Value(bias_value));
        }

        let op = Operation::CustomCall(CustomCallSpec {
            target: TARGET_CONV2D_NHWC_F32_V1.to_string(),
            attrs,
        });

        let Ok((_new_inst, new_value)) =
            rewriter.insert_before(output_inst, op, operands, output_ty)
        else {
            return false;
        };

        rewriter.replace_all_uses(output_value, new_value);
        for result_id in &mut rewriter.func.result_ids {
            if *result_id == output_value {
                *result_id = new_value;
            }
        }

        if let Some(old_inst) = rewriter.inst_of(output_value) {
            if rewriter.users_of(output_value).is_empty() {
                rewriter.erase_inst(old_inst);
            }
        }

        true
    }
}

pub struct CConv2dCustomCallFusionPass {
    config: GreedyConfig,
}

impl CConv2dCustomCallFusionPass {
    const NAME: &'static str = "c-conv2d-custom-call-fusion";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for CConv2dCustomCallFusionPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl FunctionPass<crate::CBackend> for CConv2dCustomCallFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::CBackend>,
    ) -> PassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<Conv2dPattern, _>(LowerCConv2dNhwcF32);
        let frozen = patterns.freeze();
        let stats = apply_patterns_and_fold_greedily(function, &frozen, &self.config);
        PassResult {
            changed: stats.applied > 0 || stats.dce_removed > 0,
            iterations: stats.iterations,
            rewrites_applied: stats.applied,
            erased_insts: stats.dce_removed,
        }
    }
}
