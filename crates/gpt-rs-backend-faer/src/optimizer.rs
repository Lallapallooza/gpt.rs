use std::collections::BTreeMap;
use std::sync::Arc;

use gpt_rs::backend::{
    driver::{apply_patterns_and_fold_greedily, GreedyConfig},
    index::InstId,
    optimizer::{FunctionPass, OptimizeContext, PassResult},
    pattern::{ElementwiseBinaryOpView, ExtractPatchesOpView, OpRewritePattern, PatternSet},
    pipeline::{BackendPipeline, PipelineBuilder},
    ptir_utils::tensor_spec_static,
    rewriter::ProgramRewriter,
    spec::{
        CustomCallAttr, CustomCallSpec, DType, Dimension, ElementwiseBinaryOp, ExtractPatchesSpec,
        Function, Literal, Operand, Operation, ReshapeDim, ReshapeSpec, TensorSpec, TransposeSpec,
        ValueId, ValueType,
    },
};
use gpt_rs::ops::functional::conv::Conv2dPattern;

pub const TARGET_CONV2D_NHWC_F32_V1: &str = "gpt_rs.faer.conv2d.nhwc.f32.v1";
pub const TARGET_DEPTHWISE_CONV2D_NHWC_F32_V1: &str = "gpt_rs.faer.depthwise_conv2d.nhwc.f32.v1";

fn single_user(rewriter: &ProgramRewriter, value: ValueId) -> Option<InstId> {
    let users = rewriter.users_of(value);
    if users.len() == 1 {
        Some(users[0])
    } else {
        None
    }
}

fn tensor_spec_of(rewriter: &ProgramRewriter, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}

fn static_dims(spec: &TensorSpec) -> Option<Vec<usize>> {
    spec.shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => Some(*v),
            Dimension::Dynamic(_) => None,
        })
        .collect()
}

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
    let add_view = ElementwiseBinaryOpView::new(add_inst, rewriter)?;
    if add_view.op != ElementwiseBinaryOp::Add {
        return None;
    }
    let [Operand::Value(lhs), Operand::Value(rhs)] = add_view.operands.as_slice() else {
        return None;
    };
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

    Some((add_inst, bias_value, add_view.result))
}

struct LowerFaerConv2dNhwcF32;

impl OpRewritePattern<Conv2dPattern> for LowerFaerConv2dNhwcF32 {
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

struct LowerFaerDepthwiseConv2dNhwcF32;

impl OpRewritePattern<ExtractPatchesOpView> for LowerFaerDepthwiseConv2dNhwcF32 {
    fn match_and_rewrite(
        &self,
        view: ExtractPatchesOpView,
        rewriter: &mut ProgramRewriter,
    ) -> bool {
        if !matches!(view.spec.pad_value, Literal::Float(v) if v == 0.0) {
            return false;
        }
        let Some(attrs) = conv2d_attrs_from_extract(&view.spec) else {
            return false;
        };

        let [Operand::Value(input_value)] = view.operands.as_slice() else {
            return false;
        };
        let input_spec = match tensor_spec_of(rewriter, *input_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(in_dims) = static_dims(&input_spec) else {
            return false;
        };
        let [n, _h, _w, c] = in_dims.as_slice() else {
            return false;
        };
        let (n, c) = (*n, *c);

        let output_patches_spec = match &view.result_type {
            ValueType::Tensor(spec) if spec.dtype == DType::F32 => spec.clone(),
            _ => return false,
        };
        let Some(patch_dims) = static_dims(&output_patches_spec) else {
            return false;
        };
        if patch_dims.len() != 4 {
            return false;
        }

        let (k_h, k_w) = match view.spec.window.as_slice() {
            [k_h, k_w] => (*k_h, *k_w),
            _ => return false,
        };
        let Some(khkw) = k_h.checked_mul(k_w) else {
            return false;
        };
        if patch_dims[3] != khkw.saturating_mul(c) {
            return false;
        }

        let reshape_patches_inst = match single_user(rewriter, view.result) {
            Some(inst) => inst,
            None => return false,
        };
        let reshape_patches_value = rewriter.value_of(reshape_patches_inst);
        if !matches!(rewriter.op(reshape_patches_inst), Operation::Reshape(_)) {
            return false;
        }
        if rewriter.operands(reshape_patches_inst) != [Operand::Value(view.result)] {
            return false;
        }
        let patches_spec = match tensor_spec_of(rewriter, reshape_patches_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(patches_dims) = static_dims(&patches_spec) else {
            return false;
        };
        // Depthwise portable lowering uses grouped conv with shape
        // [N, OH, OW, KH, KW, G=C, C_in/G=1].
        if patches_dims.as_slice() != [n, patch_dims[1], patch_dims[2], k_h, k_w, c, 1] {
            return false;
        }

        let dot_inst = match single_user(rewriter, reshape_patches_value) {
            Some(inst) => inst,
            None => return false,
        };
        let (weight_value, dot_value) = match (rewriter.op(dot_inst), rewriter.operands(dot_inst)) {
            (Operation::DotGeneral(spec), [Operand::Value(lhs), Operand::Value(rhs)])
                if *lhs == reshape_patches_value
                    && spec.batch_lhs.as_slice() == [5]
                    && spec.batch_rhs.as_slice() == [0]
                    && spec.contract_lhs.as_slice() == [3, 4, 6]
                    && spec.contract_rhs.as_slice() == [3, 4, 2] =>
            {
                (*rhs, rewriter.value_of(dot_inst))
            }
            _ => return false,
        };

        // Expect `weight_value` to be a reshape of the canonical OIHW weight.
        let Some(weight_reshape_inst) = rewriter.inst_of(weight_value) else {
            return false;
        };
        let original_weight_value = match (
            rewriter.op(weight_reshape_inst),
            rewriter.operands(weight_reshape_inst),
        ) {
            (Operation::Reshape(_), [Operand::Value(value)]) => *value,
            _ => return false,
        };

        let original_weight_spec = match tensor_spec_of(rewriter, original_weight_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(original_weight_dims) = static_dims(&original_weight_spec) else {
            return false;
        };
        let [w_c_out, w_c_in_g, w_k_h, w_k_w] = original_weight_dims.as_slice() else {
            return false;
        };
        let (w_c_out, w_c_in_g, w_k_h, w_k_w) = (*w_c_out, *w_c_in_g, *w_k_h, *w_k_w);
        if w_c_out != c || w_c_in_g != 1 || w_k_h != k_h || w_k_w != k_w {
            return false;
        }

        let reshaped_weight_spec = match tensor_spec_of(rewriter, weight_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(reshaped_weight_dims) = static_dims(&reshaped_weight_spec) else {
            return false;
        };
        if reshaped_weight_dims.as_slice() != [c, 1, 1, k_h, k_w] {
            return false;
        }

        let transpose_inst = match single_user(rewriter, dot_value) {
            Some(inst) => inst,
            None => return false,
        };
        let transpose_value = rewriter.value_of(transpose_inst);
        match (
            rewriter.op(transpose_inst),
            rewriter.operands(transpose_inst),
        ) {
            (Operation::Transpose(spec), [Operand::Value(src)])
                if *src == dot_value && spec.perm.as_slice() == [1, 2, 3, 0, 4] => {}
            _ => return false,
        }

        let reshape_out_inst = match single_user(rewriter, transpose_value) {
            Some(inst) => inst,
            None => return false,
        };
        let reshape_out_value = rewriter.value_of(reshape_out_inst);
        if !matches!(rewriter.op(reshape_out_inst), Operation::Reshape(_)) {
            return false;
        }
        if rewriter.operands(reshape_out_inst) != [Operand::Value(transpose_value)] {
            return false;
        }

        let output_spec = match tensor_spec_of(rewriter, reshape_out_value) {
            Some(spec) if spec.dtype == DType::F32 => spec,
            _ => return false,
        };
        let Some(out_dims) = static_dims(&output_spec) else {
            return false;
        };
        if out_dims.as_slice() != [n, patch_dims[1], patch_dims[2], c] {
            return false;
        }

        let (output_inst, output_value, bias_value) =
            match match_bias_add(rewriter, reshape_out_value) {
                Some((add_inst, bias_value, add_result)) => {
                    (add_inst, add_result, Some(bias_value))
                }
                None => (reshape_out_inst, reshape_out_value, None),
            };

        let output_ty = match rewriter.type_of(output_value) {
            Some(ty) => ty.clone(),
            None => return false,
        };

        let packed_dims = [c, khkw];
        let packed_spec = tensor_spec_static(original_weight_spec.dtype, &packed_dims);
        let reshape_op = Operation::Reshape(ReshapeSpec {
            new_shape: vec![
                ReshapeDim::Explicit(Dimension::Static(c)),
                ReshapeDim::Explicit(Dimension::Static(khkw)),
            ],
        });
        let Ok((_r_inst, packed_weight_value)) = rewriter.insert_before(
            output_inst,
            reshape_op,
            vec![Operand::Value(original_weight_value)],
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
            target: TARGET_DEPTHWISE_CONV2D_NHWC_F32_V1.to_string(),
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

/// Lowers portable `conv2d` graphs into explicit faer `custom_call` kernels.
pub struct FaerCustomCallFusionPass {
    config: GreedyConfig,
}

impl FaerCustomCallFusionPass {
    const NAME: &'static str = "faer-custom-call-fusion";

    pub fn new(config: GreedyConfig) -> Self {
        Self { config }
    }
}

impl Default for FaerCustomCallFusionPass {
    fn default() -> Self {
        Self::new(GreedyConfig::default())
    }
}

impl FunctionPass<crate::FaerPortableBackend> for FaerCustomCallFusionPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(
        &self,
        function: &mut Function,
        _cx: &mut OptimizeContext<crate::FaerPortableBackend>,
    ) -> PassResult {
        let mut patterns = PatternSet::new();
        patterns.insert_view::<ExtractPatchesOpView, _>(LowerFaerDepthwiseConv2dNhwcF32);
        patterns.insert_view::<Conv2dPattern, _>(LowerFaerConv2dNhwcF32);
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

#[derive(Default)]
pub struct FaerPipeline;

impl BackendPipeline<crate::FaerPortableBackend> for FaerPipeline {
    fn populate_legalize(&self, p: &mut PipelineBuilder<crate::FaerPortableBackend>) {
        p.pass(Arc::new(FaerCustomCallFusionPass::default()));
    }
}
