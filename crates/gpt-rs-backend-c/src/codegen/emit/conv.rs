use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    CustomCallSpec, DType, ExtractPatchesSpec, Instruction, Operand, Operation, TensorSpec,
};

use super::super::profile::{backend_operation_label, emit_profiled_op, register_op_profile_unary};
use super::super::types::{MatmulCacheEntry, ValueInfo, ValueKey};
use super::super::utils::{
    dims_usize, emit_loops_with_indices, format_f32, linear_index_expr, literal_to_f32_scalar,
    push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_input_index, operand_spec, output_info,
    LiteralCache,
};
use super::{custom_call_attr_i64_array, EmitContext};

pub(super) fn emit_instruction(
    inst: &Instruction,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<bool> {
    let EmitContext {
        module,
        value_infos,
        literal_cache,
        matmul_profile,
        ..
    } = ctx;

    match &inst.op {
        Operation::ExtractPatches(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "extract_patches input must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "extract_patches output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.spec, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_extract_patches(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &in_spec,
                    spec,
                )
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

fn emit_extract_patches(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &ExtractPatchesSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims.len() < 3 {
        return Err(ConversionError::new("extract_patches requires rank >= 3"));
    }
    let spatial_rank = in_dims.len() - 2;
    if spec.window.len() != spatial_rank
        || spec.strides.len() != spatial_rank
        || spec.dilation.len() != spatial_rank
        || spec.padding.len() != spatial_rank
    {
        return Err(ConversionError::new("extract_patches spec rank mismatch"));
    }
    let channels = *in_dims.last().expect("channels dim");
    let window_elems: usize = spec.window.iter().product();
    let expected_last = window_elems
        .checked_mul(channels)
        .ok_or_else(|| ConversionError::new("extract_patches window overflow"))?;
    if *out_dims.last().unwrap_or(&0) != expected_last {
        return Err(ConversionError::new(
            "extract_patches output shape mismatch",
        ));
    }

    let pad_value = literal_to_f32_scalar(&spec.pad_value)?;
    let pad = format_f32(pad_value);

    emit_loops_with_indices(module, &out_dims, 2, "o", |module, indices, indent| {
        let out_idx = linear_index_expr(&out_dims, indices);
        let patch_index = if indices.is_empty() {
            "0".to_string()
        } else {
            indices.last().unwrap().clone()
        };
        let header = format!(
            r#"
                size_t patch = {patch_index};
                size_t window_linear = patch / {channels};
                size_t channel = patch % {channels};
            "#
        );
        push_block(module, indent, &header);

        let mut rem_var = "window_linear".to_string();
        let mut window_indices = Vec::with_capacity(spatial_rank);
        for (idx, dim) in spec.window.iter().enumerate() {
            let stride: usize = spec.window[idx + 1..].iter().product();
            let w_name = format!("w{idx}");
            if stride > 1 {
                let rem_name = format!("rem{idx}");
                let block = format!(
                    r#"
                        size_t {w_name} = {rem_var} / {stride};
                        size_t {rem_name} = {rem_var} % {stride};
                    "#
                );
                push_block(module, indent, &block);
                rem_var = rem_name;
            } else {
                push_block(
                    module,
                    indent,
                    &format!(
                        r#"
                            size_t {w_name} = {rem_var};
                        "#
                    ),
                );
                rem_var = w_name.clone();
            }
            if *dim == 1 {
                push_block(
                    module,
                    indent,
                    &format!(
                        r#"
                            {w_name} = 0;
                        "#
                    ),
                );
            }
            window_indices.push(w_name);
        }

        let mut in_indices = Vec::with_capacity(in_dims.len());
        in_indices.push(indices[0].clone());
        let mut bounds_checks = Vec::new();
        for d in 0..spatial_rank {
            let in_name = format!("sp{d}");
            let out_index = indices[d + 1].clone();
            let pad_before = spec.padding[d].0 as i64;
            let stride = spec.strides[d] as i64;
            let dilation = spec.dilation[d] as i64;
            let w_name = &window_indices[d];
            push_block(
                module,
                indent,
                &format!(
                    "int64_t {in_name} = (int64_t){out_index} * {stride} - {pad_before} + (int64_t){w_name} * {dilation};"
                ),
            );
            let dim = in_dims[d + 1];
            bounds_checks.push(format!("{in_name} < 0 || {in_name} >= {dim}"));
            in_indices.push(in_name);
        }
        in_indices.push("channel".to_string());

        if !bounds_checks.is_empty() {
            let bounds = bounds_checks.join(" || ");
            let block = format!(
                r#"
                    if ({bounds}) {{
                      {out}[{out_idx}] = {pad};
                    }} else {{
                "#
            );
            push_block(module, indent, &block);
        }
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        if bounds_checks.is_empty() {
            push_block(
                module,
                indent,
                &format!(
                    r#"
                        {out}[{out_idx}] = {input}[{in_idx}];
                    "#
                ),
            );
        } else {
            push_block(
                module,
                indent + 1,
                &format!(
                    r#"
                            {out}[{out_idx}] = {input}[{in_idx}];
                        "#
                ),
            );
        }
        if !bounds_checks.is_empty() {
            push_block(
                module,
                indent,
                r#"
                    }
                "#,
            );
        }
    });

    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_custom_call_conv2d(
    module: &mut String,
    spec: &CustomCallSpec,
    operands: &[Operand],
    out_info: &ValueInfo,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
    op_id: usize,
    mut matmul_caches: Option<&mut Vec<MatmulCacheEntry>>,
) -> ConversionResult<()> {
    ensure_dtype(out_info.spec.dtype, DType::F32, "conv2d output must be f32")?;
    if operands.len() < 2 || operands.len() > 3 {
        return Err(ConversionError::new(
            "conv2d custom_call expects 2 or 3 operands",
        ));
    }
    let in_dtype = operand_dtype(&operands[0], value_infos)?;
    let w_dtype = operand_dtype(&operands[1], value_infos)?;
    ensure_dtype(in_dtype, DType::F32, "conv2d input must be f32")?;
    ensure_dtype(w_dtype, DType::F32, "conv2d weights must be f32")?;
    if operands.len() == 3 {
        let b_dtype = operand_dtype(&operands[2], value_infos)?;
        ensure_dtype(b_dtype, DType::F32, "conv2d bias must be f32")?;
    }

    let window = custom_call_attr_i64_array(spec, "window")?;
    let strides = custom_call_attr_i64_array(spec, "strides")?;
    let dilation = custom_call_attr_i64_array(spec, "dilation")?;
    let padding = custom_call_attr_i64_array(spec, "padding")?;
    if window.len() != 2 || strides.len() != 2 || dilation.len() != 2 || padding.len() != 4 {
        return Err(ConversionError::new(
            "conv2d custom_call attribute lengths invalid",
        ));
    }

    let in_spec = operand_spec(&operands[0], value_infos)?;
    let weight_spec = operand_spec(&operands[1], value_infos)?;
    let in_dims = dims_usize(&in_spec)?;
    let weight_dims = dims_usize(&weight_spec)?;
    let out_dims = dims_usize(&out_info.spec)?;

    if in_dims.len() != 4 || out_dims.len() != 4 {
        return Err(ConversionError::new(
            "conv2d custom_call expects rank-4 NHWC tensors",
        ));
    }
    if weight_dims.len() != 2 {
        return Err(ConversionError::new(
            "conv2d custom_call expects packed weight shape [K, C_out]",
        ));
    }

    let k_h = usize::try_from(window[0])
        .map_err(|_| ConversionError::new("conv2d window must be non-negative"))?;
    let k_w = usize::try_from(window[1])
        .map_err(|_| ConversionError::new("conv2d window must be non-negative"))?;
    let stride_h = usize::try_from(strides[0])
        .map_err(|_| ConversionError::new("conv2d strides must be non-negative"))?;
    let stride_w = usize::try_from(strides[1])
        .map_err(|_| ConversionError::new("conv2d strides must be non-negative"))?;
    let dilation_h = usize::try_from(dilation[0])
        .map_err(|_| ConversionError::new("conv2d dilation must be non-negative"))?;
    let dilation_w = usize::try_from(dilation[1])
        .map_err(|_| ConversionError::new("conv2d dilation must be non-negative"))?;
    let pad_top = usize::try_from(padding[0])
        .map_err(|_| ConversionError::new("conv2d padding must be non-negative"))?;
    let pad_left = usize::try_from(padding[2])
        .map_err(|_| ConversionError::new("conv2d padding must be non-negative"))?;

    let (n, in_h, in_w, c_in) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let (out_h, out_w, c_out) = (out_dims[1], out_dims[2], out_dims[3]);
    if out_dims[0] != n {
        return Err(ConversionError::new(
            "conv2d custom_call batch dimension mismatch",
        ));
    }

    let k = k_h
        .checked_mul(k_w)
        .and_then(|v| v.checked_mul(c_in))
        .ok_or_else(|| ConversionError::new("conv2d kernel size overflow"))?;
    if weight_dims[0] != k || weight_dims[1] != c_out {
        return Err(ConversionError::new("conv2d packed weight shape mismatch"));
    }

    let input = operand_expr(&operands[0], value_infos, module, literal_cache)?;
    let weight = operand_expr(&operands[1], value_infos, module, literal_cache)?;
    let bias = if operands.len() == 3 {
        Some(operand_expr(
            &operands[2],
            value_infos,
            module,
            literal_cache,
        )?)
    } else {
        None
    };

    let bias_expr = bias
        .as_ref()
        .map(|bias| format!("(const float*){bias}"))
        .unwrap_or_else(|| "NULL".to_string());
    let out_var = &out_info.var;
    let header = format!(
        r#"
            {{
              const float* in = (const float*){input};
              const float* w = (const float*){weight};
              const float* b = {bias_expr};
              float* out = {out_var};
        "#
    );
    push_block(module, 1, &header);
    let weight_input_index = matmul_caches
        .as_ref()
        .and_then(|_| operand_input_index(&operands[1], value_infos));
    let mut use_cache = false;
    if let (Some(caches), Some(rhs_index)) = (matmul_caches.as_mut(), weight_input_index) {
        (*caches).push(MatmulCacheEntry {
            op_id,
            rhs_index,
            n: c_out,
            k,
        });
        use_cache = true;
    }

    let call = if use_cache {
        format!(
            r#"
                gpt_rs_c_conv2d_nhwc_f32_cached_b(
                  in, w, b, out,
                  {n}, {in_h}, {in_w}, {c_in},
                  {out_h}, {out_w}, {c_out},
                  {k_h}, {k_w},
                  {stride_h}, {stride_w},
                  {dilation_h}, {dilation_w},
                  {pad_top}, {pad_left},
                  &gpt_rs_bcache_{op_id}
                );
            "#
        )
    } else {
        format!(
            r#"
                gpt_rs_c_conv2d_nhwc_f32(
                  in, w, b, out,
                  {n}, {in_h}, {in_w}, {c_in},
                  {out_h}, {out_w}, {c_out},
                  {k_h}, {k_w},
                  {stride_h}, {stride_w},
                  {dilation_h}, {dilation_w},
                  {pad_top}, {pad_left}
                );
            "#
        )
    };
    push_block(module, 2, &call);
    push_block(module, 1, "}");

    Ok(())
}
