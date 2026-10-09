use std::collections::HashMap;

use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::{ExtractPatches, TensorOp};
use gpt_rs_ir::module::Value;
use gpt_rs_ir::types::{DType, TensorType};

use crate::kernels::PANEL_N;

use super::super::profile::register_op_profile_unary;
use super::super::types::{Instruction, Kind, ValueInfo};
use super::super::utils::{
    dims_usize, emit_parallel_loops_with_indices, format_f32, linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_type, output_info,
};
use super::EmitContext;

pub(super) fn emit_instruction(
    inst: &Instruction<'_>,
    ctx: &mut EmitContext<'_>,
) -> Result<Option<usize>> {
    let EmitContext {
        module,
        value_infos,
        matmul_profile,
        ..
    } = ctx;

    let &Kind::Op(op) = &inst.kind else {
        return Ok(None);
    };
    let op_id = match op {
        TensorOp::ExtractPatches(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "extract_patches input must be f32")?;
            ensure_dtype(
                out_info.ty.dtype,
                DType::F32,
                "extract_patches output must be f32",
            )?;
            let label = inst.label;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.ty, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_extract_patches(module, &out_info.var, &input, &out_info.ty, &in_spec, spec)?;
            op_id
        }
        _ => return Ok(None),
    };

    Ok(Some(op_id))
}

fn emit_extract_patches(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorType,
    in_spec: &TensorType,
    spec: &ExtractPatches,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims.len() < 3 {
        return Err(anyhow!("extract_patches requires rank >= 3"));
    }
    let spatial_rank = in_dims.len() - 2;
    if spec.window.len() != spatial_rank
        || spec.strides.len() != spatial_rank
        || spec.dilation.len() != spatial_rank
        || spec.padding.len() != spatial_rank
    {
        return Err(anyhow!("extract_patches spec rank mismatch"));
    }
    let channels = *in_dims.last().expect("channels dim");
    let window_elems: usize = spec.window.iter().product();
    let expected_last = window_elems
        .checked_mul(channels)
        .ok_or_else(|| anyhow!("extract_patches window overflow"))?;
    if *out_dims.last().unwrap_or(&0) != expected_last {
        return Err(anyhow!("extract_patches output shape mismatch"));
    }

    let pad = format_f32(spec.pad_value.0 as f32);

    emit_parallel_loops_with_indices(module, &out_dims, 2, "o", |module, indices, indent| {
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
/// Emits a [`Kind::Conv2d`] instruction: the convolution kernel on `x` and `w`, plus the bias.
pub(super) fn emit_conv2d(
    module: &mut String,
    patches: &ExtractPatches,
    operands: &[Value],
    out_info: &ValueInfo,
    value_infos: &HashMap<Value, ValueInfo>,
) -> Result<()> {
    ensure_dtype(out_info.ty.dtype, DType::F32, "conv2d output must be f32")?;
    if operands.len() < 2 || operands.len() > 3 {
        return Err(anyhow!("a convolution reads [x, w] or [x, w, bias]"));
    }
    for operand in operands {
        ensure_dtype(
            operand_dtype(operand, value_infos)?,
            DType::F32,
            "a convolution is f32",
        )?;
    }
    let (
        &[k_h, k_w],
        &[stride_h, stride_w],
        &[dilation_h, dilation_w],
        &[(pad_top, _), (pad_left, _)],
    ) = (
        patches.window.as_slice(),
        patches.strides.as_slice(),
        patches.dilation.as_slice(),
        patches.padding.as_slice(),
    )
    else {
        return Err(anyhow!("a convolution has two spatial axes"));
    };

    let in_dims = dims_usize(&operand_type(&operands[0], value_infos)?)?;
    let weight_dims = dims_usize(&operand_type(&operands[1], value_infos)?)?;
    let out_dims = dims_usize(&out_info.ty)?;
    let (weight_k, c_out, packed) = match *weight_dims.as_slice() {
        [k, c_out] => (k, c_out, 0),
        [panels, k, PANEL_N] => (k, panels * PANEL_N, 1),
        _ => return Err(anyhow!("conv2d weight must be [K, C_out] or packed")),
    };
    let &[n, in_h, in_w, c_in] = in_dims.as_slice() else {
        return Err(anyhow!("a convolution reads an NHWC input"));
    };
    // The result is [N, OH, OW, C_out], or [N, OH, OW, C_out / PANEL_N, PANEL_N] in the same
    // order when the weight is packed.
    let (out_h, out_w) = match out_dims.as_slice() {
        &[out_n, out_h, out_w, ..] if out_n == n => (out_h, out_w),
        _ => {
            return Err(anyhow!(
                "the convolution result {out_dims:?} does not match {in_dims:?}"
            ))
        }
    };
    if out_dims.iter().product::<usize>() != n * out_h * out_w * c_out {
        return Err(anyhow!(
            "the convolution result {out_dims:?} has the wrong size"
        ));
    }
    if weight_k != k_h * k_w * c_in {
        return Err(anyhow!("conv2d weight shape mismatch"));
    }

    let input = operand_expr(&operands[0], value_infos)?;
    let weight = operand_expr(&operands[1], value_infos)?;
    let bias = if operands.len() == 3 {
        Some(operand_expr(&operands[2], value_infos)?)
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
    let call = format!(
        r#"
            gpt_rs_c_conv2d_nhwc_f32(
              in, w, b, out,
              {n}, {in_h}, {in_w}, {c_in},
              {out_h}, {out_w}, {c_out},
              {k_h}, {k_w},
              {stride_h}, {stride_w},
              {dilation_h}, {dilation_w},
              {pad_top}, {pad_left},
              {packed}
            );
        "#
    );
    push_block(module, 2, &call);
    push_block(module, 1, "}");

    Ok(())
}
