use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    ComparisonOp, DType, ElementwiseBinaryOp, ElementwiseUnaryOp, Instruction, Operand, Operation,
};

use crate::targets::{binary_expr_from_code, unary_expr_from_code};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_binary,
    register_op_profile_generic, register_op_profile_unary,
};
use super::super::types::{ValueInfo, ValueKey};
use super::super::utils::{
    c_type, dims_usize, emit_loops_with_indices, emit_memcpy, linear_index_expr, push_block,
};
use super::super::value_info::LiteralCache;
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_spec, operand_specs, output_info,
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
        Operation::StopGradient => {
            let out_info = output_info(value_infos, inst.id)?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_memcpy(module, &out_info.var, &input, out_info.byte_len);
                Ok(())
            })?;
        }
        Operation::ElementwiseUnary(op) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(
                input_dtype,
                DType::F32,
                "elementwise unary input must be f32",
            )?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "elementwise unary output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_unary(module, op, &out_info.var, &input, out_info.elem_count)
            })?;
        }
        Operation::ElementwiseBinary(op) => {
            let out_info = output_info(value_infos, inst.id)?;
            let lhs_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let rhs_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_dtype(lhs_dtype, DType::F32, "elementwise binary lhs must be f32")?;
            ensure_dtype(rhs_dtype, DType::F32, "elementwise binary rhs must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "elementwise binary output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let lhs_spec = operand_spec(&inst.operands[0], value_infos)?;
            let rhs_spec = operand_spec(&inst.operands[1], value_infos)?;
            let op_id = register_op_profile_binary(
                matmul_profile,
                label,
                &out_info.spec,
                &lhs_spec,
                &rhs_spec,
            )?;
            let lhs = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let rhs = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_binary(module, op, &out_info.var, &lhs, &rhs, out_info.elem_count)
            })?;
        }
        Operation::Cast(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            if spec.dtype != out_info.spec.dtype {
                return Err(ConversionError::new(
                    "cast output dtype does not match instruction type",
                ));
            }
            emit_profiled_op(module, op_id, |module| {
                emit_cast(
                    module,
                    &out_info.var,
                    &input,
                    out_info.spec.dtype,
                    input_dtype,
                    out_info.elem_count,
                )
            })?;
        }
        Operation::Compare(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let lhs_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let rhs_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            if lhs_dtype != rhs_dtype {
                return Err(ConversionError::new(
                    "compare operands must have the same dtype",
                ));
            }
            match lhs_dtype {
                DType::F32 | DType::Si32 | DType::I1 => {}
                _ => {
                    return Err(ConversionError::new(
                        "compare operands must be f32, si32, or i1",
                    ))
                }
            }
            ensure_dtype(out_info.spec.dtype, DType::I1, "compare output must be i1")?;
            let label = backend_operation_label(&inst.op);
            let lhs_spec = operand_spec(&inst.operands[0], value_infos)?;
            let rhs_spec = operand_spec(&inst.operands[1], value_infos)?;
            let op_id = register_op_profile_binary(
                matmul_profile,
                label,
                &out_info.spec,
                &lhs_spec,
                &rhs_spec,
            )?;
            let lhs = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let rhs = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_compare(
                    module,
                    spec,
                    &out_info.var,
                    &lhs,
                    &rhs,
                    out_info.elem_count,
                    lhs_dtype,
                )
            })?;
        }
        Operation::Select => {
            let out_info = output_info(value_infos, inst.id)?;
            let pred_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let t_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let f_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            ensure_dtype(pred_dtype, DType::I1, "select predicate must be i1")?;
            ensure_dtype(t_dtype, DType::F32, "select on_true must be f32")?;
            ensure_dtype(f_dtype, DType::F32, "select on_false must be f32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "select output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let pred = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let on_true = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let on_false = operand_expr(&inst.operands[2], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_select(
                    module,
                    &out_info.var,
                    &pred,
                    &on_true,
                    &on_false,
                    out_info.elem_count,
                )
            })?;
        }
        Operation::Quantize(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            if spec.output_dtype != out_info.spec.dtype {
                return Err(ConversionError::new(
                    "quantize output dtype does not match instruction type",
                ));
            }
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            emit_profiled_op(module, op_id, |module| {
                emit_cast(
                    module,
                    &out_info.var,
                    &input,
                    out_info.spec.dtype,
                    input_dtype,
                    out_info.elem_count,
                )
            })?;
        }
        Operation::Dequantize(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let output_dtype = spec.output_dtype.unwrap_or(DType::F32);
            if out_info.spec.dtype != output_dtype {
                return Err(ConversionError::new(
                    "dequantize output dtype does not match instruction type",
                ));
            }
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_cast(
                    module,
                    &out_info.var,
                    &input,
                    out_info.spec.dtype,
                    input_dtype,
                    out_info.elem_count,
                )
            })?;
        }
        Operation::Requantize(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            if out_info.spec.dtype != spec.output_dtype {
                return Err(ConversionError::new(
                    "requantize output dtype does not match instruction type",
                ));
            }
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_cast(
                    module,
                    &out_info.var,
                    &input,
                    out_info.spec.dtype,
                    input_dtype,
                    out_info.elem_count,
                )
            })?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn emit_unary(
    module: &mut String,
    op: &ElementwiseUnaryOp,
    out: &str,
    input: &str,
    elem_count: usize,
) -> ConversionResult<()> {
    let expr = match op {
        ElementwiseUnaryOp::Neg => "-src[i]",
        ElementwiseUnaryOp::Abs => "fabsf(src[i])",
        ElementwiseUnaryOp::Exp => "expf(src[i])",
        ElementwiseUnaryOp::Log => "logf(src[i])",
        ElementwiseUnaryOp::Tanh => "tanhf(src[i])",
        ElementwiseUnaryOp::Erf => "erff(src[i])",
        ElementwiseUnaryOp::Rsqrt => "1.0f / sqrtf(src[i])",
        ElementwiseUnaryOp::Reciprocal => "1.0f / src[i]",
    };
    let block = format!(
        r#"
            {{
              const float* src = (const float*){input};
              float* out = {out};
              for (size_t i = 0; i < {elem_count}; ++i) {{
                out[i] = {expr};
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
fn emit_binary(
    module: &mut String,
    op: &ElementwiseBinaryOp,
    out: &str,
    lhs: &str,
    rhs: &str,
    elem_count: usize,
) -> ConversionResult<()> {
    let expr = match op {
        ElementwiseBinaryOp::Add => "lhs_ptr[i] + rhs_ptr[i]",
        ElementwiseBinaryOp::Sub => "lhs_ptr[i] - rhs_ptr[i]",
        ElementwiseBinaryOp::Mul => "lhs_ptr[i] * rhs_ptr[i]",
        ElementwiseBinaryOp::Div => "lhs_ptr[i] / rhs_ptr[i]",
        ElementwiseBinaryOp::Maximum => "lhs_ptr[i] > rhs_ptr[i] ? lhs_ptr[i] : rhs_ptr[i]",
        ElementwiseBinaryOp::Minimum => "lhs_ptr[i] < rhs_ptr[i] ? lhs_ptr[i] : rhs_ptr[i]",
    };
    let block = format!(
        r#"
            {{
              const float* lhs_ptr = (const float*){lhs};
              const float* rhs_ptr = (const float*){rhs};
              float* out = {out};
              for (size_t i = 0; i < {elem_count}; ++i) {{
                out[i] = {expr};
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
fn emit_cast(
    module: &mut String,
    out: &str,
    input: &str,
    out_dtype: DType,
    in_dtype: DType,
    elem_count: usize,
) -> ConversionResult<()> {
    if out_dtype == in_dtype {
        let elem_size = in_dtype
            .size_in_bytes()
            .ok_or_else(|| ConversionError::new("cast dtype size unknown"))?;
        let byte_len = elem_count
            .checked_mul(elem_size)
            .ok_or_else(|| ConversionError::new("cast byte length overflow"))?;
        emit_memcpy(module, out, input, byte_len);
        return Ok(());
    }
    let in_ctype = c_type(in_dtype)?;
    let out_ctype = c_type(out_dtype)?;
    let body = match (in_dtype, out_dtype) {
        (DType::F32, DType::Si32) => r#"
                float v = in[i];
                if (!isfinite(v)) { v = 0.0f; }
                if (v > (float)INT32_MAX) { v = (float)INT32_MAX; }
                if (v < (float)INT32_MIN) { v = (float)INT32_MIN; }
                out[i] = (int32_t)v;
            "#
        .to_string(),
        (DType::F32, DType::I1) => r#"
                float v = in[i];
                if (!isfinite(v)) { v = 0.0f; }
                out[i] = (uint8_t)(v != 0.0f);
            "#
        .to_string(),
        (DType::Si32, DType::F32) => r#"
                out[i] = (float)in[i];
            "#
        .to_string(),
        (DType::Si32, DType::I1) => r#"
                out[i] = (uint8_t)(in[i] != 0);
            "#
        .to_string(),
        (DType::I1, DType::F32) => r#"
                out[i] = in[i] ? 1.0f : 0.0f;
            "#
        .to_string(),
        (DType::I1, DType::Si32) => r#"
                out[i] = in[i] ? 1 : 0;
            "#
        .to_string(),
        _ => {
            return Err(ConversionError::new(
                "cast dtype combination not supported by C codegen",
            ));
        }
    };
    let block = format!(
        r#"
            {{
              const {in_ctype}* in = (const {in_ctype}*){input};
              {out_ctype}* out = {out};
              for (size_t i = 0; i < {elem_count}; ++i) {{
{body}              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
fn emit_compare(
    module: &mut String,
    spec: &gpt_rs::backend::spec::CompareSpec,
    out: &str,
    lhs: &str,
    rhs: &str,
    elem_count: usize,
    dtype: DType,
) -> ConversionResult<()> {
    let expr = match spec.op {
        ComparisonOp::Less => "lhs[i] < rhs[i]",
        ComparisonOp::LessEqual => "lhs[i] <= rhs[i]",
        ComparisonOp::Equal => "lhs[i] == rhs[i]",
        ComparisonOp::GreaterEqual => "lhs[i] >= rhs[i]",
        ComparisonOp::Greater => "lhs[i] > rhs[i]",
        ComparisonOp::NotEqual => "lhs[i] != rhs[i]",
    };
    let ctype = match dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        DType::I1 => "uint8_t",
        _ => {
            return Err(ConversionError::new(
                "compare operands must be f32, si32, or i1",
            ))
        }
    };
    let block = format!(
        r#"
            {{
              const {ctype}* lhs = (const {ctype}*){lhs};
              const {ctype}* rhs = (const {ctype}*){rhs};
              uint8_t* out = (uint8_t*){out};
              for (size_t i = 0; i < {elem_count}; ++i) {{
                out[i] = {expr} ? 1 : 0;
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
fn emit_select(
    module: &mut String,
    out: &str,
    pred: &str,
    on_true: &str,
    on_false: &str,
    elem_count: usize,
) -> ConversionResult<()> {
    let block = format!(
        r#"
            {{
              const uint8_t* pred = (const uint8_t*){pred};
              const float* on_true = (const float*){on_true};
              const float* on_false = (const float*){on_false};
              float* out = (float*){out};
              for (size_t i = 0; i < {elem_count}; ++i) {{
                out[i] = pred[i] ? on_true[i] : on_false[i];
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
fn broadcast_index_expr(out_dims: &[usize], out_indices: &[String], in_dims: &[usize]) -> String {
    let mut padded_in_dims = vec![1usize; out_dims.len().saturating_sub(in_dims.len())];
    padded_in_dims.extend(in_dims);
    let mut in_indices = Vec::with_capacity(out_indices.len());
    for (idx, dim) in padded_in_dims.iter().enumerate() {
        if *dim == 1 {
            in_indices.push("0".to_string());
        } else {
            in_indices.push(out_indices[idx].clone());
        }
    }
    linear_index_expr(&padded_in_dims, &in_indices)
}
pub(super) fn emit_custom_call_elementwise(
    module: &mut String,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
    operands: &[Operand],
    out_info: &ValueInfo,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    ensure_dtype(
        out_info.spec.dtype,
        DType::F32,
        "fused elementwise output must be f32",
    )?;
    let ops_kind = custom_call_attr_i64_array(spec, "ops_kind")?;
    let ops_code = custom_call_attr_i64_array(spec, "ops_code")?;
    let lhs = custom_call_attr_i64_array(spec, "lhs")?;
    let rhs = custom_call_attr_i64_array(spec, "rhs")?;
    let node_count = ops_kind.len();
    if ops_code.len() != node_count || lhs.len() != node_count || rhs.len() != node_count {
        return Err(ConversionError::new(
            "fused elementwise attrs length mismatch",
        ));
    }
    if node_count == 0 {
        return Err(ConversionError::new(
            "fused elementwise custom_call must contain at least one op",
        ));
    }

    let out_dims = dims_usize(&out_info.spec)?;

    struct InputInfo {
        var: String,
        dims: Vec<usize>,
    }

    let mut inputs: Vec<InputInfo> = Vec::with_capacity(operands.len());
    for operand in operands {
        let dtype = operand_dtype(operand, value_infos)?;
        ensure_dtype(dtype, DType::F32, "fused elementwise operand must be f32")?;
        let spec = operand_spec(operand, value_infos)?;
        let dims = dims_usize(&spec)?;
        if out_dims.len() < dims.len() {
            return Err(ConversionError::new(
                "fused elementwise operand rank exceeds output rank",
            ));
        }
        let offset = out_dims.len() - dims.len();
        for (idx, dim) in dims.iter().enumerate() {
            let out_dim = out_dims[idx + offset];
            if *dim != 1 && *dim != out_dim {
                return Err(ConversionError::new(
                    "fused elementwise operand shape is not broadcastable",
                ));
            }
        }
        let var = operand_expr(operand, value_infos, module, literal_cache)?;
        inputs.push(InputInfo { var, dims });
    }

    let input_count = inputs.len() as i64;
    let mut node_exprs: Vec<String> = Vec::with_capacity(node_count);
    for node_idx in 0..node_count {
        let kind = ops_kind[node_idx];
        let code = ops_code[node_idx];
        let lhs_idx = lhs[node_idx];
        let rhs_idx = rhs[node_idx];

        let expr_for_idx = |idx: i64, node_idx: usize| -> ConversionResult<String> {
            if idx < 0 {
                return Err(ConversionError::new("fused elementwise index is negative"));
            }
            if idx < input_count {
                Ok(format!("in{idx}v"))
            } else {
                let local = idx - input_count;
                if local >= node_idx as i64 {
                    return Err(ConversionError::new(
                        "fused elementwise index refers to future op",
                    ));
                }
                Ok(format!("t{local}"))
            }
        };

        let lhs_expr = expr_for_idx(lhs_idx, node_idx)?;
        let expr = match kind {
            0 => unary_expr_from_code(code, &lhs_expr)
                .ok_or_else(|| ConversionError::new("unknown fused elementwise unary op"))?,
            1 => {
                let rhs_expr = expr_for_idx(rhs_idx, node_idx)?;
                binary_expr_from_code(code, &lhs_expr, &rhs_expr)
                    .ok_or_else(|| ConversionError::new("unknown fused elementwise binary op"))?
            }
            _ => {
                return Err(ConversionError::new(
                    "fused elementwise kind must be unary or binary",
                ))
            }
        };

        node_exprs.push(expr);
    }

    let input_decls = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let var = &input.var;
            format!("const float* arg{index} = (const float*){var};")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let out_var = &out_info.var;
    push_block(
        module,
        1,
        r#"
            {
        "#,
    );
    if !input_decls.is_empty() {
        push_block(module, 2, &input_decls);
    }
    push_block(module, 2, &format!("float* out = {out_var};"));

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let input_lines = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let idx_expr = broadcast_index_expr(&out_dims, indices, &input.dims);
                format!("float in{index}v = arg{index}[{idx_expr}];")
            })
            .collect::<Vec<_>>()
            .join("\n");
        push_block(module, indent, &input_lines);

        let node_lines = node_exprs
            .iter()
            .enumerate()
            .map(|(node_idx, expr)| format!("float t{node_idx} = {expr};"))
            .collect::<Vec<_>>()
            .join("\n");
        push_block(module, indent, &node_lines);

        let out_idx = linear_index_expr(&out_dims, indices);
        let last = node_count - 1;
        push_block(module, indent, &format!("out[{out_idx}] = t{last};"));
    });

    push_block(module, 1, "}");
    Ok(())
}
