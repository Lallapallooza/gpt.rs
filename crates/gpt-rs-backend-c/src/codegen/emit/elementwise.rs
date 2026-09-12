use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    ComparisonOp, DType, ElementwiseBinaryOp, ElementwiseUnaryOp, Instruction, Operand, Operation,
};

use crate::targets::{binary_expr_from_code, fused_input_fits, unary_expr_from_code};

use super::super::profile::{
    backend_operation_label, register_op_profile_binary, register_op_profile_generic,
    register_op_profile_unary,
};
use super::super::types::{ValueInfo, ValueKey};
use super::super::utils::{
    c_type, dims_usize, emit_flat_loop, emit_memcpy, emit_parallel_loops_with_indices,
    linear_index_expr, push_block,
};
use super::super::value_info::LiteralCache;
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_spec, operand_specs, output_info,
};
use super::{custom_call_attr_i64_array, EmitContext};

pub(super) fn emit_instruction(
    inst: &Instruction,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<Option<usize>> {
    let EmitContext {
        module,
        value_infos,
        literal_cache,
        matmul_profile,
        ..
    } = ctx;

    let op_id = match &inst.op {
        Operation::StopGradient => {
            let out_info = output_info(value_infos, inst.id)?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_memcpy(module, &out_info.var, &input, out_info.byte_len);
            op_id
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
            emit_unary(module, op, &out_info.var, &input, out_info.elem_count)?;
            op_id
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
            emit_binary(module, op, &out_info.var, &lhs, &rhs, out_info.elem_count)?;
            op_id
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
            emit_cast(
                module,
                &out_info.var,
                &input,
                out_info.spec.dtype,
                input_dtype,
                out_info.elem_count,
            )?;
            op_id
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
            emit_compare(
                module,
                spec,
                &out_info.var,
                &lhs,
                &rhs,
                out_info.elem_count,
                lhs_dtype,
            )?;
            op_id
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
            emit_select(
                module,
                &out_info.var,
                &pred,
                &on_true,
                &on_false,
                out_info.elem_count,
            )?;
            op_id
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
            emit_cast(
                module,
                &out_info.var,
                &input,
                out_info.spec.dtype,
                input_dtype,
                out_info.elem_count,
            )?;
            op_id
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
            emit_cast(
                module,
                &out_info.var,
                &input,
                out_info.spec.dtype,
                input_dtype,
                out_info.elem_count,
            )?;
            op_id
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
            emit_cast(
                module,
                &out_info.var,
                &input,
                out_info.spec.dtype,
                input_dtype,
                out_info.elem_count,
            )?;
            op_id
        }
        _ => return Ok(None),
    };
    Ok(Some(op_id))
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
    emit_flat_loop(
        module,
        &[("float", "src", input)],
        ("float", out),
        elem_count,
        &format!("out[i] = {expr};"),
    );
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
    emit_flat_loop(
        module,
        &[("float", "lhs_ptr", lhs), ("float", "rhs_ptr", rhs)],
        ("float", out),
        elem_count,
        &format!("out[i] = {expr};"),
    );
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
        (DType::Bf16, DType::F32) => r#"
                out[i] = gpt_rs_bf16_to_f32(in[i]);
            "#
        .to_string(),
        (DType::F32, DType::Bf16) => r#"
                out[i] = gpt_rs_f32_to_bf16(in[i]);
            "#
        .to_string(),
        _ => {
            return Err(ConversionError::new(
                "cast dtype combination not supported by C codegen",
            ));
        }
    };
    emit_flat_loop(
        module,
        &[(in_ctype, "in", input)],
        (out_ctype, out),
        elem_count,
        &body,
    );
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
    emit_flat_loop(
        module,
        &[(ctype, "lhs", lhs), (ctype, "rhs", rhs)],
        ("uint8_t", out),
        elem_count,
        &format!("out[i] = {expr} ? 1 : 0;"),
    );
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
    emit_flat_loop(
        module,
        &[
            ("uint8_t", "pred", pred),
            ("float", "on_true", on_true),
            ("float", "on_false", on_false),
        ],
        ("float", out),
        elem_count,
        "out[i] = pred[i] ? on_true[i] : on_false[i];",
    );
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
/// Decodes the `input_slice_starts` attribute into per-input offsets. The attribute holds
/// `[input, start_0, .., start_{rank-1}]` groups.
fn fused_input_slice_starts(
    spec: &gpt_rs::backend::spec::CustomCallSpec,
    input_count: usize,
    rank: usize,
) -> ConversionResult<Vec<Option<Vec<usize>>>> {
    let mut starts = vec![None; input_count];
    if !spec.attrs.contains_key("input_slice_starts") {
        return Ok(starts);
    }
    let flat = custom_call_attr_i64_array(spec, "input_slice_starts")?;
    if flat.len() % (rank + 1) != 0 {
        return Err(ConversionError::new(
            "fused elementwise input_slice_starts length mismatch",
        ));
    }
    for group in flat.chunks(rank + 1) {
        let slot = usize::try_from(group[0])
            .ok()
            .and_then(|index| starts.get_mut(index))
            .ok_or_else(|| ConversionError::new("fused elementwise sliced input out of range"))?;
        if slot.is_some() {
            return Err(ConversionError::new(format!(
                "fused elementwise input {} has more than one slice offset",
                group[0]
            )));
        }
        let offsets = group[1..]
            .iter()
            .map(|start| usize::try_from(*start))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| ConversionError::new("fused elementwise slice start is negative"))?;
        *slot = Some(offsets);
    }
    Ok(starts)
}

pub(super) fn emit_custom_call_elementwise(
    module: &mut String,
    spec: &gpt_rs::backend::spec::CustomCallSpec,
    operands: &[Operand],
    out_info: &ValueInfo,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    let store_bf16 = match out_info.spec.dtype {
        DType::F32 => false,
        DType::Bf16 => true,
        _ => {
            return Err(ConversionError::new(
                "fused elementwise output must be f32 or bf16",
            ))
        }
    };
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
    let slice_starts = fused_input_slice_starts(spec, operands.len(), out_dims.len())?;

    struct InputInfo {
        var: String,
        dims: Vec<usize>,
        starts: Option<Vec<usize>>,
    }

    let mut inputs: Vec<InputInfo> = Vec::with_capacity(operands.len());
    for (operand, starts) in operands.iter().zip(slice_starts) {
        let dtype = operand_dtype(operand, value_infos)?;
        ensure_dtype(dtype, DType::F32, "fused elementwise operand must be f32")?;
        let dims = dims_usize(&operand_spec(operand, value_infos)?)?;
        if !fused_input_fits(&out_dims, &dims, starts.as_deref()) {
            return Err(ConversionError::new(format!(
                "fused elementwise operand {dims:?} (slice starts {starts:?}) does not fit output {out_dims:?}"
            )));
        }
        let var = operand_expr(operand, value_infos, module, literal_cache)?;
        inputs.push(InputInfo { var, dims, starts });
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

    // When every input has the output's shape or holds one element, all inputs index like the
    // output, so one flat loop is enough.
    let flat = inputs.iter().all(|input| {
        input.starts.is_none()
            && (input.dims == out_dims || input.dims.iter().product::<usize>() == 1)
    });
    let out_dims = if flat {
        for input in &mut inputs {
            input.dims = vec![input.dims.iter().product()];
        }
        vec![out_info.elem_count]
    } else {
        out_dims
    };

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
    let out_type = c_type(out_info.spec.dtype)?;
    push_block(
        module,
        2,
        &format!("{out_type}* out = ({out_type}*){out_var};"),
    );

    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let input_lines = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let idx_expr = match &input.starts {
                    Some(starts) => {
                        let shifted = indices
                            .iter()
                            .zip(starts)
                            .map(|(idx, start)| {
                                if *start == 0 {
                                    idx.clone()
                                } else {
                                    format!("{start} + {idx}")
                                }
                            })
                            .collect::<Vec<_>>();
                        linear_index_expr(&input.dims, &shifted)
                    }
                    None => broadcast_index_expr(&out_dims, indices, &input.dims),
                };
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
        let result = if store_bf16 {
            format!("gpt_rs_f32_to_bf16(t{last})")
        } else {
            format!("t{last}")
        };
        push_block(module, indent, &format!("out[{out_idx}] = {result};"));
    });

    push_block(module, 1, "}");
    Ok(())
}
