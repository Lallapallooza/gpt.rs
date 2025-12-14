use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    ConcatSpec, DType, Instruction, IotaSpec, Operand, Operation, PadSpec, SliceSpec, TensorSpec,
    TileSpec, TransposeSpec,
};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_generic,
    register_op_profile_unary,
};
use super::super::types::{ValueInfo, ValueKey};
use super::super::utils::{
    axis_index, dims_usize, emit_loops_with_indices, emit_memcpy, format_f32, linear_index_expr,
    literal_to_f32_scalar, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_elem_count, operand_expr, operand_spec, operand_specs,
    output_info, LiteralCache,
};
use super::EmitContext;

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
        Operation::Reshape(_spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let in_elem = operand_elem_count(&inst.operands[0], value_infos)?;
            if out_info.elem_count != in_elem {
                return Err(ConversionError::new("reshape element count mismatch"));
            }
            emit_profiled_op(module, op_id, |module| {
                emit_memcpy(module, &out_info.var, &input, out_info.byte_len);
                Ok(())
            })?;
        }
        Operation::BroadcastTo(_spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            match input_dtype {
                DType::F32 | DType::Si32 | DType::I1 => {}
                _ => {
                    return Err(ConversionError::new(
                        "broadcast input must be f32, si32, or i1",
                    ))
                }
            }
            ensure_dtype(
                out_info.spec.dtype,
                input_dtype,
                "broadcast output must match input dtype",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_broadcast(module, &out_info.var, &input, &out_info.spec, &input_spec)
            })?;
        }
        Operation::Transpose(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "transpose input must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "transpose output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.spec, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_transpose(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &in_spec,
                    spec,
                )
            })?;
        }
        Operation::Slice(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "slice input must be f32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "slice output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.spec, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_slice(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &in_spec,
                    spec,
                )
            })?;
        }
        Operation::Concat(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            for operand in &inst.operands {
                let dtype = operand_dtype(operand, value_infos)?;
                ensure_dtype(dtype, DType::F32, "concat operands must be f32")?;
            }
            ensure_dtype(out_info.spec.dtype, DType::F32, "concat output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            emit_profiled_op(module, op_id, |module| {
                emit_concat(
                    module,
                    &out_info.var,
                    &out_info.spec,
                    &inst.operands,
                    spec,
                    value_infos,
                    literal_cache,
                )
            })?;
        }
        Operation::Pad(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "pad input must be f32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "pad output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_pad(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &input_spec,
                    spec,
                )
            })?;
        }
        Operation::Tile(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "tile input must be f32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "tile output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_tile(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &input_spec,
                    spec,
                )
            })?;
        }
        Operation::Iota(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            ensure_dtype(out_info.spec.dtype, DType::Si32, "iota output must be si32")?;
            let label = backend_operation_label(&inst.op);
            let op_id = register_op_profile_generic(matmul_profile, label, &out_info.spec, &[])?;
            emit_profiled_op(module, op_id, |module| {
                emit_iota(module, &out_info.var, &out_info.spec, spec)
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

fn emit_broadcast(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if out_dims.len() < in_dims.len() {
        return Err(ConversionError::new(
            "broadcast output rank is smaller than input rank",
        ));
    }
    let mut padded_in_dims = vec![1usize; out_dims.len() - in_dims.len()];
    padded_in_dims.extend(in_dims);

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = Vec::with_capacity(indices.len());
        for (idx, dim) in padded_in_dims.iter().enumerate() {
            if *dim == 1 {
                in_indices.push("0".to_string());
            } else {
                in_indices.push(indices[idx].clone());
            }
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&padded_in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    Ok(())
}
fn emit_transpose(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &TransposeSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.perm.len() != out_dims.len() || in_dims.len() != out_dims.len() {
        return Err(ConversionError::new("transpose rank mismatch"));
    }

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = vec!["0".to_string(); in_dims.len()];
        for (out_axis, in_axis) in spec.perm.iter().enumerate() {
            if *in_axis >= in_indices.len() {
                continue;
            }
            in_indices[*in_axis] = indices[out_axis].clone();
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    Ok(())
}
fn emit_slice(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &SliceSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.starts.len() != out_dims.len() || spec.sizes.len() != out_dims.len() {
        return Err(ConversionError::new("slice spec rank mismatch"));
    }
    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = Vec::with_capacity(indices.len());
        for (idx, start) in spec.starts.iter().enumerate() {
            let base = start;
            let offset = &indices[idx];
            in_indices.push(format!("{base} + {offset}"));
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    Ok(())
}
fn emit_concat(
    module: &mut String,
    out: &str,
    out_spec: &TensorSpec,
    operands: &[Operand],
    spec: &ConcatSpec,
    values: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let axis = axis_index(spec.axis, out_dims.len())?;
    let mut operand_exprs = Vec::with_capacity(operands.len());
    let mut operand_dims = Vec::with_capacity(operands.len());
    let mut axis_offsets = Vec::with_capacity(operands.len());
    let mut axis_cursor = 0usize;

    for operand in operands {
        let input = operand_expr(operand, values, module, literal_cache)?;
        let in_spec = operand_spec(operand, values)?;
        let in_dims = dims_usize(&in_spec)?;
        if in_dims.len() != out_dims.len() {
            return Err(ConversionError::new("concat operand rank mismatch"));
        }
        for (idx, (in_dim, out_dim)) in in_dims.iter().zip(out_dims.iter()).enumerate() {
            if idx != axis && in_dim != out_dim {
                return Err(ConversionError::new(
                    "concat operand shape mismatch on non-concat axis",
                ));
            }
        }
        axis_offsets.push(axis_cursor);
        axis_cursor += in_dims[axis];
        operand_exprs.push(input);
        operand_dims.push(in_dims);
    }

    if axis_cursor != out_dims[axis] {
        return Err(ConversionError::new(
            "concat axis size does not match output shape",
        ));
    }

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let out_idx = linear_index_expr(&out_dims, indices);
        let axis_idx = &indices[axis];
        for (op_idx, (input, in_dims)) in operand_exprs.iter().zip(operand_dims.iter()).enumerate()
        {
            let offset = axis_offsets[op_idx];
            let axis_size = in_dims[axis];
            let limit = offset + axis_size;
            let cond = format!("{axis_idx} < {limit}");
            let cond_line = if op_idx == 0 {
                format!("if ({cond}) {{")
            } else if op_idx + 1 == operand_exprs.len() {
                "else {".to_string()
            } else {
                format!("else if ({cond}) {{")
            };

            let mut in_indices = Vec::with_capacity(indices.len());
            for (dim_idx, _) in indices.iter().enumerate() {
                if dim_idx == axis {
                    if offset == 0 {
                        in_indices.push(axis_idx.clone());
                    } else {
                        in_indices.push(format!("{axis_idx} - {offset}"));
                    }
                } else {
                    in_indices.push(indices[dim_idx].clone());
                }
            }
            let in_idx = linear_index_expr(in_dims, &in_indices);
            let block = format!(
                r#"{cond_line}
  {out}[{out_idx}] = {input}[{in_idx}];
}}"#
            );
            push_block(module, indent, &block);
        }
    });
    Ok(())
}
fn emit_pad(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &PadSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.low.len() != out_dims.len()
        || spec.high.len() != out_dims.len()
        || spec.interior.len() != out_dims.len()
    {
        return Err(ConversionError::new("pad spec rank mismatch"));
    }
    let pad_value = literal_to_f32_scalar(&spec.pad_value)?;
    let pad = format_f32(pad_value);

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        push_block(module, indent, "int in_bounds = 1;");
        let mut in_indices: Vec<String> = Vec::with_capacity(indices.len());
        for (axis, idx) in indices.iter().enumerate() {
            let low = spec.low[axis];
            let interior = spec.interior[axis];
            let step = interior + 1;
            let in_dim = in_dims[axis];
            let in_name = format!("pad{axis}");
            let block = format!(
                r#"size_t {in_name} = 0;
if ({idx} < {low}) {{
  in_bounds = 0;
}}
if (in_bounds) {{
  size_t rel = {idx} - {low};
  if (rel % {step} != 0) {{
    in_bounds = 0;
  }} else {{
    {in_name} = rel / {step};
    if ({in_name} >= {in_dim}) {{
      in_bounds = 0;
    }}
  }}
}}"#
            );
            push_block(module, indent, &block);
            in_indices.push(in_name);
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        let block = format!(
            r#"if (in_bounds) {{
  {out}[{out_idx}] = {input}[{in_idx}];
}} else {{
  {out}[{out_idx}] = {pad};
}}"#
        );
        push_block(module, indent, &block);
    });
    Ok(())
}
fn emit_tile(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &TileSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.repeats.len() != out_dims.len() || in_dims.len() != out_dims.len() {
        return Err(ConversionError::new("tile rank mismatch"));
    }
    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = Vec::with_capacity(indices.len());
        for (axis, idx) in indices.iter().enumerate() {
            let dim = in_dims[axis];
            in_indices.push(format!("{idx} % {dim}"));
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    Ok(())
}
fn emit_iota(
    module: &mut String,
    out: &str,
    out_spec: &TensorSpec,
    spec: &IotaSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    let axis = axis_index(spec.axis as isize, out_dims.len())?;
    let assign = match out_spec.dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        _ => return Err(ConversionError::new("iota output must be f32 or si32")),
    };
    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let out_idx = linear_index_expr(&out_dims, indices);
        let axis_idx = &indices[axis];
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = ({assign}){axis_idx};"),
        );
    });
    Ok(())
}
