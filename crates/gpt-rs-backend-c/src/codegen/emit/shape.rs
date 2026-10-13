use std::collections::HashMap;

use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::{axis_index, Concat, Iota, Slice, TensorOp, Transpose};
use gpt_rs_ir::ir::{DType, TensorType, Value};

use super::super::profile::{register_op_profile_generic, register_op_profile_unary};
use super::super::types::{Instruction, Kind, ValueInfo};
use super::super::utils::{
    dims_usize, emit_memcpy, emit_parallel_loops_with_indices, ensure_copy_dtypes,
    linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_elem_count, operand_expr, operand_type, operand_types,
    output_info,
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
        TensorOp::Reshape(_) => {
            let out_info = output_info(value_infos, inst.result())?;
            let label = inst.label;
            let input_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.ty, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            let in_elem = operand_elem_count(&inst.operands[0], value_infos)?;
            if out_info.elem_count != in_elem {
                return Err(anyhow!("reshape element count mismatch"));
            }
            emit_memcpy(module, &out_info.var, &input, out_info.byte_len);
            op_id
        }
        TensorOp::BroadcastTo(_) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_copy_dtypes("broadcast", &[input_dtype], out_info.ty.dtype)?;
            let label = inst.label;
            let input_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.ty, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_broadcast(module, &out_info.var, &input, &out_info.ty, &input_spec)?;
            op_id
        }
        TensorOp::Transpose(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_copy_dtypes("transpose", &[input_dtype], out_info.ty.dtype)?;
            let label = inst.label;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.ty, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_transpose(module, &out_info.var, &input, &out_info.ty, &in_spec, spec)?;
            op_id
        }
        TensorOp::Slice(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_copy_dtypes("slice", &[input_dtype], out_info.ty.dtype)?;
            let label = inst.label;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.ty, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_slice(module, &out_info.var, &input, &out_info.ty, &in_spec, spec)?;
            op_id
        }
        TensorOp::Concat(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            let dtypes = inst
                .operands
                .iter()
                .map(|operand| operand_dtype(operand, value_infos))
                .collect::<Result<Vec<_>>>()?;
            ensure_copy_dtypes("concat", &dtypes, out_info.ty.dtype)?;
            let label = inst.label;
            let input_specs = operand_types(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.ty, &input_specs)?;
            emit_concat(
                module,
                &out_info.var,
                &out_info.ty,
                &inst.operands,
                spec,
                value_infos,
            )?;
            op_id
        }
        TensorOp::Iota(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            ensure_dtype(out_info.ty.dtype, DType::Si32, "iota output must be si32")?;
            let label = inst.label;
            let op_id = register_op_profile_generic(matmul_profile, label, &out_info.ty, &[])?;
            emit_iota(module, &out_info.var, &out_info.ty, spec)?;
            op_id
        }
        _ => return Ok(None),
    };

    Ok(Some(op_id))
}

fn emit_broadcast(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorType,
    in_spec: &TensorType,
) -> Result<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if out_dims.len() < in_dims.len() {
        return Err(anyhow!("broadcast output rank is smaller than input rank"));
    }
    let mut padded_in_dims = vec![1usize; out_dims.len() - in_dims.len()];
    padded_in_dims.extend(in_dims);

    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
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
    out_spec: &TensorType,
    in_spec: &TensorType,
    spec: &Transpose,
) -> Result<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.perm.len() != out_dims.len() || in_dims.len() != out_dims.len() {
        return Err(anyhow!("transpose rank mismatch"));
    }

    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
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
    out_spec: &TensorType,
    in_spec: &TensorType,
    spec: &Slice,
) -> Result<()> {
    let out_dims = dims_usize(out_spec)?;
    let in_dims = dims_usize(in_spec)?;
    if spec.starts.len() != out_dims.len() || spec.sizes.len() != out_dims.len() {
        return Err(anyhow!("slice spec rank mismatch"));
    }
    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = Vec::with_capacity(indices.len());
        for (axis, start) in spec.starts.iter().enumerate() {
            let stride = spec.stride(axis);
            in_indices.push(format!("{start} + {stride} * {}", indices[axis]));
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
    out_spec: &TensorType,
    operands: &[Value],
    spec: &Concat,
    values: &HashMap<Value, ValueInfo>,
) -> Result<()> {
    let out_dims = dims_usize(out_spec)?;
    let axis = axis_index(spec.axis, out_dims.len())?;
    let mut operand_exprs = Vec::with_capacity(operands.len());
    let mut operand_dims = Vec::with_capacity(operands.len());
    let mut axis_offsets = Vec::with_capacity(operands.len());
    let mut axis_cursor = 0usize;

    for operand in operands {
        let input = operand_expr(operand, values)?;
        let in_spec = operand_type(operand, values)?;
        let in_dims = dims_usize(&in_spec)?;
        if in_dims.len() != out_dims.len() {
            return Err(anyhow!("concat operand rank mismatch"));
        }
        for (idx, (in_dim, out_dim)) in in_dims.iter().zip(out_dims.iter()).enumerate() {
            if idx != axis && in_dim != out_dim {
                return Err(anyhow!("concat operand shape mismatch on non-concat axis"));
            }
        }
        axis_offsets.push(axis_cursor);
        axis_cursor += in_dims[axis];
        operand_exprs.push(input);
        operand_dims.push(in_dims);
    }

    if axis_cursor != out_dims[axis] {
        return Err(anyhow!("concat axis size does not match output shape"));
    }

    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
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
fn emit_iota(module: &mut String, out: &str, out_spec: &TensorType, spec: &Iota) -> Result<()> {
    let out_dims = dims_usize(out_spec)?;
    let axis = axis_index(spec.axis, out_dims.len())?;
    let assign = match out_spec.dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        _ => return Err(anyhow!("iota output must be f32 or si32")),
    };
    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
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
