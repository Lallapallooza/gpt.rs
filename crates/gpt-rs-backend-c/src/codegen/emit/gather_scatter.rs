use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::{DynamicSlice, TensorOp};
use gpt_rs_ir::types::{DType, TensorType};

use super::super::profile::register_op_profile_generic;
use super::super::types::{Instruction, Kind};
use super::super::utils::{
    c_type, dims_usize, emit_loops_with_indices, emit_parallel_loops_with_indices,
    ensure_copy_dtypes, linear_index_expr, push_block, sizing,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_type, operand_types, output_info,
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
        TensorOp::Take(_) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_copy_dtypes("take", &[input_dtype], out_info.ty.dtype)?;
            ensure_dtype(indices_dtype, DType::Si32, "take indices must be si32")?;
            let label = inst.label;
            let input_specs = operand_types(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.ty, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            let indices = operand_expr(&inst.operands[1], value_infos)?;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let idx_spec = operand_type(&inst.operands[1], value_infos)?;
            emit_take(
                module,
                &out_info.var,
                &input,
                &indices,
                &out_info.ty,
                &in_spec,
                &idx_spec,
            )?;
            op_id
        }
        TensorOp::DynamicSlice(spec) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let start_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_copy_dtypes("dynamic_slice", &[input_dtype], out_info.ty.dtype)?;
            ensure_dtype(start_dtype, DType::Si32, "dynamic_slice start must be si32")?;
            let label = inst.label;
            let input_specs = operand_types(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.ty, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            let start = operand_expr(&inst.operands[1], value_infos)?;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            emit_dynamic_slice(
                module,
                &out_info.var,
                &input,
                &start,
                &out_info.ty,
                &in_spec,
                spec,
                input_dtype,
            )?;
            op_id
        }
        TensorOp::DynamicUpdateSlice(_) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let update_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let start_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            ensure_copy_dtypes(
                "dynamic_update_slice",
                &[input_dtype, update_dtype],
                out_info.ty.dtype,
            )?;
            ensure_dtype(
                start_dtype,
                DType::Si32,
                "dynamic_update_slice start must be si32",
            )?;
            let label = inst.label;
            let input_specs = operand_types(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.ty, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            let update = operand_expr(&inst.operands[1], value_infos)?;
            let start = operand_expr(&inst.operands[2], value_infos)?;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let update_spec = operand_type(&inst.operands[1], value_infos)?;
            emit_dynamic_update_slice(
                module,
                &out_info.var,
                &input,
                &update,
                &start,
                &out_info.ty,
                &in_spec,
                &update_spec,
                input_dtype,
            )?;
            op_id
        }
        _ => return Ok(None),
    };

    Ok(Some(op_id))
}

fn emit_take(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    out_spec: &TensorType,
    in_spec: &TensorType,
    indices_spec: &TensorType,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let idx_dims = dims_usize(indices_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims.is_empty() {
        return Err(anyhow!("take input must be at least rank-1"));
    }
    let slice = in_dims[1..].iter().product::<usize>();
    let indices_count: usize = idx_dims.iter().product();
    let expected_out: Vec<usize> = idx_dims
        .iter()
        .copied()
        .chain(in_dims[1..].iter().copied())
        .collect();
    if out_dims != expected_out {
        return Err(anyhow!("take output shape mismatch"));
    }
    let axis_len = in_dims[0];
    let ctype = c_type(out_spec.dtype)?;

    let block = format!(
        r#"
            {{
              const {ctype}* input = (const {ctype}*){input};
              const int32_t* indices = (const int32_t*){indices};
              {ctype}* out = ({ctype}*){out};
              for (size_t i = 0; i < {indices_count}; ++i) {{
                int32_t idx = indices[i];
                if (idx < 0 || idx >= {axis_len}) {{ return -5; }}
                memcpy(out + i * {slice}, input + ((size_t)idx) * {slice}, {slice} * sizeof({ctype}));
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_dynamic_slice(
    module: &mut String,
    out: &str,
    input: &str,
    start: &str,
    out_spec: &TensorType,
    in_spec: &TensorType,
    spec: &DynamicSlice,
    dtype: DType,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if spec.sizes.len() != in_dims.len() {
        return Err(anyhow!("dynamic_slice rank mismatch"));
    }
    if out_dims[..] != spec.sizes[..] {
        return Err(anyhow!("dynamic_slice output shape mismatch"));
    }

    let ctype = c_type(dtype)?;
    let header = format!(
        r#"
            {{
              const {ctype}* input = (const {ctype}*){input};
              const int32_t* start = (const int32_t*){start};
        "#
    );
    push_block(module, 1, &header);
    for (axis, dim) in in_dims.iter().enumerate() {
        let size = spec.sizes[axis];
        let max_start = dim.saturating_sub(size);
        let block = format!(
            r#"
                int32_t start{axis} = start[{axis}];
                if (start{axis} < 0) start{axis} = 0;
                if (start{axis} > {max_start}) start{axis} = {max_start};
            "#
        );
        push_block(module, 2, &block);
    }

    emit_parallel_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let mut in_indices = Vec::with_capacity(indices.len());
        for (axis, idx) in indices.iter().enumerate() {
            let start_axis = axis;
            let offset = idx;
            in_indices.push(format!("start{start_axis} + {offset}"));
        }
        let out_idx = linear_index_expr(&out_dims, indices);
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    push_block(module, 1, "}");
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_dynamic_update_slice(
    module: &mut String,
    out: &str,
    input: &str,
    update: &str,
    start: &str,
    out_spec: &TensorType,
    in_spec: &TensorType,
    update_spec: &TensorType,
    dtype: DType,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let update_dims = dims_usize(update_spec)?;
    if update_dims.len() != in_dims.len() {
        return Err(anyhow!("dynamic_update_slice rank mismatch"));
    }
    if out_dims != in_dims {
        return Err(anyhow!("dynamic_update_slice output shape mismatch"));
    }

    let ctype = c_type(dtype)?;
    let header = format!(
        r#"
            {{
              const {ctype}* input = (const {ctype}*){input};
              const {ctype}* update = (const {ctype}*){update};
              const int32_t* start = (const int32_t*){start};
              {ctype}* out = ({ctype}*){out};
        "#
    );
    push_block(module, 1, &header);
    let (_, byte_len) = sizing(out_spec)?;
    push_block(module, 2, &format!("gpt_rs_copy(out, input, {byte_len});"));
    for (axis, dim) in in_dims.iter().enumerate() {
        let size = update_dims[axis];
        let max_start = dim.saturating_sub(size);
        let block = format!(
            r#"
                int32_t start{axis} = start[{axis}];
                if (start{axis} < 0) start{axis} = 0;
                if (start{axis} > {max_start}) start{axis} = {max_start};
            "#
        );
        push_block(module, 2, &block);
    }
    emit_loops_with_indices(module, &update_dims, 2, "i", |module, indices, indent| {
        let mut out_indices = Vec::with_capacity(indices.len());
        for (axis, idx) in indices.iter().enumerate() {
            let start_axis = axis;
            let offset = idx;
            out_indices.push(format!("start{start_axis} + {offset}"));
        }
        let update_idx = linear_index_expr(&update_dims, indices);
        let out_idx = linear_index_expr(&out_dims, &out_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {update}[{update_idx}];"),
        );
    });
    push_block(module, 1, "}");
    Ok(())
}
