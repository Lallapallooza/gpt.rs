use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    DType, DynamicSliceSpec, DynamicUpdateSliceSpec, GatherSpec, Instruction, Operation,
    ScatterReduceKind, ScatterReduceSpec, ScatterSpec, TensorSpec,
};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_generic,
};
use super::super::utils::{
    axis_index, dims_usize, emit_loops_with_indices, linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_spec, operand_specs, output_info,
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
        Operation::Take => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "take input must be f32")?;
            ensure_dtype(indices_dtype, DType::Si32, "take indices must be si32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "take output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let indices = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let idx_spec = operand_spec(&inst.operands[1], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_take(
                    module,
                    &out_info.var,
                    &input,
                    &indices,
                    &out_info.spec,
                    &in_spec,
                    &idx_spec,
                )
            })?;
        }
        Operation::Gather(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "gather input must be f32")?;
            ensure_dtype(indices_dtype, DType::Si32, "gather indices must be si32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "gather output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let indices = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let idx_spec = operand_spec(&inst.operands[1], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_gather(
                    module,
                    &out_info.var,
                    &input,
                    &indices,
                    &out_info.spec,
                    &in_spec,
                    &idx_spec,
                    spec,
                )
            })?;
        }
        Operation::ScatterAdd(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let x_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let updates_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            ensure_dtype(x_dtype, DType::F32, "scatter_add input must be f32")?;
            ensure_dtype(
                indices_dtype,
                DType::Si32,
                "scatter_add indices must be si32",
            )?;
            ensure_dtype(updates_dtype, DType::F32, "scatter_add updates must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "scatter_add output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let x = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let indices = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let updates = operand_expr(&inst.operands[2], value_infos, module, literal_cache)?;
            let x_spec = operand_spec(&inst.operands[0], value_infos)?;
            let idx_spec = operand_spec(&inst.operands[1], value_infos)?;
            let updates_spec = operand_spec(&inst.operands[2], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_scatter_add(
                    module,
                    &out_info.var,
                    &x,
                    &indices,
                    &updates,
                    &out_info.spec,
                    &x_spec,
                    &idx_spec,
                    &updates_spec,
                    spec,
                )
            })?;
        }
        Operation::ScatterReduce(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let x_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let updates_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            ensure_dtype(x_dtype, DType::F32, "scatter_reduce input must be f32")?;
            ensure_dtype(
                indices_dtype,
                DType::Si32,
                "scatter_reduce indices must be si32",
            )?;
            ensure_dtype(
                updates_dtype,
                DType::F32,
                "scatter_reduce updates must be f32",
            )?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "scatter_reduce output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let x = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let indices = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let updates = operand_expr(&inst.operands[2], value_infos, module, literal_cache)?;
            let x_spec = operand_spec(&inst.operands[0], value_infos)?;
            let idx_spec = operand_spec(&inst.operands[1], value_infos)?;
            let updates_spec = operand_spec(&inst.operands[2], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_scatter_reduce(
                    module,
                    &out_info.var,
                    &x,
                    &indices,
                    &updates,
                    &out_info.spec,
                    &x_spec,
                    &idx_spec,
                    &updates_spec,
                    spec,
                )
            })?;
        }
        Operation::DynamicSlice(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let start_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            match input_dtype {
                DType::F32 | DType::Si32 | DType::I1 => {}
                _ => {
                    return Err(ConversionError::new(
                        "dynamic_slice input must be f32, si32, or i1",
                    ))
                }
            }
            ensure_dtype(start_dtype, DType::Si32, "dynamic_slice start must be si32")?;
            ensure_dtype(
                out_info.spec.dtype,
                input_dtype,
                "dynamic_slice output must match input dtype",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let start = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_dynamic_slice(
                    module,
                    &out_info.var,
                    &input,
                    &start,
                    &out_info.spec,
                    &in_spec,
                    spec,
                    input_dtype,
                )
            })?;
        }
        Operation::DynamicUpdateSlice(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let update_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let start_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            match input_dtype {
                DType::F32 | DType::Si32 | DType::I1 => {}
                _ => {
                    return Err(ConversionError::new(
                        "dynamic_update_slice input must be f32, si32, or i1",
                    ))
                }
            }
            ensure_dtype(
                update_dtype,
                input_dtype,
                "dynamic_update_slice update must match input dtype",
            )?;
            ensure_dtype(
                start_dtype,
                DType::Si32,
                "dynamic_update_slice start must be si32",
            )?;
            ensure_dtype(
                out_info.spec.dtype,
                input_dtype,
                "dynamic_update_slice output must match input dtype",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let update = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let start = operand_expr(&inst.operands[2], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let update_spec = operand_spec(&inst.operands[1], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_dynamic_update_slice(
                    module,
                    &out_info.var,
                    &input,
                    &update,
                    &start,
                    &out_info.spec,
                    &in_spec,
                    &update_spec,
                    spec,
                    input_dtype,
                )
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

fn emit_take(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    indices_spec: &TensorSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let idx_dims = dims_usize(indices_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims.is_empty() {
        return Err(ConversionError::new("take input must be at least rank-1"));
    }
    let slice = in_dims[1..].iter().product::<usize>();
    let indices_count: usize = idx_dims.iter().product();
    let expected_out: Vec<usize> = idx_dims
        .iter()
        .copied()
        .chain(in_dims[1..].iter().copied())
        .collect();
    if out_dims != expected_out {
        return Err(ConversionError::new("take output shape mismatch"));
    }
    let axis_len = in_dims[0];

    let block = format!(
        r#"
            {{
              const float* input = (const float*){input};
              const int32_t* indices = (const int32_t*){indices};
              float* out = (float*){out};
              for (size_t i = 0; i < {indices_count}; ++i) {{
                int32_t idx = indices[i];
                if (idx < 0 || idx >= {axis_len}) {{ return -5; }}
                memcpy(out + i * {slice}, input + ((size_t)idx) * {slice}, {slice} * sizeof(float));
              }}
            }}
        "#
    );
    push_block(module, 1, &block);
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_gather(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    indices_spec: &TensorSpec,
    spec: &GatherSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let idx_dims = dims_usize(indices_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims.len() != idx_dims.len() {
        return Err(ConversionError::new("gather rank mismatch"));
    }
    if out_dims != idx_dims {
        return Err(ConversionError::new("gather output shape mismatch"));
    }
    let axis = axis_index(spec.axis, in_dims.len())?;

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices_vec, indent| {
        let out_idx = linear_index_expr(&out_dims, indices_vec);
        let dim = in_dims[axis];
        let header = format!(
            r#"
                int32_t idx = ((const int32_t*){indices})[{out_idx}];
                if (idx < 0 || idx >= {dim}) {{ return -5; }}
            "#
        );
        push_block(module, indent, &header);
        let mut in_indices = Vec::with_capacity(in_dims.len());
        for (pos, idx_name) in indices_vec.iter().enumerate() {
            if pos == axis {
                in_indices.push("idx".to_string());
            } else {
                in_indices.push(idx_name.clone());
            }
        }
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        push_block(
            module,
            indent,
            &format!("{out}[{out_idx}] = {input}[{in_idx}];"),
        );
    });
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_dynamic_slice(
    module: &mut String,
    out: &str,
    input: &str,
    start: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &DynamicSliceSpec,
    dtype: DType,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if spec.sizes.len() != in_dims.len() {
        return Err(ConversionError::new("dynamic_slice rank mismatch"));
    }
    if out_dims != spec.sizes {
        return Err(ConversionError::new("dynamic_slice output shape mismatch"));
    }

    let ctype = match dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        DType::I1 => "uint8_t",
        _ => {
            return Err(ConversionError::new(
                "dynamic_slice dtype must be f32, si32, or i1",
            ))
        }
    };
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

    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
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
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    update_spec: &TensorSpec,
    spec: &DynamicUpdateSliceSpec,
    dtype: DType,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let update_dims = dims_usize(update_spec)?;
    if spec.sizes.len() != in_dims.len() || spec.sizes != update_dims {
        return Err(ConversionError::new("dynamic_update_slice size mismatch"));
    }
    if out_dims != in_dims {
        return Err(ConversionError::new(
            "dynamic_update_slice output shape mismatch",
        ));
    }

    let ctype = match dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        DType::I1 => "uint8_t",
        _ => {
            return Err(ConversionError::new(
                "dynamic_update_slice dtype must be f32, si32, or i1",
            ))
        }
    };
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
    let byte_len = out_spec
        .byte_len()
        .ok_or_else(|| ConversionError::new("dynamic_update_slice byte length unknown"))?;
    push_block(module, 2, &format!("memcpy(out, input, {byte_len});"));
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
#[allow(clippy::too_many_arguments)]
fn emit_scatter_reduce(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    updates: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    indices_spec: &TensorSpec,
    updates_spec: &TensorSpec,
    spec: &ScatterReduceSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let idx_dims = dims_usize(indices_spec)?;
    let updates_dims = dims_usize(updates_spec)?;
    if in_dims != out_dims {
        return Err(ConversionError::new("scatter_reduce output shape mismatch"));
    }
    if idx_dims != updates_dims {
        return Err(ConversionError::new(
            "scatter_reduce indices/update shape mismatch",
        ));
    }
    if idx_dims.len() != in_dims.len() {
        return Err(ConversionError::new("scatter_reduce rank mismatch"));
    }
    let axis = axis_index(spec.axis, in_dims.len())?;

    let header = format!(
        r#"
            {{
              const float* input = (const float*){input};
              const int32_t* indices = (const int32_t*){indices};
              const float* updates = (const float*){updates};
              float* out = (float*){out};
        "#
    );
    push_block(module, 1, &header);
    let byte_len = out_spec
        .byte_len()
        .ok_or_else(|| ConversionError::new("scatter_reduce byte length unknown"))?;
    push_block(module, 2, &format!("memcpy(out, input, {byte_len});"));

    emit_loops_with_indices(module, &idx_dims, 2, "i", |module, idx_vec, indent| {
        let linear_idx = linear_index_expr(&idx_dims, idx_vec);
        let dim = in_dims[axis];
        let header = format!(
            r#"
                int32_t scatter_idx = indices[{linear_idx}];
                if (scatter_idx < 0 || scatter_idx >= {dim}) {{ return -6; }}
            "#
        );
        push_block(module, indent, &header);
        let mut out_indices = idx_vec.to_vec();
        out_indices[axis] = "scatter_idx".to_string();
        let out_idx = linear_index_expr(&out_dims, &out_indices);
        let update_expr = match spec.reduce {
            ScatterReduceKind::Add => format!("out[{out_idx}] += updates[{linear_idx}];"),
            ScatterReduceKind::Max => format!(
                "if (updates[{linear_idx}] > out[{out_idx}]) out[{out_idx}] = updates[{linear_idx}];"
            ),
            ScatterReduceKind::Min => format!(
                "if (updates[{linear_idx}] < out[{out_idx}]) out[{out_idx}] = updates[{linear_idx}];"
            ),
            ScatterReduceKind::Replace => format!("out[{out_idx}] = updates[{linear_idx}];"),
        };
        push_block(module, indent, &update_expr);
    });

    push_block(module, 1, "}");
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_scatter_add(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    updates: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    indices_spec: &TensorSpec,
    updates_spec: &TensorSpec,
    spec: &ScatterSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if in_dims != out_dims {
        return Err(ConversionError::new("scatter_add output shape mismatch"));
    }
    let idx_dims = dims_usize(indices_spec)?;
    let updates_dims = dims_usize(updates_spec)?;
    let idx_rank = idx_dims.len();
    let updates_rank = updates_dims.len();

    let index_vector_dim = spec.index_vector_dim.unwrap_or(idx_rank);
    if index_vector_dim > idx_rank {
        return Err(ConversionError::new("scatter_add index_vector_dim invalid"));
    }
    let index_vector_size = if index_vector_dim < idx_rank {
        idx_dims[index_vector_dim]
    } else {
        1
    };
    if spec.scatter_dims_to_operand_dims.len() != index_vector_size {
        return Err(ConversionError::new(
            "scatter_add scatter_dims_to_operand_dims size mismatch",
        ));
    }

    let mut update_window_set = std::collections::HashSet::new();
    for &dim in &spec.update_window_dims {
        if dim >= updates_rank || !update_window_set.insert(dim) {
            return Err(ConversionError::new(
                "scatter_add update_window_dims invalid",
            ));
        }
    }
    let mut inserted_set = std::collections::HashSet::new();
    for &dim in &spec.inserted_window_dims {
        if dim >= in_dims.len() || !inserted_set.insert(dim) {
            return Err(ConversionError::new(
                "scatter_add inserted_window_dims invalid",
            ));
        }
    }

    let update_scatter_dims: Vec<usize> = (0..updates_rank)
        .filter(|dim| !update_window_set.contains(dim))
        .collect();
    let scatter_indices_dims: Vec<usize> = (0..idx_rank)
        .filter(|dim| *dim != index_vector_dim)
        .collect();
    if update_scatter_dims.len() != scatter_indices_dims.len() {
        return Err(ConversionError::new(
            "scatter_add scatter dimension rank mismatch",
        ));
    }

    let mut operand_window_dims = Vec::new();
    for dim in 0..in_dims.len() {
        if spec.scatter_dims_to_operand_dims.contains(&dim) || inserted_set.contains(&dim) {
            continue;
        }
        operand_window_dims.push(dim);
    }
    if operand_window_dims.len() != spec.update_window_dims.len() {
        return Err(ConversionError::new(
            "scatter_add operand window rank mismatch",
        ));
    }

    let header = format!(
        r#"
            {{
              const float* input = (const float*){input};
              const int32_t* indices = (const int32_t*){indices};
              const float* updates = (const float*){updates};
              float* out = (float*){out};
        "#
    );
    push_block(module, 1, &header);
    let byte_len = out_spec
        .byte_len()
        .ok_or_else(|| ConversionError::new("scatter_add byte length unknown"))?;
    push_block(module, 2, &format!("memcpy(out, input, {byte_len});"));

    emit_loops_with_indices(
        module,
        &updates_dims,
        2,
        "u",
        |module, u_indices, indent| {
            let update_idx = linear_index_expr(&updates_dims, u_indices);
            for (vec_idx, &operand_dim) in spec.scatter_dims_to_operand_dims.iter().enumerate() {
                let mut idx_indices = Vec::with_capacity(idx_rank);
                let mut scatter_pos = 0usize;
                for dim in 0..idx_rank {
                    if dim == index_vector_dim {
                        idx_indices.push(vec_idx.to_string());
                    } else {
                        let update_dim = update_scatter_dims[scatter_pos];
                        idx_indices.push(u_indices[update_dim].clone());
                        scatter_pos += 1;
                    }
                }
                let idx_expr = linear_index_expr(&idx_dims, &idx_indices);
                let dim = in_dims[operand_dim];
                let header = format!(
                    r#"
                        int32_t scatter_{vec_idx} = indices[{idx_expr}];
                        if (scatter_{vec_idx} < 0 || scatter_{vec_idx} >= {dim}) {{ return -6; }}
                    "#
                );
                push_block(module, indent, &header);
            }

            let mut out_indices = vec!["0".to_string(); in_dims.len()];
            for (pos, &operand_dim) in spec.scatter_dims_to_operand_dims.iter().enumerate() {
                out_indices[operand_dim] = format!("scatter_{pos}");
            }
            for (pos, &operand_dim) in operand_window_dims.iter().enumerate() {
                let update_dim = spec.update_window_dims[pos];
                out_indices[operand_dim] = u_indices[update_dim].clone();
            }
            for &operand_dim in &spec.inserted_window_dims {
                out_indices[operand_dim] = "0".to_string();
            }
            let out_idx = linear_index_expr(&out_dims, &out_indices);
            push_block(
                module,
                indent,
                &format!("out[{out_idx}] += updates[{update_idx}];"),
            );
        },
    );

    push_block(module, 1, "}");
    Ok(())
}
