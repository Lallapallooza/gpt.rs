use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    ArgMaxSpec, DType, Instruction, Operation, ReduceKind, ReduceSpec, ReduceWindowSpec,
    SegmentReduceKind, SegmentReduceSpec, TensorSpec, TopKSpec,
};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_generic,
    register_op_profile_multi_output, register_op_profile_unary,
};
use super::super::types::ValueInfo;
use super::super::utils::{
    axis_index, dims_usize, emit_loops_with_indices, linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_spec, operand_specs, output_info,
    output_infos,
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
        Operation::Reduce(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "reduce input must be f32")?;
            ensure_dtype(out_info.spec.dtype, DType::F32, "reduce output must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_reduce(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &input_spec,
                    spec,
                )
            })?;
        }
        Operation::ArgMax(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "argmax input must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::Si32,
                "argmax output must be si32",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.spec, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_argmax(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &input_spec,
                    spec,
                )
            })?;
        }
        Operation::ReduceWindow(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "reduce_window input must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "reduce_window output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.spec, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            emit_profiled_op(module, op_id, |module| {
                emit_reduce_window(
                    module,
                    &out_info.var,
                    &input,
                    &out_info.spec,
                    &in_spec,
                    spec,
                )
            })?;
        }
        Operation::TopK(spec) => {
            let outputs = output_infos(value_infos, inst.id)?;
            if outputs.len() != 2 {
                return Err(ConversionError::new("top_k expects two outputs"));
            }
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "top_k input must be f32")?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_multi_output(matmul_profile, label, &outputs, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_topk(module, &input, &in_spec, outputs[0], outputs[1], spec)
            })?;
        }
        Operation::SegmentReduce(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let indices_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "segment_reduce input must be f32")?;
            ensure_dtype(
                indices_dtype,
                DType::Si32,
                "segment_reduce indices must be si32",
            )?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "segment_reduce output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.spec, &input_specs)?;
            let input = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let indices = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let in_spec = operand_spec(&inst.operands[0], value_infos)?;
            let idx_spec = operand_spec(&inst.operands[1], value_infos)?;
            emit_profiled_op(module, op_id, |module| {
                emit_segment_reduce(
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
        _ => return Ok(false),
    }

    Ok(true)
}

fn emit_reduce(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &ReduceSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let rank = in_dims.len();
    let mut axes_set = vec![false; rank];
    let mut reduce_axes = Vec::new();
    for axis in &spec.axes {
        let idx = axis_index(*axis as isize, rank)?;
        if !axes_set[idx] {
            axes_set[idx] = true;
            reduce_axes.push(idx);
        }
    }
    let reduce_dims: Vec<usize> = reduce_axes.iter().map(|&i| in_dims[i]).collect();
    let mut input_axis_to_output = vec![None; rank];
    if spec.keepdims {
        for axis in 0..rank {
            if !axes_set[axis] {
                input_axis_to_output[axis] = Some(axis);
            }
        }
    } else {
        let mut out_pos = 0usize;
        for axis in 0..rank {
            if !axes_set[axis] {
                input_axis_to_output[axis] = Some(out_pos);
                out_pos += 1;
            }
        }
    }

    emit_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
        let init = match spec.kind {
            ReduceKind::Sum => "0.0f",
            ReduceKind::Max => "-INFINITY",
            ReduceKind::Min => "INFINITY",
        };
        push_block(module, indent, &format!("float acc = {init};"));

        emit_loops_with_indices(
            module,
            &reduce_dims,
            indent,
            "r",
            |module, red_indices, indent| {
                let mut in_indices = vec!["0".to_string(); rank];
                for axis in 0..rank {
                    if axes_set[axis] {
                        let pos = reduce_axes.iter().position(|&a| a == axis).unwrap_or(0);
                        in_indices[axis] = red_indices[pos].clone();
                    } else {
                        let out_pos = input_axis_to_output[axis].unwrap_or(0);
                        in_indices[axis] = out_indices[out_pos].clone();
                    }
                }
                let in_idx = linear_index_expr(&in_dims, &in_indices);
                let update = match spec.kind {
                    ReduceKind::Sum => format!("acc += {input}[{in_idx}];"),
                    ReduceKind::Max => {
                        format!("if ({input}[{in_idx}] > acc) acc = {input}[{in_idx}];")
                    }
                    ReduceKind::Min => {
                        format!("if ({input}[{in_idx}] < acc) acc = {input}[{in_idx}];")
                    }
                };
                push_block(module, indent, &update);
            },
        );

        let out_idx = linear_index_expr(&out_dims, out_indices);
        push_block(module, indent, &format!("{out}[{out_idx}] = acc;"));
    });
    Ok(())
}
fn emit_argmax(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &ArgMaxSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let axis = axis_index(spec.axis, in_dims.len())?;
    let out_dims = dims_usize(out_spec)?;
    let mut expected = Vec::with_capacity(in_dims.len());
    for (idx, dim) in in_dims.iter().enumerate() {
        if idx == axis {
            if spec.keepdims {
                expected.push(1);
            }
        } else {
            expected.push(*dim);
        }
    }
    if out_dims != expected {
        return Err(ConversionError::new("argmax output shape mismatch"));
    }
    let outer = if axis == 0 {
        1usize
    } else {
        in_dims[..axis].iter().product::<usize>()
    };
    let inner = if axis + 1 >= in_dims.len() {
        1usize
    } else {
        in_dims[axis + 1..].iter().product::<usize>()
    };
    let axis_len = in_dims[axis];

    let block = format!(
        r#"
            {{
              const float* input = (const float*){input};
              int32_t* out = (int32_t*){out};
              for (size_t o = 0; o < {outer}; ++o) {{
                for (size_t i = 0; i < {inner}; ++i) {{
                  size_t base = o * {axis_len} * {inner} + i;
                  float max_val = input[base];
                  int32_t max_idx = 0;
                  for (size_t a = 1; a < {axis_len}; ++a) {{
                    float v = input[base + a * {inner}];
                    if (v > max_val) {{ max_val = v; max_idx = (int32_t)a; }}
                  }}
                  size_t out_index = o * {inner} + i;
                  out[out_index] = max_idx;
                }}
              }}
            }}
        "#
    );
    push_block(module, 1, &block);

    Ok(())
}
fn emit_reduce_window(
    module: &mut String,
    out: &str,
    input: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    spec: &ReduceWindowSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if spec.window_dims.len() != in_dims.len()
        || spec.strides.len() != in_dims.len()
        || spec.padding.len() != in_dims.len()
        || spec.base_dilation.len() != in_dims.len()
        || spec.window_dilation.len() != in_dims.len()
    {
        return Err(ConversionError::new("reduce_window rank mismatch"));
    }
    if spec.base_dilation.iter().any(|&d| d != 1) || spec.window_dilation.iter().any(|&d| d != 1) {
        return Err(ConversionError::new(
            "reduce_window dilation not supported yet",
        ));
    }

    emit_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
        let init = match spec.reduce {
            ReduceKind::Sum => "0.0f",
            ReduceKind::Max => "-INFINITY",
            ReduceKind::Min => "INFINITY",
        };
        push_block(module, indent, &format!("float acc = {init};"));

        emit_loops_with_indices(
            module,
            &spec.window_dims,
            indent,
            "w",
            |module, win_indices, indent| {
                let mut in_indices = Vec::with_capacity(in_dims.len());
                for axis in 0..in_dims.len() {
                    let stride = spec.strides[axis];
                    let pad_low = spec.padding[axis].0;
                    let out_index = &out_indices[axis];
                    let win_index = &win_indices[axis];
                    let expr =
                        format!("((int64_t){out_index} * {stride} + {win_index} - {pad_low})");
                    in_indices.push(expr);
                }

                let in_bounds = in_indices
                    .iter()
                    .enumerate()
                    .map(|(axis, idx)| {
                        let dim = in_dims[axis];
                        format!("({idx} >= 0 && {idx} < {dim})")
                    })
                    .collect::<Vec<_>>()
                    .join(" && ");
                let in_idx = linear_index_expr(&in_dims, &in_indices);
                let update = match spec.reduce {
                    ReduceKind::Sum => format!("acc += {input}[{in_idx}];"),
                    ReduceKind::Max => {
                        format!("if ({input}[{in_idx}] > acc) acc = {input}[{in_idx}];")
                    }
                    ReduceKind::Min => {
                        format!("if ({input}[{in_idx}] < acc) acc = {input}[{in_idx}];")
                    }
                };
                let block = format!(
                    r#"
                        if ({in_bounds}) {{
                          {update}
                        }}
                    "#
                );
                push_block(module, indent, &block);
            },
        );

        let out_idx = linear_index_expr(&out_dims, out_indices);
        push_block(module, indent, &format!("{out}[{out_idx}] = acc;"));
    });

    Ok(())
}
fn emit_topk(
    module: &mut String,
    input: &str,
    in_spec: &TensorSpec,
    out_values: &ValueInfo,
    out_indices: &ValueInfo,
    spec: &TopKSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(&out_values.spec)?;
    let out_idx_dims = dims_usize(&out_indices.spec)?;
    if out_dims != out_idx_dims {
        return Err(ConversionError::new("top_k output shape mismatch"));
    }
    if out_dims.len() != in_dims.len() {
        return Err(ConversionError::new("top_k rank mismatch"));
    }
    let axis = axis_index(spec.axis, in_dims.len())?;
    if out_dims[axis] != spec.k {
        return Err(ConversionError::new("top_k output axis mismatch"));
    }
    if in_dims[axis] < spec.k {
        return Err(ConversionError::new("top_k k exceeds axis size"));
    }
    if out_indices.spec.dtype != spec.indices_dtype {
        return Err(ConversionError::new("top_k indices dtype mismatch"));
    }
    if out_values.spec.dtype != DType::F32 {
        return Err(ConversionError::new(
            "top_k values dtype must be f32 for C backend",
        ));
    }
    if out_indices.spec.dtype != DType::Si32 {
        return Err(ConversionError::new(
            "top_k indices dtype must be si32 for C backend",
        ));
    }

    let mut outer_dims = Vec::new();
    for (idx, dim) in in_dims.iter().enumerate() {
        if idx != axis {
            outer_dims.push(*dim);
        }
    }
    let mut in_indices = Vec::with_capacity(in_dims.len());
    let mut out_indices_vec = Vec::with_capacity(out_dims.len());
    let mut outer_pos = 0usize;
    for d in 0..in_dims.len() {
        if d == axis {
            in_indices.push("a".to_string());
            out_indices_vec.push("kk".to_string());
        } else {
            in_indices.push(format!("o{outer_pos}"));
            out_indices_vec.push(format!("o{outer_pos}"));
            outer_pos += 1;
        }
    }
    let in_idx_expr = linear_index_expr(&in_dims, &in_indices);
    let out_idx_expr = linear_index_expr(&out_dims, &out_indices_vec);

    let out_values_var = &out_values.var;
    let out_indices_var = &out_indices.var;
    let axis_dim = in_dims[axis];
    let k = spec.k;
    let header = format!(
        r#"
            {{
              const float* input = (const float*){input};
              float* out_values = (float*){out_values_var};
              int32_t* out_indices = (int32_t*){out_indices_var};
              const size_t axis_dim = {axis_dim};
              const size_t k = {k};
        "#
    );
    push_block(module, 1, &header);

    emit_loops_with_indices(
        module,
        &outer_dims,
        2,
        "o",
        |module, _outer_indices, indent| {
            push_block(
                module,
                indent,
                r#"
                    uint8_t* selected = (uint8_t*)calloc(axis_dim, 1);
                    if (!selected) { return -4; }
                    for (size_t kk = 0; kk < k; ++kk) {
                "#,
            );
            if spec.largest {
                let block = format!(
                    r#"
                        float best = -INFINITY;
                        int32_t best_idx = 0;
                        for (size_t a = 0; a < axis_dim; ++a) {{
                          if (selected[a]) {{ continue; }}
                          float v = input[{in_idx_expr}];
                          if (v > best) {{ best = v; best_idx = (int32_t)a; }}
                        }}
"#
                );
                push_block(module, indent + 1, &block);
            } else {
                let block = format!(
                    r#"
                        float best = INFINITY;
                        int32_t best_idx = 0;
                        for (size_t a = 0; a < axis_dim; ++a) {{
                          if (selected[a]) {{ continue; }}
                          float v = input[{in_idx_expr}];
                          if (v < best) {{ best = v; best_idx = (int32_t)a; }}
                        }}
"#
                );
                push_block(module, indent + 1, &block);
            }
            let tail = format!(
                r#"
                        selected[best_idx] = 1;
                        out_values[{out_idx_expr}] = best;
                        out_indices[{out_idx_expr}] = best_idx;
"#
            );
            push_block(module, indent + 1, &tail);
            push_block(module, indent, "    }");
            push_block(module, indent, "    free(selected);");
        },
    );

    push_block(module, 1, "    }");
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_segment_reduce(
    module: &mut String,
    out: &str,
    input: &str,
    indices: &str,
    out_spec: &TensorSpec,
    in_spec: &TensorSpec,
    indices_spec: &TensorSpec,
    spec: &SegmentReduceSpec,
) -> ConversionResult<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let idx_dims = dims_usize(indices_spec)?;
    if idx_dims.len() != 1 {
        return Err(ConversionError::new(
            "segment_reduce indices must be rank-1",
        ));
    }
    if in_dims.is_empty() {
        return Err(ConversionError::new(
            "segment_reduce input must be rank >= 1",
        ));
    }
    if idx_dims[0] != in_dims[0] {
        return Err(ConversionError::new(
            "segment_reduce indices length mismatch",
        ));
    }
    if out_dims.len() != in_dims.len() {
        return Err(ConversionError::new("segment_reduce output rank mismatch"));
    }
    if out_dims[0] != spec.num_segments {
        return Err(ConversionError::new("segment_reduce num_segments mismatch"));
    }
    if out_dims[1..] != in_dims[1..] {
        return Err(ConversionError::new("segment_reduce output shape mismatch"));
    }

    let inner_dims = &in_dims[1..];
    let init_value = match spec.kind {
        SegmentReduceKind::Sum => "0.0f",
        SegmentReduceKind::Max => "-INFINITY",
    };

    let out_elems = out_spec
        .shape
        .element_count()
        .ok_or_else(|| ConversionError::new("segment_reduce dynamic shape"))?;
    let in_len = in_dims[0];
    let segment_count = spec.num_segments;
    let header = format!(
        r#"
            {{
              const float* input = (const float*){input};
              const int32_t* indices = (const int32_t*){indices};
              float* out = (float*){out};
              for (size_t i = 0; i < {out_elems}; ++i) {{ out[i] = {init_value}; }}
              for (size_t s = 0; s < {in_len}; ++s) {{
                int32_t seg = indices[s];
                if (seg < 0 || seg >= {segment_count}) {{ return -6; }}
        "#
    );
    push_block(module, 1, &header);
    emit_loops_with_indices(module, inner_dims, 3, "k", |module, inner_idx, indent| {
        let mut in_indices = Vec::with_capacity(in_dims.len());
        in_indices.push("s".to_string());
        in_indices.extend(inner_idx.iter().cloned());
        let mut out_indices = Vec::with_capacity(out_dims.len());
        out_indices.push("seg".to_string());
        out_indices.extend(inner_idx.iter().cloned());
        let in_idx = linear_index_expr(&in_dims, &in_indices);
        let out_idx = linear_index_expr(&out_dims, &out_indices);
        let update = match spec.kind {
            SegmentReduceKind::Sum => format!("out[{out_idx}] += input[{in_idx}];"),
            SegmentReduceKind::Max => {
                format!("if (input[{in_idx}] > out[{out_idx}]) out[{out_idx}] = input[{in_idx}];")
            }
        };
        push_block(module, indent, &update);
    });
    push_block(module, 2, "}");
    push_block(module, 1, "}");
    Ok(())
}
