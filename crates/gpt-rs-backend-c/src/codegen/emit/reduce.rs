use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::{ReduceAttrs, ReduceWindowAttrs, TensorOp};
use gpt_rs_ir::dialects::tile::CombineKind;
use gpt_rs_ir::ir::{DType, TensorType};

use super::super::profile::register_op_profile_unary;
use super::super::types::{Instruction, Kind};
use super::super::utils::{
    dims_usize, emit_loops_with_indices, emit_parallel_loops_with_indices, linear_index_expr,
    push_block,
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
    let op_id = if let Some((spec, fold)) = reduction(op) {
        {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "reduce input must be f32")?;
            ensure_dtype(out_info.ty.dtype, DType::F32, "reduce output must be f32")?;
            ensure_dtype(
                spec.accum.unwrap_or(DType::F32),
                DType::F32,
                "reduce accumulator must be f32",
            )?;
            let label = inst.label;
            let input_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.ty, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_reduce(
                module,
                &out_info.var,
                &input,
                (&out_info.ty, &input_spec),
                spec,
                &fold,
            )?;
            op_id
        }
    } else if let Some((spec, fold)) = window_reduction(op) {
        {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(input_dtype, DType::F32, "reduce_window input must be f32")?;
            ensure_dtype(
                spec.accum.unwrap_or(DType::F32),
                DType::F32,
                "reduce_window accumulator must be f32",
            )?;
            ensure_dtype(
                out_info.ty.dtype,
                DType::F32,
                "reduce_window output must be f32",
            )?;
            let label = inst.label;
            let in_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id = register_op_profile_unary(matmul_profile, label, &out_info.ty, &in_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            let types = (&out_info.ty, &in_spec);
            emit_reduce_window(module, &out_info.var, &input, types, spec, &fold)?;
            op_id
        }
    } else {
        return Ok(None);
    };

    Ok(Some(op_id))
}

/// The attributes and the fold of a reduction that the C backend emits: an f32 sum, maximum or
/// minimum.
fn reduction(op: &TensorOp) -> Option<(&ReduceAttrs, CombineKind)> {
    Some(match op {
        TensorOp::ReduceSum(spec) => (&spec.0, CombineKind::Sum),
        TensorOp::ReduceMax(spec) => (&spec.0, CombineKind::Max),
        TensorOp::ReduceMin(spec) => (&spec.0, CombineKind::Min),
        _ => return None,
    })
}

/// As [`reduction`], for a reduction of windows.
fn window_reduction(op: &TensorOp) -> Option<(&ReduceWindowAttrs, CombineKind)> {
    Some(match op {
        TensorOp::ReduceWindowSum(spec) => (&spec.0, CombineKind::Sum),
        TensorOp::ReduceWindowMax(spec) => (&spec.0, CombineKind::Max),
        TensorOp::ReduceWindowMin(spec) => (&spec.0, CombineKind::Min),
        _ => return None,
    })
}

/// The panic message for a combine kind other than `sum`, `max` and `min`: the C backend reduces
/// f32 only.
const F32_ONLY: &str = "the C backend's reductions sum, max or min f32 values";

fn emit_reduce(
    module: &mut String,
    out: &str,
    input: &str,
    (out_spec, in_spec): (&TensorType, &TensorType),
    spec: &ReduceAttrs,
    fold: &CombineKind,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let rank = in_dims.len();
    let mut axes_set = vec![false; rank];
    for axis in spec.reduced_axes(rank)? {
        axes_set[axis] = true;
    }
    // Reduced axes in memory order, so the innermost loop walks the fastest-varying one.
    let reduce_axes: Vec<usize> = (0..rank).filter(|axis| axes_set[*axis]).collect();
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

    emit_parallel_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
        push_block(module, indent, accumulator_decl(fold));
        let update = |module: &mut String, red_indices: &[String], indent: usize| {
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
            let update = accumulator_update(fold, &format!("{input}[{in_idx}]"));
            push_block(module, indent, &update);
        };
        match reduce_dims.split_last() {
            None => update(module, &[], indent),
            Some((&inner, outer)) => {
                emit_loops_with_indices(module, outer, indent, "r", |module, outer_idx, indent| {
                    // When the innermost reduced axis is contiguous, the compiler may reassociate
                    // this one accumulator, so the loop vectorizes. That is exact for the order
                    // keys of `max` and `min`. No other value in the module is reassociated.
                    if reduce_axes.last() == Some(&(rank - 1)) {
                        let op = match fold {
                            CombineKind::Sum => "+",
                            CombineKind::Max => "max",
                            CombineKind::Min => "min",
                            _ => unreachable!("{F32_ONLY}"),
                        };
                        push_block(
                            module,
                            indent,
                            &format!("#pragma omp simd reduction({op}:acc)"),
                        );
                    }
                    let r = format!("r{}", outer.len());
                    push_block(
                        module,
                        indent,
                        &format!("for (size_t {r} = 0; {r} < {inner}; ++{r}) {{"),
                    );
                    let mut red_indices = outer_idx.to_vec();
                    red_indices.push(r);
                    update(module, &red_indices, indent + 1);
                    push_block(module, indent, "}");
                });
            }
        }

        let out_idx = linear_index_expr(&out_dims, out_indices);
        let result = accumulator_result(fold);
        push_block(module, indent, &format!("{out}[{out_idx}] = {result};"));
    });
    Ok(())
}

/// Declares the accumulator `acc` of a reduction, holding its identity. `max` and `min`
/// accumulate the order keys from the kernel preamble, whose unsigned max and min propagate NaN
/// and order `-0.0` below `+0.0`.
fn accumulator_decl(kind: &CombineKind) -> &'static str {
    match kind {
        CombineKind::Sum => "float acc = 0.0f;",
        CombineKind::Max => "uint32_t acc = gpt_rs_f32_to_max_key(-INFINITY);",
        CombineKind::Min => "uint32_t acc = gpt_rs_f32_to_min_key(INFINITY);",
        _ => unreachable!("{F32_ONLY}"),
    }
}

/// Folds the f32 expression `value` into `acc`.
fn accumulator_update(kind: &CombineKind, value: &str) -> String {
    match kind {
        CombineKind::Sum => format!("acc += {value};"),
        CombineKind::Max => format!(
            "{{ uint32_t key = gpt_rs_f32_to_max_key({value}); acc = key > acc ? key : acc; }}"
        ),
        CombineKind::Min => format!(
            "{{ uint32_t key = gpt_rs_f32_to_min_key({value}); acc = key < acc ? key : acc; }}"
        ),
        _ => unreachable!("{F32_ONLY}"),
    }
}

/// The f32 value of `acc`.
fn accumulator_result(kind: &CombineKind) -> &'static str {
    match kind {
        CombineKind::Sum => "acc",
        CombineKind::Max | CombineKind::Min => "gpt_rs_key_to_f32(acc)",
        _ => unreachable!("{F32_ONLY}"),
    }
}

fn emit_reduce_window(
    module: &mut String,
    out: &str,
    input: &str,
    (out_spec, in_spec): (&TensorType, &TensorType),
    spec: &ReduceWindowAttrs,
    fold: &CombineKind,
) -> Result<()> {
    let in_dims = dims_usize(in_spec)?;
    let out_dims = dims_usize(out_spec)?;
    if spec.window_dims.len() != in_dims.len()
        || spec.strides.len() != in_dims.len()
        || spec.padding.len() != in_dims.len()
        || spec.base_dilation.len() != in_dims.len()
        || spec.window_dilation.len() != in_dims.len()
    {
        return Err(anyhow!("reduce_window rank mismatch"));
    }
    if spec.base_dilation.iter().any(|&d| d != 1) || spec.window_dilation.iter().any(|&d| d != 1) {
        return Err(anyhow!("reduce_window dilation not supported yet"));
    }

    emit_parallel_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
        push_block(module, indent, accumulator_decl(fold));
        // Every window of an empty input is empty, so each output is the value `acc` starts with.
        if !in_dims.contains(&0) {
            emit_loops_with_indices(
                module,
                &spec.window_dims,
                indent,
                "w",
                |module, win_indices, indent| {
                    let indices = (out_indices, win_indices);
                    let step = window_fold(spec, fold, input, &in_dims, &out_dims, indices);
                    push_block(module, indent, &step);
                },
            );
        }
        let out_idx = linear_index_expr(&out_dims, out_indices);
        let result = accumulator_result(fold);
        push_block(module, indent, &format!("{out}[{out_idx}] = {result};"));
    });

    Ok(())
}

/// C statements that fold the element of `input` at the output and window loop `indices` into
/// `acc`. On the axes where a window can leave the input, the load reads a clamped index and a
/// select replaces an element outside the input with the reduction's identity. The statements
/// have no branch, so the loop over the innermost output axis vectorizes.
fn window_fold(
    spec: &ReduceWindowAttrs,
    fold: &CombineKind,
    input: &str,
    in_dims: &[usize],
    out_dims: &[usize],
    (out_indices, win_indices): (&[String], &[String]),
) -> String {
    let mut lines = Vec::new();
    let mut indices = Vec::with_capacity(in_dims.len());
    let mut inside = Vec::new();
    for (axis, &dim) in in_dims.iter().enumerate() {
        let (out_index, win_index) = (&out_indices[axis], &win_indices[axis]);
        let (stride, pad_low) = (spec.strides[axis], spec.padding[axis].0);
        let index = format!("((int64_t){out_index} * {stride} + {win_index} - {pad_low})");
        if always_inside(spec, axis, out_dims[axis], dim) {
            indices.push(index);
            continue;
        }
        lines.push(format!("int64_t i{axis} = {index};"));
        inside.push(format!("(i{axis} >= 0) & (i{axis} < {dim})"));
        let last = dim - 1;
        indices.push(format!(
            "(i{axis} < 0 ? 0 : i{axis} < {dim} ? i{axis} : {last})"
        ));
    }
    let index = linear_index_expr(in_dims, &indices);
    lines.push(format!("float x = {input}[{index}];"));
    let value = if inside.is_empty() {
        "x".to_string()
    } else {
        format!("(({}) ? x : {})", inside.join(" & "), identity(fold))
    };
    lines.push(accumulator_update(fold, &value));
    lines.join("\n")
}

/// Whether every window of `axis` reads inside the input, so its index needs no bounds check.
fn always_inside(spec: &ReduceWindowAttrs, axis: usize, out_dim: usize, in_dim: usize) -> bool {
    let (Some(out_last), Some(window_last)) = (
        out_dim.checked_sub(1),
        spec.window_dims[axis].checked_sub(1),
    ) else {
        return true;
    };
    spec.padding[axis].0 == 0 && out_last * spec.strides[axis] + window_last < in_dim
}

/// The f32 value that leaves any accumulator of `kind` unchanged.
fn identity(kind: &CombineKind) -> &'static str {
    match kind {
        CombineKind::Sum => "-0.0f",
        CombineKind::Max => "-INFINITY",
        CombineKind::Min => "INFINITY",
        _ => unreachable!("{F32_ONLY}"),
    }
}
