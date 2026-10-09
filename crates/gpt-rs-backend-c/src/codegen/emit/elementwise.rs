use std::collections::HashMap;

use anyhow::{anyhow, bail, ensure, Result};
use gpt_rs_ir::dialects::tensor::{BinaryKind, CompareKind, TensorOp, UnaryKind};
use gpt_rs_ir::module::Value;
use gpt_rs_ir::types::DType;

use crate::targets::{binary_expr, fused_input_fits, unary_expr};

use super::super::profile::{
    register_op_profile_binary, register_op_profile_generic, register_op_profile_unary,
};
use super::super::types::{Elementwise, Instruction, Kind, NodeOp, NodeRef, ValueInfo};
use super::super::utils::{
    c_type, dims_usize, emit_flat_loop, emit_memcpy, emit_parallel_loops_with_indices,
    linear_index_expr, push_block,
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
        TensorOp::Unary(unary) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(
                input_dtype,
                DType::F32,
                "elementwise unary input must be f32",
            )?;
            ensure_dtype(
                out_info.ty.dtype,
                DType::F32,
                "elementwise unary output must be f32",
            )?;
            let label = inst.label;
            let input_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.ty, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            emit_unary(
                module,
                unary.kind,
                &out_info.var,
                &input,
                out_info.elem_count,
            )?;
            op_id
        }
        TensorOp::Binary(binary) => {
            let out_info = output_info(value_infos, inst.result())?;
            let lhs_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let rhs_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let dtype = out_info.ty.dtype;
            ensure!(
                matches!(dtype, DType::F32 | DType::Si32) && lhs_dtype == dtype && rhs_dtype == dtype,
                "elementwise binary ops take f32 or si32 operands of the result dtype, got {lhs_dtype} and {rhs_dtype} to {dtype}"
            );
            let label = inst.label;
            let lhs_spec = operand_type(&inst.operands[0], value_infos)?;
            let rhs_spec = operand_type(&inst.operands[1], value_infos)?;
            let op_id = register_op_profile_binary(
                matmul_profile,
                label,
                &out_info.ty,
                &lhs_spec,
                &rhs_spec,
            )?;
            let lhs = operand_expr(&inst.operands[0], value_infos)?;
            let rhs = operand_expr(&inst.operands[1], value_infos)?;
            emit_binary(
                module,
                binary.kind,
                dtype,
                &out_info.var,
                (&lhs, &rhs),
                out_info.elem_count,
            )?;
            op_id
        }
        TensorOp::Cast(cast) => {
            let out_info = output_info(value_infos, inst.result())?;
            let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let label = inst.label;
            let input_spec = operand_type(&inst.operands[0], value_infos)?;
            let op_id =
                register_op_profile_unary(matmul_profile, label, &out_info.ty, &input_spec)?;
            let input = operand_expr(&inst.operands[0], value_infos)?;
            if cast.to != out_info.ty.dtype {
                return Err(anyhow!("cast output dtype does not match instruction type"));
            }
            emit_cast(
                module,
                &out_info.var,
                &input,
                out_info.ty.dtype,
                input_dtype,
                out_info.elem_count,
            )?;
            op_id
        }
        TensorOp::Compare(compare) => {
            let out_info = output_info(value_infos, inst.result())?;
            let lhs_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let rhs_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            if lhs_dtype != rhs_dtype {
                return Err(anyhow!("compare operands must have the same dtype"));
            }
            match lhs_dtype {
                DType::F32 | DType::Si32 | DType::I1 => {}
                _ => return Err(anyhow!("compare operands must be f32, si32, or i1")),
            }
            ensure_dtype(out_info.ty.dtype, DType::I1, "compare output must be i1")?;
            let label = inst.label;
            let lhs_spec = operand_type(&inst.operands[0], value_infos)?;
            let rhs_spec = operand_type(&inst.operands[1], value_infos)?;
            let op_id = register_op_profile_binary(
                matmul_profile,
                label,
                &out_info.ty,
                &lhs_spec,
                &rhs_spec,
            )?;
            let lhs = operand_expr(&inst.operands[0], value_infos)?;
            let rhs = operand_expr(&inst.operands[1], value_infos)?;
            emit_compare(
                module,
                compare.kind,
                &out_info.var,
                &lhs,
                &rhs,
                out_info.elem_count,
                lhs_dtype,
            )?;
            op_id
        }
        TensorOp::Select(_) => {
            let out_info = output_info(value_infos, inst.result())?;
            let pred_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let t_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            let f_dtype = operand_dtype(&inst.operands[2], value_infos)?;
            ensure_dtype(pred_dtype, DType::I1, "select predicate must be i1")?;
            ensure_dtype(t_dtype, DType::F32, "select on_true must be f32")?;
            ensure_dtype(f_dtype, DType::F32, "select on_false must be f32")?;
            ensure_dtype(out_info.ty.dtype, DType::F32, "select output must be f32")?;
            let label = inst.label;
            let input_specs = operand_types(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_generic(matmul_profile, label, &out_info.ty, &input_specs)?;
            let pred = operand_expr(&inst.operands[0], value_infos)?;
            let on_true = operand_expr(&inst.operands[1], value_infos)?;
            let on_false = operand_expr(&inst.operands[2], value_infos)?;
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
        _ => return Ok(None),
    };
    Ok(Some(op_id))
}

fn emit_unary(
    module: &mut String,
    op: UnaryKind,
    out: &str,
    input: &str,
    elem_count: usize,
) -> Result<()> {
    let expr = unary_expr(op, "src[i]");
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
    op: BinaryKind,
    dtype: DType,
    out: &str,
    (lhs, rhs): (&str, &str),
    elem_count: usize,
) -> Result<()> {
    let (ctype, expr) = match dtype {
        DType::Si32 => ("int32_t", int_binary_expr(op, "lhs_ptr[i]", "rhs_ptr[i]")?),
        _ => ("float", binary_expr(op, "lhs_ptr[i]", "rhs_ptr[i]")),
    };
    emit_flat_loop(
        module,
        &[(ctype, "lhs_ptr", lhs), (ctype, "rhs_ptr", rhs)],
        (ctype, out),
        elem_count,
        &format!("out[i] = {expr};"),
    );
    Ok(())
}

/// `lhs op rhs` of `int32_t` operands. Sums, differences and products wrap, as the spec says, so
/// they compute on `uint32_t`.
fn int_binary_expr(op: BinaryKind, lhs: &str, rhs: &str) -> Result<String> {
    let wrapping = |symbol: &str| format!("(int32_t)((uint32_t){lhs} {symbol} (uint32_t){rhs})");
    Ok(match op {
        BinaryKind::Add => wrapping("+"),
        BinaryKind::Sub => wrapping("-"),
        BinaryKind::Mul => wrapping("*"),
        BinaryKind::Maximum => format!("({lhs} > {rhs} ? {lhs} : {rhs})"),
        BinaryKind::Minimum => format!("({lhs} < {rhs} ? {lhs} : {rhs})"),
        other => bail!("the C backend has no si32 {other:?}"),
    })
}
fn emit_cast(
    module: &mut String,
    out: &str,
    input: &str,
    out_dtype: DType,
    in_dtype: DType,
    elem_count: usize,
) -> Result<()> {
    if out_dtype == in_dtype {
        let elem_size = in_dtype
            .size_in_bytes()
            .ok_or_else(|| anyhow!("cast dtype size unknown"))?;
        let byte_len = elem_count
            .checked_mul(elem_size)
            .ok_or_else(|| anyhow!("cast byte length overflow"))?;
        emit_memcpy(module, out, input, byte_len);
        return Ok(());
    }
    let in_ctype = c_type(in_dtype)?;
    let out_ctype = c_type(out_dtype)?;
    let body = match (in_dtype, out_dtype) {
        // Truncates toward zero and saturates. NaN gives 0. 2^31 is the first f32 above
        // INT32_MAX, and -2^31 is INT32_MIN itself.
        (DType::F32, DType::Si32) => r#"
                float v = in[i];
                out[i] = isnan(v) ? 0
                    : v >= 2147483648.0f ? INT32_MAX
                    : v < -2147483648.0f ? INT32_MIN
                    : (int32_t)v;
            "#
        .to_string(),
        // Every value but zero is true, NaN included.
        (DType::F32, DType::I1) => r#"
                out[i] = (uint8_t)(in[i] != 0.0f);
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
        (DType::F8E4M3, DType::F32) => r#"
                out[i] = gpt_rs_f8e4m3_to_f32(in[i]);
            "#
        .to_string(),
        // Every float8_e4m3fn value is a bf16 value, so the narrowing is exact.
        (DType::F8E4M3, DType::Bf16) => r#"
                out[i] = gpt_rs_f32_to_bf16(gpt_rs_f8e4m3_to_f32(in[i]));
            "#
        .to_string(),
        _ => {
            return Err(anyhow!("cast dtype combination not supported by C codegen"));
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
    kind: CompareKind,
    out: &str,
    lhs: &str,
    rhs: &str,
    elem_count: usize,
    dtype: DType,
) -> Result<()> {
    let expr = match kind {
        CompareKind::Less => "lhs[i] < rhs[i]",
        CompareKind::LessEqual => "lhs[i] <= rhs[i]",
        CompareKind::Equal => "lhs[i] == rhs[i]",
        CompareKind::GreaterEqual => "lhs[i] >= rhs[i]",
        CompareKind::Greater => "lhs[i] > rhs[i]",
        CompareKind::NotEqual => "lhs[i] != rhs[i]",
    };
    let ctype = match dtype {
        DType::F32 => "float",
        DType::Si32 => "int32_t",
        DType::I1 => "uint8_t",
        _ => return Err(anyhow!("compare operands must be f32, si32, or i1")),
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
) -> Result<()> {
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
/// Emits a [`Kind::Elementwise`] instruction: one loop over the result that computes the chain's
/// ops on its inputs.
pub(super) fn emit_elementwise(
    module: &mut String,
    chain: &Elementwise,
    operands: &[Value],
    out_info: &ValueInfo,
    value_infos: &HashMap<Value, ValueInfo>,
) -> Result<()> {
    let expected = if chain.store_bf16 {
        DType::Bf16
    } else {
        DType::F32
    };
    ensure_dtype(
        out_info.ty.dtype,
        expected,
        "a fused elementwise result is f32 or stored as bf16",
    )?;
    let out_dims = dims_usize(&out_info.ty)?;

    struct InputInfo {
        var: String,
        dims: Vec<usize>,
        starts: Option<Vec<usize>>,
    }

    let mut inputs: Vec<InputInfo> = Vec::with_capacity(operands.len());
    for (operand, input) in operands.iter().zip(&chain.inputs) {
        let dtype = operand_dtype(operand, value_infos)?;
        ensure_dtype(dtype, DType::F32, "fused elementwise operand must be f32")?;
        let dims = dims_usize(&operand_type(operand, value_infos)?)?;
        let starts = input.starts.clone();
        if !fused_input_fits(&out_dims, &dims, starts.as_deref()) {
            return Err(anyhow!(
                "fused elementwise operand {dims:?} (slice starts {starts:?}) does not fit output {out_dims:?}"
            ));
        }
        let var = operand_expr(operand, value_infos)?;
        inputs.push(InputInfo { var, dims, starts });
    }

    let reference = |node: NodeRef| match node {
        NodeRef::Input(index) => format!("in{index}v"),
        NodeRef::Node(index) => format!("t{index}"),
    };
    let node_exprs: Vec<String> = chain
        .nodes
        .iter()
        .map(|node| {
            let lhs = reference(node.lhs);
            match (node.op, node.rhs) {
                (NodeOp::Unary(kind), _) => unary_expr(kind, &lhs),
                (NodeOp::Binary(kind), Some(rhs)) => binary_expr(kind, &lhs, &reference(rhs)),
                (NodeOp::Binary(_), None) => unreachable!("a binary node has two operands"),
            }
        })
        .collect();
    let node_count = node_exprs.len();
    let store_bf16 = chain.store_bf16;

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
    let out_type = c_type(out_info.ty.dtype)?;
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
