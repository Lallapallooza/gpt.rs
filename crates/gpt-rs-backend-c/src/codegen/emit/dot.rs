use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, DotGeneralSpec, Instruction, Operation, TensorSpec};
use gpt_rs::profiling::tensor_spec_signature;

use super::super::profile::{
    backend_operation_label, emit_profiled_op, matmul_work_stats, register_op_profile_binary,
    OpProfile,
};
use super::super::types::MatmulCacheEntry;
use super::super::utils::{dims_usize, emit_loops_with_indices, linear_index_expr, push_block};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_input_index, operand_spec, output_info,
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
        matmul_caches,
        ..
    } = ctx;

    match &inst.op {
        Operation::DotGeneral(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            let lhs_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            let rhs_dtype = operand_dtype(&inst.operands[1], value_infos)?;
            ensure_dtype(lhs_dtype, DType::F32, "dot_general lhs must be f32")?;
            ensure_dtype(rhs_dtype, DType::F32, "dot_general rhs must be f32")?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "dot_general output must be f32",
            )?;
            let lhs = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
            let rhs = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
            let lhs_spec = operand_spec(&inst.operands[0], value_infos)?;
            let rhs_spec = operand_spec(&inst.operands[1], value_infos)?;
            let rhs_input_index = matmul_caches
                .as_ref()
                .and_then(|_| operand_input_index(&inst.operands[1], value_infos));
            let caches = matmul_caches.as_deref_mut();
            if !emit_matmul_if_supported(
                module,
                &out_info.var,
                &lhs,
                &rhs,
                &out_info.spec,
                &lhs_spec,
                &rhs_spec,
                spec,
                matmul_profile,
                caches,
                rhs_input_index,
            )? {
                let label = backend_operation_label(&inst.op);
                let op_id = register_op_profile_binary(
                    matmul_profile,
                    label,
                    &out_info.spec,
                    &lhs_spec,
                    &rhs_spec,
                )?;
                emit_profiled_op(module, op_id, |module| {
                    emit_dot_general(
                        module,
                        &out_info.var,
                        &lhs,
                        &rhs,
                        &out_info.spec,
                        &lhs_spec,
                        &rhs_spec,
                        spec,
                    )
                })?;
            }
        }
        _ => return Ok(false),
    }

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn emit_dot_general(
    module: &mut String,
    out: &str,
    lhs: &str,
    rhs: &str,
    out_spec: &TensorSpec,
    lhs_spec: &TensorSpec,
    rhs_spec: &TensorSpec,
    spec: &DotGeneralSpec,
) -> ConversionResult<()> {
    let lhs_dims = dims_usize(lhs_spec)?;
    let rhs_dims = dims_usize(rhs_spec)?;
    let out_dims = dims_usize(out_spec)?;

    let lhs_rank = lhs_dims.len();
    let rhs_rank = rhs_dims.len();

    let batch_count = spec.batch_lhs.len();
    if batch_count != spec.batch_rhs.len() {
        return Err(ConversionError::new("dot_general batch rank mismatch"));
    }

    let mut is_lhs_batch = vec![false; lhs_rank];
    let mut is_rhs_batch = vec![false; rhs_rank];
    for (lhs_axis, rhs_axis) in spec.batch_lhs.iter().zip(spec.batch_rhs.iter()) {
        if *lhs_axis >= lhs_rank || *rhs_axis >= rhs_rank {
            return Err(ConversionError::new("dot_general batch axis out of range"));
        }
        is_lhs_batch[*lhs_axis] = true;
        is_rhs_batch[*rhs_axis] = true;
    }

    let mut is_lhs_contract = vec![false; lhs_rank];
    let mut is_rhs_contract = vec![false; rhs_rank];
    for (lhs_axis, rhs_axis) in spec.contract_lhs.iter().zip(spec.contract_rhs.iter()) {
        if *lhs_axis >= lhs_rank || *rhs_axis >= rhs_rank {
            return Err(ConversionError::new(
                "dot_general contract axis out of range",
            ));
        }
        is_lhs_contract[*lhs_axis] = true;
        is_rhs_contract[*rhs_axis] = true;
    }

    let lhs_free: Vec<usize> = (0..lhs_rank)
        .filter(|axis| !is_lhs_batch[*axis] && !is_lhs_contract[*axis])
        .collect();
    let rhs_free: Vec<usize> = (0..rhs_rank)
        .filter(|axis| !is_rhs_batch[*axis] && !is_rhs_contract[*axis])
        .collect();

    let contract_dims: Vec<usize> = spec
        .contract_lhs
        .iter()
        .map(|&axis| lhs_dims[axis])
        .collect();

    let expected_out_rank = batch_count + lhs_free.len() + rhs_free.len();
    if out_dims.len() != expected_out_rank {
        return Err(ConversionError::new("dot_general output rank mismatch"));
    }

    emit_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
        push_block(module, indent, "float acc = 0.0f;");
        emit_loops_with_indices(
            module,
            &contract_dims,
            indent,
            "k",
            |module, k_indices, indent| {
                let mut lhs_indices = vec!["0".to_string(); lhs_rank];
                let mut rhs_indices = vec!["0".to_string(); rhs_rank];

                for (pos, axis) in spec.batch_lhs.iter().enumerate() {
                    lhs_indices[*axis] = out_indices[pos].clone();
                }
                for (pos, axis) in spec.batch_rhs.iter().enumerate() {
                    rhs_indices[*axis] = out_indices[pos].clone();
                }

                let mut out_pos = batch_count;
                for axis in &lhs_free {
                    lhs_indices[*axis] = out_indices[out_pos].clone();
                    out_pos += 1;
                }
                for axis in &rhs_free {
                    rhs_indices[*axis] = out_indices[out_pos].clone();
                    out_pos += 1;
                }

                for (pos, axis) in spec.contract_lhs.iter().enumerate() {
                    lhs_indices[*axis] = k_indices[pos].clone();
                }
                for (pos, axis) in spec.contract_rhs.iter().enumerate() {
                    rhs_indices[*axis] = k_indices[pos].clone();
                }

                let lhs_idx = linear_index_expr(&lhs_dims, &lhs_indices);
                let rhs_idx = linear_index_expr(&rhs_dims, &rhs_indices);
                push_block(
                    module,
                    indent,
                    &format!("acc += {lhs}[{lhs_idx}] * {rhs}[{rhs_idx}];"),
                );
            },
        );
        let out_idx = linear_index_expr(&out_dims, out_indices);
        push_block(module, indent, &format!("{out}[{out_idx}] = acc;"));
    });
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn emit_matmul_if_supported(
    module: &mut String,
    out: &str,
    lhs: &str,
    rhs: &str,
    out_spec: &TensorSpec,
    lhs_spec: &TensorSpec,
    rhs_spec: &TensorSpec,
    spec: &DotGeneralSpec,
    matmul_profile: &mut OpProfile,
    mut matmul_caches: Option<&mut Vec<MatmulCacheEntry>>,
    rhs_input_index: Option<usize>,
) -> ConversionResult<bool> {
    let lhs_dims = dims_usize(lhs_spec)?;
    let rhs_dims = dims_usize(rhs_spec)?;
    let out_dims = dims_usize(out_spec)?;
    let lhs_rank = lhs_dims.len();
    let rhs_rank = rhs_dims.len();

    if spec.batch_lhs.is_empty()
        && spec.batch_rhs.is_empty()
        && lhs_rank == 2
        && rhs_rank == 2
        && spec.contract_lhs.as_slice() == [1]
        && spec.contract_rhs.as_slice() == [0]
    {
        let (m, k) = (lhs_dims[0], lhs_dims[1]);
        let (k2, n) = (rhs_dims[0], rhs_dims[1]);
        if k != k2 || out_dims.as_slice() != [m, n] {
            return Ok(false);
        }
        let lhs_sig = tensor_spec_signature(lhs_spec);
        let rhs_sig = tensor_spec_signature(rhs_spec);
        let out_sig = tensor_spec_signature(out_spec);
        let signature = format!("lhs={lhs_sig} rhs={rhs_sig} out={out_sig}");
        let work = matmul_work_stats(1, m, n, k);
        let op_id = matmul_profile.register("backend.dot_general", signature, work);
        let mut use_cache = false;
        if let (Some(caches), Some(rhs_index)) = (matmul_caches.as_mut(), rhs_input_index) {
            (*caches).push(MatmulCacheEntry {
                op_id,
                rhs_index,
                n,
                k,
            });
            use_cache = true;
        }
        let header = format!(
            r#"
                {{
                  const float* a = (const float*){lhs};
                  const float* b = (const float*){rhs};
                  float* c = {out};
            "#
        );
        push_block(module, 1, &header);
        emit_profiled_op(module, op_id, |module| {
            let call = if use_cache {
                format!(
                    r#"
                        gpt_rs_c_matmul_f32_cached_b(a, b, c, {m}, {n}, {k}, &gpt_rs_bcache_{op_id});
                    "#
                )
            } else {
                format!(
                    r#"
                        gpt_rs_c_matmul_f32(a, b, c, {m}, {n}, {k});
                    "#
                )
            };
            push_block(module, 2, &call);
            Ok(())
        })?;
        push_block(module, 1, "}");
        return Ok(true);
    }

    if spec.batch_lhs.as_slice() == [0]
        && spec.batch_rhs.as_slice() == [0]
        && lhs_rank == 3
        && rhs_rank == 3
        && spec.contract_lhs.as_slice() == [2]
        && spec.contract_rhs.as_slice() == [1]
    {
        let (batch, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
        let (batch2, k2, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);
        if batch != batch2 || k != k2 || out_dims.as_slice() != [batch, m, n] {
            return Ok(false);
        }
        let lhs_sig = tensor_spec_signature(lhs_spec);
        let rhs_sig = tensor_spec_signature(rhs_spec);
        let out_sig = tensor_spec_signature(out_spec);
        let signature = format!("lhs={lhs_sig} rhs={rhs_sig} out={out_sig}");
        let work = matmul_work_stats(batch, m, n, k);
        let op_id = matmul_profile.register("backend.dot_general", signature, work);
        let lhs_stride = m * k;
        let rhs_stride = k * n;
        let out_stride = m * n;
        let header = format!(
            r#"
                {{
                  const float* a = (const float*){lhs};
                  const float* b = (const float*){rhs};
                  float* c = {out};
            "#
        );
        push_block(module, 1, &header);
        emit_profiled_op(module, op_id, |module| {
            let loop_block = format!(
                r#"
                    for (size_t batch = 0; batch < {batch}; ++batch) {{
                      const float* a_base = a + batch * {lhs_stride};
                      const float* b_base = b + batch * {rhs_stride};
                      float* c_base = c + batch * {out_stride};
                      gpt_rs_c_matmul_f32(a_base, b_base, c_base, {m}, {n}, {k});
                    }}
                "#
            );
            push_block(module, 2, &loop_block);
            Ok(())
        })?;
        push_block(module, 1, "}");
        return Ok(true);
    }

    Ok(false)
}
