use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::shape_helpers::contiguous_strides_or_error;
use gpt_rs::backend::spec::{DType, DotGeneralSpec, Instruction, Operation, TensorSpec};

use crate::kernels::LINEAR_PACK_MIN_ROWS;
use crate::targets::TARGET_LINEAR_NT_F32_BF16;

use super::super::profile::{
    backend_operation_label, bpack_cache_arg, matmul_work_stats, register_op_profile_binary,
    signature_binary,
};
use super::super::utils::{
    c_type, dims_usize, emit_loops_with_indices, emit_parallel_loops_with_indices,
    linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_input_index, operand_spec, output_info,
};
use super::EmitContext;

pub(super) fn emit_instruction(
    inst: &Instruction,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<Option<usize>> {
    let linear_spec;
    let spec = match &inst.op {
        Operation::DotGeneral(spec) => spec,
        Operation::CustomCall(call) if call.target == TARGET_LINEAR_NT_F32_BF16 => {
            // `x [M, K] . w [N, K]`.
            linear_spec = DotGeneralSpec {
                batch_lhs: vec![],
                batch_rhs: vec![],
                contract_lhs: vec![1],
                contract_rhs: vec![1],
                accum_dtype: Some(DType::F32),
                out_dtype: Some(DType::F32),
            };
            &linear_spec
        }
        _ => return Ok(None),
    };
    let [lhs, rhs] = inst.operands.as_slice() else {
        return Err(ConversionError::new("dot_general expects two operands"));
    };
    let lhs_dtype = operand_dtype(lhs, ctx.value_infos)?;
    let rhs_dtype = operand_dtype(rhs, ctx.value_infos)?;
    if lhs_dtype != rhs_dtype && matches!(inst.op, Operation::DotGeneral(_)) {
        return Err(ConversionError::new(format!(
            "dot_general operands must share a dtype, got {lhs_dtype:?} x {rhs_dtype:?}"
        )));
    }
    emit_dot(inst, spec, ctx).map(Some)
}

fn emit_dot(
    inst: &Instruction,
    spec: &DotGeneralSpec,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<usize> {
    let EmitContext {
        module,
        value_infos,
        literal_cache,
        matmul_profile,
        matmul_caches,
        ..
    } = ctx;
    let out_info = output_info(value_infos, inst.id)?;
    ensure_dtype(
        out_info.spec.dtype,
        DType::F32,
        "dot_general output must be f32",
    )?;
    let lhs_spec = operand_spec(&inst.operands[0], value_infos)?;
    let rhs_spec = operand_spec(&inst.operands[1], value_infos)?;
    let lhs = operand_expr(&inst.operands[0], value_infos, module, literal_cache)?;
    let rhs = operand_expr(&inst.operands[1], value_infos, module, literal_cache)?;
    let label = backend_operation_label(&inst.op);
    let out = &out_info.var;

    let layout = strided_batched_layout(
        &dims_usize(&lhs_spec)?,
        &dims_usize(&rhs_spec)?,
        &dims_usize(&out_info.spec)?,
        spec,
    )?;
    let no_kernel = || {
        ConversionError::new(format!(
            "no C kernel for {:?} x {:?} dot_general with {spec:?}",
            lhs_spec.dtype, rhs_spec.dtype
        ))
    };
    let Some(layout) = layout else {
        if lhs_spec.dtype != DType::F32 || rhs_spec.dtype != DType::F32 {
            return Err(no_kernel());
        }
        let op_id = register_op_profile_binary(
            matmul_profile,
            label,
            &out_info.spec,
            &lhs_spec,
            &rhs_spec,
        )?;
        emit_dot_general(
            module,
            out,
            &lhs,
            &rhs,
            &out_info.spec,
            &lhs_spec,
            &rhs_spec,
            spec,
        )?;
        return Ok(op_id);
    };

    let (form, kernel) =
        dot_kernel(&layout, lhs_spec.dtype, rhs_spec.dtype).ok_or_else(no_kernel)?;
    let elem_bytes = |spec: &TensorSpec| {
        spec.dtype
            .size_in_bytes()
            .ok_or_else(|| ConversionError::new("dot_general operand dtype has no byte size"))
    };
    let StridedBatchedLayout {
        batch,
        m,
        n,
        k,
        sab,
        sam,
        sak,
        sbb,
        sbk,
        sbn,
    } = layout;
    let work = matmul_work_stats(
        batch,
        m,
        n,
        k,
        elem_bytes(&lhs_spec)?,
        elem_bytes(&rhs_spec)?,
    );
    let signature = signature_binary(&out_info.spec, &lhs_spec, &rhs_spec);
    let op_id = matmul_profile.register(label, signature, work);
    let (xtype, wtype) = (c_type(lhs_spec.dtype)?, c_type(rhs_spec.dtype)?);
    let args =
        format!("(const {xtype}*){lhs}, (const {wtype}*){rhs}, {out}, {batch}, {m}, {n}, {k}");
    let call = match form {
        DotForm::Nt => format!("{kernel}({args}, {sab}, {sam}, {sbb}, {sbn});"),
        DotForm::Gemm => {
            // When b is an entry input, the kernel keeps a packed copy of b across calls. A single
            // row of a gets no copy, because the kernel multiplies it with b in place.
            let rhs_input =
                operand_input_index(&inst.operands[1], value_infos).filter(|_| batch == 1 && m > 1);
            let cache = bpack_cache_arg(
                matmul_caches.as_deref_mut(),
                rhs_input,
                op_id,
                n,
                k,
                sbk,
                sbn,
            );
            format!("{kernel}({args}, {sab}, {sam}, {sak}, {sbb}, {sbk}, {sbn}, {cache}, NULL);")
        }
    };
    push_block(module, 1, &call);
    Ok(op_id)
}

/// Kernel families for single-contraction dots.
#[derive(Debug, Clone, Copy)]
enum DotForm {
    /// The linear kernels. Both operands are contiguous along K, as in the PyTorch linear layout
    /// `x [.., M, K] . w [.., N, K]`.
    Nt,
    /// The packed GEMM, whose packing absorbs any operand strides.
    Gemm,
}

fn dot_kernel(
    layout: &StridedBatchedLayout,
    lhs: DType,
    rhs: DType,
) -> Option<(DotForm, &'static str)> {
    let nt = layout.sak == 1 && layout.sbk == 1;
    match (lhs, rhs) {
        (DType::F32, DType::F32) if nt && layout.m < LINEAR_PACK_MIN_ROWS => {
            Some((DotForm::Nt, "gpt_rs_c_linear_nt_f32"))
        }
        (DType::F32, DType::F32) => Some((DotForm::Gemm, "gpt_rs_c_matmul_f32")),
        (DType::F32, DType::Bf16) if nt => Some((DotForm::Nt, "gpt_rs_c_linear_nt_f32_bf16")),
        (DType::Bf16, DType::Bf16) if nt => Some((DotForm::Nt, "gpt_rs_c_linear_nt_bf16_bf16")),
        _ => None,
    }
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

    emit_parallel_loops_with_indices(module, &out_dims, 2, "o", |module, out_indices, indent| {
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
/// Row-major strides of a single-contraction `dot_general` viewed as
/// `c[b][i][j] = sum_p a[b*sab + i*sam + p*sak] * b[b*sbb + p*sbk + j*sbn]` with `c` contiguous
/// `[batch, m, n]`.
struct StridedBatchedLayout {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    sab: usize,
    sam: usize,
    sak: usize,
    sbb: usize,
    sbk: usize,
    sbn: usize,
}

struct OperandAxes {
    batch: Option<usize>,
    contract: usize,
    free: Option<usize>,
}

/// Returns the axes of one operand, or `None` when the operand has more than one free axis.
fn operand_axes(
    dims: &[usize],
    batch: Option<usize>,
    contract: usize,
) -> ConversionResult<Option<OperandAxes>> {
    let rank = dims.len();
    if batch.is_some_and(|axis| axis >= rank) || contract >= rank || batch == Some(contract) {
        return Err(ConversionError::new(format!(
            "dot_general axes (batch {batch:?}, contract {contract}) are invalid for rank {rank}"
        )));
    }
    let free: Vec<usize> = (0..rank)
        .filter(|axis| Some(*axis) != batch && *axis != contract)
        .collect();
    Ok(match free.as_slice() {
        [] => Some(OperandAxes {
            batch,
            contract,
            free: None,
        }),
        [axis] => Some(OperandAxes {
            batch,
            contract,
            free: Some(*axis),
        }),
        _ => None,
    })
}

/// Maps a `dot_general` onto [`StridedBatchedLayout`]. Each operand must have at most one batch
/// dim, exactly one contraction dim, and at most one free dim. Returns `None` for other specs.
fn strided_batched_layout(
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    spec: &DotGeneralSpec,
) -> ConversionResult<Option<StridedBatchedLayout>> {
    let (([], [], [contract_lhs], [contract_rhs]) | ([_], [_], [contract_lhs], [contract_rhs])) = (
        spec.batch_lhs.as_slice(),
        spec.batch_rhs.as_slice(),
        spec.contract_lhs.as_slice(),
        spec.contract_rhs.as_slice(),
    ) else {
        return Ok(None);
    };
    let (Some(lhs), Some(rhs)) = (
        operand_axes(lhs_dims, spec.batch_lhs.first().copied(), *contract_lhs)?,
        operand_axes(rhs_dims, spec.batch_rhs.first().copied(), *contract_rhs)?,
    ) else {
        return Ok(None);
    };
    let overflow = || ConversionError::new("dot_general operand size overflows");
    let lhs_strides = contiguous_strides_or_error(lhs_dims, overflow)?;
    let rhs_strides = contiguous_strides_or_error(rhs_dims, overflow)?;
    let extent = |dims: &[usize], strides: &[usize], axis: Option<usize>| {
        axis.map_or((1, 0), |axis| (dims[axis], strides[axis]))
    };
    let k = lhs_dims[lhs.contract];
    let (batch, sab) = extent(lhs_dims, &lhs_strides, lhs.batch);
    let (rhs_batch, sbb) = extent(rhs_dims, &rhs_strides, rhs.batch);
    let (m, sam) = extent(lhs_dims, &lhs_strides, lhs.free);
    let (n, sbn) = extent(rhs_dims, &rhs_strides, rhs.free);
    let expected_out: Vec<usize> = [
        lhs.batch.map(|_| batch),
        lhs.free.map(|_| m),
        rhs.free.map(|_| n),
    ]
    .into_iter()
    .flatten()
    .collect();
    if rhs_dims[rhs.contract] != k || rhs_batch != batch || expected_out != out_dims {
        return Err(ConversionError::new(format!(
            "dot_general shapes {lhs_dims:?} x {rhs_dims:?} -> {out_dims:?} do not match {spec:?}"
        )));
    }
    Ok(Some(StridedBatchedLayout {
        batch,
        m,
        n,
        k,
        sab,
        sam,
        sak: lhs_strides[lhs.contract],
        sbb,
        sbk: rhs_strides[rhs.contract],
        sbn,
    }))
}
