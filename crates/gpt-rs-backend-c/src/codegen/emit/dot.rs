use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::{DotAxes, DotGeneral, TensorOp};
use gpt_rs_ir::module::Value;
use gpt_rs_ir::types::{DType, TensorType};

use super::super::profile::{matmul_work_stats, register_op_profile_binary, signature_binary};
use super::super::types::{Instruction, Kind};
use super::super::utils::{
    c_type, contiguous_strides, dims_usize, emit_loops_with_indices,
    emit_parallel_loops_with_indices, linear_index_expr, push_block,
};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_type, output_info,
};
use super::EmitContext;
use crate::kernels::{LINEAR_PACK_MIN_ROWS, PANEL_N};
use crate::targets::linear_kernel;

pub(super) fn emit_instruction(
    inst: &Instruction<'_>,
    ctx: &mut EmitContext<'_>,
) -> Result<Option<usize>> {
    let &Kind::Op(TensorOp::DotGeneral(spec)) = &inst.kind else {
        return Ok(None);
    };
    let &[lhs, rhs] = inst.operands.as_slice() else {
        return Err(anyhow!("dot_general expects two operands"));
    };
    if reads_panels(
        spec,
        &operand_type(&lhs, ctx.value_infos)?,
        &operand_type(&rhs, ctx.value_infos)?,
    ) {
        return emit_matmul_packed(inst, ctx).map(Some);
    }
    let lhs_dtype = operand_dtype(&lhs, ctx.value_infos)?;
    let rhs_dtype = operand_dtype(&rhs, ctx.value_infos)?;
    if lhs_dtype != rhs_dtype {
        return Err(anyhow!(
            "dot_general operands must share a dtype, got {lhs_dtype} x {rhs_dtype}"
        ));
    }
    emit_dot(inst, spec, None, ctx).map(Some)
}

/// Emits a [`Kind::Linear`] instruction, `x [M, K] . w [N, K]` with `w` as stored, scaled by the
/// third operand when there is one.
pub(super) fn emit_linear(inst: &Instruction<'_>, ctx: &mut EmitContext<'_>) -> Result<usize> {
    let scale = match inst.operands.as_slice() {
        [_, _] => None,
        [_, _, scale] => Some(scale),
        _ => return Err(anyhow!("a linear kernel reads [x, w] or [x, w, s]")),
    };
    let spec = DotGeneral::contract([1], [1]).with_dtypes(DType::F32, DType::F32);
    emit_dot(inst, &spec, scale, ctx)
}

/// Whether `spec` contracts the last axis of `x` with a weight in the GEMM's panels,
/// `[N / PANEL_N, K, PANEL_N]`, as the C backend's packing pass lays it out.
fn reads_panels(spec: &DotGeneral, x: &TensorType, w: &TensorType) -> bool {
    let panels = matches!(w.static_dims().as_deref(), Some(&[_, _, PANEL_N]));
    let Ok(axes) = spec.axes(x.rank(), w.rank()) else {
        return false;
    };
    panels
        && axes.batch_lhs.is_empty()
        && axes.batch_rhs.is_empty()
        && axes.contract_rhs[..] == [1]
        && axes.contract_lhs[..] == [x.rank() - 1]
}

/// Emits `lhs . rhs`. A `scale` operand is the one-element scale of a dequantized `rhs`.
fn emit_dot(
    inst: &Instruction<'_>,
    spec: &DotGeneral,
    scale: Option<&Value>,
    ctx: &mut EmitContext<'_>,
) -> Result<usize> {
    let EmitContext {
        module,
        value_infos,
        matmul_profile,
    } = ctx;
    let out_info = output_info(value_infos, inst.result())?;
    ensure_dtype(
        out_info.ty.dtype,
        DType::F32,
        "dot_general output must be f32",
    )?;
    let lhs_spec = operand_type(&inst.operands[0], value_infos)?;
    let rhs_spec = operand_type(&inst.operands[1], value_infos)?;
    let lhs = operand_expr(&inst.operands[0], value_infos)?;
    let rhs = operand_expr(&inst.operands[1], value_infos)?;
    let scale_arg = match scale {
        Some(scale) => {
            let scale_spec = operand_type(scale, value_infos)?;
            if scale_spec.element_count() != Some(1) {
                return Err(anyhow!("a linear weight scale must have one element"));
            }
            let expr = operand_expr(scale, value_infos)?;
            let arg = format!(", (const {}*){expr}", c_type(scale_spec.dtype)?);
            Some((scale_spec.dtype, arg))
        }
        None => None,
    };
    let label = inst.label;
    let out = &out_info.var;
    let axes = spec.axes(lhs_spec.rank(), rhs_spec.rank())?;

    let layout = strided_batched_layout(
        &dims_usize(&lhs_spec)?,
        &dims_usize(&rhs_spec)?,
        &dims_usize(&out_info.ty)?,
        &axes,
    )?;
    let no_kernel = || {
        anyhow!(
            "no C kernel for {} x {} dot_general with {spec:?}",
            lhs_spec.dtype,
            rhs_spec.dtype
        )
    };
    let Some(layout) = layout else {
        if lhs_spec.dtype != DType::F32 || rhs_spec.dtype != DType::F32 || scale_arg.is_some() {
            return Err(no_kernel());
        }
        let op_id =
            register_op_profile_binary(matmul_profile, label, &out_info.ty, &lhs_spec, &rhs_spec)?;
        emit_dot_general(
            module,
            out,
            &lhs,
            &rhs,
            &out_info.ty,
            &lhs_spec,
            &rhs_spec,
            &axes,
        )?;
        return Ok(op_id);
    };

    let scale_dtype = scale_arg.as_ref().map(|(dtype, _)| *dtype);
    let (form, kernel) =
        dot_kernel(&layout, lhs_spec.dtype, rhs_spec.dtype, scale_dtype).ok_or_else(no_kernel)?;
    let elem_bytes = |spec: &TensorType| {
        spec.dtype
            .size_in_bytes()
            .ok_or_else(|| anyhow!("dot_general operand dtype has no byte size"))
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
    let signature = signature_binary(&out_info.ty, &lhs_spec, &rhs_spec);
    let op_id = matmul_profile.register(label, signature, work);
    let (xtype, wtype) = (c_type(lhs_spec.dtype)?, c_type(rhs_spec.dtype)?);
    let args =
        format!("(const {xtype}*){lhs}, (const {wtype}*){rhs}, {out}, {batch}, {m}, {n}, {k}");
    let call = match form {
        DotForm::Nt => {
            let scale_suffix = scale_arg.as_ref().map_or("", |(_, arg)| arg.as_str());
            format!("{kernel}({args}, {sab}, {sam}, {sbb}, {sbn}{scale_suffix});")
        }
        DotForm::Gemm => {
            format!("{kernel}({args}, {sab}, {sam}, {sak}, {sbb}, {sbk}, {sbn}, 0, NULL);")
        }
    };
    push_block(module, 1, &call);
    Ok(op_id)
}

/// Emits the GEMM of `x [.., K]` and `w` packed as `[N / PANEL_N, K, PANEL_N]`, whose result
/// `[.., N / PANEL_N, PANEL_N]` holds `x . w` as `[.., N]`.
fn emit_matmul_packed(inst: &Instruction<'_>, ctx: &mut EmitContext<'_>) -> Result<usize> {
    let EmitContext {
        module,
        value_infos,
        matmul_profile,
    } = ctx;
    let [x, packed] = inst.operands.as_slice() else {
        return Err(anyhow!("the packed GEMM reads [x, packed]"));
    };
    let out_info = output_info(value_infos, inst.result())?;
    let x_spec = operand_type(x, value_infos)?;
    let packed_spec = operand_type(packed, value_infos)?;
    for spec in [&out_info.ty, &x_spec, &packed_spec] {
        ensure_dtype(spec.dtype, DType::F32, "the packed GEMM is f32")?;
    }
    let (x_dims, packed_dims, out_dims) = (
        dims_usize(&x_spec)?,
        dims_usize(&packed_spec)?,
        dims_usize(&out_info.ty)?,
    );
    let mismatch =
        || anyhow!("the packed GEMM does not take {x_dims:?} x {packed_dims:?} -> {out_dims:?}");
    let (Some((&k, rows)), &[panels, packed_k, PANEL_N]) =
        (x_dims.split_last(), packed_dims.as_slice())
    else {
        return Err(mismatch());
    };
    let m: usize = rows.iter().product();
    let n = panels * PANEL_N;
    let expected: Vec<usize> = rows.iter().copied().chain([panels, PANEL_N]).collect();
    if packed_k != k || out_dims != expected {
        return Err(mismatch());
    }
    let label = inst.label;
    let signature = signature_binary(&out_info.ty, &x_spec, &packed_spec);
    let op_id = matmul_profile.register(label, signature, matmul_work_stats(1, m, n, k, 4, 4));
    let x = operand_expr(x, value_infos)?;
    let packed = operand_expr(packed, value_infos)?;
    let out = &out_info.var;
    push_block(
        module,
        1,
        &format!(
            "gpt_rs_c_matmul_f32((const float*){x}, NULL, {out}, 1, {m}, {n}, {k}, 0, {k}, 1, 0, 0, 0, (const float*){packed}, NULL);"
        ),
    );
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

/// The kernel for `lhs . rhs`, where `rhs` is dequantized with a scale of dtype `scale` when one
/// is given.
fn dot_kernel(
    layout: &StridedBatchedLayout,
    lhs: DType,
    rhs: DType,
    scale: Option<DType>,
) -> Option<(DotForm, &'static str)> {
    let nt = layout.sak == 1 && layout.sbk == 1;
    match (lhs, rhs, scale) {
        (DType::F32, DType::F32, None) if nt && layout.m < LINEAR_PACK_MIN_ROWS => {
            Some((DotForm::Nt, "gpt_rs_c_linear_nt_f32"))
        }
        (DType::F32, DType::F32, None) => Some((DotForm::Gemm, "gpt_rs_c_matmul_f32")),
        (DType::Bf16, DType::Bf16, None) => Some((DotForm::Gemm, "gpt_rs_c_matmul_bf16")),
        _ if nt => Some((DotForm::Nt, linear_kernel(lhs, rhs, scale)?)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_dot_general(
    module: &mut String,
    out: &str,
    lhs: &str,
    rhs: &str,
    out_spec: &TensorType,
    lhs_spec: &TensorType,
    rhs_spec: &TensorType,
    spec: &DotAxes,
) -> Result<()> {
    let lhs_dims = dims_usize(lhs_spec)?;
    let rhs_dims = dims_usize(rhs_spec)?;
    let out_dims = dims_usize(out_spec)?;

    let lhs_rank = lhs_dims.len();
    let rhs_rank = rhs_dims.len();

    let batch_count = spec.batch_lhs.len();
    if batch_count != spec.batch_rhs.len() {
        return Err(anyhow!("dot_general batch rank mismatch"));
    }

    let mut is_lhs_batch = vec![false; lhs_rank];
    let mut is_rhs_batch = vec![false; rhs_rank];
    for (lhs_axis, rhs_axis) in spec.batch_lhs.iter().zip(spec.batch_rhs.iter()) {
        if *lhs_axis >= lhs_rank || *rhs_axis >= rhs_rank {
            return Err(anyhow!("dot_general batch axis out of range"));
        }
        is_lhs_batch[*lhs_axis] = true;
        is_rhs_batch[*rhs_axis] = true;
    }

    let mut is_lhs_contract = vec![false; lhs_rank];
    let mut is_rhs_contract = vec![false; rhs_rank];
    for (lhs_axis, rhs_axis) in spec.contract_lhs.iter().zip(spec.contract_rhs.iter()) {
        if *lhs_axis >= lhs_rank || *rhs_axis >= rhs_rank {
            return Err(anyhow!("dot_general contract axis out of range"));
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
        return Err(anyhow!("dot_general output rank mismatch"));
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
) -> Result<Option<OperandAxes>> {
    let rank = dims.len();
    if batch.is_some_and(|axis| axis >= rank) || contract >= rank || batch == Some(contract) {
        return Err(anyhow!(
            "dot_general axes (batch {batch:?}, contract {contract}) are invalid for rank {rank}"
        ));
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
    spec: &DotAxes,
) -> Result<Option<StridedBatchedLayout>> {
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
    let lhs_strides = contiguous_strides(lhs_dims)?;
    let rhs_strides = contiguous_strides(rhs_dims)?;
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
        return Err(anyhow!(
            "dot_general shapes {lhs_dims:?} x {rhs_dims:?} -> {out_dims:?} do not match {spec:?}"
        ));
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
