use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    DType, Instruction, Operation, RngNormalSpec, RngUniformSpec, TensorSpec,
};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_generic,
};
use super::super::utils::{dims_usize, emit_loops_with_indices, linear_index_expr, push_block};
use super::super::value_info::{ensure_dtype, output_info};
use super::EmitContext;

pub(super) fn emit_instruction(
    inst: &Instruction,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<bool> {
    let EmitContext {
        module,
        value_infos,
        matmul_profile,
        ..
    } = ctx;

    match &inst.op {
        Operation::RngUniform(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "rng_uniform output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let op_id = register_op_profile_generic(matmul_profile, label, &out_info.spec, &[])?;
            emit_profiled_op(module, op_id, |module| {
                emit_rng_uniform(module, &out_info.var, &out_info.spec, spec)
            })?;
        }
        Operation::RngNormal(spec) => {
            let out_info = output_info(value_infos, inst.id)?;
            ensure_dtype(
                out_info.spec.dtype,
                DType::F32,
                "rng_normal output must be f32",
            )?;
            let label = backend_operation_label(&inst.op);
            let op_id = register_op_profile_generic(matmul_profile, label, &out_info.spec, &[])?;
            emit_profiled_op(module, op_id, |module| {
                emit_rng_normal(module, &out_info.var, &out_info.spec, spec)
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

fn emit_rng_uniform(
    module: &mut String,
    out: &str,
    out_spec: &TensorSpec,
    spec: &RngUniformSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    if out_dims != dims_usize(&TensorSpec::new(spec.dtype, spec.shape.clone()))? {
        return Err(ConversionError::new("rng_uniform output shape mismatch"));
    }
    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let out_idx = linear_index_expr(&out_dims, indices);
        let block = format!(
            r#"
                float u = fabsf(sinf((float)({out_idx} + 1) * 12.9898f) * 43758.5453f);
                u = u - floorf(u);
                {out}[{out_idx}] = u;
            "#
        );
        push_block(module, indent, &block);
    });
    Ok(())
}
fn emit_rng_normal(
    module: &mut String,
    out: &str,
    out_spec: &TensorSpec,
    spec: &RngNormalSpec,
) -> ConversionResult<()> {
    let out_dims = dims_usize(out_spec)?;
    if out_dims != dims_usize(&TensorSpec::new(spec.dtype, spec.shape.clone()))? {
        return Err(ConversionError::new("rng_normal output shape mismatch"));
    }
    emit_loops_with_indices(module, &out_dims, 2, "i", |module, indices, indent| {
        let out_idx = linear_index_expr(&out_dims, indices);
        let block = format!(
            r#"
                float u1 = fabsf(sinf((float)({out_idx} + 1) * 12.9898f) * 43758.5453f);
                float u2 = fabsf(sinf((float)({out_idx} + 7) * 78.233f) * 12345.6789f);
                u1 = u1 - floorf(u1);
                u2 = u2 - floorf(u2);
                float r = sqrtf(-2.0f * logf(u1));
                float theta = 6.2831853f * u2;
                float z = r * cosf(theta);
                {out}[{out_idx}] = z;
            "#
        );
        push_block(module, indent, &block);
    });
    Ok(())
}
