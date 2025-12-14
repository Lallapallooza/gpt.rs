mod control_flow;
mod conv;
mod dot;
mod elementwise;
mod gather_scatter;
mod reduce;
mod rng;
mod shape;

use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    CustomCallAttr, CustomCallSpec, Instruction, Operand, Operation, Program,
};

use crate::targets::{TARGET_CONV2D_NHWC_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1};

use super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_custom_call, OpProfile,
};
use super::types::{MatmulCacheEntry, ValueInfo, ValueKey};
use super::value_info::{operand_specs, output_info, LiteralCache};

pub(super) use control_flow::emit_region_function;

pub(super) struct EmitContext<'a> {
    pub(super) module: &'a mut String,
    pub(super) value_infos: &'a HashMap<ValueKey, ValueInfo>,
    pub(super) literal_cache: &'a mut LiteralCache,
    pub(super) program: &'a Program,
    pub(super) matmul_profile: &'a mut OpProfile,
    pub(super) matmul_caches: Option<&'a mut Vec<MatmulCacheEntry>>,
}

pub(super) fn emit_instructions(
    module: &mut String,
    instructions: &[Instruction],
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
    program: &Program,
    matmul_profile: &mut OpProfile,
    matmul_caches: Option<&mut Vec<MatmulCacheEntry>>,
) -> ConversionResult<()> {
    let mut ctx = EmitContext {
        module,
        value_infos,
        literal_cache,
        program,
        matmul_profile,
        matmul_caches,
    };

    for inst in instructions {
        if matches!(inst.op, Operation::Constant(_)) {
            continue;
        }
        if let Operation::CustomCall(spec) = &inst.op {
            let out_info = output_info(ctx.value_infos, inst.id)?;
            let label = backend_operation_label(&inst.op);
            let input_specs = operand_specs(&inst.operands, ctx.value_infos)?;
            let op_id = register_op_profile_custom_call(
                ctx.matmul_profile,
                label,
                &out_info.spec,
                &input_specs,
                &spec.target,
            )?;
            emit_profiled_op(ctx.module, op_id, |module| {
                emit_custom_call(
                    module,
                    spec,
                    &inst.operands,
                    out_info,
                    ctx.value_infos,
                    ctx.literal_cache,
                    op_id,
                    ctx.matmul_caches.as_deref_mut(),
                )
            })?;
            continue;
        }

        if elementwise::emit_instruction(inst, &mut ctx)?
            || shape::emit_instruction(inst, &mut ctx)?
            || reduce::emit_instruction(inst, &mut ctx)?
            || gather_scatter::emit_instruction(inst, &mut ctx)?
            || dot::emit_instruction(inst, &mut ctx)?
            || conv::emit_instruction(inst, &mut ctx)?
            || rng::emit_instruction(inst, &mut ctx)?
            || control_flow::emit_instruction(inst, &mut ctx)?
        {
            continue;
        }

        return Err(ConversionError::new("unsupported instruction"));
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_custom_call(
    module: &mut String,
    spec: &CustomCallSpec,
    operands: &[Operand],
    out_info: &ValueInfo,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
    op_id: usize,
    matmul_caches: Option<&mut Vec<MatmulCacheEntry>>,
) -> ConversionResult<()> {
    match spec.target.as_str() {
        TARGET_ELEMENTWISE_FUSED_F32_V1 => elementwise::emit_custom_call_elementwise(
            module,
            spec,
            operands,
            out_info,
            value_infos,
            literal_cache,
        ),
        TARGET_CONV2D_NHWC_F32_V1 => conv::emit_custom_call_conv2d(
            module,
            spec,
            operands,
            out_info,
            value_infos,
            literal_cache,
            op_id,
            matmul_caches,
        ),
        _ => {
            let target = &spec.target;
            Err(ConversionError::new(format!(
                "custom_call target '{target}' not supported by C codegen"
            )))
        }
    }
}

pub(super) fn custom_call_attr_i64_array(
    spec: &CustomCallSpec,
    key: &str,
) -> ConversionResult<Vec<i64>> {
    let attr = spec
        .attrs
        .get(key)
        .ok_or_else(|| ConversionError::new(format!("custom_call missing attr '{key}'")))?;
    match attr {
        CustomCallAttr::I64Array(values) => Ok(values.clone()),
        _ => Err(ConversionError::new(format!(
            "custom_call attr '{key}' must be i64 array"
        ))),
    }
}
