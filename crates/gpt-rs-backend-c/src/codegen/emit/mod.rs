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
use gpt_rs::backend::spec::{CustomCallAttr, CustomCallSpec, Instruction, Operation, Program};

use crate::targets::{TARGET_CONV2D_NHWC_F32_V1, TARGET_ELEMENTWISE_FUSED};

use super::outline::Outliner;
use super::profile::{backend_operation_label, register_op_profile_custom_call, OpProfile};
use super::types::{MatmulCacheEntry, ValueInfo, ValueKey};
use super::utils::push_block;
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

/// Packed-weight caches and op outlining, used when emitting the entry function.
pub(super) struct EntryEmit<'a> {
    pub(super) matmul_caches: &'a mut Vec<MatmulCacheEntry>,
    pub(super) outliner: &'a mut Outliner,
}

/// Emits `instructions` into `module`. With `entry`, each op body is outlined into a shared static
/// function over renamed buffers.
pub(super) fn emit_instructions(
    module: &mut String,
    instructions: &[Instruction],
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
    program: &Program,
    matmul_profile: &mut OpProfile,
    mut entry: Option<EntryEmit<'_>>,
) -> ConversionResult<()> {
    for inst in instructions {
        if matches!(inst.op, Operation::Constant(_)) {
            continue;
        }
        let renamed = match &entry {
            Some(entry) => Some(entry.outliner.rename(inst, value_infos)?),
            None => None,
        };
        let mut body = String::new();
        let mut ctx = EmitContext {
            module: &mut body,
            value_infos: renamed.as_ref().map_or(value_infos, |r| &r.value_infos),
            literal_cache,
            program,
            matmul_profile,
            matmul_caches: entry.as_mut().map(|entry| &mut *entry.matmul_caches),
        };
        let op_id = emit_instruction(inst, &mut ctx)?;
        let code = match (&mut entry, &renamed) {
            (Some(entry), Some(renamed)) => entry.outliner.outline(&body, renamed),
            _ => body,
        };
        push_block(module, 1, &format!("GPTRS_OP_BEGIN({op_id});"));
        push_block(module, 1, &code);
        push_block(module, 1, &format!("GPTRS_OP_END({op_id});"));
    }
    Ok(())
}

/// Emits one instruction and returns the id of the profile op it registered.
fn emit_instruction(inst: &Instruction, ctx: &mut EmitContext<'_>) -> ConversionResult<usize> {
    type Emit = fn(&Instruction, &mut EmitContext<'_>) -> ConversionResult<Option<usize>>;
    let emitters: [Emit; 8] = [
        elementwise::emit_instruction,
        shape::emit_instruction,
        reduce::emit_instruction,
        gather_scatter::emit_instruction,
        dot::emit_instruction,
        conv::emit_instruction,
        rng::emit_instruction,
        control_flow::emit_instruction,
    ];
    for emit in emitters {
        if let Some(op_id) = emit(inst, ctx)? {
            return Ok(op_id);
        }
    }
    match &inst.op {
        Operation::CustomCall(spec) => emit_custom_call(inst, spec, ctx),
        _ => Err(ConversionError::new("unsupported instruction")),
    }
}

fn emit_custom_call(
    inst: &Instruction,
    spec: &CustomCallSpec,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<usize> {
    let out_info = output_info(ctx.value_infos, inst.id)?;
    let input_specs = operand_specs(&inst.operands, ctx.value_infos)?;
    let op_id = register_op_profile_custom_call(
        ctx.matmul_profile,
        backend_operation_label(&inst.op),
        &out_info.spec,
        &input_specs,
        &spec.target,
    )?;
    match spec.target.as_str() {
        TARGET_ELEMENTWISE_FUSED => elementwise::emit_custom_call_elementwise(
            ctx.module,
            spec,
            &inst.operands,
            out_info,
            ctx.value_infos,
            ctx.literal_cache,
        )?,
        TARGET_CONV2D_NHWC_F32_V1 => conv::emit_custom_call_conv2d(
            ctx.module,
            spec,
            &inst.operands,
            out_info,
            ctx.value_infos,
            ctx.literal_cache,
            op_id,
            ctx.matmul_caches.as_deref_mut(),
        )?,
        target => {
            return Err(ConversionError::new(format!(
                "custom_call target '{target}' not supported by C codegen"
            )))
        }
    }
    Ok(op_id)
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
