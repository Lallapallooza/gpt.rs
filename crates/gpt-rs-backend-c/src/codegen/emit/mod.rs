mod conv;
mod dot;
mod elementwise;
mod gather_scatter;
mod random;
mod reduce;
mod search;
mod shape;

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::ir::Value;

use super::outline::Outliner;
use super::profile::{register_op_profile_generic, OpProfile};
use super::types::{Instruction, Kind, ValueInfo};
use super::utils::push_block;
use super::value_info::{operand_types, output_info};

pub(super) struct EmitContext<'a> {
    pub(super) module: &'a mut String,
    pub(super) value_infos: &'a HashMap<Value, ValueInfo>,
    pub(super) matmul_profile: &'a mut OpProfile,
}

/// Emits `instructions` into `module`. Each op body is outlined into a shared static function over
/// renamed buffers.
pub(super) fn emit_instructions(
    module: &mut String,
    instructions: &[Instruction<'_>],
    value_infos: &HashMap<Value, ValueInfo>,
    matmul_profile: &mut OpProfile,
    outliner: &mut Outliner,
) -> Result<()> {
    for inst in instructions {
        if matches!(inst.kind, Kind::Op(TensorOp::Constant(_))) {
            continue;
        }
        let renamed = outliner.rename(inst, value_infos)?;
        let mut body = String::new();
        let mut ctx = EmitContext {
            module: &mut body,
            value_infos: &renamed.value_infos,
            matmul_profile,
        };
        let op_id = emit_instruction(inst, &mut ctx)?;
        let code = outliner.outline(&body, &renamed);
        push_block(module, 1, &format!("GPTRS_OP_BEGIN({op_id});"));
        push_block(module, 1, &code);
        push_block(module, 1, &format!("GPTRS_OP_END({op_id});"));
    }
    Ok(())
}

/// Emits one instruction and returns the id of the profile op it registered.
fn emit_instruction(inst: &Instruction<'_>, ctx: &mut EmitContext<'_>) -> Result<usize> {
    match &inst.kind {
        Kind::Op(_) => emit_op(inst, ctx),
        Kind::Linear => dot::emit_linear(inst, ctx),
        Kind::Conv2d(patches) => {
            let op_id = register_kernel(inst, ctx)?;
            let out_info = output_info(ctx.value_infos, inst.result())?;
            conv::emit_conv2d(
                ctx.module,
                patches,
                &inst.operands,
                out_info,
                ctx.value_infos,
            )?;
            Ok(op_id)
        }
        Kind::GroupedConv2d { patches, groups } => {
            let op_id = register_kernel(inst, ctx)?;
            let out_info = output_info(ctx.value_infos, inst.result())?;
            conv::emit_grouped_conv2d(
                ctx.module,
                patches,
                *groups,
                &inst.operands,
                out_info,
                ctx.value_infos,
            )?;
            Ok(op_id)
        }
        Kind::Elementwise(chain) => {
            let op_id = register_kernel(inst, ctx)?;
            let out_info = output_info(ctx.value_infos, inst.result())?;
            elementwise::emit_elementwise(
                ctx.module,
                chain,
                &inst.operands,
                out_info,
                ctx.value_infos,
            )?;
            Ok(op_id)
        }
    }
}

fn emit_op(inst: &Instruction<'_>, ctx: &mut EmitContext<'_>) -> Result<usize> {
    type Emit = fn(&Instruction<'_>, &mut EmitContext<'_>) -> Result<Option<usize>>;
    let emitters: [Emit; 8] = [
        elementwise::emit_instruction,
        shape::emit_instruction,
        reduce::emit_instruction,
        gather_scatter::emit_instruction,
        dot::emit_instruction,
        conv::emit_instruction,
        search::emit_instruction,
        random::emit_instruction,
    ];
    for emit in emitters {
        if let Some(op_id) = emit(inst, ctx)? {
            return Ok(op_id);
        }
    }
    Err(anyhow!("the C backend has no kernel for {}", inst.label))
}

/// Registers the profile op of a kernel that reads `inst.operands`.
fn register_kernel(inst: &Instruction<'_>, ctx: &mut EmitContext<'_>) -> Result<usize> {
    let out_info = output_info(ctx.value_infos, inst.result())?;
    let input_types = operand_types(&inst.operands, ctx.value_infos)?;
    register_op_profile_generic(ctx.matmul_profile, inst.label, &out_info.ty, &input_types)
}
