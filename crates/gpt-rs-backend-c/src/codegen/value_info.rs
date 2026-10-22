use std::collections::HashMap;

use anyhow::{anyhow, ensure, Result};
use gpt_rs_ir::dialects::tensor::TensorOps;
use gpt_rs_ir::ir::{DType, Dense, Module, TensorType, Value};

use super::plan::{plan, Planned};
use super::types::{Instruction, Kind, ValueInfo, ValueStorage};
use super::utils::{format_f32, push_block, sizing};

pub(super) fn tensor_type(module: &Module, value: Value) -> Result<TensorType> {
    let ty = module.ty(value);
    ty.as_tensor()
        .cloned()
        .ok_or_else(|| anyhow!("the C backend has no {ty} values"))
}

/// Where the function keeps each of its values, and the constants it holds. The arguments are
/// the inputs, and `results` the outputs.
pub(super) fn build_value_infos(
    module: &Module,
    args: &[Value],
    insts: &[Instruction],
    results: &[Value],
) -> Result<(HashMap<Value, ValueInfo>, HashMap<Value, Dense>)> {
    let mut infos: HashMap<Value, ValueInfo> = HashMap::new();
    for (index, &arg) in args.iter().enumerate() {
        let info = value_info(
            module,
            arg,
            ValueStorage::Input { index },
            format!("in{index}"),
        )?;
        infos.insert(arg, info);
    }
    let planned = plan(insts, results, |value| {
        let ty = tensor_type(module, value)?;
        Ok((ty.dtype, sizing(&ty)?.1))
    })?;
    let mut constants = HashMap::new();
    for (number, inst) in insts.iter().enumerate() {
        for (result, &value) in inst.results.iter().enumerate() {
            let (storage, var) = match planned[&value] {
                Planned::Output { index } => {
                    (ValueStorage::Output { index }, format!("out{index}"))
                }
                Planned::Temp { slot } => (ValueStorage::Temp { slot }, format!("s{slot}")),
                Planned::Const => (ValueStorage::Const, format!("v{number}")),
                // A reshape's operand comes before it, so the operand's variable is known.
                Planned::Alias { of } => (ValueStorage::Alias, infos[&of].var.clone()),
            };
            let mut info = value_info(module, value, storage, var)?;
            if let (Kind::Op(TensorOps::Constant(constant)), 0) = (&inst.kind, result) {
                info.const_name = Some(format!("kConst_{number}"));
                constants.insert(value, constant.value.clone());
            }
            infos.insert(value, info);
        }
    }
    Ok((infos, constants))
}

fn value_info(
    module: &Module,
    value: Value,
    storage: ValueStorage,
    var: String,
) -> Result<ValueInfo> {
    let ty = tensor_type(module, value)?;
    let (elem_count, byte_len) = sizing(&ty)?;
    Ok(ValueInfo {
        storage,
        ty,
        elem_count,
        byte_len,
        var,
        const_name: None,
    })
}

pub(super) fn emit_tensor_dims(
    module: &mut String,
    prefix: &str,
    types: &[TensorType],
) -> Result<Vec<String>> {
    let mut names = Vec::with_capacity(types.len());
    for (index, ty) in types.iter().enumerate() {
        let dims = ty
            .static_dims()
            .ok_or_else(|| anyhow!("{ty} has a dynamic dimension"))?;
        let name = format!("{prefix}_{index}");
        let dims_str = dims
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        push_block(
            module,
            0,
            &format!("static const int64_t {name}[] = {{{dims_str}}};"),
        );
        names.push(name);
    }
    if !types.is_empty() {
        module.push('\n');
    }
    Ok(names)
}

pub(super) fn output_info(values: &HashMap<Value, ValueInfo>, id: Value) -> Result<&ValueInfo> {
    values
        .get(&id)
        .ok_or_else(|| anyhow!("{id:?} has no buffer"))
}

pub(super) fn operand_dtype(operand: &Value, values: &HashMap<Value, ValueInfo>) -> Result<DType> {
    output_info(values, *operand).map(|info| info.ty.dtype)
}

pub(super) fn operand_type(
    operand: &Value,
    values: &HashMap<Value, ValueInfo>,
) -> Result<TensorType> {
    output_info(values, *operand).map(|info| info.ty.clone())
}

pub(super) fn operand_types(
    operands: &[Value],
    values: &HashMap<Value, ValueInfo>,
) -> Result<Vec<TensorType>> {
    operands
        .iter()
        .map(|operand| operand_type(operand, values))
        .collect()
}

pub(super) fn operand_elem_count(
    operand: &Value,
    values: &HashMap<Value, ValueInfo>,
) -> Result<usize> {
    output_info(values, *operand).map(|info| info.elem_count)
}

pub(super) fn operand_expr(operand: &Value, values: &HashMap<Value, ValueInfo>) -> Result<String> {
    output_info(values, *operand).map(|info| info.var.clone())
}

pub(super) fn ensure_dtype(actual: DType, expected: DType, message: &str) -> Result<()> {
    ensure!(actual == expected, "{message}");
    Ok(())
}

/// The elements of `constant` as C literals.
pub(super) fn constant_values(constant: &Dense) -> Result<Vec<String>> {
    let bytes = &constant.bytes;
    Ok(match constant.dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| format_f32(f32::from_le_bytes(chunk)))
            .collect(),
        DType::Si32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| i32::from_le_bytes(chunk).to_string())
            .collect(),
        DType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&chunk| format!("0x{:04x}u", u16::from_le_bytes(chunk)))
            .collect(),
        DType::I1 => bytes
            .iter()
            .map(|&byte| if byte == 0 { "0" } else { "1" }.to_string())
            .collect(),
        dtype => return Err(anyhow!("the C backend has no {dtype} constants")),
    })
}
