use std::collections::{HashMap, HashSet};

use gpt_rs::backend::conversion::{
    AliasKind, BufferKey, ConversionError, ConversionResult, FunctionBufferPlan,
};
use gpt_rs::backend::spec::{
    DType, Instruction, Operand, Operation, TensorLiteral, TensorSpec, ValueId, ValueType,
};

use super::types::{ResultBinding, ValueInfo, ValueKey, ValueStorage};
use super::utils::{
    c_type, dims_from_shape, emit_value_array, format_f32, push_block, sizing_from_spec,
};
use crate::dtype::dtype_tag;

#[derive(Default)]
pub(super) struct LiteralCache {
    next_id: usize,
    names: HashMap<LiteralKey, String>,
}

#[derive(Hash, PartialEq, Eq)]
struct LiteralKey {
    dtype: u32,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl LiteralCache {
    pub(super) fn get_or_emit(
        &mut self,
        literal: &TensorLiteral,
        module: &mut String,
    ) -> ConversionResult<String> {
        let key = literal_key(literal)?;
        if let Some(name) = self.names.get(&key) {
            return Ok(name.clone());
        }
        let lit_id = self.next_id;
        let name = format!("kLit_{lit_id}");
        self.next_id += 1;
        let values = literal_to_values(literal, None)?;
        let ctype = c_type(literal.spec.dtype)?;
        let values_str = emit_value_array(&values);
        let block = format!(
            r#"
                static const {ctype} {name}[] = {{{values_str}}};
            "#
        );
        push_block(module, 1, &block);
        self.names.insert(key, name.clone());
        Ok(name)
    }
}

fn literal_key(literal: &TensorLiteral) -> ConversionResult<LiteralKey> {
    let dtype = dtype_tag(literal.spec.dtype)
        .ok_or_else(|| ConversionError::new("literal dtype not supported by C codegen"))?;
    let mut shape = Vec::with_capacity(literal.spec.shape.rank());
    for dim in literal.spec.shape.dims() {
        match dim {
            gpt_rs::backend::spec::Dimension::Static(value) => shape.push(*value),
            gpt_rs::backend::spec::Dimension::Dynamic(_) => {
                return Err(ConversionError::new("dynamic literal shape not supported"))
            }
        }
    }
    Ok(LiteralKey {
        dtype,
        shape,
        bytes: literal.bytes.as_ref().to_vec(),
    })
}

pub(super) fn operand_specs(
    operands: &[Operand],
    value_infos: &HashMap<ValueKey, ValueInfo>,
) -> ConversionResult<Vec<TensorSpec>> {
    let mut specs = Vec::with_capacity(operands.len());
    for operand in operands {
        specs.push(operand_spec(operand, value_infos)?);
    }
    Ok(specs)
}
pub(super) fn build_value_infos(
    parameter_ids: &[ValueId],
    parameters: &[ValueType],
    instructions: &[Instruction],
    plan: &FunctionBufferPlan,
    output_indices: &HashMap<ValueKey, usize>,
) -> ConversionResult<(
    HashMap<ValueKey, ValueInfo>,
    HashMap<ValueId, TensorLiteral>,
)> {
    let mut value_infos: HashMap<ValueKey, ValueInfo> = HashMap::new();
    let mut const_literals: HashMap<ValueId, TensorLiteral> = HashMap::new();

    for (index, (id, ty)) in parameter_ids.iter().zip(parameters.iter()).enumerate() {
        let spec = tensor_spec_of(ty)?;
        let buffers = plan.buffers_for_value(*id);
        if buffers.len() != 1 {
            return Err(ConversionError::new(
                "tuple-typed parameters are not supported by C codegen yet",
            ));
        }
        let buffer = buffers
            .into_iter()
            .next()
            .ok_or_else(|| ConversionError::new("buffer plan missing parameter value"))?;
        let (elem_count, byte_len) = sizing_from_spec(&spec)?;
        value_infos.insert(
            ValueKey::tensor(*id),
            ValueInfo {
                storage: ValueStorage::Input { index },
                spec,
                elem_count,
                byte_len: buffer.byte_len.unwrap_or(byte_len),
                var: format!("in{index}"),
                const_name: None,
            },
        );
    }

    for inst in instructions {
        let buffers = plan.buffers_for_value(inst.id);
        if buffers.is_empty() {
            return Err(ConversionError::new(
                "buffer plan missing instruction output",
            ));
        }
        let is_const = matches!(inst.op, Operation::Constant(_));
        if is_const && buffers.len() != 1 {
            return Err(ConversionError::new(
                "constant instruction cannot produce tuple outputs",
            ));
        }
        if let Operation::Constant(literal) = &inst.op {
            const_literals.insert(inst.id, literal.clone());
        }
        for buffer in buffers {
            let spec = TensorSpec::new(buffer.dtype, buffer.shape.clone());
            let (elem_count, byte_len) = sizing_from_spec(&spec)?;
            let key = ValueKey::new(inst.id, buffer.path.clone());
            let storage = if is_const {
                ValueStorage::Const
            } else if let Some(index) = output_indices.get(&key).copied() {
                ValueStorage::Output { index }
            } else if let Some(slot) = buffer.slot {
                ValueStorage::Temp { slot }
            } else if buffer.alias_kind == AliasKind::Identity && buffer.alias_of.is_some() {
                ValueStorage::Alias
            } else {
                return Err(ConversionError::new("buffer plan missing slot assignment"));
            };
            let var_name = match storage {
                ValueStorage::Temp { slot } => format!("s{slot}"),
                ValueStorage::Output { index } => format!("out{index}"),
                ValueStorage::Const => value_var_name(inst.id, &buffer.path),
                ValueStorage::Alias => value_var_name(inst.id, &buffer.path),
                ValueStorage::Input { index } => format!("in{index}"),
            };
            let const_name = if matches!(storage, ValueStorage::Const) {
                Some({
                    let inst_id = inst.id.0;
                    let suffix = path_suffix(&buffer.path);
                    format!("kConst_{inst_id}{suffix}")
                })
            } else {
                None
            };
            value_infos.insert(
                key,
                ValueInfo {
                    storage,
                    spec,
                    elem_count,
                    byte_len: buffer.byte_len.unwrap_or(byte_len),
                    var: var_name,
                    const_name,
                },
            );
        }
    }

    let mut keys: Vec<ValueKey> = value_infos.keys().cloned().collect();
    keys.sort_by(|a, b| match a.value.0.cmp(&b.value.0) {
        std::cmp::Ordering::Equal => a.path.cmp(&b.path),
        other => other,
    });

    let mut alias_memo: HashMap<ValueKey, (String, usize)> = HashMap::new();
    let mut alias_visiting: HashSet<ValueKey> = HashSet::new();
    for key in &keys {
        let is_alias = value_infos
            .get(key)
            .map(|info| info.storage == ValueStorage::Alias)
            .unwrap_or(false);
        if !is_alias {
            continue;
        }
        let (resolved_var, resolved_len) = resolve_alias_value_info(
            key,
            &value_infos,
            plan,
            &mut alias_memo,
            &mut alias_visiting,
        )?;
        if let Some(entry) = value_infos.get_mut(key) {
            entry.var = resolved_var;
            entry.byte_len = resolved_len;
        }
    }

    Ok((value_infos, const_literals))
}

fn resolve_alias_value_info(
    key: &ValueKey,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    plan: &FunctionBufferPlan,
    memo: &mut HashMap<ValueKey, (String, usize)>,
    visiting: &mut HashSet<ValueKey>,
) -> ConversionResult<(String, usize)> {
    if let Some(found) = memo.get(key) {
        return Ok(found.clone());
    }
    if !visiting.insert(key.clone()) {
        return Err(ConversionError::new("alias buffer cycle detected"));
    }

    let info = value_infos
        .get(key)
        .ok_or_else(|| ConversionError::new("alias source missing value info"))?;
    if info.storage != ValueStorage::Alias {
        let resolved = (info.var.clone(), info.byte_len);
        memo.insert(key.clone(), resolved.clone());
        let _ = visiting.remove(key);
        return Ok(resolved);
    }

    let buffer = plan
        .values
        .get(&BufferKey {
            value: key.value,
            path: key.path.clone(),
        })
        .and_then(|index| plan.buffers.get(*index))
        .ok_or_else(|| ConversionError::new("missing alias buffer spec"))?;
    let alias_of = buffer
        .alias_of
        .clone()
        .ok_or_else(|| ConversionError::new("alias buffer missing source"))?;
    let alias_key = ValueKey::new(alias_of.value, alias_of.path);
    let resolved = resolve_alias_value_info(&alias_key, value_infos, plan, memo, visiting)?;
    memo.insert(key.clone(), resolved.clone());
    let _ = visiting.remove(key);
    Ok(resolved)
}
pub(super) fn tensor_spec_of(ty: &ValueType) -> ConversionResult<TensorSpec> {
    match ty {
        ValueType::Tensor(spec) => Ok(spec.clone()),
        ValueType::Tuple(_) => Err(ConversionError::new("tuple values are not supported")),
    }
}
pub(super) fn value_var_name(value: ValueId, path: &[usize]) -> String {
    if path.is_empty() {
        {
            let value_id = value.0;
            format!("v{value_id}")
        }
    } else {
        let suffix = path
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("_");
        {
            let value_id = value.0;
            format!("v{value_id}_{suffix}")
        }
    }
}
pub(super) fn path_suffix(path: &[usize]) -> String {
    if path.is_empty() {
        String::new()
    } else {
        let suffix = path
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("_");
        format!("_{suffix}")
    }
}
pub(super) fn is_tuple_type(ty: &ValueType) -> bool {
    matches!(ty, ValueType::Tuple(_))
}
pub(super) fn flatten_value_types(types: &[ValueType]) -> ConversionResult<Vec<TensorSpec>> {
    let mut out = Vec::new();
    for ty in types {
        flatten_value_type(ty, &mut Vec::new(), &mut out)?;
    }
    Ok(out)
}
pub(super) fn flatten_value_type(
    ty: &ValueType,
    path: &mut Vec<usize>,
    out: &mut Vec<TensorSpec>,
) -> ConversionResult<()> {
    match ty {
        ValueType::Tensor(spec) => {
            let _ = path;
            out.push(spec.clone());
            Ok(())
        }
        ValueType::Tuple(elements) => {
            for (index, element) in elements.iter().enumerate() {
                path.push(index);
                flatten_value_type(element, path, out)?;
                path.pop();
            }
            Ok(())
        }
    }
}
pub(super) fn output_indices_map(bindings: &[ResultBinding]) -> HashMap<ValueKey, usize> {
    let mut map = HashMap::new();
    for (index, binding) in bindings.iter().enumerate() {
        map.insert(ValueKey::new(binding.value, binding.path.clone()), index);
    }
    map
}
pub(super) fn flatten_result_bindings(
    result_ids: &[ValueId],
    result_types: &[ValueType],
) -> ConversionResult<Vec<ResultBinding>> {
    if result_ids.len() != result_types.len() {
        return Err(ConversionError::new("result ids/types length mismatch"));
    }
    let mut out = Vec::new();
    for (value, ty) in result_ids.iter().copied().zip(result_types.iter()) {
        flatten_result_binding(value, ty, &mut Vec::new(), &mut out)?;
    }
    Ok(out)
}
pub(super) fn flatten_result_binding(
    value: ValueId,
    ty: &ValueType,
    path: &mut Vec<usize>,
    out: &mut Vec<ResultBinding>,
) -> ConversionResult<()> {
    match ty {
        ValueType::Tensor(_) => {
            out.push(ResultBinding {
                value,
                path: path.clone(),
            });
            Ok(())
        }
        ValueType::Tuple(elements) => {
            for (index, element) in elements.iter().enumerate() {
                path.push(index);
                flatten_result_binding(value, element, path, out)?;
                path.pop();
            }
            Ok(())
        }
    }
}
pub(super) fn emit_tensor_dims(
    module: &mut String,
    prefix: &str,
    specs: &[TensorSpec],
) -> ConversionResult<Vec<String>> {
    let mut names = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let dims = dims_from_shape(spec)?;
        let name = format!("{prefix}_{index}");
        let dims_str = dims
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let block = format!(
            r#"
                static const int64_t {name}[] = {{{dims_str}}};
            "#
        );
        push_block(module, 0, &block);
        names.push(name);
    }
    if !specs.is_empty() {
        module.push('\n');
    }
    Ok(names)
}
pub(super) fn value_info(
    values: &HashMap<ValueKey, ValueInfo>,
    id: ValueId,
) -> ConversionResult<&ValueInfo> {
    value_info_for_path(values, id, &[])
}
pub(super) fn value_info_for_path<'a>(
    values: &'a HashMap<ValueKey, ValueInfo>,
    id: ValueId,
    path: &[usize],
) -> ConversionResult<&'a ValueInfo> {
    values
        .get(&ValueKey::new(id, path.to_vec()))
        .ok_or_else(|| ConversionError::new("value info missing in map"))
}
pub(super) fn output_infos(
    values: &HashMap<ValueKey, ValueInfo>,
    id: ValueId,
) -> ConversionResult<Vec<&ValueInfo>> {
    let mut matches: Vec<(&ValueKey, &ValueInfo)> = values
        .iter()
        .filter_map(|(key, info)| {
            if key.value == id {
                Some((key, info))
            } else {
                None
            }
        })
        .collect();
    if matches.is_empty() {
        return Err(ConversionError::new("missing output value info"));
    }
    matches.sort_by(|(a_key, _), (b_key, _)| a_key.path.cmp(&b_key.path));
    Ok(matches.into_iter().map(|(_, info)| info).collect())
}
pub(super) fn output_info(
    values: &HashMap<ValueKey, ValueInfo>,
    id: ValueId,
) -> ConversionResult<&ValueInfo> {
    let infos = output_infos(values, id)?;
    if infos.len() != 1 {
        return Err(ConversionError::new(
            "expected a single output value for instruction",
        ));
    }
    Ok(infos[0])
}
pub(super) fn operand_dtype(
    operand: &Operand,
    values: &HashMap<ValueKey, ValueInfo>,
) -> ConversionResult<DType> {
    match operand {
        Operand::Value(id) => value_info(values, *id).map(|info| info.spec.dtype),
        Operand::TupleElement { tuple, index } => {
            value_info_for_path(values, *tuple, &[*index]).map(|info| info.spec.dtype)
        }
        Operand::Literal(literal) => Ok(literal.spec.dtype),
    }
}
pub(super) fn operand_spec(
    operand: &Operand,
    values: &HashMap<ValueKey, ValueInfo>,
) -> ConversionResult<TensorSpec> {
    match operand {
        Operand::Value(id) => value_info(values, *id).map(|info| info.spec.clone()),
        Operand::TupleElement { tuple, index } => {
            value_info_for_path(values, *tuple, &[*index]).map(|info| info.spec.clone())
        }
        Operand::Literal(literal) => Ok(literal.spec.clone()),
    }
}
pub(super) fn operand_input_index(
    operand: &Operand,
    values: &HashMap<ValueKey, ValueInfo>,
) -> Option<usize> {
    let info = match operand {
        Operand::Value(id) => value_info(values, *id).ok(),
        Operand::TupleElement { tuple, index } => {
            value_info_for_path(values, *tuple, &[*index]).ok()
        }
        Operand::Literal(_) => None,
    }?;
    match info.storage {
        ValueStorage::Input { index } => Some(index),
        _ => None,
    }
}
pub(super) fn operand_elem_count(
    operand: &Operand,
    values: &HashMap<ValueKey, ValueInfo>,
) -> ConversionResult<usize> {
    let spec = operand_spec(operand, values)?;
    spec.shape
        .element_count()
        .ok_or_else(|| ConversionError::new("dynamic shape not supported"))
}
pub(super) fn ensure_dtype(actual: DType, expected: DType, message: &str) -> ConversionResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(ConversionError::new(message))
    }
}
pub(super) fn operand_expr(
    operand: &Operand,
    values: &HashMap<ValueKey, ValueInfo>,
    module: &mut String,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<String> {
    match operand {
        Operand::Value(id) => value_info(values, *id).map(|info| info.var.clone()),
        Operand::TupleElement { tuple, index } => {
            value_info_for_path(values, *tuple, &[*index]).map(|info| info.var.clone())
        }
        Operand::Literal(literal) => literal_cache.get_or_emit(literal, module),
    }
}
pub(super) fn literal_to_values(
    literal: &TensorLiteral,
    expected_len: Option<usize>,
) -> ConversionResult<Vec<String>> {
    let elem_count = literal
        .spec
        .shape
        .element_count()
        .ok_or_else(|| ConversionError::new("dynamic literal shape not supported"))?;
    let expected = expected_len.unwrap_or(elem_count);
    if elem_count != expected {
        return Err(ConversionError::new("literal element count mismatch"));
    }

    match literal.spec.dtype {
        DType::F32 => {
            if literal.bytes.len() != elem_count * 4 {
                return Err(ConversionError::new("literal byte length mismatch"));
            }
            let mut values = Vec::with_capacity(elem_count);
            for chunk in literal.bytes.chunks_exact(4) {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                values.push(format_f32(f32::from_le_bytes(bytes)));
            }
            Ok(values)
        }
        DType::Si32 => {
            if literal.bytes.len() != elem_count * 4 {
                return Err(ConversionError::new("literal byte length mismatch"));
            }
            let mut values = Vec::with_capacity(elem_count);
            for chunk in literal.bytes.chunks_exact(4) {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                values.push(i32::from_le_bytes(bytes).to_string());
            }
            Ok(values)
        }
        DType::I1 => {
            if literal.bytes.len() != elem_count {
                return Err(ConversionError::new("literal byte length mismatch"));
            }
            let mut values = Vec::with_capacity(elem_count);
            for byte in literal.bytes.iter() {
                values.push(if *byte == 0 { "0".into() } else { "1".into() });
            }
            Ok(values)
        }
        _ => Err(ConversionError::new(
            "literal dtype not supported by C codegen",
        )),
    }
}
