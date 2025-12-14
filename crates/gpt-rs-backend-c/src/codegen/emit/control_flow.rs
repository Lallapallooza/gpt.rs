use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{
    CondSpec, DType, Instruction, Operand, Operation, Program, Region, RegionId, ScanSpec, ValueId,
    WhileSpec,
};

use super::super::profile::{
    backend_operation_label, emit_profiled_op, register_op_profile_multi_output, OpProfile,
};
use super::super::types::{ValueInfo, ValueKey, ValueStorage};
use super::super::utils::{c_type, dims_usize, emit_value_array, push_block};
use super::super::value_info::{
    build_value_infos, ensure_dtype, flatten_result_bindings, flatten_value_types,
    literal_to_values, operand_dtype, operand_expr, operand_spec, operand_specs,
    output_indices_map, output_infos, value_info_for_path, LiteralCache,
};
use super::{emit_instructions, EmitContext};

pub(super) fn emit_instruction(
    inst: &Instruction,
    ctx: &mut EmitContext<'_>,
) -> ConversionResult<bool> {
    let EmitContext {
        module,
        value_infos,
        literal_cache,
        program,
        matmul_profile,
        ..
    } = ctx;

    match &inst.op {
        Operation::Cond(spec) => {
            let pred_dtype = operand_dtype(&inst.operands[0], value_infos)?;
            ensure_dtype(pred_dtype, DType::I1, "cond predicate must be i1")?;
            let label = backend_operation_label(&inst.op);
            let outputs = output_infos(value_infos, inst.id)?;
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_multi_output(matmul_profile, label, &outputs, &input_specs)?;
            emit_profiled_op(module, op_id, |module| {
                emit_cond(
                    module,
                    spec,
                    &inst.operands,
                    outputs,
                    value_infos,
                    literal_cache,
                )
            })?;
        }
        Operation::While(spec) => {
            let label = backend_operation_label(&inst.op);
            let outputs = output_infos(value_infos, inst.id)?;
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_multi_output(matmul_profile, label, &outputs, &input_specs)?;
            emit_profiled_op(module, op_id, |module| {
                emit_while(
                    module,
                    spec,
                    &inst.operands,
                    outputs,
                    program,
                    value_infos,
                    literal_cache,
                )
            })?;
        }
        Operation::Scan(spec) => {
            let label = backend_operation_label(&inst.op);
            let outputs = output_infos(value_infos, inst.id)?;
            let input_specs = operand_specs(&inst.operands, value_infos)?;
            let op_id =
                register_op_profile_multi_output(matmul_profile, label, &outputs, &input_specs)?;
            emit_profiled_op(module, op_id, |module| {
                emit_scan(
                    module,
                    spec,
                    &inst.operands,
                    outputs,
                    program,
                    value_infos,
                    literal_cache,
                )
            })?;
        }
        _ => return Ok(false),
    }

    Ok(true)
}

fn region_fn_name(id: RegionId) -> String {
    let region_id = id.0;
    format!("region_r{region_id}")
}
fn emit_cond(
    module: &mut String,
    spec: &CondSpec,
    operands: &[Operand],
    outputs: Vec<&ValueInfo>,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    if operands.is_empty() {
        return Err(ConversionError::new("cond requires a predicate operand"));
    }
    let pred = operand_expr(&operands[0], value_infos, module, literal_cache)?;
    let mut args = Vec::with_capacity(operands.len().saturating_sub(1));
    for operand in operands.iter().skip(1) {
        args.push(operand_expr(operand, value_infos, module, literal_cache)?);
    }
    let out_vars: Vec<String> = outputs.iter().map(|info| info.var.clone()).collect();
    let true_fn = region_fn_name(spec.true_region);
    let false_fn = region_fn_name(spec.false_region);

    let cond_inputs = if args.is_empty() {
        "const void* const* cond_inputs = NULL;".to_string()
    } else {
        let args_join = args.join(", ");
        format!("const void* cond_inputs[] = {{{args_join}}};")
    };
    let cond_outputs = if out_vars.is_empty() {
        "void* const* cond_outputs = NULL;".to_string()
    } else {
        let out_join = out_vars.join(", ");
        format!("void* cond_outputs[] = {{{out_join}}};")
    };
    let block = format!(
        r#"
            {{
              const uint8_t* pred = (const uint8_t*){pred};
        "#
    );
    push_block(module, 1, &block);
    push_block(module, 2, &cond_inputs);
    push_block(module, 2, &cond_outputs);
    let tail = format!(
        r#"
            int cond_status = pred[0]
              ? {true_fn}(cond_inputs, cond_outputs)
              : {false_fn}(cond_inputs, cond_outputs);
            if (cond_status != 0) {{
              return cond_status;
            }}
        "#
    );
    push_block(module, 2, &tail);
    push_block(module, 1, "}");
    Ok(())
}
#[allow(clippy::needless_range_loop)]
fn emit_while(
    module: &mut String,
    spec: &WhileSpec,
    operands: &[Operand],
    outputs: Vec<&ValueInfo>,
    program: &Program,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    let carry_count = operands.len();
    if outputs.len() != carry_count {
        return Err(ConversionError::new("while carry/output mismatch"));
    }
    let cond_region = find_region(program, spec.cond_region)?;
    let body_region = find_region(program, spec.body_region)?;
    let cond_specs = flatten_value_types(&cond_region.results)?;
    if cond_specs.len() != carry_count + 1 {
        return Err(ConversionError::new(
            "while cond region must return predicate and carries",
        ));
    }
    let body_specs = flatten_value_types(&body_region.results)?;
    if body_specs.len() != carry_count {
        return Err(ConversionError::new(
            "while body region must return carries",
        ));
    }
    if cond_specs[0].dtype != DType::I1 || cond_specs[0].shape.element_count() != Some(1) {
        return Err(ConversionError::new(
            "while cond predicate must be i1 scalar",
        ));
    }

    for (idx, out_info) in outputs.iter().enumerate() {
        if cond_specs[idx + 1].dtype != out_info.spec.dtype
            || cond_specs[idx + 1].shape != out_info.spec.shape
        {
            return Err(ConversionError::new("while cond carry spec mismatch"));
        }
        if body_specs[idx].dtype != out_info.spec.dtype
            || body_specs[idx].shape != out_info.spec.shape
        {
            return Err(ConversionError::new("while body carry spec mismatch"));
        }
    }

    let mut carry_inputs = Vec::with_capacity(carry_count);
    for operand in operands {
        carry_inputs.push(operand_expr(operand, value_infos, module, literal_cache)?);
    }
    let out_vars: Vec<String> = outputs.iter().map(|info| info.var.clone()).collect();

    let mut carry_sizes = Vec::with_capacity(carry_count);
    for out_info in outputs.iter() {
        carry_sizes.push(out_info.byte_len);
    }
    let mut cond_sizes = Vec::with_capacity(cond_specs.len());
    for spec in &cond_specs {
        cond_sizes.push(
            spec.byte_len()
                .ok_or_else(|| ConversionError::new("while cond byte length unknown"))?,
        );
    }
    let mut body_sizes = Vec::with_capacity(body_specs.len());
    for spec in &body_specs {
        body_sizes.push(
            spec.byte_len()
                .ok_or_else(|| ConversionError::new("while body byte length unknown"))?,
        );
    }

    push_block(module, 1, "{");
    if carry_count > 0 {
        push_block(module, 2, &format!("void* carry_buffers[{carry_count}];"));
        for idx in 0..carry_count {
            let size = carry_sizes[idx];
            let input = &carry_inputs[idx];
            let block = format!(
                r#"
                    carry_buffers[{idx}] = malloc({size});
                    if (!carry_buffers[{idx}]) {{ return -4; }}
                    memcpy(carry_buffers[{idx}], {input}, {size});
                "#
            );
            push_block(module, 2, &block);
        }
    } else {
        push_block(module, 2, "void** carry_buffers = NULL;");
    }

    let cond_count = cond_specs.len();
    push_block(module, 2, &format!("void* cond_outputs[{cond_count}];"));
    for idx in 0..cond_count {
        let size = cond_sizes[idx];
        let block = format!(
            r#"
                cond_outputs[{idx}] = malloc({size});
                if (!cond_outputs[{idx}]) {{ return -4; }}
            "#
        );
        push_block(module, 2, &block);
    }

    if carry_count > 0 {
        push_block(module, 2, &format!("void* body_outputs[{carry_count}];"));
        for idx in 0..carry_count {
            let size = body_sizes[idx];
            let block = format!(
                r#"
                    body_outputs[{idx}] = malloc({size});
                    if (!body_outputs[{idx}]) {{ return -4; }}
                "#
            );
            push_block(module, 2, &block);
        }
    } else {
        push_block(module, 2, "void** body_outputs = NULL;");
    }

    if carry_count > 0 {
        let cond_inputs = (0..carry_count)
            .map(|idx| format!("carry_buffers[{idx}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let body_inputs = (0..carry_count)
            .map(|idx| {
                let out_idx = idx + 1;
                format!("cond_outputs[{out_idx}]")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let arrays = format!(
            r#"
                const void* cond_inputs[] = {{{cond_inputs}}};
                const void* body_inputs[] = {{{body_inputs}}};
            "#
        );
        push_block(module, 2, &arrays);
    } else {
        push_block(
            module,
            2,
            r#"
                const void* const* cond_inputs = NULL;
                const void* const* body_inputs = NULL;
            "#,
        );
    }

    let cond_fn = region_fn_name(spec.cond_region);
    let body_fn = region_fn_name(spec.body_region);
    push_block(module, 2, "for (;;) {");
    push_block(
        module,
        3,
        &format!("int cond_status = {cond_fn}(cond_inputs, cond_outputs);"),
    );
    push_block(
        module,
        3,
        r#"
            if (cond_status != 0) {
              return cond_status;
            }
        "#,
    );
    push_block(
        module,
        3,
        "const uint8_t* pred = (const uint8_t*)cond_outputs[0];",
    );
    push_block(module, 3, "if (!pred[0]) {");
    for idx in 0..carry_count {
        let out_var = &out_vars[idx];
        let size = carry_sizes[idx];
        let cond_idx = idx + 1;
        push_block(
            module,
            4,
            &format!("memcpy({out_var}, cond_outputs[{cond_idx}], {size});"),
        );
    }
    push_block(module, 4, "break;");
    push_block(module, 3, "}");
    push_block(
        module,
        3,
        &format!("int body_status = {body_fn}(body_inputs, body_outputs);"),
    );
    push_block(
        module,
        3,
        r#"
            if (body_status != 0) {
              return body_status;
            }
        "#,
    );
    for idx in 0..carry_count {
        let size = carry_sizes[idx];
        push_block(
            module,
            3,
            &format!("memcpy(carry_buffers[{idx}], body_outputs[{idx}], {size});"),
        );
    }
    push_block(module, 2, "}");

    for idx in 0..cond_count {
        push_block(module, 2, &format!("free(cond_outputs[{idx}]);"));
    }
    for idx in 0..carry_count {
        push_block(module, 2, &format!("free(carry_buffers[{idx}]);"));
        push_block(module, 2, &format!("free(body_outputs[{idx}]);"));
    }
    push_block(module, 1, "}");
    Ok(())
}
#[allow(clippy::needless_range_loop)]
fn emit_scan(
    module: &mut String,
    spec: &ScanSpec,
    operands: &[Operand],
    outputs: Vec<&ValueInfo>,
    program: &Program,
    value_infos: &HashMap<ValueKey, ValueInfo>,
    literal_cache: &mut LiteralCache,
) -> ConversionResult<()> {
    let carry_count = spec.carry_count;
    let scan_out_count = spec.scan_output_count;
    if operands.len() < carry_count {
        return Err(ConversionError::new("scan operand count mismatch"));
    }
    if outputs.len() != carry_count + scan_out_count {
        return Err(ConversionError::new("scan output count mismatch"));
    }
    let scan_input_count = operands.len() - carry_count;

    let body_region = find_region(program, spec.body_region)?;
    let body_specs = flatten_value_types(&body_region.results)?;
    if body_specs.len() != carry_count + scan_out_count {
        return Err(ConversionError::new(
            "scan body region result count mismatch",
        ));
    }

    let carry_outputs = &outputs[..carry_count];
    let scan_outputs = &outputs[carry_count..];

    for (idx, out_info) in carry_outputs.iter().enumerate() {
        if body_specs[idx].dtype != out_info.spec.dtype
            || body_specs[idx].shape != out_info.spec.shape
        {
            return Err(ConversionError::new("scan carry output spec mismatch"));
        }
    }

    let mut scan_input_specs = Vec::with_capacity(scan_input_count);
    let mut scan_inputs = Vec::with_capacity(scan_input_count);
    let mut time_dim: Option<usize> = None;
    for operand in operands.iter().skip(carry_count) {
        let spec = operand_spec(operand, value_infos)?;
        let dims = dims_usize(&spec)?;
        if dims.is_empty() {
            return Err(ConversionError::new("scan inputs must be at least rank-1"));
        }
        let t = dims[0];
        if let Some(existing) = time_dim {
            if existing != t {
                return Err(ConversionError::new("scan input time dims mismatch"));
            }
        } else {
            time_dim = Some(t);
        }
        scan_input_specs.push(spec);
        scan_inputs.push(operand_expr(operand, value_infos, module, literal_cache)?);
    }

    let t_len = time_dim.ok_or_else(|| ConversionError::new("scan time dimension missing"))?;

    for (idx, out_info) in scan_outputs.iter().enumerate() {
        let step_spec = &body_specs[carry_count + idx];
        let out_dims = dims_usize(&out_info.spec)?;
        let step_dims = dims_usize(step_spec)?;
        if out_dims.len() != step_dims.len() + 1 {
            return Err(ConversionError::new(
                "scan output rank must be body rank + 1",
            ));
        }
        if out_dims[0] != t_len {
            return Err(ConversionError::new("scan output time dimension mismatch"));
        }
        if out_dims[1..] != step_dims[..] {
            return Err(ConversionError::new("scan output shape mismatch"));
        }
        if out_info.spec.dtype != step_spec.dtype {
            return Err(ConversionError::new("scan output dtype mismatch"));
        }
    }

    let mut carry_inputs = Vec::with_capacity(carry_count);
    for operand in operands.iter().take(carry_count) {
        carry_inputs.push(operand_expr(operand, value_infos, module, literal_cache)?);
    }

    let mut carry_sizes = Vec::with_capacity(carry_count);
    for out_info in carry_outputs {
        carry_sizes.push(out_info.byte_len);
    }
    let mut body_sizes = Vec::with_capacity(body_specs.len());
    for spec in &body_specs {
        body_sizes.push(
            spec.byte_len()
                .ok_or_else(|| ConversionError::new("scan body byte length unknown"))?,
        );
    }

    let mut scan_slice_sizes = Vec::with_capacity(scan_input_specs.len());
    for spec in &scan_input_specs {
        let dims = dims_usize(spec)?;
        let slice_elems: usize = dims.iter().skip(1).product();
        let elem_size = spec
            .dtype
            .size_in_bytes()
            .ok_or_else(|| ConversionError::new("scan input dtype size unknown"))?;
        scan_slice_sizes.push(slice_elems * elem_size);
    }

    let mut scan_step_sizes = Vec::with_capacity(scan_out_count);
    for spec in &body_specs[carry_count..] {
        scan_step_sizes.push(
            spec.byte_len()
                .ok_or_else(|| ConversionError::new("scan step byte length unknown"))?,
        );
    }

    push_block(module, 1, "{");
    if carry_count > 0 {
        push_block(module, 2, &format!("void* carry_buffers[{carry_count}];"));
        for idx in 0..carry_count {
            let size = carry_sizes[idx];
            let input = &carry_inputs[idx];
            let block = format!(
                r#"
                    carry_buffers[{idx}] = malloc({size});
                    if (!carry_buffers[{idx}]) {{ return -4; }}
                    memcpy(carry_buffers[{idx}], {input}, {size});
                "#
            );
            push_block(module, 2, &block);
        }
    } else {
        push_block(module, 2, "void** carry_buffers = NULL;");
    }

    let body_count = body_specs.len();
    push_block(module, 2, &format!("void* body_outputs[{body_count}];"));
    for idx in 0..body_count {
        let size = body_sizes[idx];
        let block = format!(
            r#"
                body_outputs[{idx}] = malloc({size});
                if (!body_outputs[{idx}]) {{ return -4; }}
            "#
        );
        push_block(module, 2, &block);
    }

    for (idx, scan_input) in scan_inputs.iter().enumerate() {
        let block = format!(
            r#"
                const char* scan_base_{idx} = (const char*){scan_input};
            "#
        );
        push_block(module, 2, &block);
    }

    let body_fn = region_fn_name(spec.body_region);
    let total_inputs = carry_count + scan_input_count;
    let loop_header = format!(
        r#"
            for (size_t t = 0; t < {t_len}; ++t) {{
        "#
    );
    push_block(module, 2, &loop_header);
    if total_inputs > 0 {
        push_block(
            module,
            3,
            &format!("const void* body_inputs[{total_inputs}];"),
        );
        for idx in 0..carry_count {
            push_block(
                module,
                3,
                &format!("body_inputs[{idx}] = carry_buffers[{idx}];"),
            );
        }
        for (idx, slice_size) in scan_slice_sizes.iter().enumerate() {
            let input_index = carry_count + idx;
            push_block(
                module,
                3,
                &format!("body_inputs[{input_index}] = scan_base_{idx} + t * {slice_size};"),
            );
        }
        push_block(
            module,
            3,
            &format!("int body_status = {body_fn}(body_inputs, body_outputs);"),
        );
    } else {
        let call = format!(
            r#"
                const void* const* body_inputs = NULL;
                int body_status = {body_fn}(body_inputs, body_outputs);
            "#
        );
        push_block(module, 3, &call);
    }
    push_block(
        module,
        3,
        r#"
            if (body_status != 0) {
              return body_status;
            }
        "#,
    );

    for idx in 0..carry_count {
        let size = carry_sizes[idx];
        push_block(
            module,
            3,
            &format!("memcpy(carry_buffers[{idx}], body_outputs[{idx}], {size});"),
        );
    }
    for (idx, out_info) in scan_outputs.iter().enumerate() {
        let out_idx = carry_count + idx;
        let out_var = &out_info.var;
        let step_size = scan_step_sizes[idx];
        push_block(
            module,
            3,
            &format!(
                "memcpy((char*){out_var} + t * {step_size}, body_outputs[{out_idx}], {step_size});"
            ),
        );
    }
    push_block(module, 2, "}");

    for (idx, out_info) in carry_outputs.iter().enumerate() {
        let out_var = &out_info.var;
        let size = carry_sizes[idx];
        push_block(
            module,
            2,
            &format!("memcpy({out_var}, carry_buffers[{idx}], {size});"),
        );
    }
    for idx in 0..body_count {
        push_block(module, 2, &format!("free(body_outputs[{idx}]);"));
    }
    for idx in 0..carry_count {
        push_block(module, 2, &format!("free(carry_buffers[{idx}]);"));
    }
    push_block(module, 1, "}");
    Ok(())
}
fn region_result_ids(region: &Region) -> ConversionResult<Vec<ValueId>> {
    if region.body.len() < region.results.len() {
        return Err(ConversionError::new("region body shorter than result list"));
    }
    let start = region.body.len() - region.results.len();
    Ok(region.body[start..].iter().map(|inst| inst.id).collect())
}
fn find_region(program: &Program, id: RegionId) -> ConversionResult<&Region> {
    program
        .regions
        .iter()
        .find(|region| region.id == id)
        .ok_or_else(|| ConversionError::new("region not found"))
}
pub(crate) fn emit_region_function(
    module: &mut String,
    region: &Region,
    plan: &gpt_rs::backend::conversion::FunctionBufferPlan,
    program: &Program,
    matmul_profile: &mut OpProfile,
) -> ConversionResult<()> {
    let region_id = region.id.0;
    let name = format!("region_r{region_id}");
    let parameter_ids: Vec<ValueId> = (0..region.parameters.len())
        .map(|idx| ValueId(idx as u32))
        .collect();
    let result_ids = region_result_ids(region)?;
    let result_bindings = flatten_result_bindings(&result_ids, &region.results)?;
    let output_indices = output_indices_map(&result_bindings);
    let (value_infos, const_literals) = build_value_infos(
        &parameter_ids,
        &region.parameters,
        &region.body,
        plan,
        &output_indices,
    )?;

    let header = format!("static int {name}(const void* const* inputs, void* const* outputs) {{");
    push_block(module, 0, &header);

    let mut value_keys: Vec<ValueKey> = value_infos.keys().cloned().collect();
    value_keys.sort_by(|a, b| match a.value.0.cmp(&b.value.0) {
        std::cmp::Ordering::Equal => a.path.cmp(&b.path),
        other => other,
    });

    let mut declared = std::collections::HashSet::new();
    for value_key in value_keys.iter() {
        let value_info = value_infos.get(value_key).expect("value id must exist");
        if matches!(value_info.storage, ValueStorage::Alias) {
            continue;
        }
        if !declared.insert(value_info.var.clone()) {
            continue;
        }
        let ctype = c_type(value_info.spec.dtype)?;
        match value_info.storage {
            ValueStorage::Input { index } => {
                let var = &value_info.var;
                push_block(
                    module,
                    1,
                    &format!("const {ctype}* {var} = (const {ctype}*)inputs[{index}];"),
                );
            }
            ValueStorage::Output { index } => {
                let var = &value_info.var;
                push_block(
                    module,
                    1,
                    &format!("{ctype}* {var} = ({ctype}*)outputs[{index}];"),
                );
            }
            ValueStorage::Temp { .. } => {
                let var = &value_info.var;
                let byte_len = value_info.byte_len;
                let block = format!(
                    r#"
                        {ctype}* {var} = ({ctype}*)malloc({byte_len});
                        if (!{var}) {{ return -4; }}
                    "#
                );
                push_block(module, 1, &block);
            }
            ValueStorage::Const => {
                let const_name = value_info.const_name.as_ref().expect("const name");
                let literal = const_literals
                    .get(&value_key.value)
                    .ok_or_else(|| ConversionError::new("missing const literal"))?
                    .clone();
                let values = literal_to_values(&literal, Some(value_info.elem_count))?;
                let values_str = emit_value_array(&values);
                let var = &value_info.var;
                let block = format!(
                    r#"
                        static const {ctype} {const_name}[] = {{{values_str}}};
                        const {ctype}* {var} = {const_name};
                    "#
                );
                push_block(module, 1, &block);
            }
            ValueStorage::Alias => {}
        }
    }

    if !value_infos.is_empty() {
        module.push('\n');
    }

    let mut literal_cache = LiteralCache::default();
    emit_instructions(
        module,
        &region.body,
        &value_infos,
        &mut literal_cache,
        program,
        matmul_profile,
        None,
    )?;

    if !result_bindings.is_empty() {
        module.push('\n');
    }

    for (index, binding) in result_bindings.iter().enumerate() {
        let info = value_info_for_path(&value_infos, binding.value, &binding.path)
            .map_err(|_| ConversionError::new("missing region result value info"))?;
        if matches!(
            info.storage,
            ValueStorage::Output { .. } | ValueStorage::Alias
        ) {
            continue;
        }
        let var = &info.var;
        let byte_len = info.byte_len;
        push_block(
            module,
            1,
            &format!("memcpy(outputs[{index}], {var}, {byte_len});"),
        );
    }

    let mut freed = std::collections::HashSet::new();
    for value_key in value_keys.iter() {
        let value_info = value_infos.get(value_key).expect("value id must exist");
        if matches!(value_info.storage, ValueStorage::Temp { .. })
            && freed.insert(value_info.var.clone())
        {
            let var = &value_info.var;
            push_block(module, 1, &format!("free({var});"));
        }
    }

    push_block(module, 1, "return 0;");
    push_block(module, 0, "}");
    module.push('\n');
    module.push('\n');

    Ok(())
}
