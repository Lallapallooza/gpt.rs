use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{DType, Dimension, Literal, TensorSpec};

pub(super) fn escape_c_string(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            '\\' => "\\\\".to_string(),
            '"' => "\\\"".to_string(),
            '\n' => "\\n".to_string(),
            '\r' => "\\r".to_string(),
            '\t' => "\\t".to_string(),
            _ => ch.to_string(),
        })
        .collect::<Vec<_>>()
        .join("")
}
pub(super) fn sizing_from_spec(spec: &TensorSpec) -> ConversionResult<(usize, usize)> {
    let elem_count = spec
        .shape
        .element_count()
        .ok_or_else(|| ConversionError::new("dynamic shape not supported"))?;
    let byte_len = spec
        .byte_len()
        .ok_or_else(|| ConversionError::new("unknown dtype size"))?;
    Ok((elem_count, byte_len))
}
pub(super) fn dims_from_shape(spec: &TensorSpec) -> ConversionResult<Vec<i64>> {
    let mut dims = Vec::with_capacity(spec.shape.rank());
    for dim in spec.shape.dims() {
        match dim {
            Dimension::Static(value) => dims.push(*value as i64),
            Dimension::Dynamic(_) => {
                return Err(ConversionError::new("dynamic dimensions not supported"))
            }
        }
    }
    Ok(dims)
}
pub(super) fn emit_value_array(values: &[String]) -> String {
    values.join(", ")
}
pub(super) fn format_f32(value: f32) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            "-INFINITY".to_string()
        } else {
            "INFINITY".to_string()
        }
    } else {
        let base = value.to_string();
        let needs_decimal = !base.contains('.') && !base.contains('e') && !base.contains('E');
        let suffix = if needs_decimal { ".0f" } else { "f" };
        format!("{base}{suffix}")
    }
}
pub(super) fn c_type(dtype: DType) -> ConversionResult<&'static str> {
    match dtype {
        DType::F32 => Ok("float"),
        DType::Si32 => Ok("int32_t"),
        DType::I1 => Ok("uint8_t"),
        _ => Err(ConversionError::new("dtype not supported by C codegen")),
    }
}
pub(super) fn emit_memcpy(module: &mut String, out: &str, input: &str, byte_len: usize) {
    if out == input {
        return;
    }
    let block = format!(
        r#"
            memcpy({out}, {input}, {byte_len});
        "#
    );
    push_block(module, 1, &block);
}
pub(super) fn dims_usize(spec: &TensorSpec) -> ConversionResult<Vec<usize>> {
    let mut dims = Vec::with_capacity(spec.shape.rank());
    for dim in spec.shape.dims() {
        match dim {
            Dimension::Static(value) => dims.push(*value),
            Dimension::Dynamic(_) => {
                return Err(ConversionError::new("dynamic dimensions not supported"))
            }
        }
    }
    Ok(dims)
}
pub(super) fn axis_index(axis: isize, rank: usize) -> ConversionResult<usize> {
    let axis = if axis < 0 {
        let shifted = rank as isize + axis;
        if shifted < 0 {
            return Err(ConversionError::new("invalid axis"));
        }
        shifted as usize
    } else {
        axis as usize
    };
    if axis >= rank {
        return Err(ConversionError::new("axis out of range"));
    }
    Ok(axis)
}
pub(super) fn linear_index_expr(dims: &[usize], indices: &[String]) -> String {
    if dims.is_empty() {
        return "0".to_string();
    }
    let first = &indices[0];
    let mut expr = format!("({first})");
    for (dim, idx) in dims.iter().skip(1).zip(indices.iter().skip(1)) {
        expr = format!("({expr} * {dim} + ({idx}))");
    }
    expr
}
pub(super) fn emit_loops_with_indices<F>(
    module: &mut String,
    dims: &[usize],
    indent: usize,
    prefix: &str,
    body: F,
) where
    F: FnOnce(&mut String, &[String], usize),
{
    let indices: Vec<String> = (0..dims.len()).map(|i| format!("{prefix}{i}")).collect();
    if dims.is_empty() {
        body(module, &indices, indent);
        return;
    }
    for (idx, dim) in dims.iter().enumerate() {
        let idx_name = &indices[idx];
        push_line(
            module,
            indent + idx,
            &format!("for (size_t {idx_name} = 0; {idx_name} < {dim}; ++{idx_name}) {{"),
        );
    }
    body(module, &indices, indent + dims.len());
    for idx in (0..dims.len()).rev() {
        push_line(module, indent + idx, "}");
    }
}
pub(super) fn push_line(module: &mut String, indent: usize, line: &str) {
    push_block(module, indent, line);
}
pub(super) fn push_block(module: &mut String, indent: usize, block: &str) {
    if block.is_empty() {
        return;
    }
    let pad = "  ".repeat(indent);
    let mut lines: Vec<&str> = block.split('\n').collect();
    if matches!(lines.first(), Some(line) if line.trim().is_empty()) {
        lines.remove(0);
    }
    if matches!(lines.last(), Some(line) if line.trim().is_empty()) {
        lines.pop();
    }

    let mut min_indent = usize::MAX;
    for line in &lines {
        if line.trim().is_empty() {
            continue;
        }
        let count = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
        min_indent = min_indent.min(count);
    }
    if min_indent == usize::MAX {
        min_indent = 0;
    }

    for line in lines {
        if line.is_empty() {
            module.push('\n');
        } else {
            let trimmed = if min_indent > 0 && line.len() >= min_indent {
                &line[min_indent..]
            } else {
                line
            };
            if trimmed.is_empty() {
                module.push('\n');
                continue;
            }
            module.push_str(&pad);
            module.push_str(trimmed);
            module.push('\n');
        }
    }
}
pub(super) fn literal_to_f32_scalar(value: &Literal) -> ConversionResult<f32> {
    match value {
        Literal::I1(v) => Ok(if *v { 1.0 } else { 0.0 }),
        Literal::Signed(v) => Ok(*v as f32),
        Literal::Unsigned(v) => Ok(*v as f32),
        Literal::Float(v) => Ok(*v as f32),
        Literal::Complex { .. } => {
            Err(ConversionError::new("complex pad values are not supported"))
        }
    }
}
