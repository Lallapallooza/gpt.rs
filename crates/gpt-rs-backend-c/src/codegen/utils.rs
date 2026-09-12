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
        // bfloat16 values are carried as raw bit patterns.
        DType::Bf16 => Ok("uint16_t"),
        _ => Err(ConversionError::new("dtype not supported by C codegen")),
    }
}

/// Validates the operands of pure data-movement ops. These ops copy elements through `c_type`
/// pointers, so they work for any representable dtype when the input and output dtypes agree.
pub(super) fn ensure_copy_dtypes(
    op: &str,
    inputs: &[DType],
    output: DType,
) -> ConversionResult<()> {
    c_type(output)?;
    for dtype in inputs {
        if *dtype != output {
            return Err(ConversionError::new(format!(
                "{op} operands must match the output dtype ({output:?}), got {dtype:?}"
            )));
        }
    }
    Ok(())
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
/// Returns the `#pragma omp parallel for` line for `collapse` nested loops over `total`
/// independent output elements. Returns `None` when the work is too small to pay for the fork/join.
pub(super) fn omp_parallel_pragma(total: usize, collapse: usize) -> Option<String> {
    if total < crate::kernels::PARALLEL_MIN_WORK || collapse == 0 {
        return None;
    }
    let collapse = if collapse > 1 {
        format!(" collapse({collapse})")
    } else {
        String::new()
    };
    Some(format!(
        "#pragma omp parallel for{collapse} schedule(static)"
    ))
}

// Threads also share long innermost loops, in chunks that are long enough to stay vectorised.
const PARALLEL_INNER_CHUNK: usize = 2048;

/// Like [`emit_loops_with_indices`], for loop nests whose iterations write distinct output
/// elements, so they are safe to run concurrently. Never use it for reductions or scatters. The
/// innermost loop is never collapsed, so the compiler can still vectorise it.
pub(super) fn emit_parallel_loops_with_indices<F>(
    module: &mut String,
    dims: &[usize],
    indent: usize,
    prefix: &str,
    body: F,
) where
    F: FnOnce(&mut String, &[String], usize),
{
    let total: usize = dims.iter().product();
    let Some((&inner, outer)) = dims.split_last() else {
        return emit_loops_with_indices(module, dims, indent, prefix, body);
    };
    let chunked = !outer.is_empty() && inner > PARALLEL_INNER_CHUNK;
    let collapse = if outer.is_empty() {
        1
    } else {
        outer.len() + usize::from(chunked)
    };
    let Some(pragma) = omp_parallel_pragma(total, collapse) else {
        return emit_loops_with_indices(module, dims, indent, prefix, body);
    };
    push_line(module, indent, &pragma);
    if !chunked {
        return emit_loops_with_indices(module, dims, indent, prefix, body);
    }
    let idx = format!("{prefix}{}", dims.len() - 1);
    let chunk = PARALLEL_INNER_CHUNK;
    let chunk_loop = format!("for (size_t {idx}_c = 0; {idx}_c < {inner}; {idx}_c += {chunk}) {{");
    let inner_loop = format!(
        "for (size_t {idx} = {idx}_c; {idx} < GPTRS_MIN({idx}_c + {chunk}, {inner}); ++{idx}) {{"
    );
    emit_loops_with_indices(
        module,
        outer,
        indent,
        prefix,
        |module, outer_indices, indent| {
            push_line(module, indent, &chunk_loop);
            push_line(module, indent + 1, &inner_loop);
            let mut indices = outer_indices.to_vec();
            indices.push(idx.clone());
            body(module, &indices, indent + 2);
            push_line(module, indent + 1, "}");
            push_line(module, indent, "}");
        },
    );
}

/// Emits `for (size_t i = 0; i < elem_count; ++i) { body }` over flat buffers, split across
/// threads when large. Each input binds to a `const <ctype>*` with the given name. `out` binds to
/// a `<ctype>*` named `out`.
pub(super) fn emit_flat_loop(
    module: &mut String,
    inputs: &[(&str, &str, &str)],
    out: (&str, &str),
    elem_count: usize,
    body: &str,
) {
    let mut block = String::from("{\n");
    for (ctype, name, expr) in inputs {
        block.push_str(&format!(
            "  const {ctype}* {name} = (const {ctype}*){expr};\n"
        ));
    }
    let (out_ctype, out_var) = out;
    block.push_str(&format!("  {out_ctype}* out = ({out_ctype}*){out_var};\n"));
    if let Some(pragma) = omp_parallel_pragma(elem_count, 1) {
        block.push_str(&format!("  {pragma}\n"));
    }
    block.push_str(&format!("  for (size_t i = 0; i < {elem_count}; ++i) {{\n"));
    for line in body.lines().map(str::trim).filter(|line| !line.is_empty()) {
        block.push_str(&format!("    {line}\n"));
    }
    block.push_str("  }\n}");
    push_block(module, 1, &block);
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
