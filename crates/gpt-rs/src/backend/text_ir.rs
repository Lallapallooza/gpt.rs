use std::collections::HashMap;

use thiserror::Error;

use crate::backend::ptir_utils;
use crate::backend::spec::{
    BroadcastToSpec, CastSpec, DType, DotGeneralSpec, ElementwiseBinaryOp, ElementwiseUnaryOp,
    Operand, Operation, Program, ProgramBuilder, ReduceKind, ReduceSpec, TensorSpec, ValueId,
    ValueType,
};

/// Errors raised while parsing the lightweight PTIR text format used in tests.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TextIrError {
    #[error("{0}")]
    Message(String),
}

impl TextIrError {
    fn new(msg: impl Into<String>) -> Self {
        TextIrError::Message(msg.into())
    }
}

/// Parses a single PTIR function described using a compact MLIR-inspired syntax.
///
/// # Example
/// ```
/// use gpt_rs::backend::text_ir::parse_program;
///
/// let program = parse_program(r#"
/// func @redundant_cast(%x: tensor<f32, 2x2>) -> tensor<f32, 2x2> {
///   %cast = cast %x -> tensor<f32, 2x2>
///   return %cast
/// }
/// "#).expect("valid program");
/// assert_eq!(program.entry, "redundant_cast");
/// assert_eq!(program.functions[0].body.len(), 1);
/// ```
pub fn parse_program(src: &str) -> Result<Program, TextIrError> {
    Parser::new(src).parse().map(|parsed| parsed.program)
}

/// Parses a PTIR program and returns a [`ParsedProgram`] that tracks value name mappings.
pub fn parse_program_with_symbols(src: &str) -> Result<ParsedProgram, TextIrError> {
    Parser::new(src).parse()
}

/// Program paired with the mapping from textual value names to SSA identifiers.
#[derive(Debug, Clone)]
pub struct ParsedProgram {
    pub program: Program,
    pub value_names: HashMap<String, ValueId>,
}

struct Parser<'a> {
    source: &'a str,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source }
    }

    fn parse(&self) -> Result<ParsedProgram, TextIrError> {
        let trimmed = self.source.trim();
        if trimmed.is_empty() {
            return Err(TextIrError::new("input is empty"));
        }
        let header_end = trimmed
            .find('{')
            .ok_or_else(|| TextIrError::new("missing `{` to start function body"))?;
        let header = trimmed[..header_end].trim();
        let body_start = header_end + 1;
        let body_end = trimmed
            .rfind('}')
            .ok_or_else(|| TextIrError::new("missing `}` to end function body"))?;
        let body = trimmed[body_start..body_end].trim();

        let (name, params, result_ty) = self.parse_function_header(header)?;
        let mut builder = ProgramBuilder::new();
        let mut value_map: HashMap<String, ValueId> = HashMap::new();

        for Parameter { name, ty } in params {
            let parsed_ty = parse_type(&ty)?;
            let id = builder.add_parameter(parsed_ty.clone());
            value_map.insert(name, id);
        }

        let (results, statements) = self.parse_body(body, &mut builder, &mut value_map)?;
        let expected_result_ty = parse_type(&result_ty)?;
        let result_ids = results
            .into_iter()
            .map(|name| {
                value_map
                    .get(&name)
                    .copied()
                    .ok_or_else(|| TextIrError::new(format!("unknown return value `{name}`")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let function = builder.finish(name.clone(), result_ids.clone());
        match expected_result_ty {
            ValueType::Tensor(expected) => {
                if function.results.len() != 1
                    || result_ids.len() != 1
                    || function.results[0] != ValueType::Tensor(expected.clone())
                {
                    return Err(TextIrError::new(
                        "declared result type does not match returned value type",
                    ));
                }
            }
            ValueType::Tuple(elements) => {
                if function.results.len() != elements.len() || result_ids.len() != elements.len() {
                    return Err(TextIrError::new(
                        "declared tuple result arity does not match returned values",
                    ));
                }
                for (actual, expected) in function.results.iter().zip(elements.iter()) {
                    if actual != expected {
                        return Err(TextIrError::new(
                            "declared tuple result type does not match returned value type",
                        ));
                    }
                }
            }
        };

        // Sanity-check the statements count: builder produces instructions as side effect.
        if let Some(expected) = statements {
            if expected != function.body.len() {
                return Err(TextIrError::new("internal statement count mismatch"));
            }
        }

        Ok(ParsedProgram {
            program: Program::new(name).with_functions(vec![function]),
            value_names: value_map,
        })
    }

    fn parse_function_header(
        &self,
        header: &str,
    ) -> Result<(String, Vec<Parameter>, String), TextIrError> {
        let header = header.trim();
        let header = header
            .strip_prefix("func")
            .or_else(|| header.strip_prefix("function"))
            .ok_or_else(|| TextIrError::new("function header must start with `func`"))?
            .trim_start();
        let open_paren = header
            .find('(')
            .ok_or_else(|| TextIrError::new("missing `(` in function header"))?;
        let close_paren = find_matching_paren(header, open_paren)
            .ok_or_else(|| TextIrError::new("missing `)` to close parameter list"))?;

        let name_section = header[..open_paren].trim();
        let name = name_section
            .strip_prefix('@')
            .unwrap_or(name_section)
            .trim();
        if name.is_empty() {
            return Err(TextIrError::new("function name cannot be empty"));
        }
        let params_str = &header[open_paren + 1..close_paren];
        let params = self.parse_parameters(params_str)?;

        let arrow_section = header[close_paren + 1..].trim();
        let result_ty = arrow_section
            .strip_prefix("->")
            .ok_or_else(|| TextIrError::new("missing `->` and return type in header"))?
            .trim()
            .to_string();

        Ok((name.to_string(), params, result_ty))
    }

    fn parse_parameters(&self, params: &str) -> Result<Vec<Parameter>, TextIrError> {
        let params = params.trim();
        if params.is_empty() {
            return Ok(Vec::new());
        }
        split_top_level(params, ',')
            .into_iter()
            .map(|raw| {
                let decl = raw.trim();
                let (name, ty) = decl
                    .split_once(':')
                    .ok_or_else(|| TextIrError::new("parameter must be `name: type`"))?;
                let name = normalize_value_name(name.trim());
                if name.is_empty() {
                    return Err(TextIrError::new("parameter name cannot be empty"));
                }
                let ty = ty.trim();
                if ty.is_empty() {
                    return Err(TextIrError::new("parameter type cannot be empty"));
                }
                Ok(Parameter {
                    name,
                    ty: ty.to_string(),
                })
            })
            .collect()
    }

    fn parse_body(
        &self,
        body: &str,
        builder: &mut ProgramBuilder,
        value_map: &mut HashMap<String, ValueId>,
    ) -> Result<(Vec<String>, Option<usize>), TextIrError> {
        if body.is_empty() {
            return Err(TextIrError::new("function body cannot be empty"));
        }
        let mut statements = 0usize;
        let mut return_values: Vec<String> = Vec::new();

        for line in body.lines() {
            let statement = line.trim();
            if statement.is_empty() {
                continue;
            }
            if statement.starts_with("return") {
                if !return_values.is_empty() {
                    return Err(TextIrError::new(
                        "multiple `return` statements are not allowed",
                    ));
                }
                let values = statement
                    .strip_prefix("return")
                    .ok_or_else(|| TextIrError::new("malformed `return` statement"))?
                    .trim()
                    .trim_end_matches(';');
                if values.is_empty() {
                    return Err(TextIrError::new(
                        "`return` must reference at least one value",
                    ));
                }
                return_values = split_top_level(values, ',')
                    .into_iter()
                    .map(|value| normalize_value_name(value.trim()))
                    .collect();
                continue;
            }
            statements += 1;
            self.parse_statement(statement, builder, value_map)?;
        }

        if return_values.is_empty() {
            return Err(TextIrError::new(
                "function body must end with a `return` statement",
            ));
        }

        Ok((return_values, Some(statements)))
    }

    fn parse_statement(
        &self,
        statement: &str,
        builder: &mut ProgramBuilder,
        value_map: &mut HashMap<String, ValueId>,
    ) -> Result<(), TextIrError> {
        let (result_name_raw, rest) = statement
            .split_once('=')
            .ok_or_else(|| TextIrError::new("statements must be of the form `%result = ...`"))?;
        let result_name = normalize_value_name(result_name_raw.trim());
        if result_name.is_empty() {
            return Err(TextIrError::new("result identifier cannot be empty"));
        }
        let rest = rest.trim();

        let (op_name, remainder) = parse_op_name(rest)?;
        let (operand_specs, attributes, type_section) = split_operands_attrs_and_type(remainder)?;
        let result_type = parse_type(type_section)?;
        let operands = convert_operands(operand_specs, value_map)?;
        let operation = build_operation(op_name, &operands, &result_type, &attributes)?;

        let id = builder.emit_single(operation, operands, result_type.clone());
        value_map.insert(result_name, id);
        Ok(())
    }
}

fn normalize_value_name(name: &str) -> String {
    name.trim_start_matches('%').to_string()
}

fn find_matching_paren(src: &str, open_index: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in src[open_index..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(open_index + offset);
                }
            }
            _ => {}
        }
    }
    None
}

struct Parameter {
    name: String,
    ty: String,
}

fn parse_type(src: &str) -> Result<ValueType, TextIrError> {
    let trimmed = src.trim();
    if let Some(body) = trimmed.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
        let inner = body.trim();
        if inner.is_empty() {
            return Err(TextIrError::new(
                "tuple types must contain at least one element",
            ));
        }
        let elements = split_top_level(inner, ',')
            .into_iter()
            .map(parse_type)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(ValueType::Tuple(elements));
    }
    if let Some(body) = trimmed
        .strip_prefix("tensor<")
        .or_else(|| trimmed.strip_prefix("Tensor<"))
        .and_then(|s| s.strip_suffix('>'))
    {
        return parse_tensor_type(body);
    }
    Err(TextIrError::new(format!(
        "unsupported type `{trimmed}`; only `tensor<...>` is implemented"
    )))
}

fn parse_tensor_type(body: &str) -> Result<ValueType, TextIrError> {
    let mut parts = body.split(',');
    let dtype_str = parts
        .next()
        .ok_or_else(|| TextIrError::new("tensor type must specify a dtype"))?
        .trim();
    let dtype = parse_dtype(dtype_str)?;
    let dims_str = parts.next().unwrap_or("").trim();
    let dims = if dims_str.is_empty() {
        Vec::new()
    } else {
        parse_dimensions(dims_str)?
    };
    if parts.next().is_some() {
        return Err(TextIrError::new(
            "tensor type accepts only `tensor<dtype, dims>` form",
        ));
    }
    Ok(ptir_utils::value_type_tensor(
        ptir_utils::tensor_spec_static(dtype, &dims),
    ))
}

fn parse_dtype(src: &str) -> Result<DType, TextIrError> {
    let normalized = src.trim().to_ascii_uppercase();
    match normalized.as_str() {
        "I1" => Ok(DType::I1),
        "SI4" => Ok(DType::Si4),
        "UI4" => Ok(DType::Ui4),
        "SI8" => Ok(DType::Si8),
        "UI8" => Ok(DType::Ui8),
        "SI16" => Ok(DType::Si16),
        "UI16" => Ok(DType::Ui16),
        "SI32" => Ok(DType::Si32),
        "UI32" => Ok(DType::Ui32),
        "SI64" => Ok(DType::Si64),
        "UI64" => Ok(DType::Ui64),
        "FP8E4M3" => Ok(DType::Fp8E4M3),
        "FP8E5M2" => Ok(DType::Fp8E5M2),
        "BF16" => Ok(DType::Bf16),
        "F16" => Ok(DType::F16),
        "F32" => Ok(DType::F32),
        "F64" => Ok(DType::F64),
        "CF32" => Ok(DType::Cf32),
        "CF64" => Ok(DType::Cf64),
        other => Err(TextIrError::new(format!("unsupported dtype `{other}`"))),
    }
}

fn parse_dimensions(src: &str) -> Result<Vec<usize>, TextIrError> {
    src.split('x')
        .map(|dim| {
            let dim = dim.trim();
            if dim.is_empty() {
                return Err(TextIrError::new("dimension sizes cannot be empty"));
            }
            dim.parse::<usize>()
                .map_err(|_| TextIrError::new(format!("invalid dimension `{dim}`")))
        })
        .collect()
}

fn split_top_level(input: &str, delimiter: char) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (idx, ch) in input.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth = depth.saturating_sub(1);
            }
            c if c == delimiter && depth == 0 => {
                if start != idx {
                    pieces.push(input[start..idx].trim());
                }
                start = idx + c.len_utf8();
            }
            _ => {}
        }
    }
    if start < input.len() {
        pieces.push(input[start..].trim());
    } else if input.trim().is_empty() {
        // no parameters
    }
    pieces
}

fn parse_op_name(src: &str) -> Result<(String, &str), TextIrError> {
    let trimmed = src.trim_start();
    if trimmed.is_empty() {
        return Err(TextIrError::new("operation name is missing"));
    }
    let mut end = trimmed.len();
    for (idx, ch) in trimmed.char_indices() {
        if ch.is_whitespace() || ch == '(' || ch == '%' {
            end = idx;
            break;
        }
    }
    let name = trimmed[..end].trim();
    if name.is_empty() {
        return Err(TextIrError::new("operation name cannot be empty"));
    }
    let remainder = trimmed[end..].trim_start();
    Ok((name.to_string(), remainder))
}

#[derive(Debug, Clone)]
struct AttributeExpr {
    name: String,
    value: String,
}

fn split_operands_attrs_and_type(
    remainder: &str,
) -> Result<(Vec<OperandExpr>, Vec<AttributeExpr>, &str), TextIrError> {
    let (operands_section, ty_section) = remainder
        .split_once("->")
        .ok_or_else(|| TextIrError::new("operations must specify result type with `->`"))?;
    let (operand_tokens, attrs) = parse_operands_and_attributes(operands_section)?;
    let operand_exprs = operand_tokens
        .into_iter()
        .map(|token| parse_operand_expr(&token))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((operand_exprs, attrs, ty_section.trim()))
}

fn parse_operands_and_attributes(
    section: &str,
) -> Result<(Vec<String>, Vec<AttributeExpr>), TextIrError> {
    let mut operands = Vec::new();
    let mut attrs = Vec::new();
    let chars = section.trim();
    let bytes = chars.as_bytes();
    let len = bytes.len();
    let mut idx = 0usize;

    while idx < len {
        while idx < len && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= len {
            break;
        }
        match bytes[idx] {
            b'(' => {
                let end = find_matching(chars, idx, '(', ')')
                    .ok_or_else(|| TextIrError::new("unmatched `(` in operand list"))?;
                let inner = &chars[idx + 1..end];
                for token in split_top_level(inner, ',') {
                    let trimmed = token.trim();
                    if !trimmed.is_empty() {
                        operands.push(trimmed.to_string());
                    }
                }
                idx = end + 1;
            }
            b'%' | b'-' | b'0'..=b'9' => {
                let start = idx;
                idx += 1;
                while idx < len
                    && !bytes[idx].is_ascii_whitespace()
                    && bytes[idx] != b','
                    && bytes[idx] != b')'
                    && bytes[idx] != b'['
                {
                    idx += 1;
                }
                operands.push(chars[start..idx].trim().to_string());
            }
            b',' => {
                idx += 1;
            }
            _ => {
                let attr_start = idx;
                while idx < len && (bytes[idx].is_ascii_alphabetic() || bytes[idx] == b'_') {
                    idx += 1;
                }
                if idx >= len || bytes[idx] != b'[' {
                    return Err(TextIrError::new(format!(
                        "unexpected token starting at `{}`",
                        &chars[attr_start..]
                    )));
                }
                let name = chars[attr_start..idx].trim().to_string();
                let end = find_matching(chars, idx, '[', ']')
                    .ok_or_else(|| TextIrError::new("unmatched `[` in attribute"))?;
                let value = chars[idx + 1..end].trim().to_string();
                attrs.push(AttributeExpr { name, value });
                idx = end + 1;
            }
        }
    }

    Ok((operands, attrs))
}

fn find_matching(src: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0isize;
    let mut idx = start;
    let bytes = src.as_bytes();
    while idx < bytes.len() {
        if bytes[idx] == open as u8 {
            depth += 1;
        } else if bytes[idx] == close as u8 {
            depth -= 1;
            if depth == 0 {
                return Some(idx);
            }
        }
        idx += 1;
    }
    None
}

fn parse_operand_expr(src: &str) -> Result<OperandExpr, TextIrError> {
    let trimmed = src.trim();
    if trimmed.is_empty() {
        return Err(TextIrError::new("operand cannot be empty"));
    }
    let without_percent = trimmed.trim_start_matches('%');
    if without_percent.is_empty() {
        return Err(TextIrError::new("operand name cannot be empty"));
    }
    if let Some(open) = without_percent.find('[') {
        let close = without_percent
            .rfind(']')
            .ok_or_else(|| TextIrError::new("tuple element operand missing `]`"))?;
        if close <= open {
            return Err(TextIrError::new("tuple element operand malformed"));
        }
        let name = without_percent[..open].trim();
        if name.is_empty() {
            return Err(TextIrError::new("tuple element base value cannot be empty"));
        }
        let index_str = without_percent[open + 1..close].trim();
        let index = index_str
            .parse::<usize>()
            .map_err(|_| TextIrError::new(format!("invalid tuple index `{index_str}`")))?;
        Ok(OperandExpr::TupleElement {
            tuple: name.to_string(),
            index,
        })
    } else {
        Ok(OperandExpr::Value(without_percent.trim().to_string()))
    }
}

fn convert_operands(
    specs: Vec<OperandExpr>,
    value_map: &HashMap<String, ValueId>,
) -> Result<Vec<Operand>, TextIrError> {
    specs
        .into_iter()
        .map(|spec| match spec {
            OperandExpr::Value(name) => value_map
                .get(&name)
                .copied()
                .map(Operand::Value)
                .ok_or_else(|| TextIrError::new(format!("unknown operand `%{name}`"))),
            OperandExpr::TupleElement { tuple, index } => value_map
                .get(&tuple)
                .copied()
                .map(|id| Operand::TupleElement { tuple: id, index })
                .ok_or_else(|| TextIrError::new(format!("unknown tuple value `%{tuple}`"))),
        })
        .collect()
}

fn build_operation(
    name: String,
    operands: &[Operand],
    result_type: &ValueType,
    attrs: &[AttributeExpr],
) -> Result<Operation, TextIrError> {
    let attr_map: HashMap<&str, &str> = attrs
        .iter()
        .map(|attr| (attr.name.as_str(), attr.value.as_str()))
        .collect();
    match name.as_str() {
        "cast" => {
            let ValueType::Tensor(spec) = result_type else {
                return Err(TextIrError::new(
                    "cast result type must be a tensor in the PTIR text format",
                ));
            };
            Ok(Operation::Cast(CastSpec { dtype: spec.dtype }))
        }
        "reduce_max" | "reduce_sum" => {
            let ValueType::Tensor(_spec) = result_type else {
                return Err(TextIrError::new(
                    "reduce result type must be a tensor in PTIR text",
                ));
            };
            let Some(axes) = attr_map.get("axes") else {
                return Err(TextIrError::new(
                    "reduce operations require `axes[...]` attribute",
                ));
            };
            let keepdims = attr_map
                .get("keepdims")
                .map(|v| parse_bool(v))
                .transpose()?
                .unwrap_or(false);
            let accum_dtype = match attr_map.get("accum_dtype") {
                Some(value) if !value.trim().is_empty() => Some(parse_dtype(value)?),
                _ => None,
            };
            let out_dtype = match attr_map.get("out_dtype") {
                Some(value) if !value.trim().is_empty() => Some(parse_dtype(value)?),
                _ => None,
            };
            let axes_values = parse_usize_list(axes)?;
            let kind = if name == "reduce_max" {
                ReduceKind::Max
            } else {
                ReduceKind::Sum
            };
            Ok(Operation::Reduce(ReduceSpec {
                kind,
                axes: axes_values,
                keepdims,
                accum_dtype,
                out_dtype,
            }))
        }
        "broadcast_to" => {
            let Some(result_shape_raw) = attr_map.get("shape").or(attr_map.get("result_shape"))
            else {
                return Err(TextIrError::new(
                    "broadcast_to requires `shape[...]` attribute",
                ));
            };
            let shape_values = parse_usize_list(result_shape_raw)?;
            Ok(Operation::BroadcastTo(BroadcastToSpec {
                result_shape: ptir_utils::shape_static(&shape_values),
            }))
        }
        "dot_general" => {
            if operands.len() != 2 {
                return Err(TextIrError::new(
                    "dot_general expects exactly two operands in PTIR text",
                ));
            }
            let Some(contract_lhs_raw) = attr_map.get("contract_lhs") else {
                return Err(TextIrError::new(
                    "dot_general requires `contract_lhs[...]` attribute",
                ));
            };
            let Some(contract_rhs_raw) = attr_map.get("contract_rhs") else {
                return Err(TextIrError::new(
                    "dot_general requires `contract_rhs[...]` attribute",
                ));
            };
            let contract_lhs = parse_usize_list(contract_lhs_raw)?;
            let contract_rhs = parse_usize_list(contract_rhs_raw)?;
            let batch_shared = attr_map
                .get("batch")
                .map(|value| parse_usize_list(value))
                .transpose()?;
            let batch_lhs = attr_map
                .get("batch_lhs")
                .map(|value| parse_usize_list(value))
                .transpose()?;
            let batch_rhs = attr_map
                .get("batch_rhs")
                .map(|value| parse_usize_list(value))
                .transpose()?;
            let lhs_axes = batch_lhs.or(batch_shared.clone()).unwrap_or_default();
            let rhs_axes = batch_rhs.or(batch_shared).unwrap_or_default();
            let accum_dtype = attr_map
                .get("accum_dtype")
                .map(|value| parse_dtype(value))
                .transpose()?;
            let out_dtype = attr_map
                .get("out_dtype")
                .map(|value| parse_dtype(value))
                .transpose()?;
            Ok(Operation::DotGeneral(DotGeneralSpec {
                batch_lhs: lhs_axes,
                batch_rhs: rhs_axes,
                contract_lhs,
                contract_rhs,
                accum_dtype,
                out_dtype,
            }))
        }
        "sub" | "add" | "div" | "mul" => {
            let op = match name.as_str() {
                "sub" => ElementwiseBinaryOp::Sub,
                "add" => ElementwiseBinaryOp::Add,
                "div" => ElementwiseBinaryOp::Div,
                "mul" => ElementwiseBinaryOp::Mul,
                _ => unreachable!(),
            };
            Ok(Operation::ElementwiseBinary(op))
        }
        "exp" => Ok(Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp)),
        "neg" => Ok(Operation::ElementwiseUnary(ElementwiseUnaryOp::Neg)),
        "stop_gradient" => {
            if operands.len() != 1 {
                return Err(TextIrError::new(
                    "stop_gradient expects exactly one operand in PTIR text",
                ));
            }
            Ok(Operation::StopGradient)
        }
        other => Err(TextIrError::new(format!(
            "unsupported operation `{other}` in PTIR text parser"
        ))),
    }
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, TextIrError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let inner = trimmed
        .trim_start_matches(['[', '(', '{'])
        .trim_end_matches([']', ')', '}']);
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|token| {
            let tok = token.trim();
            if tok.is_empty() {
                return Err(TextIrError::new("empty entry in integer list"));
            }
            tok.parse::<usize>()
                .map_err(|_| TextIrError::new(format!("invalid integer `{tok}`")))
        })
        .collect()
}

fn parse_bool(value: &str) -> Result<bool, TextIrError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(TextIrError::new(format!(
            "invalid boolean literal `{other}`"
        ))),
    }
}

enum OperandExpr {
    Value(String),
    TupleElement { tuple: String, index: usize },
}

/// Builds a `Program` from the lightweight PTIR syntax at compile-time.
#[macro_export]
macro_rules! ptir_program {
    ($src:expr) => {{
        $crate::backend::text_ir::parse_program($src).expect("failed to parse PTIR text program")
    }};
}

/// Builds a reusable PTIR snippet with placeholder support.
#[macro_export]
macro_rules! ptir_snippet {
    ($src:expr) => {{
        $crate::backend::text_ir::PtirSnippet::new($src)
    }};
}

/// Borrowed snippet source that can be instantiated with runtime bindings.
#[derive(Debug, Clone, Copy)]
pub struct PtirSnippet {
    source: &'static str,
}

impl PtirSnippet {
    pub const fn new(source: &'static str) -> Self {
        Self { source }
    }

    pub fn instantiate(&self, bindings: &SnippetBindings) -> Result<ParsedProgram, TextIrError> {
        let mut text = self.source.to_string();
        for (name, value) in &bindings.replacements {
            let needle = format!("{{{{{name}}}}}");
            text = text.replace(&needle, value);
        }
        if text.contains("{{") {
            return Err(TextIrError::new(format!(
                "unbound snippet placeholder in `{text}`"
            )));
        }
        parse_program_with_symbols(&text)
    }
}

/// Typed bindings supplied to instantiate a [`PtirSnippet`].
#[derive(Debug, Default, Clone)]
pub struct SnippetBindings {
    replacements: HashMap<String, String>,
    value_bindings: HashMap<String, ValueId>,
}

impl SnippetBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(mut self, name: &str, value: ValueId) -> Self {
        self.value_bindings.insert(name.to_string(), value);
        self
    }

    pub fn int<I: Into<usize>>(mut self, name: &str, value: I) -> Self {
        self.replacements
            .insert(name.to_string(), value.into().to_string());
        self
    }

    pub fn bool(mut self, name: &str, value: bool) -> Self {
        self.replacements
            .insert(name.to_string(), value.to_string());
        self
    }

    pub fn dtype(mut self, name: &str, dtype: DType) -> Self {
        self.replacements
            .insert(name.to_string(), dtype_to_string(dtype).to_string());
        self
    }

    pub fn shape<S>(mut self, name: &str, dims: S) -> Self
    where
        S: IntoIterator<Item = usize>,
    {
        let rendered = dims
            .into_iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        self.replacements.insert(name.to_string(), rendered);
        self
    }

    pub fn dims<S>(mut self, name: &str, dims: S) -> Self
    where
        S: IntoIterator<Item = usize>,
    {
        let rendered = dims
            .into_iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.replacements.insert(name.to_string(), rendered);
        self
    }

    pub fn shape_list<S>(mut self, name: &str, dims: S) -> Self
    where
        S: IntoIterator<Item = usize>,
    {
        let rendered = dims
            .into_iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.replacements.insert(name.to_string(), rendered);
        self
    }

    pub(crate) fn bound_values(&self) -> impl Iterator<Item = (&str, ValueId)> {
        self.value_bindings.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

fn dtype_to_string(dtype: DType) -> &'static str {
    match dtype {
        DType::I1 => "i1",
        DType::Si4 => "si4",
        DType::Ui4 => "ui4",
        DType::Si8 => "si8",
        DType::Ui8 => "ui8",
        DType::Si16 => "si16",
        DType::Ui16 => "ui16",
        DType::Si32 => "si32",
        DType::Ui32 => "ui32",
        DType::Si64 => "si64",
        DType::Ui64 => "ui64",
        DType::Fp8E4M3 => "fp8_e4m3",
        DType::Fp8E5M2 => "fp8_e5m2",
        DType::Bf16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::Cf32 => "cf32",
        DType::Cf64 => "cf64",
    }
}

/// Result returned after emitting a snippet into a graph builder.
#[derive(Debug, Clone)]
pub struct SnippetResult {
    value_ids: Vec<ValueId>,
    value_specs: Vec<TensorSpec>,
}

impl SnippetResult {
    pub fn new(value_ids: Vec<ValueId>, value_specs: Vec<TensorSpec>) -> Self {
        debug_assert_eq!(value_ids.len(), value_specs.len());
        Self {
            value_ids,
            value_specs,
        }
    }

    pub fn results(&self) -> &[ValueId] {
        &self.value_ids
    }

    pub fn specs(&self) -> &[TensorSpec] {
        &self.value_specs
    }

    pub fn into_parts(self) -> (Vec<ValueId>, Vec<TensorSpec>) {
        (self.value_ids, self.value_specs)
    }

    pub fn single(self) -> ValueId {
        assert_eq!(self.value_ids.len(), 1);
        self.value_ids.into_iter().next().unwrap()
    }
}
