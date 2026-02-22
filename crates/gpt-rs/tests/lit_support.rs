use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use gpt_rs::backend::optimizer::{
    EntryParam, EntrySignature, FunctionPass, OptimizeConfig, OptimizeContext, OptimizeServices,
};
use gpt_rs::backend::param_resolver::InMemoryParamResolver;
use gpt_rs::backend::param_resolver::ParamResolver;
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{
    BroadcastToSpec, Dimension, ElementwiseUnaryOp, Function, Instruction, Operand, Operation,
    PortableBackend, Program, ReduceSpec, ReshapeDim, ReshapeSpec, Shape, SliceSpec, TensorInit,
    TensorLiteral, TensorSpec, TransposeSpec, ValueId, ValueType,
};
use gpt_rs::tensor::InputRole;

pub struct Case {
    pub name: String,
    pub meta: String,
    pub input: String,
    pub expected: String,
}

/// Fixture format:
/// EXPECTED:
/// <text PTIR>
/// INPUT:
/// <text PTIR>
/// run_test
/// (repeat)
pub fn load_cases(fixture_rel: &str) -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(fixture_rel);
    let contents =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {:?}: {}", path, e));
    let mut cases = Vec::new();
    let mut meta = String::new();
    let mut expected = String::new();
    let mut input = String::new();
    let mut in_meta = false;
    let mut in_expected = false;
    let mut in_input = false;
    let mut counter = 0;
    for line in contents.lines() {
        match line.trim_end() {
            "META:" => {
                in_meta = true;
                in_expected = false;
                in_input = false;
                meta.clear();
            }
            "EXPECTED:" => {
                in_expected = true;
                in_meta = false;
                in_input = false;
                expected.clear();
            }
            "INPUT:" => {
                in_input = true;
                in_meta = false;
                in_expected = false;
                input.clear();
            }
            "run_test" => {
                counter += 1;
                cases.push(Case {
                    name: format!("case_{counter}"),
                    meta: meta.clone(),
                    input: input.clone(),
                    expected: expected.clone(),
                });
                in_meta = false;
                in_expected = false;
                in_input = false;
            }
            other => {
                if in_meta {
                    meta.push_str(other);
                    meta.push('\n');
                } else if in_expected {
                    expected.push_str(other);
                    expected.push('\n');
                } else if in_input {
                    input.push_str(other);
                    input.push('\n');
                }
            }
        }
    }
    cases
}

pub fn run_case_with_passes<B: PortableBackend + 'static>(
    backend: &B,
    passes: &[&dyn FunctionPass<B>],
    case: &Case,
) {
    let mut prog = parse_program(&case.input);
    let func = prog
        .functions
        .get_mut(0)
        .expect("fixture must have a function");

    let (entry, resolver) = entry_signature_for_case(backend, func, &case.meta);
    let services = OptimizeServices {
        params: Some(&resolver),
    };
    let cfg = OptimizeConfig::default();
    let mut cx = OptimizeContext::new(backend, services, entry, cfg);

    let mut changed = false;
    for pass in passes {
        let res = pass.run(func, &mut cx);
        changed |= res.changed;
    }

    assert!(changed, "pass must change IR for {}", case.name);
    assert!(
        ProgramRewriter::new(func).expect("index").verify(),
        "SSA verification failed for {}",
        case.name
    );
    let out = prog.to_text();
    assert_eq!(
        out.trim(),
        case.expected.trim(),
        "output mismatch for {}",
        case.name
    );
}

// ---------------------- tiny PTIR text parser --------------------------

fn entry_signature_for_case<B: PortableBackend + 'static>(
    backend: &B,
    func: &Function,
    meta: &str,
) -> (EntrySignature<B>, InMemoryParamResolver<B::TensorHandle>) {
    let mut roles: Vec<InputRole> = Vec::new();
    let mut stable_ids: Vec<Option<u128>> = Vec::new();

    for line in meta.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        if let Some(rest) = line.strip_prefix("roles:") {
            roles = rest
                .split(|ch: char| ch == ',' || ch.is_whitespace())
                .filter(|v| !v.is_empty())
                .map(|token| match token {
                    "Arg" => InputRole::Arg,
                    "Param" => InputRole::Param,
                    other => panic!("unsupported role token {other}"),
                })
                .collect();
            continue;
        }
        if let Some(rest) = line.strip_prefix("stable_ids:") {
            stable_ids = rest
                .split(|ch: char| ch == ',' || ch.is_whitespace())
                .filter(|v| !v.is_empty())
                .map(|token| Some(token.parse::<u128>().expect("stable id")))
                .collect();
            continue;
        }
    }

    let resolver = InMemoryParamResolver::<B::TensorHandle>::new();
    let mut entry_params = Vec::with_capacity(func.parameter_ids.len());

    for (idx, (param_id, param_ty)) in func
        .parameter_ids
        .iter()
        .copied()
        .zip(func.parameters.iter().cloned())
        .enumerate()
    {
        let role = roles.get(idx).copied().unwrap_or(InputRole::Arg);
        let stable_id = stable_ids
            .get(idx)
            .copied()
            .flatten()
            .or(Some(u128::from(param_id.0)));

        if role == InputRole::Param {
            let stable_id = stable_id.expect("param stable id");
            let spec = match &param_ty {
                ValueType::Tensor(spec) => spec.clone(),
                other => panic!("param-only folding expects tensor params, got {other:?}"),
            };
            let handle = backend
                .materialize(TensorInit::Zeroed(spec))
                .expect("materialize zeroed param");
            resolver.set(stable_id, handle);
        }

        entry_params.push(EntryParam {
            id: param_id,
            ty: param_ty,
            role,
            stable_id,
        });
    }

    (EntrySignature::new(entry_params), resolver)
}

fn parse_program(text: &str) -> Program {
    let mut lines = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>();
    assert!(
        lines.remove(0).starts_with("program @"),
        "program header missing"
    );
    assert!(
        lines.remove(0).starts_with("func @"),
        "function header missing"
    );

    let mut params = Vec::new();
    let mut param_ids = Vec::new();
    let mut body = Vec::new();
    let mut results = Vec::new();
    let mut result_ids = Vec::new();

    while let Some(line) = lines.first().cloned() {
        lines.remove(0);
        if line.starts_with("params:") {
            continue;
        }
        if line.starts_with("body:") {
            break;
        }
        if line.starts_with('%') {
            let (id, ty) = parse_binding(line);
            param_ids.push(ValueId(id));
            params.push(ty);
        }
    }

    while let Some(line) = lines.first().cloned() {
        if line.starts_with("results:") {
            lines.remove(0);
            break;
        }
        lines.remove(0);
        if line.starts_with('%') {
            body.push(parse_instruction(line));
        }
    }

    for line in lines {
        if line.starts_with('%') {
            let (id, ty) = parse_binding(line);
            result_ids.push(ValueId(id));
            results.push(ty);
        }
    }

    Program::new("captured").with_functions(vec![Function {
        name: "captured".into(),
        parameters: params,
        parameter_ids: param_ids,
        body,
        hints: vec![],
        results,
        result_ids,
    }])
}

fn parse_binding(line: &str) -> (u32, ValueType) {
    let line = line.trim_end_matches(',');
    let (id_part, ty_part) = line.split_once(':').expect("binding colon");
    let id: u32 = id_part.trim_start_matches('%').trim().parse().expect("id");
    let ty = parse_type(ty_part.trim());
    (id, ty)
}

fn parse_type(s: &str) -> ValueType {
    let inner = s.trim().trim_start_matches("tensor<").trim_end_matches('>');
    let (dtype_str, shape_str) = inner.split_once(" x ").unwrap_or((inner, "[]"));
    let dtype = match dtype_str.trim() {
        "F32" => gpt_rs::backend::spec::DType::F32,
        other => panic!("unsupported dtype {other}"),
    };
    let dims = shape_str.trim();
    let dims_vec = if dims == "[]" {
        Vec::new()
    } else {
        dims.split('x')
            .map(|d| {
                let d = d.trim();
                if let Some(rest) = d.strip_prefix('?') {
                    Dimension::Dynamic(gpt_rs::backend::spec::DimSymbol::new(rest))
                } else {
                    Dimension::Static(d.parse().expect("static dim"))
                }
            })
            .collect()
    };
    ValueType::Tensor(TensorSpec {
        dtype,
        shape: Shape::new(dims_vec),
    })
}

fn parse_instruction(line: &str) -> Instruction {
    let (id_part, rest) = line.split_once('=').expect("inst split");
    let id: u32 = id_part
        .trim_start_matches('%')
        .trim()
        .parse()
        .expect("inst id");
    let rest = rest.trim();
    let (op_part, rest) = rest.split_once('(').expect("op split");
    let (operands_tail, ty_part) = rest.rsplit_once(") ->").expect("split operands/type");
    let (op_attr, operands_part) = match operands_tail.rfind('(') {
        Some(idx) => (operands_tail[..idx].trim(), &operands_tail[idx..]),
        None => (operands_tail.trim(), ""),
    };

    let operands = parse_operands(operands_part);
    let output_ty = parse_type(ty_part.trim());
    let op = parse_op(op_part.trim(), op_attr);

    Instruction {
        id: ValueId(id),
        op,
        operands,
        output: output_ty,
    }
}

fn parse_operands(s: &str) -> Vec<Operand> {
    let mut ops_str = s.trim();
    if let Some(stripped) = ops_str.strip_prefix('(') {
        ops_str = stripped;
    }
    if let Some(stripped) = ops_str.strip_suffix(')') {
        ops_str = stripped;
    }
    if ops_str.is_empty() {
        return Vec::new();
    }
    ops_str
        .split(',')
        .map(|o| Operand::Value(ValueId(o.trim().trim_start_matches('%').parse().unwrap())))
        .collect()
}

fn parse_op(name: &str, op_repr: &str) -> Operation {
    match name {
        "Reshape" => {
            let dims = extract_numbers(op_repr)
                .into_iter()
                .map(|n| ReshapeDim::Explicit(Dimension::Static(n)))
                .collect();
            Operation::Reshape(ReshapeSpec { new_shape: dims })
        }
        "Transpose" => {
            let perm = extract_numbers(op_repr);
            Operation::Transpose(TransposeSpec { perm })
        }
        "Slice" => {
            let starts = extract_usize_list(op_repr, "starts");
            let sizes = extract_usize_list(op_repr, "sizes");
            Operation::Slice(SliceSpec { starts, sizes })
        }
        "BroadcastTo" => {
            let result_shape = extract_usize_list(op_repr, "result_shape");
            Operation::BroadcastTo(BroadcastToSpec {
                result_shape: Shape::new(
                    result_shape
                        .into_iter()
                        .map(Dimension::Static)
                        .collect::<Vec<_>>(),
                ),
            })
        }
        "ElementwiseBinary" => {
            if op_repr.contains("Add") {
                Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Add)
            } else if op_repr.contains("Mul") {
                Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Mul)
            } else if op_repr.contains("Sub") {
                Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Sub)
            } else {
                panic!("unsupported elementwise binary op");
            }
        }
        "ElementwiseUnary" => {
            if op_repr.contains("Rsqrt") {
                Operation::ElementwiseUnary(ElementwiseUnaryOp::Rsqrt)
            } else if op_repr.contains("Erf") {
                Operation::ElementwiseUnary(ElementwiseUnaryOp::Erf)
            } else if op_repr.contains("Exp") {
                Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp)
            } else {
                panic!("unsupported elementwise unary op");
            }
        }
        "Constant" => {
            let bytes = extract_bytes(op_repr);
            let dtype = if op_repr.contains("dtype: F32") {
                gpt_rs::backend::spec::DType::F32
            } else {
                panic!("unsupported constant dtype");
            };
            let shape_list = extract_usize_list(op_repr, "Shape");
            let lit = TensorLiteral {
                spec: TensorSpec {
                    dtype,
                    shape: Shape::new(
                        shape_list
                            .into_iter()
                            .map(Dimension::Static)
                            .collect::<Vec<_>>(),
                    ),
                },
                bytes: Arc::<[u8]>::from(bytes),
            };
            Operation::Constant(lit)
        }
        "Reduce" => {
            let axes = extract_usize_list(op_repr, "axes");
            let keepdims = op_repr.contains("keepdims: true");
            let kind = if op_repr.contains("kind: Sum") {
                gpt_rs::backend::spec::ReduceKind::Sum
            } else if op_repr.contains("kind: Max") {
                gpt_rs::backend::spec::ReduceKind::Max
            } else if op_repr.contains("kind: Min") {
                gpt_rs::backend::spec::ReduceKind::Min
            } else {
                panic!("unsupported reduce kind");
            };
            Operation::Reduce(ReduceSpec {
                kind,
                axes,
                keepdims,
                accum_dtype: None,
                out_dtype: None,
            })
        }
        _ => panic!("unsupported op {}", name),
    }
}

fn extract_usize_list(op_repr: &str, key: &str) -> Vec<usize> {
    let (_, rest) = op_repr.split_once(key).unwrap_or((op_repr, op_repr));
    let inside = rest
        .split_once('[')
        .and_then(|(_, b)| b.split_once(']'))
        .map(|(a, _)| a)
        .unwrap_or("");
    if inside.trim().is_empty() {
        return Vec::new();
    }
    extract_numbers(inside)
}

fn extract_bytes(op_repr: &str) -> Vec<u8> {
    let (_, rest) = op_repr.split_once("bytes: [").expect("bytes start");
    let inside = rest
        .split_once(']')
        .map(|(a, _)| a)
        .expect("bytes closing bracket");
    if inside.trim().is_empty() {
        return Vec::new();
    }
    inside
        .split(',')
        .map(|b| b.trim().parse::<u8>().expect("byte value"))
        .collect()
}

fn extract_numbers(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.push(cur.parse().expect("number"));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        out.push(cur.parse().expect("number"));
    }
    out
}
