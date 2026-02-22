use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use gpt_rs::backend::fusion::{
    FUSION_ATTR_KIND, FUSION_ATTR_VERSION, FUSION_KIND_ELEMENTWISE_DAG_V1,
};
use gpt_rs::backend::shape_helpers::{contiguous_strides_or_error, static_dims_or_error};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, CustomCallAttr, CustomCallSpec, DType, TensorSpec,
};

use crate::kernels::{KernelKind, KernelSpec};

#[derive(Debug, Clone)]
pub struct FusedElementwisePlan {
    ops_kind: Vec<i64>,
    ops_code: Vec<i64>,
    lhs: Vec<i64>,
    rhs: Vec<i64>,
}

impl FusedElementwisePlan {
    pub fn parse(spec: &CustomCallSpec) -> BackendResult<Self> {
        let version = custom_call_i64(spec, FUSION_ATTR_VERSION)?;
        if version != 1 {
            return Err(BackendError::execution(format!(
                "unsupported fused elementwise payload version {version}"
            )));
        }
        let kind = custom_call_string(spec, FUSION_ATTR_KIND)?;
        if kind != FUSION_KIND_ELEMENTWISE_DAG_V1 {
            return Err(BackendError::execution(format!(
                "unsupported fused elementwise payload kind '{kind}'"
            )));
        }
        let ops_kind = custom_call_i64_array(spec, "ops_kind")?.clone();
        let ops_code = custom_call_i64_array(spec, "ops_code")?.clone();
        let lhs = custom_call_i64_array(spec, "lhs")?.clone();
        let rhs = custom_call_i64_array(spec, "rhs")?.clone();
        if ops_kind.len() != ops_code.len()
            || ops_kind.len() != lhs.len()
            || ops_kind.len() != rhs.len()
        {
            return Err(BackendError::execution(
                "fused elementwise custom_call attrs length mismatch",
            ));
        }
        if ops_kind.len() < 2 {
            return Err(BackendError::execution(
                "fused elementwise custom_call requires at least two nodes",
            ));
        }
        Ok(Self {
            ops_kind,
            ops_code,
            lhs,
            rhs,
        })
    }

    pub fn build_kernel_spec(
        &self,
        out_spec: &TensorSpec,
        input_specs: &[TensorSpec],
    ) -> BackendResult<KernelSpec> {
        if out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "fused elementwise custom_call requires F32 output",
            ));
        }
        if input_specs.is_empty() {
            return Err(BackendError::execution(
                "fused elementwise custom_call requires at least one input",
            ));
        }
        for spec in input_specs {
            if spec.dtype != DType::F32 {
                return Err(BackendError::execution(
                    "fused elementwise custom_call requires F32 inputs",
                ));
            }
        }

        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution(
                "dynamic dimensions are not supported by fused elementwise runtime",
            )
        })?;
        let input_dims = input_specs
            .iter()
            .map(|spec| {
                static_dims_or_error(&spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by fused elementwise runtime",
                    )
                })
            })
            .collect::<BackendResult<Vec<_>>>()?;

        let source = emit_kernel_source(
            out_dims.as_slice(),
            input_dims.as_slice(),
            self.ops_kind.as_slice(),
            self.ops_code.as_slice(),
            self.lhs.as_slice(),
            self.rhs.as_slice(),
        )?;
        let source_hash = fused_kernel_hash(
            &source,
            out_dims.as_slice(),
            input_dims.as_slice(),
            self.ops_kind.as_slice(),
            self.ops_code.as_slice(),
            self.lhs.as_slice(),
            self.rhs.as_slice(),
        );
        let symbol = format!("gpt_rs_triton_fused_elementwise_f32_{source_hash:016x}");
        Ok(KernelSpec {
            id: format!("gpt_rs.triton.kernel.fused_elementwise_f32.v1.{source_hash:016x}"),
            kind: KernelKind::FusedElementwiseF32,
            source,
            symbol,
        })
    }
}

fn custom_call_i64(spec: &CustomCallSpec, key: &str) -> BackendResult<i64> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::I64(value)) => Ok(*value),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing i64 attr '{key}'"
        ))),
    }
}

fn custom_call_string<'a>(spec: &'a CustomCallSpec, key: &str) -> BackendResult<&'a str> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::String(value)) => Ok(value.as_str()),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing string attr '{key}'"
        ))),
    }
}

fn custom_call_i64_array<'a>(spec: &'a CustomCallSpec, key: &str) -> BackendResult<&'a Vec<i64>> {
    match spec.attrs.get(key) {
        Some(CustomCallAttr::I64Array(values)) => Ok(values),
        _ => Err(BackendError::execution(format!(
            "triton custom_call missing i64 array attr '{key}'"
        ))),
    }
}

fn emit_kernel_source(
    out_dims: &[usize],
    input_dims: &[Vec<usize>],
    ops_kind: &[i64],
    ops_code: &[i64],
    lhs: &[i64],
    rhs: &[i64],
) -> BackendResult<String> {
    if out_dims.is_empty() || out_dims.len() > 4 {
        return Err(BackendError::execution(format!(
            "fused elementwise output rank must be in [1,4], got {}",
            out_dims.len()
        )));
    }
    for dims in input_dims {
        if dims.len() > out_dims.len() {
            return Err(BackendError::execution(
                "fused elementwise input rank exceeds output rank",
            ));
        }
    }
    validate_refs(input_dims.len(), ops_kind, lhs, rhs)?;

    let mut out = TritonSourceWriter::default();
    let symbol = "gpt_rs_triton_fused_elementwise_f32";
    out.line(format!("# gpt_rs.kernel: {}", "fused_elementwise_f32"));
    out.line(format!("# gpt_rs.symbol: {symbol}"));
    out.line(format!(
        "# gpt_rs.signature: {},out_ptr=*fp32,n=i32",
        join_input_sig(input_dims.len())
    ));
    out.line(format!(
        "# gpt_rs.param_abi: {},*fp32,u32,*opaque",
        join_input_abi(input_dims.len())
    ));
    out.line(format!("# gpt_rs.constexpr: BLOCK_SIZE={}", 256));
    out.line(format!("# gpt_rs.num_warps: {}", 8));
    out.blank();
    out.line("import triton");
    out.line("import triton.language as tl");
    out.blank();
    out.line("@triton.jit");
    out.line(format!(
        "def {symbol}({}, out_ptr, n, BLOCK_SIZE: tl.constexpr):",
        join_input_params(input_dims.len())
    ));
    out.line("    pid = tl.program_id(0)");
    out.line("    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)");
    out.line("    mask = offs < n");
    out.line("    rem = offs");
    for axis in (0..out_dims.len()).rev() {
        let dim = out_dims[axis];
        out.line(format!("    idx_{axis} = rem % {dim}"));
        if axis > 0 {
            out.line(format!("    rem = rem // {dim}"));
        }
    }
    out.blank();

    for (input_idx, dims) in input_dims.iter().enumerate() {
        let aligned = align_shape_to_rank(dims, out_dims.len())?;
        let strides = contiguous_strides_or_error(aligned.as_slice(), || {
            BackendError::execution("stride overflow")
        })?;
        let index = input_index_expr(aligned.as_slice(), strides.as_slice(), out_dims.len());
        out.line(format!(
            "    in{input_idx} = tl.load(in{input_idx}_ptr + ({index}), mask=mask, other=0.0)"
        ));
    }
    out.blank();

    for node_idx in 0..ops_kind.len() {
        let lhs_expr = ref_expr(lhs[node_idx], input_dims.len())?;
        let expr = match ops_kind[node_idx] {
            0 => unary_expr_from_code(ops_code[node_idx], lhs_expr.as_str())?,
            1 => {
                let rhs_expr = ref_expr(rhs[node_idx], input_dims.len())?;
                binary_expr_from_code(ops_code[node_idx], lhs_expr.as_str(), rhs_expr.as_str())?
            }
            other => {
                return Err(BackendError::execution(format!(
                    "unsupported fused node kind {other} at node {node_idx}"
                )))
            }
        };
        out.line(format!("    node_{node_idx} = {expr}"));
    }
    out.line(format!("    out = node_{}", ops_kind.len() - 1));
    out.line("    tl.store(out_ptr + offs, out, mask=mask)");
    Ok(out.finish())
}

fn validate_refs(
    input_count: usize,
    ops_kind: &[i64],
    lhs: &[i64],
    rhs: &[i64],
) -> BackendResult<()> {
    for idx in 0..ops_kind.len() {
        let max_ref = i64::try_from(input_count + idx)
            .map_err(|_| BackendError::execution("fused elementwise reference overflow"))?;
        if lhs[idx] < 0 || lhs[idx] >= max_ref {
            return Err(BackendError::execution(format!(
                "fused elementwise lhs reference {} out of range at node {idx}",
                lhs[idx]
            )));
        }
        if ops_kind[idx] == 1 {
            if rhs[idx] < 0 || rhs[idx] >= max_ref {
                return Err(BackendError::execution(format!(
                    "fused elementwise rhs reference {} out of range at node {idx}",
                    rhs[idx]
                )));
            }
        } else if rhs[idx] != -1 {
            return Err(BackendError::execution(format!(
                "fused unary node {idx} must use rhs=-1"
            )));
        }
    }
    Ok(())
}

fn join_input_sig(input_count: usize) -> String {
    (0..input_count)
        .map(|idx| format!("in{idx}_ptr=*fp32"))
        .collect::<Vec<_>>()
        .join(",")
}

fn join_input_abi(input_count: usize) -> String {
    vec!["*fp32"; input_count].join(",")
}

fn join_input_params(input_count: usize) -> String {
    (0..input_count)
        .map(|idx| format!("in{idx}_ptr"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn input_index_expr(aligned_dims: &[usize], strides: &[usize], rank: usize) -> String {
    let mut terms = Vec::new();
    for axis in 0..rank {
        if aligned_dims[axis] != 1 {
            let stride = strides[axis];
            if stride == 1 {
                terms.push(format!("idx_{axis}"));
            } else {
                terms.push(format!("idx_{axis} * {stride}"));
            }
        }
    }
    if terms.is_empty() {
        // Keep the index block-shaped for masked loads from broadcast scalars.
        "offs * 0".to_string()
    } else {
        terms.join(" + ")
    }
}

fn ref_expr(reference: i64, input_count: usize) -> BackendResult<String> {
    let idx = usize::try_from(reference)
        .map_err(|_| BackendError::execution("fused elementwise reference conversion failed"))?;
    if idx < input_count {
        Ok(format!("in{idx}"))
    } else {
        Ok(format!("node_{}", idx - input_count))
    }
}

fn unary_expr_from_code(code: i64, arg: &str) -> BackendResult<String> {
    let expr = match code {
        0 => format!("-({arg})"),
        1 => format!("tl.abs({arg})"),
        2 => format!("tl.exp({arg})"),
        3 => format!("tl.log({arg})"),
        4 => format!("((tl.exp(2.0 * ({arg})) - 1.0) / (tl.exp(2.0 * ({arg})) + 1.0))"),
        5 => erf_expr(arg),
        6 => format!("1.0 / tl.sqrt({arg})"),
        7 => format!("1.0 / ({arg})"),
        _ => {
            return Err(BackendError::execution(format!(
                "unsupported fused unary code {code}"
            )))
        }
    };
    Ok(expr)
}

fn erf_expr(arg: &str) -> String {
    let sign = format!("tl.where(({arg}) < 0.0, -1.0, 1.0)");
    let ax = format!("tl.abs({arg})");
    let t = format!("(1.0 / (1.0 + 0.3275911 * ({ax})))");
    let poly = format!(
        "((((((1.061405429 * ({t}) - 1.453152027) * ({t}) + 1.421413741) * ({t}) - 0.284496736) * ({t}) + 0.254829592) * ({t})))"
    );
    format!("({sign} * (1.0 - ({poly}) * tl.exp(-(({ax}) * ({ax})))))")
}

fn binary_expr_from_code(code: i64, lhs: &str, rhs: &str) -> BackendResult<String> {
    let expr = match code {
        0 => format!("({lhs}) + ({rhs})"),
        1 => format!("({lhs}) - ({rhs})"),
        2 => format!("({lhs}) * ({rhs})"),
        3 => format!("({lhs}) / ({rhs})"),
        4 => format!("tl.maximum({lhs}, {rhs})"),
        5 => format!("tl.minimum({lhs}, {rhs})"),
        _ => {
            return Err(BackendError::execution(format!(
                "unsupported fused binary code {code}"
            )))
        }
    };
    Ok(expr)
}

fn align_shape_to_rank(dims: &[usize], rank: usize) -> BackendResult<Vec<usize>> {
    if dims.len() > rank {
        return Err(BackendError::execution(format!(
            "cannot align rank {} tensor into rank {} output",
            dims.len(),
            rank
        )));
    }
    let mut aligned = vec![1usize; rank];
    let start = rank - dims.len();
    aligned[start..].copy_from_slice(dims);
    Ok(aligned)
}

fn fused_kernel_hash(
    source: &str,
    out_dims: &[usize],
    input_dims: &[Vec<usize>],
    ops_kind: &[i64],
    ops_code: &[i64],
    lhs: &[i64],
    rhs: &[i64],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    out_dims.hash(&mut hasher);
    input_dims.hash(&mut hasher);
    ops_kind.hash(&mut hasher);
    ops_code.hash(&mut hasher);
    lhs.hash(&mut hasher);
    rhs.hash(&mut hasher);
    hasher.finish()
}

#[derive(Default)]
struct TritonSourceWriter {
    source: String,
}

impl TritonSourceWriter {
    fn line(&mut self, line: impl AsRef<str>) {
        writeln!(&mut self.source, "{}", line.as_ref()).expect("write to string");
    }

    fn blank(&mut self) {
        self.line("");
    }

    fn finish(self) -> String {
        self.source
    }
}
