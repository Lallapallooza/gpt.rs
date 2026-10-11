//! `argmax` and `top_k` of f32 tensors: a scan of each line along the axis.

use anyhow::{ensure, Result};
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::types::{DType, TensorType};

use super::super::profile::register_op_profile_generic;
use super::super::types::{Instruction, Kind};
use super::super::utils::{c_type, dims_usize, push_block};
use super::super::value_info::{
    ensure_dtype, operand_dtype, operand_expr, operand_types, output_info,
};
use super::EmitContext;
use crate::kernels::PARALLEL_MIN_WORK;

/// The most shares that one long line splits into.
const MAX_SHARES: usize = 64;

pub(super) fn emit_instruction(
    inst: &Instruction<'_>,
    ctx: &mut EmitContext<'_>,
) -> Result<Option<usize>> {
    let EmitContext {
        module,
        value_infos,
        matmul_profile,
        ..
    } = ctx;
    let &Kind::Op(op) = &inst.kind else {
        return Ok(None);
    };
    let (axis, k, largest) = match op {
        TensorOp::ArgMax(spec) => (spec.axis, None, true),
        TensorOp::TopK(spec) => (spec.axis, Some(spec.k), true),
        TensorOp::BottomK(spec) => (spec.axis, Some(spec.k), false),
        _ => return Ok(None),
    };
    let input_dtype = operand_dtype(&inst.operands[0], value_infos)?;
    ensure_dtype(input_dtype, DType::F32, "argmax and top_k read f32")?;
    let inputs = operand_types(&inst.operands, value_infos)?;
    let (outer, n, inner) = lines(&inputs[0], axis)?;
    let input = operand_expr(&inst.operands[0], value_infos)?;
    let first = output_info(value_infos, inst.results[0])?;
    let op_id = register_op_profile_generic(matmul_profile, inst.label, &first.ty, &inputs)?;
    let block = match k {
        None => {
            let index_type = c_type(first.ty.dtype)?;
            argmax(&input, &first.var, index_type, (outer, n, inner))
        }
        Some(k) => {
            let indices = output_info(value_infos, inst.results[1])?;
            let index_type = c_type(indices.ty.dtype)?;
            let lines = (outer, n, inner);
            top_k(
                &input,
                &first.var,
                &indices.var,
                index_type,
                lines,
                k,
                largest,
            )
        }
    };
    push_block(module, 1, &block);
    Ok(Some(op_id))
}

/// The extents `(outer, n, inner)` of `ty` around `axis`, whose lines are `n` long.
fn lines(ty: &TensorType, axis: i64) -> Result<(usize, usize, usize)> {
    let dims = dims_usize(ty)?;
    let rank = dims.len() as i64;
    let axis = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&axis),
        "axis {axis} is out of range for {ty}"
    );
    let axis = axis as usize;
    let outer = dims[..axis].iter().product();
    let inner = dims[axis + 1..].iter().product();
    Ok((outer, dims[axis], inner))
}

/// `better(v, w)`: whether `v` comes before `w`. NaN is larger than every number, and of equal
/// values the earlier one comes first, so the scan replaces only on a strict win.
fn better(largest: bool) -> &'static str {
    if largest {
        "((v != v && w == w) || v > w)"
    } else {
        "((w != w && v == v) || v < w)"
    }
}

/// Scans each line for its best element. Many lines split across the team by line. One long line,
/// like the logits of a decode step, splits into shares whose winners then meet in index order.
fn argmax(
    input: &str,
    out: &str,
    index_type: &str,
    (outer, n, inner): (usize, usize, usize),
) -> String {
    let better = better(true);
    let lines = outer * inner;
    let parallel = lines * n >= PARALLEL_MIN_WORK;
    let (line_split, max_parts) = match (parallel, lines) {
        (false, _) => ("", 1),
        (true, 1) => ("", MAX_SHARES),
        (true, _) => ("_Pragma(\"omp parallel for schedule(static)\")", 1),
    };
    format!(
        r#"
        {{
          const float* x = (const float*){input};
          {index_type}* out = ({index_type}*){out};
          const size_t parts = GPTRS_MIN((size_t)gpt_rs_c_team_size(), (size_t){max_parts});
          const size_t share = ({n} + parts - 1) / parts;
          {line_split}
          for (size_t l = 0; l < {lines}; ++l) {{
            const float* line = x + (l / {inner}) * {n} * {inner} + l % {inner};
            float part_value[{max_parts}];
            size_t part_index[{max_parts}];
            _Pragma("omp parallel for schedule(static) if (parts > 1)")
            for (size_t p = 0; p < parts; ++p) {{
              const size_t begin = p * share;
              const size_t end = GPTRS_MIN(begin + share, (size_t){n});
              if (begin >= end) {{ continue; }}
              float w = line[begin * {inner}];
              size_t best = begin;
              for (size_t j = begin + 1; j < end; ++j) {{
                float v = line[j * {inner}];
                if {better} {{ w = v; best = j; }}
              }}
              part_value[p] = w;
              part_index[p] = best;
            }}
            float w = part_value[0];
            size_t best = part_index[0];
            for (size_t p = 1; p < parts && p * share < {n}; ++p) {{
              float v = part_value[p];
              if {better} {{ w = v; best = part_index[p]; }}
            }}
            out[l] = ({index_type})best;
          }}
        }}
        "#
    )
}

/// Keeps the best `k` elements of each line in order as it scans it: an element that beats the
/// last kept one moves in at its place, and the last one drops out.
fn top_k(
    input: &str,
    values: &str,
    indices: &str,
    index_type: &str,
    (outer, n, inner): (usize, usize, usize),
    k: usize,
    largest: bool,
) -> String {
    let better = better(largest);
    format!(
        r#"
        {{
          const float* x = (const float*){input};
          float* values = (float*){values};
          {index_type}* indices = ({index_type}*){indices};
          for (size_t o = 0; o < {outer}; ++o) {{
            for (size_t i = 0; i < {inner}; ++i) {{
              const float* line = x + o * {n} * {inner} + i;
              float* top = values + o * {k} * {inner} + i;
              {index_type}* top_index = indices + o * {k} * {inner} + i;
              size_t kept = 0;
              for (size_t j = 0; j < {n}; ++j) {{
                float v = line[j * {inner}];
                if (kept == {k}) {{
                  float w = top[({k} - 1) * {inner}];
                  if (!{better}) {{ continue; }}
                  kept -= 1;
                }}
                size_t at = kept;
                while (at > 0) {{
                  float w = top[(at - 1) * {inner}];
                  if (!{better}) {{ break; }}
                  top[at * {inner}] = w;
                  top_index[at * {inner}] = top_index[(at - 1) * {inner}];
                  at -= 1;
                }}
                top[at * {inner}] = v;
                top_index[at * {inner}] = ({index_type})j;
                kept += 1;
              }}
            }}
          }}
        }}
        "#
    )
}
