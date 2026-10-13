//! `rng_uniform` of f32 values: the Philox4x32-10 stream of the state, as the spec defines it.

use anyhow::{bail, ensure, Result};
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::ir::DType;

use super::super::profile::register_op_profile_generic;
use super::super::types::{Instruction, Kind};
use super::super::utils::push_block;
use super::super::value_info::{operand_expr, operand_types, output_info};
use super::EmitContext;

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
    let spec = match op {
        TensorOp::RngUniform(spec) => spec,
        TensorOp::RngNormal(_) => bail!("the C backend draws only uniform values, not normal"),
        _ => return Ok(None),
    };
    ensure!(
        spec.dtype == DType::F32,
        "the C backend draws only f32 values, not {}",
        spec.dtype
    );
    let inputs = operand_types(&inst.operands, value_infos)?;
    let state = operand_expr(&inst.operands[0], value_infos)?;
    let new_state = output_info(value_infos, inst.results[0])?;
    let values = output_info(value_infos, inst.results[1])?;
    let op_id = register_op_profile_generic(matmul_profile, inst.label, &values.ty, &inputs)?;
    let (new_state, values, count) = (&new_state.var, &values.var, values.elem_count);
    // Each f32 is the top 24 bits of one word over 2^24.
    let block = format!(
        r#"
        {{
          const uint64_t* state = (const uint64_t*){state};
          uint64_t* new_state = (uint64_t*){new_state};
          float* values = (float*){values};
          const uint64_t key = state[0], counter = state[1];
          const size_t blocks = ({count} + 3) / 4;
          for (size_t b = 0; b < blocks; ++b) {{
            uint32_t words[4];
            gpt_rs_philox4x32_10(key, counter, (uint64_t)b, words);
            for (size_t w = 0; w < 4 && b * 4 + w < {count}; ++w) {{
              values[b * 4 + w] = (float)(words[w] >> 8) * 0x1p-24f;
            }}
          }}
          new_state[0] = key;
          new_state[1] = counter + (uint64_t)blocks;
        }}
        "#
    );
    push_block(module, 1, &block);
    Ok(Some(op_id))
}
