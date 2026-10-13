//! Buffer planning: where the generated function keeps each value it computes. A `reshape` shares
//! the buffer of its operand. Every other value that is neither a result nor a constant gets a
//! temporary slot, which values of the same dtype and size reuse once its last reader has run.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::tensor::TensorOp;
use gpt_rs_ir::ir::{DType, Value};

use super::types::{Instruction, Kind};

/// The storage the plan gives a value an instruction computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Planned {
    /// Output `index` of the function.
    Output { index: usize },
    /// The temporary buffer `slot`.
    Temp { slot: usize },
    /// A constant of the function.
    Const,
    /// The buffer of the operand of the `reshape` that computes the value.
    Alias { of: Value },
}

/// The storage of each value of `insts`, which compute `results` from the function's arguments.
/// `size` gives the dtype and the byte length of a value.
pub(super) fn plan(
    insts: &[Instruction<'_>],
    results: &[Value],
    size: impl Fn(Value) -> Result<(DType, usize)>,
) -> Result<HashMap<Value, Planned>> {
    // The position that defines each value, and the last one that reads it. The arguments are
    // defined at 0, instruction `i` at `i + 1`, and the results are read after the last one.
    let mut live: HashMap<Value, (usize, usize)> = HashMap::new();
    for (index, inst) in insts.iter().enumerate() {
        for &operand in &inst.operands {
            live.entry(operand).or_insert((0, 0)).1 = index + 1;
        }
        for &result in &inst.results {
            live.insert(result, (index + 1, index + 1));
        }
    }
    for &result in results {
        live.entry(result).or_insert((0, 0)).1 = insts.len() + 1;
    }

    let outputs: HashMap<Value, usize> = results
        .iter()
        .enumerate()
        .map(|(index, &value)| (value, index))
        .collect();
    let mut planned = HashMap::new();
    let mut alias_root: HashMap<Value, Value> = HashMap::new();
    for inst in insts {
        for &value in &inst.results {
            // A constant is a static array, which the function copies when it returns the
            // constant.
            if matches!(inst.kind, Kind::Op(TensorOp::Constant(_))) {
                planned.insert(value, Planned::Const);
            } else if let Some(&index) = outputs.get(&value) {
                planned.insert(value, Planned::Output { index });
            } else if let (Kind::Op(TensorOp::Reshape(_)), [operand]) =
                (&inst.kind, inst.operands.as_slice())
            {
                planned.insert(value, Planned::Alias { of: *operand });
                let root = alias_root.get(operand).copied().unwrap_or(*operand);
                alias_root.insert(value, root);
            }
        }
    }

    // A value lives as long as any alias of it.
    for (&alias, &root) in &alias_root {
        let alias_end = live[&alias].1;
        let range = live
            .get_mut(&root)
            .ok_or_else(|| anyhow!("{root:?} has no live range"))?;
        range.1 = range.1.max(alias_end);
    }

    let mut temps: Vec<(DType, usize, usize, usize, Value)> = Vec::new();
    for &value in insts.iter().flat_map(|inst| &inst.results) {
        if planned.contains_key(&value) {
            continue;
        }
        let (dtype, byte_len) = size(value)?;
        let (start, end) = live[&value];
        temps.push((dtype, byte_len, start, end, value));
    }
    temps.sort_by_key(|&(dtype, byte_len, start, ..)| (dtype, byte_len, start));
    // Each slot's dtype, byte length, and the last position that reads it.
    let mut slots: Vec<(DType, usize, usize)> = Vec::new();
    for (dtype, byte_len, start, end, value) in temps {
        let free = slots
            .iter()
            .position(|&(d, len, last)| d == dtype && len == byte_len && last < start);
        let slot = match free {
            Some(slot) => {
                slots[slot].2 = end;
                slot
            }
            None => {
                slots.push((dtype, byte_len, end));
                slots.len() - 1
            }
        };
        planned.insert(value, Planned::Temp { slot });
    }
    Ok(planned)
}
