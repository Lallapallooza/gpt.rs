use std::collections::HashMap;

use crate::backend::{
    optimizer::OptimizeContext,
    rewriter::ProgramRewriter,
    spec::{Function, Operation, PortableBackend, ValueId},
};

use super::{FunctionPass, FunctionPassResult};

fn is_side_effecting(op: &Operation) -> bool {
    matches!(
        op,
        Operation::ScatterAdd(_)
            | Operation::Cond(_)
            | Operation::While(_)
            | Operation::Scan(_)
            | Operation::RngUniform(_)
            | Operation::RngNormal(_)
            | Operation::TopK(_)
            | Operation::Quantize(_)
            | Operation::Dequantize(_)
            | Operation::Requantize(_)
    )
}

/// Common-subexpression elimination for pure operations.
///
/// Hashes `(op, operands, result_type)` for every side-effect-free instruction
/// and replaces later duplicates with the first occurrence.
#[derive(Default)]
pub struct CommonSubexpressionEliminationPass;

impl CommonSubexpressionEliminationPass {
    const NAME: &'static str = "cse";
}

impl<B: PortableBackend + 'static> FunctionPass<B> for CommonSubexpressionEliminationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, _cx: &mut OptimizeContext<B>) -> FunctionPassResult {
        let mut rewriter =
            ProgramRewriter::new(function).expect("failed to build rewriter for CSE");

        let mut seen: HashMap<Vec<u8>, ValueId> = HashMap::new();
        let mut stats = FunctionPassResult::default();

        for inst in rewriter.insts_in_order() {
            if !rewriter.contains(inst) {
                continue;
            }
            stats.iterations = stats.iterations.saturating_add(1);

            let op = rewriter.op(inst).clone();
            if is_side_effecting(&op) {
                continue;
            }

            let operands = rewriter.operands(inst).to_vec();
            let Some(result_ty) = rewriter.type_of(rewriter.value_of(inst)).cloned() else {
                continue;
            };

            let key_bytes = match bincode::serialize(&(&op, &operands, &result_ty)) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };

            let value = rewriter.value_of(inst);
            if let Some(existing) = seen.get(&key_bytes).copied() {
                replace_value(&mut rewriter, value, existing, &mut stats);
                continue;
            }

            seen.insert(key_bytes, value);
        }

        stats
    }
}

fn replace_value(
    rewriter: &mut ProgramRewriter,
    from: ValueId,
    to: ValueId,
    stats: &mut FunctionPassResult,
) {
    if from == to {
        return;
    }
    rewriter.replace_all_uses(from, to);
    for result_id in &mut rewriter.func.result_ids {
        if *result_id == from {
            *result_id = to;
        }
    }
    if let Some(inst) = rewriter.inst_of(from) {
        rewriter.erase_inst(inst);
    }
    stats.changed = true;
    stats.rewrites_applied += 1;
    stats.erased_insts += 1;
}
