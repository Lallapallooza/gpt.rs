use std::collections::{HashMap, HashSet};

use crate::backend::optimizer::{FunctionPass, OptimizeContext, PassResult};
use crate::backend::spec::{Function, Operand, Operation, PortableBackend, ValueId};

#[derive(Default)]
pub struct DeadCodeEliminationPass;

impl DeadCodeEliminationPass {
    const NAME: &'static str = "dce";
}

impl<B: PortableBackend + 'static> FunctionPass<B> for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn run(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult {
        let mut def_map: HashMap<ValueId, usize> = HashMap::with_capacity(function.body.len());
        for (idx, inst) in function.body.iter().enumerate() {
            def_map.insert(inst.id, idx);
        }

        let mut worklist: Vec<ValueId> = function.result_ids.clone();
        for inst in &function.body {
            if is_side_effecting(&inst.op) {
                worklist.push(inst.id);
            }
        }

        let mut live: HashSet<ValueId> = HashSet::new();
        while let Some(value) = worklist.pop() {
            if !live.insert(value) {
                continue;
            }
            let Some(&idx) = def_map.get(&value) else {
                continue;
            };
            let inst = &function.body[idx];
            for operand in &inst.operands {
                match operand {
                    Operand::Value(dep) => worklist.push(*dep),
                    Operand::TupleElement { tuple, .. } => worklist.push(*tuple),
                    Operand::Literal(_) => {}
                }
            }
        }

        let before = function.body.len();
        function
            .body
            .retain(|inst| live.contains(&inst.id) || is_side_effecting(&inst.op));
        let removed = before.saturating_sub(function.body.len());

        cx.entry_mut().remove_params_by_live_set(function, &live);

        PassResult {
            changed: removed > 0,
            iterations: 0,
            rewrites_applied: 0,
            erased_insts: removed,
        }
    }
}

fn is_side_effecting(op: &Operation) -> bool {
    matches!(
        op,
        Operation::ScatterAdd(_)
            | Operation::Cond(_)
            | Operation::While(_)
            | Operation::Scan(_)
            | Operation::RngUniform(_)
            | Operation::RngNormal(_)
    )
}
