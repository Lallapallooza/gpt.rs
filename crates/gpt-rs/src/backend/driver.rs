use std::collections::{HashMap, VecDeque};

use crate::backend::{
    index::InstId,
    pattern::FrozenPatternSet,
    rewriter::ProgramRewriter,
    spec::{Function, Operation, ValueId},
};

#[derive(Debug, Clone)]
pub struct GreedyConfig {
    pub max_iterations: usize,
    pub enable_dce: bool,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            max_iterations: usize::MAX,
            enable_dce: true,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GreedyRewriteStats {
    pub iterations: usize,
    pub applied: usize,
    pub dce_removed: usize,
}

pub fn apply_patterns_and_fold_greedily(
    func: &mut Function,
    patterns: &FrozenPatternSet,
    cfg: &GreedyConfig,
) -> GreedyRewriteStats {
    if patterns.is_empty() {
        return GreedyRewriteStats::default();
    }

    let mut rewriter = ProgramRewriter::new(func).expect("failed to build rewriter");
    let mut worklist = VecDeque::new();
    seed_worklist(&rewriter, &mut worklist);

    let mut failure_cache: HashMap<(usize, InstId), u32> = HashMap::new();
    let mut stats = GreedyRewriteStats::default();

    while let Some(inst) = worklist.pop_front() {
        if stats.iterations >= cfg.max_iterations {
            break;
        }
        if !rewriter.contains(inst) {
            continue;
        }

        let version = rewriter.version(inst).unwrap_or(0);
        let op_snapshot = rewriter.op(inst).clone();
        let mut applied_pattern = false;

        for (idx, pattern) in patterns.matching(&op_snapshot) {
            if failure_cache.get(&(idx, inst)).copied() == Some(version) {
                continue;
            }

            if pattern.match_and_rewrite(inst, &mut rewriter) {
                applied_pattern = true;
                stats.applied += 1;
                stats.iterations = stats.iterations.saturating_add(1);
                clear_failure_entries(inst, &mut failure_cache);
                seed_worklist(&rewriter, &mut worklist);
                if stats.iterations >= cfg.max_iterations {
                    break;
                }
                break;
            } else {
                failure_cache.insert((idx, inst), version);
            }
        }

        if applied_pattern && stats.iterations >= cfg.max_iterations {
            break;
        }
    }

    if cfg.enable_dce {
        stats.dce_removed = run_dce(&mut rewriter);
    }

    stats
}

fn seed_worklist(rewriter: &ProgramRewriter, worklist: &mut VecDeque<InstId>) {
    worklist.clear();
    for inst in rewriter.insts_in_order() {
        worklist.push_back(inst);
    }
}

fn clear_failure_entries(inst: InstId, cache: &mut HashMap<(usize, InstId), u32>) {
    cache.retain(|(_, cached_inst), _| *cached_inst != inst);
}

fn run_dce(rewriter: &mut ProgramRewriter) -> usize {
    let mut removed_total = 0;
    loop {
        let mut removed_in_pass = 0;
        for inst in rewriter.insts_in_order().into_iter().rev() {
            if !rewriter.contains(inst) {
                continue;
            }
            if is_side_effecting(rewriter.op(inst)) {
                continue;
            }
            let value = rewriter.value_of(inst);
            if is_function_result(rewriter, value) {
                continue;
            }
            if rewriter.users_of(value).is_empty() {
                rewriter.erase_inst(inst);
                removed_in_pass += 1;
            }
        }
        if removed_in_pass == 0 {
            break;
        }
        removed_total += removed_in_pass;
    }
    removed_total
}

fn is_function_result(rewriter: &ProgramRewriter, value: ValueId) -> bool {
    rewriter.func.result_ids.contains(&value)
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
