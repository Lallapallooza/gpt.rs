use std::cmp::Reverse;
use std::collections::HashSet;

use crate::backend::spec::ValueId;

use super::SelectedFusion;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SelectionResult {
    pub selected: Vec<usize>,
    pub rejected_overlap: usize,
}

pub fn select_non_overlapping(candidates: &[SelectedFusion]) -> SelectionResult {
    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_by_key(|&idx| {
        let candidate = &candidates[idx];
        (
            Reverse(candidate.score),
            Reverse(candidate.candidate.inst_values.len()),
            candidate.candidate.anchor.0,
            candidate.candidate.id,
        )
    });

    let mut occupied = HashSet::<ValueId>::new();
    let mut selected = Vec::new();
    let mut rejected_overlap = 0usize;
    for idx in order {
        let candidate = &candidates[idx];
        let has_overlap = candidate
            .candidate
            .inst_values
            .iter()
            .any(|value| occupied.contains(value));
        if has_overlap {
            rejected_overlap = rejected_overlap.saturating_add(1);
            continue;
        }
        for value in &candidate.candidate.inst_values {
            occupied.insert(*value);
        }
        selected.push(idx);
    }

    SelectionResult {
        selected,
        rejected_overlap,
    }
}
