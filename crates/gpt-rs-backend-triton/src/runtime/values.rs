use gpt_rs::backend::spec::{Function, Operand, ValueId};

use crate::tensor::TritonTensor;

#[derive(Clone)]
pub(crate) struct DenseValueStore {
    entries: Vec<Option<TritonTensor>>,
}

impl DenseValueStore {
    pub(crate) fn new(function: &Function) -> Self {
        let max_value = max_value_id(function);
        Self {
            entries: vec![None; max_value.saturating_add(1)],
        }
    }

    pub(crate) fn contains(&self, value: ValueId) -> bool {
        self.get(value).is_some()
    }

    pub(crate) fn get(&self, value: ValueId) -> Option<&TritonTensor> {
        self.entries
            .get(value.0 as usize)
            .and_then(|entry| entry.as_ref())
    }

    pub(crate) fn get_cloned(&self, value: ValueId) -> Option<TritonTensor> {
        self.get(value).cloned()
    }

    pub(crate) fn insert(&mut self, value: ValueId, tensor: TritonTensor) {
        let index = value.0 as usize;
        if index >= self.entries.len() {
            self.entries.resize(index + 1, None);
        }
        self.entries[index] = Some(tensor);
    }
}

fn max_value_id(function: &Function) -> usize {
    let mut max_value = 0usize;
    for value in &function.parameter_ids {
        max_value = max_value.max(value.0 as usize);
    }
    for value in &function.result_ids {
        max_value = max_value.max(value.0 as usize);
    }
    for instruction in &function.body {
        max_value = max_value.max(instruction.id.0 as usize);
        for operand in &instruction.operands {
            match operand {
                Operand::Value(value) => {
                    max_value = max_value.max(value.0 as usize);
                }
                Operand::TupleElement { tuple, .. } => {
                    max_value = max_value.max(tuple.0 as usize);
                }
                Operand::Literal(_) => {}
            }
        }
    }
    max_value
}
