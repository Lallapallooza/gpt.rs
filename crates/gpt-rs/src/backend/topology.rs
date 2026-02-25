use std::collections::HashSet;
use std::fmt;

use crate::backend::spec::{Function, Operand};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    pub missing_value: u32,
    pub instruction_id: u32,
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operand value {} is missing before instruction {}",
            self.missing_value, self.instruction_id
        )
    }
}

pub fn validate_function_topology(function: &Function) -> Result<(), TopologyError> {
    let mut available = HashSet::new();
    for id in &function.parameter_ids {
        available.insert(id.0);
    }

    for instruction in &function.body {
        for operand in &instruction.operands {
            let operand_id = match operand {
                Operand::Value(value) => Some(value.0),
                Operand::TupleElement { tuple, .. } => Some(tuple.0),
                Operand::Literal(_) => None,
            };
            if let Some(value_id) = operand_id {
                if !available.contains(&value_id) {
                    return Err(TopologyError {
                        missing_value: value_id,
                        instruction_id: instruction.id.0,
                    });
                }
            }
        }
        available.insert(instruction.id.0);
    }

    Ok(())
}
