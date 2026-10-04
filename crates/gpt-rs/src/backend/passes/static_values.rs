use std::collections::HashSet;

use crate::backend::optimizer::OptimizeContext;
use crate::backend::spec::{Function, Operand, PortableBackend, ValueId};
use crate::tensor::InputRole;

/// The values of `function` that depend on parameters and literals only: the parameter inputs and
/// the result of every instruction whose operands are all static. They are the same on every run
/// with the same parameters.
pub(crate) fn static_values<B: PortableBackend + 'static>(
    function: &Function,
    cx: &OptimizeContext<'_, B>,
) -> HashSet<ValueId> {
    let mut statics: HashSet<ValueId> = function
        .parameter_ids
        .iter()
        .copied()
        .filter(|id| cx.entry().role_of(*id) == Some(InputRole::Param))
        .collect();
    // The body defines every value before its uses.
    for inst in &function.body {
        let static_operands = inst.operands.iter().all(|operand| match operand {
            Operand::Literal(_) => true,
            Operand::Value(value) => statics.contains(value),
            Operand::TupleElement { .. } => false,
        });
        if static_operands {
            statics.insert(inst.id);
        }
    }
    statics
}
