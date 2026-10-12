//! Reading a function.

use super::ir::*;

impl Func {
    /// Whether the runtime runs this function when it loads parameters, not per call.
    pub fn is_load_time(&self) -> bool {
        !self.result_params.is_empty()
    }

    /// The parameter that argument `index` binds.
    pub fn param(&self, index: usize) -> Option<ParamId> {
        self.params.get(index).copied().flatten()
    }
}
