//! The value of `tensor.constant`.

use crate::dialects::tensor::ir::Constant;
use crate::ir::{ConstantOp, Dense};

impl ConstantOp for Constant {
    fn value(&self) -> &Dense {
        &self.value
    }
}
