//! The value of `tensor.constant`.

use crate::dialects::tensor::ir::Constant;
use crate::ir::{ConstantValue, Dense};

impl ConstantValue for Constant {
    fn value(&self) -> &Dense {
        &self.value
    }
}
