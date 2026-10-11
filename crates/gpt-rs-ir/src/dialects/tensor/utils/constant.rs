//! The value of `tensor.constant`.

use crate::dense::Dense;
use crate::dialects::tensor::ir::Constant;
use crate::op::ConstantOp;

impl ConstantOp for Constant {
    fn value(&self) -> &Dense {
        &self.value
    }
}
