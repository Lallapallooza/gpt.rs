mod custom;
mod dot;
mod elementwise;
mod reduction;
mod shape;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::ir_query;
use gpt_rs::backend::spec::{CustomCallAttr, DType, Function, Operation, TensorSpec};

pub(super) fn validate_instruction(
    function: &Function,
    instruction: &gpt_rs::backend::spec::Instruction,
) -> ConversionResult<()> {
    let cx = ValidateCtx {
        function,
        instruction,
    };
    match &instruction.op {
        Operation::CustomCall(_) => custom::validate(&cx),
        Operation::DotGeneral(_) => dot::validate(&cx),
        Operation::Reduce(_) => reduction::validate(&cx),
        Operation::ElementwiseUnary(_)
        | Operation::ElementwiseBinary(_)
        | Operation::Compare(_)
        | Operation::Select
        | Operation::Take => elementwise::validate(&cx),
        Operation::Constant(_)
        | Operation::StopGradient
        | Operation::Reshape(_)
        | Operation::BroadcastTo(_)
        | Operation::Slice(_)
        | Operation::Transpose(_)
        | Operation::Concat(_)
        | Operation::Iota(_)
        | Operation::DynamicSlice(_)
        | Operation::DynamicUpdateSlice(_)
        | Operation::ExtractPatches(_)
        | Operation::ReduceWindow(_) => shape::validate(&cx),
        other => Err(ConversionError::new(format!(
            "triton lowering does not support operation: {:?}",
            other
        ))),
    }
}

#[derive(Clone, Copy)]
pub(super) struct DTypeCapability {
    allowed: &'static [DType],
}

impl DTypeCapability {
    pub(super) const fn new(allowed: &'static [DType]) -> Self {
        Self { allowed }
    }

    pub(super) fn require(&self, spec: &TensorSpec, what: &str) -> ConversionResult<()> {
        if self.allowed.contains(&spec.dtype) {
            return Ok(());
        }
        Err(ConversionError::new(format!(
            "{what} does not support dtype {:?}; allowed: {:?}",
            spec.dtype, self.allowed
        )))
    }
}

pub(super) struct ValidateCtx<'a> {
    pub(super) function: &'a Function,
    pub(super) instruction: &'a gpt_rs::backend::spec::Instruction,
}

impl<'a> ValidateCtx<'a> {
    pub(super) fn output_tensor_spec(&self) -> ConversionResult<TensorSpec> {
        match ir_query::tensor_spec_from_value_type(&self.instruction.output) {
            Some(spec) => Ok(spec),
            None => Err(ConversionError::new(
                "tuple outputs are not supported by triton lowering",
            )),
        }
    }

    pub(super) fn operand_tensor_spec(
        &self,
        index: usize,
        what: &str,
    ) -> ConversionResult<TensorSpec> {
        ir_query::tensor_spec_for_operand(self.function, self.instruction.operands.get(index))
            .map_err(|err| {
                ConversionError::new(format!(
                    "failed to resolve operand tensor spec for {what}: {err}"
                ))
            })
    }

    pub(super) fn each_operand_tensor_spec(&self, what: &str) -> ConversionResult<Vec<TensorSpec>> {
        self.instruction
            .operands
            .iter()
            .map(|operand| {
                ir_query::tensor_spec_for_operand(self.function, Some(operand)).map_err(|err| {
                    ConversionError::new(format!(
                        "failed to resolve operand tensor spec for {what}: {err}"
                    ))
                })
            })
            .collect::<ConversionResult<Vec<_>>>()
    }

    pub(super) fn custom_call_i64_array<'b>(
        &self,
        attr: Option<&'b CustomCallAttr>,
        name: &str,
    ) -> ConversionResult<&'b [i64]> {
        match attr {
            Some(CustomCallAttr::I64Array(values)) => Ok(values.as_slice()),
            _ => Err(ConversionError::new(format!(
                "triton custom_call missing i64 array attr '{name}'"
            ))),
        }
    }

    pub(super) fn custom_call_i64(
        &self,
        attr: Option<&CustomCallAttr>,
        name: &str,
    ) -> ConversionResult<i64> {
        match attr {
            Some(CustomCallAttr::I64(value)) => Ok(*value),
            _ => Err(ConversionError::new(format!(
                "triton custom_call missing i64 attr '{name}'"
            ))),
        }
    }

    pub(super) fn custom_call_f64(
        &self,
        attr: Option<&CustomCallAttr>,
        name: &str,
    ) -> ConversionResult<f64> {
        match attr {
            Some(CustomCallAttr::F64(value)) => Ok(*value),
            _ => Err(ConversionError::new(format!(
                "triton custom_call missing f64 attr '{name}'"
            ))),
        }
    }

    pub(super) fn custom_call_string<'b>(
        &self,
        attr: Option<&'b CustomCallAttr>,
        name: &str,
    ) -> ConversionResult<&'b str> {
        match attr {
            Some(CustomCallAttr::String(value)) => Ok(value.as_str()),
            _ => Err(ConversionError::new(format!(
                "triton custom_call missing string attr '{name}'"
            ))),
        }
    }
}
