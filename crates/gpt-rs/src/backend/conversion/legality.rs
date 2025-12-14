use std::collections::HashSet;

use crate::backend::conversion::{ConversionDiagnostic, ConversionStage};
use crate::backend::spec::{DType, Operand, Operation, Program, ValueType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    Constant,
    ElementwiseUnary,
    ElementwiseBinary,
    DotGeneral,
    Reduce,
    ArgMax,
    Compare,
    Select,
    Cast,
    StopGradient,
    Reshape,
    Transpose,
    BroadcastTo,
    Slice,
    Concat,
    Pad,
    Tile,
    Iota,
    Take,
    Gather,
    ScatterAdd,
    ScatterReduce,
    DynamicSlice,
    DynamicUpdateSlice,
    Cond,
    While,
    Scan,
    ExtractPatches,
    ReduceWindow,
    RngUniform,
    RngNormal,
    TopK,
    SegmentReduce,
    Quantize,
    Dequantize,
    Requantize,
    CustomCall,
}

impl OperationKind {
    pub fn from_op(op: &Operation) -> Self {
        match op {
            Operation::Constant(_) => OperationKind::Constant,
            Operation::ElementwiseUnary(_) => OperationKind::ElementwiseUnary,
            Operation::ElementwiseBinary(_) => OperationKind::ElementwiseBinary,
            Operation::DotGeneral(_) => OperationKind::DotGeneral,
            Operation::Reduce(_) => OperationKind::Reduce,
            Operation::ArgMax(_) => OperationKind::ArgMax,
            Operation::Compare(_) => OperationKind::Compare,
            Operation::Select => OperationKind::Select,
            Operation::Cast(_) => OperationKind::Cast,
            Operation::StopGradient => OperationKind::StopGradient,
            Operation::Reshape(_) => OperationKind::Reshape,
            Operation::Transpose(_) => OperationKind::Transpose,
            Operation::BroadcastTo(_) => OperationKind::BroadcastTo,
            Operation::Slice(_) => OperationKind::Slice,
            Operation::Concat(_) => OperationKind::Concat,
            Operation::Pad(_) => OperationKind::Pad,
            Operation::Tile(_) => OperationKind::Tile,
            Operation::Iota(_) => OperationKind::Iota,
            Operation::Take => OperationKind::Take,
            Operation::Gather(_) => OperationKind::Gather,
            Operation::ScatterAdd(_) => OperationKind::ScatterAdd,
            Operation::ScatterReduce(_) => OperationKind::ScatterReduce,
            Operation::DynamicSlice(_) => OperationKind::DynamicSlice,
            Operation::DynamicUpdateSlice(_) => OperationKind::DynamicUpdateSlice,
            Operation::Cond(_) => OperationKind::Cond,
            Operation::While(_) => OperationKind::While,
            Operation::Scan(_) => OperationKind::Scan,
            Operation::ExtractPatches(_) => OperationKind::ExtractPatches,
            Operation::ReduceWindow(_) => OperationKind::ReduceWindow,
            Operation::RngUniform(_) => OperationKind::RngUniform,
            Operation::RngNormal(_) => OperationKind::RngNormal,
            Operation::TopK(_) => OperationKind::TopK,
            Operation::SegmentReduce(_) => OperationKind::SegmentReduce,
            Operation::Quantize(_) => OperationKind::Quantize,
            Operation::Dequantize(_) => OperationKind::Dequantize,
            Operation::Requantize(_) => OperationKind::Requantize,
            Operation::CustomCall(_) => OperationKind::CustomCall,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LegalitySpec {
    pub allowed_ops: Option<HashSet<OperationKind>>,
    pub allow_dynamic_dims: bool,
    pub allowed_dtypes: Option<HashSet<DType>>,
}

impl Default for LegalitySpec {
    fn default() -> Self {
        Self {
            allowed_ops: None,
            allow_dynamic_dims: true,
            allowed_dtypes: None,
        }
    }
}

impl LegalitySpec {
    pub fn allow_ops(mut self, ops: impl IntoIterator<Item = OperationKind>) -> Self {
        self.allowed_ops = Some(ops.into_iter().collect());
        self
    }

    pub fn allow_dtypes(mut self, dtypes: impl IntoIterator<Item = DType>) -> Self {
        self.allowed_dtypes = Some(dtypes.into_iter().collect());
        self
    }

    pub fn with_dynamic_dims(mut self, allowed: bool) -> Self {
        self.allow_dynamic_dims = allowed;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct LegalityReport {
    pub diagnostics: Vec<ConversionDiagnostic>,
}

impl LegalityReport {
    pub fn is_ok(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn check_program_legality(
    program: &Program,
    spec: &LegalitySpec,
) -> Result<(), LegalityReport> {
    let mut report = LegalityReport::default();

    for function in &program.functions {
        for ty in function.parameters.iter().chain(function.results.iter()) {
            check_value_type(ty, spec, &mut report, Some(function.name.clone()), None);
        }
        for (idx, inst) in function.body.iter().enumerate() {
            let kind = OperationKind::from_op(&inst.op);
            if let Some(allowed) = &spec.allowed_ops {
                if !allowed.contains(&kind) {
                    report.diagnostics.push(ConversionDiagnostic::new(
                        ConversionStage::Legalize,
                        Some(function.name.clone()),
                        Some(idx),
                        format!("operation {:?} is not allowed", kind),
                    ));
                }
            }
            check_value_type(
                &inst.output,
                spec,
                &mut report,
                Some(function.name.clone()),
                Some(idx),
            );
            for operand in &inst.operands {
                if let Operand::Literal(literal) = operand {
                    check_value_type(
                        &ValueType::Tensor(literal.spec.clone()),
                        spec,
                        &mut report,
                        Some(function.name.clone()),
                        Some(idx),
                    );
                }
            }
        }
    }

    for region in &program.regions {
        for ty in region.parameters.iter().chain(region.results.iter()) {
            check_value_type(ty, spec, &mut report, None, None);
        }
        for (idx, inst) in region.body.iter().enumerate() {
            let kind = OperationKind::from_op(&inst.op);
            if let Some(allowed) = &spec.allowed_ops {
                if !allowed.contains(&kind) {
                    report.diagnostics.push(ConversionDiagnostic::new(
                        ConversionStage::Legalize,
                        None,
                        Some(idx),
                        format!("operation {:?} is not allowed", kind),
                    ));
                }
            }
            check_value_type(&inst.output, spec, &mut report, None, Some(idx));
        }
    }

    if report.is_ok() {
        Ok(())
    } else {
        Err(report)
    }
}

fn check_value_type(
    ty: &ValueType,
    spec: &LegalitySpec,
    report: &mut LegalityReport,
    function: Option<String>,
    inst_index: Option<usize>,
) {
    match ty {
        ValueType::Tensor(tensor) => {
            if !spec.allow_dynamic_dims && tensor.shape.static_dims().is_none() {
                report.diagnostics.push(ConversionDiagnostic::new(
                    ConversionStage::Legalize,
                    function.clone(),
                    inst_index,
                    "dynamic dimensions are not allowed",
                ));
            }
            if let Some(allowed) = &spec.allowed_dtypes {
                if !allowed.contains(&tensor.dtype) {
                    report.diagnostics.push(ConversionDiagnostic::new(
                        ConversionStage::Legalize,
                        function,
                        inst_index,
                        format!("dtype {:?} is not allowed", tensor.dtype),
                    ));
                }
            }
        }
        ValueType::Tuple(values) => {
            for value in values {
                check_value_type(value, spec, report, function.clone(), inst_index);
            }
        }
    }
}
