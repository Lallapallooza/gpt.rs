use gpt_rs::backend::spec::{Dimension, TensorSpec};
use gpt_rs::backend::{rewriter::ProgramRewriter, spec::ValueType};

pub(super) fn tensor_spec_of(
    rewriter: &ProgramRewriter<'_>,
    value: gpt_rs::backend::spec::ValueId,
) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}

pub(super) fn static_dims(spec: &TensorSpec) -> Option<Vec<usize>> {
    spec.shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => Some(*v),
            Dimension::Dynamic(_) => None,
        })
        .collect()
}
