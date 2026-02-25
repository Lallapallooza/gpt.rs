use gpt_rs::backend::index::InstId;
use gpt_rs::backend::rewriter::ProgramRewriter;
use gpt_rs::backend::spec::{TensorSpec, ValueId, ValueType};

pub(super) fn single_user(rewriter: &ProgramRewriter, value: ValueId) -> Option<InstId> {
    let users = rewriter.users_of(value);
    if users.len() == 1 {
        Some(users[0])
    } else {
        None
    }
}

pub(super) fn tensor_spec_of(rewriter: &ProgramRewriter, value: ValueId) -> Option<TensorSpec> {
    match rewriter.type_of(value) {
        Some(ValueType::Tensor(spec)) => Some(spec.clone()),
        _ => None,
    }
}
