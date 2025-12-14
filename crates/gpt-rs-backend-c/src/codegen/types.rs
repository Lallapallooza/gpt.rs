use gpt_rs::backend::spec::{TensorSpec, ValueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueStorage {
    Input { index: usize },
    Output { index: usize },
    Temp { slot: usize },
    Const,
    Alias,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ValueKey {
    pub(super) value: ValueId,
    pub(super) path: Vec<usize>,
}
impl ValueKey {
    pub(super) fn new(value: ValueId, path: Vec<usize>) -> Self {
        Self { value, path }
    }

    pub(super) fn tensor(value: ValueId) -> Self {
        Self::new(value, Vec::new())
    }
}
#[derive(Debug, Clone)]
pub(super) struct ValueInfo {
    pub(super) storage: ValueStorage,
    pub(super) spec: TensorSpec,
    pub(super) elem_count: usize,
    pub(super) byte_len: usize,
    pub(super) var: String,
    pub(super) const_name: Option<String>,
}
pub(super) struct MatmulCacheEntry {
    pub(super) op_id: usize,
    pub(super) rhs_index: usize,
    pub(super) n: usize,
    pub(super) k: usize,
}
#[derive(Debug, Clone)]
pub(super) struct ResultBinding {
    pub(super) value: ValueId,
    pub(super) path: Vec<usize>,
}
