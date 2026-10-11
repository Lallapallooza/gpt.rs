use gpt_rs_ir::dialects::tensor::{ExtractPatches, TensorOp};
use gpt_rs_ir::module::Value;
use gpt_rs_ir::types::TensorType;

/// Where the generated function keeps a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueStorage {
    /// Input `index` of the function.
    Input { index: usize },
    /// Output `index` of the function.
    Output { index: usize },
    /// The temporary buffer `slot`.
    Temp { slot: usize },
    /// A constant of the function.
    Const,
    /// The buffer of another value, which a `reshape` reads unchanged.
    Alias,
}

#[derive(Debug, Clone)]
pub(super) struct ValueInfo {
    pub(super) storage: ValueStorage,
    pub(super) ty: TensorType,
    pub(super) elem_count: usize,
    pub(super) byte_len: usize,
    pub(super) var: String,
    pub(super) const_name: Option<String>,
}

/// What codegen emits for the instructions of a function: one op, or a sequence of ops that one
/// kernel computes.
pub(super) struct Instruction<'m> {
    pub(super) kind: Kind<'m>,
    /// The values the instruction reads.
    pub(super) operands: Vec<Value>,
    /// The values it gives.
    pub(super) results: Vec<Value>,
    /// Its profiler label.
    pub(super) label: &'static str,
}

impl Instruction<'_> {
    /// The value of an instruction that gives one.
    pub(super) fn result(&self) -> Value {
        self.results[0]
    }
}

pub(super) enum Kind<'m> {
    Op(&'m TensorOp),
    /// An NHWC f32 convolution with these patches. Operands `[x, w]` or `[x, w, bias]`, where `w`
    /// is `[KH * KW * C_in, C_out]` or in the GEMM's panels `[C_out / PANEL_N, K, PANEL_N]`.
    Conv2d(ExtractPatches),
    /// An NHWC f32 convolution in `groups` groups with these patches. Operands `[x, w]` or
    /// `[x, w, bias]`, where `w` is `[KH, KW, C_in / groups, groups, C_out / groups]`.
    GroupedConv2d {
        patches: ExtractPatches,
        groups: usize,
    },
    /// A projection `x [M, K] . w [N, K]` that reads `w` as stored. Operands `[x, w]` or
    /// `[x, w, s]`, where `s` is the one-element scale of an FP8 `w`.
    Linear,
    /// A chain of f32 elementwise ops in one loop, whose operands are the chain's inputs.
    Elementwise(Elementwise),
}

pub(super) struct Elementwise {
    /// The ops of the chain, each after the nodes it reads.
    pub(super) nodes: Vec<Node>,
    /// One per operand.
    pub(super) inputs: Vec<ElementwiseInput>,
    /// Whether the kernel rounds its result to bf16 as it stores it.
    pub(super) store_bf16: bool,
}

/// How the kernel reads an input: broadcast into the result's shape, or, with `starts`, at those
/// offsets inside a same-rank source, which is the input of a `slice` that the kernel covers.
pub(super) struct ElementwiseInput {
    pub(super) starts: Option<Vec<usize>>,
}

pub(super) struct Node {
    /// An elementwise op that has a C expression: of one operand, or of two with `rhs`.
    pub(super) op: TensorOp,
    pub(super) lhs: NodeRef,
    pub(super) rhs: Option<NodeRef>,
}

#[derive(Clone, Copy)]
pub(super) enum NodeRef {
    Input(usize),
    Node(usize),
}
