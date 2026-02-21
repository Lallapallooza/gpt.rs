use std::sync::Arc;

use gpt_rs::backend::param_resolver::ParamResolver;
use gpt_rs::backend::spec::{
    BackendResult, Instruction, PortableBackend, Program, TensorInit, TensorLiteral,
};
use gpt_rs_backend_ref_cpu::{CpuPortableBackend, CpuTensor};

/// Bootstrap Triton backend.
///
/// This M1 implementation intentionally delegates execution to the reference
/// CPU portable backend so the crate can be integrated end-to-end while the
/// CUDA/Triton runtime layers are developed in later milestones.
#[derive(Default)]
pub struct TritonBackend {
    inner: CpuPortableBackend,
}

impl TritonBackend {
    pub fn new() -> Self {
        Self {
            inner: CpuPortableBackend::new(),
        }
    }
}

impl PortableBackend for TritonBackend {
    type TensorHandle = CpuTensor;

    fn backend_name(&self) -> &str {
        "triton"
    }

    fn param_resolver(&self) -> Option<Arc<dyn ParamResolver<Handle = Self::TensorHandle>>> {
        self.inner.param_resolver()
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle> {
        self.inner.materialize(init)
    }

    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        self.inner.to_literal(tensor)
    }

    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        self.inner.execute_instruction(instruction, inputs)
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        self.inner.run_program(program, entry_inputs)
    }
}

/// Register the Triton backend with the global backend registry.
pub fn register_triton_backend() {
    gpt_rs::backend::registry::register_portable_backend("triton", TritonBackend::new);
}

#[gpt_rs::linkme::distributed_slice(gpt_rs::backend::registry::BACKEND_REGISTRARS)]
static REGISTER_TRITON_BACKEND: fn() = register_triton_backend;
