use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use gpt_rs::backend::spec::{BackendError, BackendResult};
use gpt_rs::profiling;

use crate::device::{CudaDriver, CudaFunction};
use crate::kernels::KernelSpec;

use super::TritonExecutor;

thread_local! {
    static EXECUTION_KERNEL_STACK: RefCell<Vec<HashMap<String, Arc<LoadedKernel>>>> =
        const { RefCell::new(Vec::new()) };
}

pub(super) struct LoadedKernel {
    #[allow(dead_code)]
    pub(super) fingerprint: u64,
    pub(super) function: CudaFunction,
    pub(super) profile_signature: Option<u32>,
}

pub(super) struct ExecutionKernelGuard;

impl ExecutionKernelGuard {
    pub(super) fn push(kernels: HashMap<String, Arc<LoadedKernel>>) -> Self {
        EXECUTION_KERNEL_STACK.with(|stack| stack.borrow_mut().push(kernels));
        Self
    }
}

impl Drop for ExecutionKernelGuard {
    fn drop(&mut self) {
        EXECUTION_KERNEL_STACK.with(|stack| {
            let _ = stack.borrow_mut().pop();
        });
    }
}

impl TritonExecutor {
    pub(super) fn load_kernel(
        &self,
        driver: &Arc<CudaDriver>,
        spec: &KernelSpec,
    ) -> BackendResult<Arc<LoadedKernel>> {
        if let Some(found) = EXECUTION_KERNEL_STACK.with(|stack| {
            stack
                .borrow()
                .last()
                .and_then(|kernels| kernels.get(spec.id.as_str()).cloned())
        }) {
            return Ok(found);
        }

        let compiled = self.compiler.compile(spec)?;
        if let Some(found) = self
            .loaded_kernels
            .lock()
            .map_err(|_| BackendError::execution("triton loaded kernel cache mutex poisoned"))?
            .get(&compiled.fingerprint)
            .cloned()
        {
            profiling::cache_event("triton_backend.module_hit_mem");
            return Ok(found);
        }
        profiling::cache_event("triton_backend.module_miss_mem");

        let function = {
            let _load_scope = profiling::compile_scope("triton_backend.dlopen");
            let module = driver.load_ptx_module(compiled.ptx.as_ref())?;
            driver.get_function(&module, compiled.symbol.as_ref())?
        };
        let loaded = Arc::new(LoadedKernel {
            fingerprint: compiled.fingerprint,
            function,
            profile_signature: profiling::signature_id(&spec.id),
        });
        self.loaded_kernels
            .lock()
            .map_err(|_| BackendError::execution("triton loaded kernel cache mutex poisoned"))?
            .insert(compiled.fingerprint, Arc::clone(&loaded));
        Ok(loaded)
    }
}
