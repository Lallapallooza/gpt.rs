pub mod cpu;

pub use cpu::{
    CpuKernelInterceptor, CpuPortableBackend, CpuTensor, GenericCpuBackend, NoopInterceptor,
    TensorData,
};

/// Register the CPU backend with the global backend registry.
///
/// This function is called automatically via a static initializer, but can also
/// be called manually to ensure the backend is registered.
/// The backend is registered under both "cpu" and "cpu-portable" names.
pub fn register_cpu_backend() {
    let constructor = || GenericCpuBackend::with_interceptor(NoopInterceptor);

    // Register under both names for convenience
    gpt_rs::backend::registry::register_portable_backend("cpu", constructor);
    gpt_rs::backend::registry::register_portable_backend("cpu-portable", constructor);
}

// Auto-register on library load
#[gpt_rs::linkme::distributed_slice(gpt_rs::backend::registry::BACKEND_REGISTRARS)]
static REGISTER_CPU_BACKEND: fn() = register_cpu_backend;
