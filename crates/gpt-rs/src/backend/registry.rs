//! Runtime backend registry for dynamic backend selection.
//!
//! This module enables registering and selecting backends by name at runtime,
//! avoiding hardcoded backend types throughout the codebase. Backends can be
//! registered from any crate (including external ones) using the global registry.

use super::spec::{BackendError, BackendResult, PortableBackend, TensorInit, TensorLiteral};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Type-erased backend handle that can be downcast to concrete backend types.
pub type BackendHandle = Box<dyn Any + Send + Sync>;

/// Factory function that creates a new backend instance.
pub type BackendConstructor = Box<dyn Fn() -> Box<dyn ErasedBackend> + Send + Sync>;

/// Type-erased backend trait that wraps PortableBackend without generic parameters.
///
/// This enables dynamic dispatch and runtime backend selection. Each method works
/// with type-erased handles (Box<dyn Any>) that get downcast internally.
pub trait ErasedBackend: Send + Sync {
    /// Returns a human-readable backend identifier (e.g., "cpu", "faer", "cuda").
    fn backend_name(&self) -> &str;

    /// Materializes a tensor handle from host initialization data.
    fn materialize(&self, init: TensorInit) -> BackendResult<BackendHandle>;

    /// Reads back a tensor handle into a dense literal (debug/development only).
    fn to_literal(&self, handle: &BackendHandle) -> BackendResult<TensorLiteral>;

    /// Executes a single PTIR instruction with type-erased handles.
    fn execute_instruction(
        &self,
        instruction: &super::spec::Instruction,
        inputs: &[BackendHandle],
    ) -> BackendResult<Vec<BackendHandle>>;

    /// Executes an entire PTIR program starting from the entry function.
    fn run_program(
        &self,
        program: &super::spec::Program,
        entry_inputs: &[BackendHandle],
    ) -> BackendResult<Vec<BackendHandle>>;

    /// Clone this backend as a trait object.
    fn clone_backend(&self) -> Box<dyn ErasedBackend>;

    /// Downcast to Any for type recovery when needed.
    fn as_any(&self) -> &dyn Any;
}

/// Wrapper that implements ErasedBackend for any concrete PortableBackend.
struct BackendWrapper<B: PortableBackend> {
    inner: Arc<B>,
}

impl<B: PortableBackend> BackendWrapper<B> {
    fn new(backend: B) -> Self {
        Self {
            inner: Arc::new(backend),
        }
    }

    /// Get reference to the inner backend.
    pub fn backend(&self) -> &Arc<B> {
        &self.inner
    }
}

impl<B: PortableBackend + 'static> ErasedBackend for BackendWrapper<B> {
    fn backend_name(&self) -> &str {
        self.inner.backend_name()
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<BackendHandle> {
        let handle = self.inner.materialize(init)?;
        Ok(Box::new(handle) as BackendHandle)
    }

    fn to_literal(&self, handle: &BackendHandle) -> BackendResult<TensorLiteral> {
        let typed_handle = handle.downcast_ref::<B::TensorHandle>().ok_or_else(|| {
            BackendError::execution(format!(
                "handle type mismatch for backend {}",
                self.backend_name()
            ))
        })?;
        self.inner.to_literal(typed_handle)
    }

    fn execute_instruction(
        &self,
        instruction: &super::spec::Instruction,
        inputs: &[BackendHandle],
    ) -> BackendResult<Vec<BackendHandle>> {
        // Downcast all input handles
        let mut typed_inputs = Vec::with_capacity(inputs.len());
        for handle in inputs {
            let typed = handle.downcast_ref::<B::TensorHandle>().ok_or_else(|| {
                BackendError::execution(format!(
                    "input handle type mismatch for backend {}",
                    self.backend_name()
                ))
            })?;
            typed_inputs.push(typed.clone());
        }

        // Execute with typed handles
        let outputs = self.inner.execute_instruction(instruction, &typed_inputs)?;

        // Type-erase outputs
        Ok(outputs
            .into_iter()
            .map(|h| Box::new(h) as BackendHandle)
            .collect())
    }

    fn run_program(
        &self,
        program: &super::spec::Program,
        entry_inputs: &[BackendHandle],
    ) -> BackendResult<Vec<BackendHandle>> {
        // Downcast all input handles
        let mut typed_inputs = Vec::with_capacity(entry_inputs.len());
        for handle in entry_inputs {
            let typed = handle.downcast_ref::<B::TensorHandle>().ok_or_else(|| {
                BackendError::execution(format!(
                    "input handle type mismatch for backend {}",
                    self.backend_name()
                ))
            })?;
            typed_inputs.push(typed.clone());
        }

        // Execute program
        let outputs = self.inner.run_program(program, &typed_inputs)?;

        // Type-erase outputs
        Ok(outputs
            .into_iter()
            .map(|h| Box::new(h) as BackendHandle)
            .collect())
    }

    fn clone_backend(&self) -> Box<dyn ErasedBackend> {
        Box::new(BackendWrapper {
            inner: Arc::clone(&self.inner),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Global backend registry mapping backend names to constructors.
struct BackendRegistry {
    backends: RwLock<HashMap<String, BackendConstructor>>,
}

impl BackendRegistry {
    fn new() -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
        }
    }

    fn register(&self, name: String, constructor: BackendConstructor) {
        self.backends.write().unwrap().insert(name, constructor);
    }

    fn create(&self, name: &str) -> Option<Box<dyn ErasedBackend>> {
        let registry = self.backends.read().unwrap();
        let constructor = registry.get(name)?;
        Some(constructor())
    }

    fn list_backends(&self) -> Vec<String> {
        self.backends.read().unwrap().keys().cloned().collect()
    }

    fn has_backend(&self, name: &str) -> bool {
        self.backends.read().unwrap().contains_key(name)
    }
}

// Global registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<BackendRegistry> = std::sync::OnceLock::new();

fn global_registry() -> &'static BackendRegistry {
    GLOBAL_REGISTRY.get_or_init(BackendRegistry::new)
}

/// Register a backend by name with a constructor function.
///
/// The constructor will be called each time the backend is requested via `create_backend()`.
/// External crates can register their backends by calling this from a module initializer.
///
/// # Example
/// ```ignore
/// use gpt_rs::backend::registry::register_backend;
///
/// // In your backend crate's lib.rs or initialization code:
/// pub fn register() {
///     register_backend("my_backend", || {
///         Box::new(MyBackendWrapper::new(MyBackend::create()))
///     });
/// }
/// ```
pub fn register_backend<F>(name: impl Into<String>, constructor: F)
where
    F: Fn() -> Box<dyn ErasedBackend> + Send + Sync + 'static,
{
    global_registry().register(name.into(), Box::new(constructor));
}

/// Register a concrete PortableBackend implementation.
///
/// This is a convenience wrapper that handles the BackendWrapper boilerplate.
pub fn register_portable_backend<B, F>(name: impl Into<String>, constructor: F)
where
    B: PortableBackend + 'static,
    F: Fn() -> B + Send + Sync + 'static,
{
    register_backend(name, move || Box::new(BackendWrapper::new(constructor())));
}

/// Create a backend instance by name.
///
/// Returns `None` if no backend with the given name has been registered.
pub fn create_backend(name: &str) -> Option<Box<dyn ErasedBackend>> {
    global_registry().create(name)
}

/// List all registered backend names.
pub fn list_backends() -> Vec<String> {
    global_registry().list_backends()
}

/// Check if a backend with the given name is registered.
pub fn has_backend(name: &str) -> bool {
    global_registry().has_backend(name)
}

/// Helper to access the typed backend from a BackendWrapper.
///
/// This is useful when you need to work with the original PortableBackend type,
/// for example when creating DeviceTensor<B>.
pub fn get_typed_backend<B: PortableBackend + 'static>(
    backend: &dyn ErasedBackend,
) -> Option<Arc<B>> {
    backend
        .as_any()
        .downcast_ref::<BackendWrapper<B>>()
        .map(|wrapper| Arc::clone(wrapper.backend()))
}
