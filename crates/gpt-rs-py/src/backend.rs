use gpt_rs::backend::registry;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::RwLock;

// Global backend state - stores the name of the currently selected backend
static CURRENT_BACKEND_NAME: RwLock<String> = RwLock::new(String::new());

/// Initialize the backend system and ensure backends are registered
fn ensure_backends_registered() {
    // Ensure built-in backends are registered
    gpt_rs_backend_ref_cpu::register_cpu_backend();

    #[cfg(feature = "faer")]
    gpt_rs_backend_faer::register_faer_backend();

    #[cfg(feature = "conversion-c")]
    gpt_rs_backend_c::register_c_backend();

    #[cfg(feature = "triton")]
    gpt_rs_backend_triton::register_triton_backend();
}

/// Get the default backend name
fn default_backend_name() -> &'static str {
    #[cfg(feature = "faer")]
    {
        "faer"
    }
    #[cfg(not(feature = "faer"))]
    {
        "cpu"
    }
}

/// Set the global backend for tensor operations
/// WARNING: This invalidates all existing Tensor and Layer instances per FR14
#[pyfunction]
pub fn set_backend(name: &str) -> PyResult<()> {
    ensure_backends_registered();

    // Check if backend exists
    if !registry::has_backend(name) {
        let available = registry::list_backends();
        return Err(PyValueError::new_err(format!(
            "unknown backend: '{}'. Available backends: {}",
            name,
            available.join(", ")
        )));
    }

    // Update global state
    let mut state = CURRENT_BACKEND_NAME
        .write()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to acquire backend lock: {}", e)))?;
    *state = name.to_string();

    Ok(())
}

/// Get the current backend name
#[pyfunction]
pub fn get_backend() -> PyResult<String> {
    ensure_backends_registered();

    let state = CURRENT_BACKEND_NAME
        .read()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to acquire backend lock: {}", e)))?;

    // Return default if not set
    if state.is_empty() {
        Ok(default_backend_name().to_string())
    } else {
        Ok(state.clone())
    }
}

/// List all available backends
#[pyfunction]
pub fn list_backends() -> PyResult<Vec<String>> {
    ensure_backends_registered();
    Ok(registry::list_backends())
}

/// Internal helper to get current backend name (for Rust code)
pub(crate) fn get_current_backend_name() -> String {
    CURRENT_BACKEND_NAME
        .read()
        .map(|state| {
            if state.is_empty() {
                default_backend_name().to_string()
            } else {
                state.clone()
            }
        })
        .unwrap_or_else(|_| default_backend_name().to_string())
}

/// Create the current backend instance
pub(crate) fn create_current_backend() -> PyResult<Box<dyn registry::ErasedBackend>> {
    ensure_backends_registered();

    let name = get_current_backend_name();
    registry::create_backend(&name)
        .ok_or_else(|| PyValueError::new_err(format!("backend '{}' not available", name)))
}
