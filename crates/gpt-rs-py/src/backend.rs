//! The backend that `load_model` uses when the call names none.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::RwLock;

/// The backend that `set_backend` chose, or `None` for the build's default.
static CURRENT_BACKEND: RwLock<Option<String>> = RwLock::new(None);

/// Makes `name` the backend of later `load_model` calls that name none.
#[pyfunction]
pub fn set_backend(name: &str) -> PyResult<()> {
    let names = gpt_rs_backends::names();
    if !names.contains(&name) {
        return Err(PyValueError::new_err(format!(
            "unknown backend '{name}'; this build has {}",
            names.join(", ")
        )));
    }
    let mut current = CURRENT_BACKEND
        .write()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to acquire backend lock: {e}")))?;
    *current = Some(name.to_string());
    Ok(())
}

/// The backend of `load_model` calls that name none.
#[pyfunction]
pub fn get_backend() -> String {
    CURRENT_BACKEND
        .read()
        .ok()
        .and_then(|current| current.clone())
        .unwrap_or_else(|| gpt_rs_backends::DEFAULT.to_string())
}

/// The backends of this build.
#[pyfunction]
pub fn list_backends() -> Vec<&'static str> {
    gpt_rs_backends::names()
}
