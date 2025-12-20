use crate::tensor::PyTensor;
use gpt_rs::backend::registry;
use gpt_rs::ops::functional;
use gpt_rs::DeviceTensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Execute an operation by dispatching to the correct backend based on the current backend name
macro_rules! dispatch_to_backend {
    ($backend:expr, $inputs:expr, $op:ident) => {{
        let backend_name = crate::backend::get_current_backend_name();

        match backend_name.as_str() {
            "cpu" | "cpu-portable" => {
                let typed = registry::get_typed_backend::<
                    gpt_rs_backend_ref_cpu::GenericCpuBackend<
                        gpt_rs_backend_ref_cpu::NoopInterceptor,
                    >,
                >($backend.as_ref())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "backend type mismatch: expected cpu-portable, got {}",
                        backend_name
                    ))
                })?;
                $op(typed, $inputs)
            }
            #[cfg(feature = "conversion-c")]
            "c" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_c::CBackend>($backend.as_ref())
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "backend type mismatch: expected c, got {}",
                                backend_name
                            ))
                        })?;
                $op(typed, $inputs)
            }
            #[cfg(feature = "faer")]
            "faer" | "faer-portable" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                        $backend.as_ref(),
                    )
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "backend type mismatch: expected faer-portable, got {}",
                            backend_name
                        ))
                    })?;
                $op(typed, $inputs)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "backend '{}' is not supported in Python bindings yet",
                    backend_name
                )));
            }
        }
    }};
}

/// Helper to convert PyTensor inputs to DeviceTensor<B> and execute an operation
fn with_device_tensors<B, F>(backend: Arc<B>, inputs: &[&PyTensor], op: F) -> PyResult<PyTensor>
where
    B: gpt_rs::backend::spec::PortableBackend + 'static,
    F: FnOnce(Arc<B>, Vec<DeviceTensor<B>>) -> PyResult<DeviceTensor<B>>,
{
    // Convert PyTensor -> host Tensor -> DeviceTensor<B>
    let mut device_tensors = Vec::with_capacity(inputs.len());
    for input in inputs {
        let host = input.to_host()?;
        let device = DeviceTensor::from_host(Arc::clone(&backend), host)
            .map_err(|e| PyValueError::new_err(format!("failed to create device tensor: {}", e)))?;
        device_tensors.push(device);
    }

    // Execute operation
    let result = op(backend, device_tensors)?;

    // Convert back to PyTensor
    let result_host = result
        .to_host()
        .map_err(|e| PyValueError::new_err(format!("failed to read result: {}", e)))?;
    PyTensor::from_host(result_host)
}

/// Softmax activation along the last dimension
#[pyfunction]
pub fn softmax_last_dim(tensor: &PyTensor) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let inputs = vec![tensor];

    dispatch_to_backend!(backend, &inputs, execute_softmax_last_dim)
}

fn execute_softmax_last_dim<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::softmax_last_dim(&*backend, &tensors[0])
            .map_err(|e| PyValueError::new_err(format!("softmax_last_dim failed: {}", e)))?;
        Ok(result)
    })
}

/// GELU activation function
#[pyfunction]
pub fn gelu(tensor: &PyTensor) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let inputs = vec![tensor];

    dispatch_to_backend!(backend, &inputs, execute_gelu)
}

fn execute_gelu<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::gelu(&*backend, &tensors[0])
            .map_err(|e| PyValueError::new_err(format!("gelu failed: {}", e)))?;
        Ok(result)
    })
}

/// Layer normalization
#[pyfunction]
#[pyo3(signature = (tensor, weight, bias, eps=1e-5))]
pub fn layer_norm(
    tensor: &PyTensor,
    weight: &PyTensor,
    bias: &PyTensor,
    eps: f32,
) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let backend_name = crate::backend::get_current_backend_name();

    match backend_name.as_str() {
        "cpu" | "cpu-portable" => {
            let typed = registry::get_typed_backend::<
                gpt_rs_backend_ref_cpu::GenericCpuBackend<gpt_rs_backend_ref_cpu::NoopInterceptor>,
            >(backend.as_ref())
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "backend type mismatch: expected cpu-portable, got {}",
                    backend_name
                ))
            })?;
            execute_layer_norm(typed, &[tensor, weight, bias], eps)
        }
        #[cfg(feature = "conversion-c")]
        "c" => {
            let typed = registry::get_typed_backend::<gpt_rs_backend_c::CBackend>(backend.as_ref())
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "backend type mismatch: expected c, got {}",
                        backend_name
                    ))
                })?;
            execute_layer_norm(typed, &[tensor, weight, bias], eps)
        }
        #[cfg(feature = "faer")]
        "faer" | "faer-portable" => {
            let typed = registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                backend.as_ref(),
            )
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "backend type mismatch: expected faer-portable, got {}",
                    backend_name
                ))
            })?;
            execute_layer_norm(typed, &[tensor, weight, bias], eps)
        }
        _ => Err(PyValueError::new_err(format!(
            "backend '{}' is not supported in Python bindings yet",
            backend_name
        ))),
    }
}

fn execute_layer_norm<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
    eps: f32,
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::layer_norm(&*backend, &tensors[0], &tensors[1], &tensors[2], eps)
            .map_err(|e| PyValueError::new_err(format!("layer_norm failed: {}", e)))?;
        Ok(result.output)
    })
}

/// Matrix multiplication
#[pyfunction]
pub fn matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let inputs = vec![a, b];

    dispatch_to_backend!(backend, &inputs, execute_matmul)
}

fn execute_matmul<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::matmul(&*backend, &tensors[0], &tensors[1])
            .map_err(|e| PyValueError::new_err(format!("matmul failed: {}", e)))?;
        Ok(result)
    })
}

/// Add bias to tensor
#[pyfunction]
pub fn add_bias(tensor: &PyTensor, bias: &PyTensor) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let inputs = vec![tensor, bias];

    dispatch_to_backend!(backend, &inputs, execute_add_bias)
}

fn execute_add_bias<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::add_bias(&*backend, &tensors[0], &tensors[1])
            .map_err(|e| PyValueError::new_err(format!("add_bias failed: {}", e)))?;
        Ok(result)
    })
}

/// Embedding lookup
#[pyfunction]
pub fn embedding_lookup(table: &PyTensor, indices: &PyTensor) -> PyResult<PyTensor> {
    let backend = crate::backend::create_current_backend()?;
    let inputs = vec![table, indices];

    dispatch_to_backend!(backend, &inputs, execute_embedding_lookup)
}

fn execute_embedding_lookup<B: gpt_rs::backend::spec::PortableBackend + 'static>(
    backend: Arc<B>,
    inputs: &[&PyTensor],
) -> PyResult<PyTensor> {
    with_device_tensors(backend.clone(), inputs, |backend, tensors| {
        let result = functional::embedding_lookup(&*backend, &tensors[0], &tensors[1])
            .map_err(|e| PyValueError::new_err(format!("embedding_lookup failed: {}", e)))?;
        Ok(result)
    })
}
