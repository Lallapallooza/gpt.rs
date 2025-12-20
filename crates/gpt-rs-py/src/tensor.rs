use gpt_rs::backend::registry::BackendHandle;
use gpt_rs::backend::spec::TensorInit;
use gpt_rs::{DType, Shape, Tensor};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Python-exposed tensor wrapper for gpt-rs
/// Stores a type-erased backend handle that works with any backend
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    // Store type-erased handle from the backend
    handle: BackendHandle,
    // Cache metadata for quick access
    shape: Vec<usize>,
    dtype: DType,
}

impl PyTensor {
    /// Create PyTensor from a host Tensor using the current backend
    pub(crate) fn from_host(host: Tensor) -> PyResult<Self> {
        let backend = crate::backend::create_current_backend()?;

        let shape = host.shape().dims().to_vec();
        let dtype = host.dtype();
        let literal = host.to_literal();

        let handle = backend
            .materialize(TensorInit::Literal(literal))
            .map_err(|e| PyValueError::new_err(format!("failed to materialize tensor: {}", e)))?;

        Ok(PyTensor {
            handle,
            shape,
            dtype,
        })
    }

    /// Convert back to host Tensor
    pub(crate) fn to_host(&self) -> PyResult<Tensor> {
        let backend = crate::backend::create_current_backend()?;

        let literal = backend.to_literal(&self.handle).map_err(|e| {
            PyValueError::new_err(format!("failed to read tensor from device: {}", e))
        })?;

        Tensor::from_literal(&literal)
            .map_err(|e| PyValueError::new_err(format!("failed to create host tensor: {}", e)))
    }

    pub(crate) fn handle(&self) -> &BackendHandle {
        &self.handle
    }

    pub(crate) fn dtype_enum(&self) -> DType {
        self.dtype
    }

    pub(crate) fn shape_vec(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn from_backend_handle(
        handle: BackendHandle,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        PyTensor {
            handle,
            shape,
            dtype,
        }
    }
}

#[pymethods]
impl PyTensor {
    /// Create a Tensor from a NumPy array (always copies data per FR11)
    #[staticmethod]
    fn from_numpy(_py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Validate dtype
        let dtype_obj = arr.getattr("dtype")?;
        let dtype_str_py = dtype_obj.str()?;
        let dtype_str = dtype_str_py.extract::<String>()?;

        // Validate that dtype is supported
        match dtype_str.as_str() {
            "float32" | "float64" | "int32" | "int64" => {}
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported dtype: {}. Supported types: float32, float64 (converts to f32), int32, int64 (converts to i32)",
                    dtype_str
                )));
            }
        };

        // Get shape
        let shape_tuple: Bound<'_, PyTuple> = arr.getattr("shape")?.extract()?;
        let shape_vec: Vec<usize> = shape_tuple.extract()?;

        // Create host tensor based on dtype (always copy as per FR11)
        let host_tensor = match dtype_str.as_str() {
            "float32" => {
                let np_arr = arr.downcast::<PyArrayDyn<f32>>()?;
                let data: Vec<f32> = np_arr.readonly().as_array().iter().copied().collect();
                Tensor::from_vec(Shape::new(shape_vec.clone()), data)
            }
            "float64" => {
                // Convert f64 to f32
                let np_arr = arr.downcast::<PyArrayDyn<f64>>()?;
                let data_f32: Vec<f32> = np_arr
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|&x| x as f32)
                    .collect();
                Tensor::from_vec(Shape::new(shape_vec.clone()), data_f32)
            }
            "int32" => {
                let np_arr = arr.downcast::<PyArrayDyn<i32>>()?;
                let data: Vec<i32> = np_arr.readonly().as_array().iter().copied().collect();
                Tensor::from_i32(Shape::new(shape_vec.clone()), data)
            }
            "int64" => {
                // Convert i64 to i32
                let np_arr = arr.downcast::<PyArrayDyn<i64>>()?;
                let data_i32: Vec<i32> = np_arr
                    .readonly()
                    .as_array()
                    .iter()
                    .map(|&x| x as i32)
                    .collect();
                Tensor::from_i32(Shape::new(shape_vec.clone()), data_i32)
            }
            _ => unreachable!(),
        }
        .map_err(|e| PyValueError::new_err(format!("failed to create tensor: {}", e)))?;

        // Store as host tensor - no backend-specific type!
        PyTensor::from_host(host_tensor)
    }

    /// Convert tensor back to NumPy array (always copies)
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let host = self.to_host()?;
        let shape = host.shape().dims();

        match host.dtype() {
            DType::F32 => {
                let data = host.data().to_vec();
                Ok(PyArray::from_vec_bound(py, data).reshape(shape)?.into_any())
            }
            DType::I32 => {
                let data = host.data_i32().to_vec();
                Ok(PyArray::from_vec_bound(py, data).reshape(shape)?.into_any())
            }
            _ => Err(PyValueError::new_err(format!(
                "unsupported dtype for numpy conversion: {:?}",
                host.dtype()
            ))),
        }
    }

    /// Convert from PyTorch tensor (CPU only, always copies)
    #[staticmethod]
    fn from_torch(py: Python<'_>, tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Check if tensor is on CPU
        let device_str_obj = tensor.getattr("device")?.str()?;
        let device_str = device_str_obj.extract::<String>()?;

        if !device_str.contains("cpu") {
            return Err(PyValueError::new_err(
                "only CPU tensors are supported. Use tensor.cpu() first.",
            ));
        }

        // Convert to numpy and use from_numpy
        let numpy_arr = tensor.call_method0("numpy")?;
        Self::from_numpy(py, &numpy_arr)
    }

    /// Convert to PyTorch tensor (CPU only, always copies)
    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // First convert to numpy
        let numpy_arr = self.to_numpy(py)?;

        // Import torch and convert
        let torch = py.import_bound("torch")?;
        torch.call_method1("from_numpy", (numpy_arr,))
    }

    /// Convert to NumPy array (convenience alias for to_numpy)
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_numpy(py)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> String {
        match self.dtype {
            DType::F32 => "f32".to_string(),
            DType::I32 => "i32".to_string(),
            DType::F16 => "f16".to_string(),
            DType::BF16 => "bf16".to_string(),
        }
    }

    #[getter]
    fn backend(&self) -> String {
        crate::backend::get_current_backend_name()
    }
}
