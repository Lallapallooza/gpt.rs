use crate::functional;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Embedding layer that maps token IDs to dense vectors
#[pyclass(name = "Embedding")]
pub struct PyEmbedding {
    weight: PyObject,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    fn new(weight: PyObject) -> PyResult<Self> {
        Ok(PyEmbedding { weight })
    }

    fn forward(&self, py: Python<'_>, indices: &PyTensor) -> PyResult<PyTensor> {
        let weight_bound = self.weight.bind(py).downcast::<PyTensor>()?;
        let weight = weight_bound.borrow();
        functional::embedding_lookup(&weight, indices)
    }
}

/// Linear layer (fully connected) with optional bias
#[pyclass(name = "Linear")]
pub struct PyLinear {
    weight: PyObject,
    bias: Option<PyObject>,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (weight, bias=None))]
    fn new(weight: PyObject, bias: Option<PyObject>) -> PyResult<Self> {
        Ok(PyLinear { weight, bias })
    }

    fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let weight_bound = self.weight.bind(py).downcast::<PyTensor>()?;
        let weight = weight_bound.borrow();
        let mut output = functional::matmul(x, &weight)?;

        if let Some(ref bias_obj) = self.bias {
            let bias_bound = bias_obj.bind(py).downcast::<PyTensor>()?;
            let bias = bias_bound.borrow();
            output = functional::add_bias(&output, &bias)?;
        }

        Ok(output)
    }
}

/// Layer normalization
#[pyclass(name = "LayerNorm")]
pub struct PyLayerNorm {
    weight: PyObject,
    bias: PyObject,
    eps: f32,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (weight, bias, eps=1e-5))]
    fn new(weight: PyObject, bias: PyObject, eps: f32) -> PyResult<Self> {
        Ok(PyLayerNorm { weight, bias, eps })
    }

    fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        let weight_bound = self.weight.bind(py).downcast::<PyTensor>()?;
        let weight = weight_bound.borrow();
        let bias_bound = self.bias.bind(py).downcast::<PyTensor>()?;
        let bias = bias_bound.borrow();
        functional::layer_norm(x, &weight, &bias, self.eps)
    }
}

/// Feed-forward network with GELU activation
#[pyclass(name = "FeedForward")]
pub struct PyFeedForward {
    w_in: PyObject,
    w_out: PyObject,
    b_in: Option<PyObject>,
    b_out: Option<PyObject>,
}

#[pymethods]
impl PyFeedForward {
    #[new]
    #[pyo3(signature = (w_in, w_out, b_in=None, b_out=None))]
    fn new(
        w_in: PyObject,
        w_out: PyObject,
        b_in: Option<PyObject>,
        b_out: Option<PyObject>,
    ) -> PyResult<Self> {
        Ok(PyFeedForward {
            w_in,
            w_out,
            b_in,
            b_out,
        })
    }

    fn forward(&self, py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
        // First linear: x @ w_in + b_in
        let w_in_bound = self.w_in.bind(py).downcast::<PyTensor>()?;
        let w_in = w_in_bound.borrow();
        let mut hidden = functional::matmul(x, &w_in)?;

        if let Some(ref b_in_obj) = self.b_in {
            let b_in_bound = b_in_obj.bind(py).downcast::<PyTensor>()?;
            let b_in = b_in_bound.borrow();
            hidden = functional::add_bias(&hidden, &b_in)?;
        }

        // GELU activation
        hidden = functional::gelu(&hidden)?;

        // Second linear: hidden @ w_out + b_out
        let w_out_bound = self.w_out.bind(py).downcast::<PyTensor>()?;
        let w_out = w_out_bound.borrow();
        let mut output = functional::matmul(&hidden, &w_out)?;

        if let Some(ref b_out_obj) = self.b_out {
            let b_out_bound = b_out_obj.bind(py).downcast::<PyTensor>()?;
            let b_out = b_out_bound.borrow();
            output = functional::add_bias(&output, &b_out)?;
        }

        Ok(output)
    }
}
