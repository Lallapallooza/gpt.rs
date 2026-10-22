use anyhow::Context as _;
use gpt_rs::gpt_rs_ir::host::TensorData;
use gpt_rs::inference::generate::{generate_tokens, Sampling};
use gpt_rs::nn::capture;
use gpt_rs::runtime::{LoadedModel, ModelHandle};
use gpt_rs::tensor::HostTensor;
use numpy::{PyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods as _};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn tensor_to_numpy<'py>(py: Python<'py>, host: &HostTensor) -> PyResult<Bound<'py, PyAny>> {
    let shape = host.dims.as_slice();
    match &host.data {
        TensorData::F32(data) => Ok(PyArray::from_slice_bound(py, data)
            .reshape(shape)?
            .into_any()),
        TensorData::Si32(data) => Ok(PyArray::from_slice_bound(py, data)
            .reshape(shape)?
            .into_any()),
        _ => Err(PyValueError::new_err(format!(
            "unsupported tensor dtype for numpy conversion: {}",
            host.dtype()
        ))),
    }
}

fn last_logits_row(logits: &HostTensor) -> PyResult<&[f32]> {
    let dims = &logits.dims;
    if dims.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "expected logits [T, V], got shape {dims:?}"
        )));
    }
    let seq_len = dims[0];
    let vocab = dims[1];
    if seq_len == 0 || vocab == 0 {
        return Err(PyValueError::new_err("logits must be non-empty"));
    }
    let data = logits
        .as_f32()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let start = (seq_len - 1) * vocab;
    Ok(&data[start..start + vocab])
}

/// The f32 NCHW pixels of `array`, which must have rank 4, copied: the model runs without the GIL,
/// while Python code may write the array.
fn pixels(array: PyReadonlyArrayDyn<f32>) -> PyResult<HostTensor> {
    let shape: Vec<usize> = array.shape().to_vec();
    if shape.len() != 4 {
        return Err(PyValueError::new_err(format!(
            "expected NCHW pixels of rank 4, got shape {shape:?}",
        )));
    }
    let values: Vec<f32> = array.as_array().iter().copied().collect();
    HostTensor::f32(shape, values)
        .map_err(|e| PyValueError::new_err(format!("failed to create host tensor: {e}")))
}

fn runtime_error(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

/// A loaded checkpoint-backed model.
///
/// The object is intentionally small: it only supports "load + run" workloads
/// (forward and generation) and hides tensor/nn/functional internals.
#[pyclass(name = "LoadedModel")]
pub struct PyLoadedModel {
    backend_name: String,
    model: ModelHandle,
}

#[pymethods]
impl PyLoadedModel {
    fn kind(&self) -> &str {
        self.model.kind()
    }

    /// Forward a token sequence through a causal LM and return logits [T, V].
    fn forward_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = &self.model;
        let logits = py
            .allow_threads(|| model.causal_lm()?.forward(&tokens))
            .map_err(runtime_error)?;
        tensor_to_numpy(py, &logits)
    }

    /// Return the last logits row [V] for a causal LM.
    fn logits<'py>(&self, py: Python<'py>, tokens: Vec<usize>) -> PyResult<Bound<'py, PyAny>> {
        let model = &self.model;
        let logits = py
            .allow_threads(|| model.causal_lm()?.forward(&tokens))
            .map_err(runtime_error)?;
        let row = last_logits_row(&logits)?;
        Ok(PyArray::from_vec_bound(py, row.to_vec()).into_any())
    }

    /// Return the output of every module in a forward pass over `tokens`, as
    /// `gpt_rs::nn::capture` records it.
    ///
    /// The returned list contains dictionaries with:
    /// - `name`: activation name, the Hugging Face module path (for example `model.layers.3`)
    /// - `tensor`: numpy array view of the host tensor
    fn debug_token_activations<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let model = &self.model;
        let (_, activations) = py
            .allow_threads(|| {
                let causal_lm = model.causal_lm()?;
                capture::module_outputs(|| causal_lm.forward(&tokens))
            })
            .map_err(runtime_error)?;

        let out = PyList::empty_bound(py);
        for (name, tensor) in activations {
            let entry = PyDict::new_bound(py);
            entry.set_item("name", name)?;
            entry.set_item("tensor", tensor_to_numpy(py, &tensor)?)?;
            out.append(entry)?;
        }
        Ok(out.into_any())
    }

    /// Generates up to `max_new_tokens` tokens after `prompt_tokens`. A temperature of zero or
    /// less picks the most likely token. A positive one draws from the softmax, with the random
    /// draws of `seed` (a fresh seed when it is `None`). With `top_k`, it draws only among the
    /// `top_k` most likely tokens. Generation stops after an end-of-sequence token of the
    /// checkpoint unless `ignore_eos` is set, and when it fills the context length.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, *, temperature=1.0, top_k=None, seed=None, ignore_eos=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate_tokens(
        &self,
        py: Python,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        seed: Option<u64>,
        ignore_eos: bool,
    ) -> PyResult<Vec<usize>> {
        let sampling = if temperature > 0.0 {
            let sampling = Sampling::random(temperature, seed.unwrap_or_else(rand::random));
            match top_k {
                Some(k) => sampling.with_top_k(k),
                None => sampling,
            }
        } else {
            Sampling::greedy()
        };
        let model = &self.model;
        let stop_tokens = if ignore_eos {
            &[][..]
        } else {
            model.eos_token_ids()
        };
        py.allow_threads(|| {
            let causal_lm = model.causal_lm()?;
            generate_tokens(
                causal_lm,
                &prompt_tokens,
                max_new_tokens,
                &sampling,
                stop_tokens,
            )
        })
        .map_err(runtime_error)
    }

    /// Forward a vision model given a float32 NCHW input and return logits [N, C].
    fn forward_vision<'py>(
        &self,
        py: Python<'py>,
        input_nchw: PyReadonlyArrayDyn<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pixels = pixels(input_nchw)?;
        let model = &self.model;
        let out = py
            .allow_threads(|| model.vision()?.forward(pixels))
            .map_err(runtime_error)?;
        tensor_to_numpy(py, &out)
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }

    /// Input dtype of the projection matmuls that the checkpoint or the load options set, such as
    /// `"bf16"` or `"f32"`, or `None` when neither sets one. `None` keeps the f32 input.
    #[getter]
    fn matmul_input_dtype(&self) -> Option<String> {
        self.model
            .runtime()
            .matmul_input_dtype
            .map(|dtype| dtype.to_string())
    }
}

/// Loads the Hugging Face model directory `checkpoint`. `config_overrides` replaces keys of the
/// model config (the text config of a multimodal checkpoint). `target` is the target of a backend
/// that compiles for one: a preset, `host` or a target file (`$GPTRS_TARGET`, else `host`).
/// `winograd` runs 3x3 convolutions with stride 1 as Winograd's F(2x2, 3x3), which rounds
/// differently from the convolution. It is off by default (`$GPTRS_WINOGRAD=1`).
#[pyfunction(signature = (
    checkpoint,
    *,
    backend=None,
    matmul_input_dtype=None,
    config_overrides=None,
    target=None,
    winograd=None
))]
pub fn load_model(
    py: Python,
    checkpoint: String,
    backend: Option<String>,
    matmul_input_dtype: Option<String>,
    config_overrides: Option<Bound<PyDict>>,
    target: Option<String>,
    winograd: Option<bool>,
) -> PyResult<PyLoadedModel> {
    let config_overrides = match config_overrides {
        Some(overrides) => {
            let json: String = py
                .import_bound("json")?
                .call_method1("dumps", (overrides,))?
                .extract()?;
            serde_json::from_str(&json).map_err(|e| PyValueError::new_err(e.to_string()))?
        }
        None => serde_json::Map::new(),
    };
    let options = gpt_rs::runtime::LoadOptions {
        namespace: None,
        runtime: gpt_rs::model::config::ModelRuntimeConfig {
            matmul_input_dtype: matmul_input_dtype
                .as_deref()
                .map(str::parse::<gpt_rs::DType>)
                .transpose()
                .map_err(|e| PyValueError::new_err(format!("{e:#}")))?,
        },
        config_overrides,
    };

    let mut backend_options = gpt_rs_backends::BackendOptions::from_env();
    if let Some(target) = target {
        backend_options.target = target;
    }
    if let Some(winograd) = winograd {
        backend_options.winograd = winograd;
    }
    let backend_name = backend.unwrap_or_else(crate::backend::get_backend);
    let model = py
        .allow_threads(|| {
            gpt_rs_backends::load_model(&backend_name, &backend_options, &checkpoint, options)
        })
        .with_context(|| format!("failed to load checkpoint {checkpoint}"))
        .map_err(runtime_error)?;
    Ok(PyLoadedModel {
        backend_name,
        model,
    })
}

#[pyfunction]
pub fn supported_model_kinds() -> Vec<&'static str> {
    gpt_rs::model::registry::model_factories::<gpt_rs_backend_ref_cpu::CpuPortableBackend>()
        .iter()
        .map(|factory| factory.kind)
        .collect()
}

#[pyfunction]
pub fn version_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("crate", env!("CARGO_PKG_NAME"))?;
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("rust", "2021")?;
    Ok(d)
}
