use std::sync::Arc;

use anyhow::Context as _;
use gpt_rs::backend::registry;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::inference::generate::{generate_tokens, GenerateConfig};
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::nn::capture;
use gpt_rs::runtime::{LoadedModel, ModelHandle, ModelInput, ModelOutput};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use numpy::{PyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods as _};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn tensor_to_numpy<'py>(py: Python<'py>, host: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let shape = host.shape().dims();
    match host.dtype() {
        gpt_rs::DType::F32 => {
            let data = host.data().to_vec();
            Ok(PyArray::from_vec_bound(py, data).reshape(shape)?.into_any())
        }
        gpt_rs::DType::I32 => {
            let data = host.data_i32().to_vec();
            Ok(PyArray::from_vec_bound(py, data).reshape(shape)?.into_any())
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported tensor dtype for numpy conversion: {other:?}",
        ))),
    }
}

fn last_logits_row(logits: &Tensor) -> PyResult<&[f32]> {
    let dims = logits.shape().dims();
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
    let data = logits.data();
    let start = (seq_len - 1) * vocab;
    Ok(&data[start..start + vocab])
}

fn numpy_to_device_tensor<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    arr: PyReadonlyArrayDyn<'_, f32>,
    expected_rank: usize,
    kind_label: &'static str,
) -> PyResult<DeviceTensor<B>> {
    let shape: Vec<usize> = arr.shape().to_vec();
    if shape.len() != expected_rank {
        return Err(PyValueError::new_err(format!(
            "expected {kind_label} input rank {expected_rank}, got shape {shape:?}",
        )));
    }
    let values: Vec<f32> = arr.as_array().iter().copied().collect();
    let host = Tensor::from_vec(Shape::new(shape), values)
        .map_err(|e| PyValueError::new_err(format!("failed to create host tensor: {e}")))?;
    DeviceTensor::from_host(Arc::clone(backend), host)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to upload tensor: {e}")))
}

type CpuBackend =
    gpt_rs_backend_ref_cpu::GenericCpuBackend<gpt_rs_backend_ref_cpu::NoopInterceptor>;

#[cfg(feature = "faer")]
type FaerBackend = gpt_rs_backend_faer::FaerPortableBackend;

#[cfg(feature = "conversion-c")]
type CBackend = gpt_rs_backend_c::CBackend;

#[cfg(feature = "triton")]
type TritonBackend = gpt_rs_backend_triton::TritonBackend;

enum PyLoadedModelInner {
    Cpu {
        backend: Arc<CpuBackend>,
        model: ModelHandle<CpuBackend>,
    },
    #[cfg(feature = "faer")]
    Faer {
        backend: Arc<FaerBackend>,
        model: ModelHandle<FaerBackend>,
    },
    #[cfg(feature = "conversion-c")]
    C {
        backend: Arc<CBackend>,
        model: ModelHandle<CBackend>,
    },
    #[cfg(feature = "triton")]
    Triton {
        backend: Arc<TritonBackend>,
        model: ModelHandle<TritonBackend>,
    },
}

/// Evaluates `$body` with `$backend` and `$model` bound to the fields of whichever backend
/// variant `$inner` holds.
macro_rules! with_model {
    ($inner:expr, |$backend:pat_param, $model:ident| $body:expr) => {
        match $inner {
            PyLoadedModelInner::Cpu {
                backend: $backend,
                model: $model,
            } => $body,
            #[cfg(feature = "faer")]
            PyLoadedModelInner::Faer {
                backend: $backend,
                model: $model,
            } => $body,
            #[cfg(feature = "conversion-c")]
            PyLoadedModelInner::C {
                backend: $backend,
                model: $model,
            } => $body,
            #[cfg(feature = "triton")]
            PyLoadedModelInner::Triton {
                backend: $backend,
                model: $model,
            } => $body,
        }
    };
}

/// A loaded checkpoint-backed model.
///
/// The object is intentionally small: it only supports "load + run" workloads
/// (forward and generation) and hides tensor/nn/functional internals.
#[pyclass(name = "LoadedModel")]
pub struct PyLoadedModel {
    backend_name: String,
    inner: PyLoadedModelInner,
}

#[pymethods]
impl PyLoadedModel {
    fn kind(&self) -> &str {
        with_model!(&self.inner, |_, model| model.kind())
    }

    /// Forward a token sequence through a causal LM and return logits [T, V].
    fn forward_tokens<'py>(
        &mut self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ModelOutput::Tensor(logits) = py
            .allow_threads(|| {
                with_model!(&mut self.inner, |_, model| LoadedModel::forward(
                    model,
                    ModelInput::Tokens(tokens)
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        tensor_to_numpy(py, &logits)
    }

    /// Return the last logits row [V] for a causal LM.
    fn logits<'py>(&mut self, py: Python<'py>, tokens: Vec<usize>) -> PyResult<Bound<'py, PyAny>> {
        let ModelOutput::Tensor(logits) = py
            .allow_threads(|| {
                with_model!(&mut self.inner, |_, model| LoadedModel::forward(
                    model,
                    ModelInput::Tokens(tokens)
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
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
        &mut self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (_, activations) = py
            .allow_threads(|| {
                with_model!(&mut self.inner, |_, model| capture::module_outputs(|| {
                    LoadedModel::forward(model, ModelInput::Tokens(tokens))
                }))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        let out = PyList::empty_bound(py);
        for (name, tensor) in activations {
            let entry = PyDict::new_bound(py);
            entry.set_item("name", name)?;
            entry.set_item("tensor", tensor_to_numpy(py, &tensor)?)?;
            out.append(entry)?;
        }
        Ok(out.into_any())
    }

    /// Generates up to `max_new_tokens` tokens after `prompt_tokens`. Generation stops after an
    /// end-of-sequence token of the checkpoint unless `ignore_eos` is set.
    #[pyo3(signature = (prompt_tokens, max_new_tokens, *, temperature=1.0, top_k=None, kv_cache=true, kv_cache_capacity=None, ignore_eos=false))]
    #[allow(clippy::too_many_arguments)]
    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        kv_cache: bool,
        kv_cache_capacity: Option<usize>,
        ignore_eos: bool,
    ) -> PyResult<Vec<usize>> {
        let sampler = match top_k {
            Some(k) => Sampler::new(temperature).with_top_k(k),
            None => Sampler::new(temperature),
        };
        with_model!(&mut self.inner, |_, model| {
            let Some(causal_lm) = model.as_causal_lm() else {
                return Err(PyValueError::new_err(
                    "model is not a causal language model",
                ));
            };
            let cfg = GenerateConfig {
                max_new_tokens,
                kv_cache,
                kv_cache_capacity,
                stop_tokens: if ignore_eos {
                    Vec::new()
                } else {
                    model.eos_token_ids().to_vec()
                },
            };
            generate_tokens(causal_lm, &prompt_tokens, &sampler, cfg)
                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    /// Forward a vision model given a float32 NCHW input and return logits [N, C].
    fn forward_vision<'py>(
        &mut self,
        py: Python<'py>,
        input_nchw: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = with_model!(&mut self.inner, |backend, model| {
            let input = numpy_to_device_tensor(backend, input_nchw, 4, "vision")?;
            py.allow_threads(|| LoadedModel::forward(model, ModelInput::Vision(input)))
        });
        let ModelOutput::Tensor(out) =
            out.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        tensor_to_numpy(py, &out)
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }
}

#[pyfunction(signature = (checkpoint, *, backend=None, matmul_input_dtype=None))]
pub fn load_model(
    checkpoint: String,
    backend: Option<String>,
    matmul_input_dtype: Option<String>,
) -> PyResult<PyLoadedModel> {
    if let Some(name) = backend.as_deref() {
        crate::backend::set_backend(name)?;
    }
    let options = gpt_rs::runtime::LoadOptions {
        namespace: None,
        matmul_input_dtype: matmul_input_dtype
            .as_deref()
            .map(str::parse::<gpt_rs::DType>)
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))?,
    };

    let erased = crate::backend::create_current_backend()?;
    let backend_name = erased.backend_name().to_string();
    macro_rules! load_as {
        ($backend_type:ty, $variant:ident) => {
            if let Some(backend) = registry::get_typed_backend::<$backend_type>(erased.as_ref()) {
                let model = gpt_rs::runtime::load_model_with_options(
                    Arc::clone(&backend),
                    &checkpoint,
                    options,
                )
                .with_context(|| format!("failed to load checkpoint {checkpoint}"))
                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                return Ok(PyLoadedModel {
                    backend_name,
                    inner: PyLoadedModelInner::$variant { backend, model },
                });
            }
        };
    }
    load_as!(CpuBackend, Cpu);
    #[cfg(feature = "faer")]
    load_as!(FaerBackend, Faer);
    #[cfg(feature = "conversion-c")]
    load_as!(CBackend, C);
    #[cfg(feature = "triton")]
    load_as!(TritonBackend, Triton);

    Err(PyRuntimeError::new_err(format!(
        "unsupported backend '{backend_name}' for runtime.load_model (missing feature build?)",
    )))
}

#[pyfunction]
pub fn supported_model_kinds() -> PyResult<Vec<String>> {
    Ok(gpt_rs::model::registry::model_factories::<CpuBackend>()
        .iter()
        .map(|factory| factory.kind.to_string())
        .collect())
}

#[pyfunction]
pub fn supported_backends() -> PyResult<Vec<String>> {
    crate::backend::list_backends()
}

#[pyfunction]
pub fn backend_features<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("cpu", true)?;
    d.set_item("faer", cfg!(feature = "faer"))?;
    d.set_item("conversion_c", cfg!(feature = "conversion-c"))?;
    d.set_item("triton", cfg!(feature = "triton"))?;
    Ok(d)
}

#[pyfunction]
pub fn version_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("crate", env!("CARGO_PKG_NAME"))?;
    d.set_item("version", env!("CARGO_PKG_VERSION"))?;
    d.set_item("rust", "2021")?;
    Ok(d)
}
