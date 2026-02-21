use std::sync::Arc;

use anyhow::Context as _;
use gpt_rs::backend::registry;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::inference::generate::{GenerateConfig, Generator};
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::runtime::{ModelInput, ModelOutput};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use numpy::{PyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods as _};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

enum PyLoadedModelInner {
    Cpu {
        backend: Arc<CpuBackend>,
        model: Box<dyn gpt_rs::runtime::LoadedModel<CpuBackend>>,
    },
    #[cfg(feature = "faer")]
    Faer {
        backend: Arc<FaerBackend>,
        model: Box<dyn gpt_rs::runtime::LoadedModel<FaerBackend>>,
    },
    #[cfg(feature = "conversion-c")]
    C {
        backend: Arc<CBackend>,
        model: Box<dyn gpt_rs::runtime::LoadedModel<CBackend>>,
    },
}

impl PyLoadedModelInner {
    fn kind(&self) -> &str {
        match self {
            Self::Cpu { model, .. } => model.kind(),
            #[cfg(feature = "faer")]
            Self::Faer { model, .. } => model.kind(),
            #[cfg(feature = "conversion-c")]
            Self::C { model, .. } => model.kind(),
        }
    }
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
        self.inner.kind()
    }

    /// Forward a token sequence through a causal LM and return logits [T, V].
    fn forward_tokens<'py>(
        &mut self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = py.allow_threads(|| match &mut self.inner {
            PyLoadedModelInner::Cpu { model, .. } => model.forward(ModelInput::Tokens(tokens)),
            #[cfg(feature = "faer")]
            PyLoadedModelInner::Faer { model, .. } => model.forward(ModelInput::Tokens(tokens)),
            #[cfg(feature = "conversion-c")]
            PyLoadedModelInner::C { model, .. } => model.forward(ModelInput::Tokens(tokens)),
        });
        let out = out.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        match out {
            ModelOutput::Tensor(t) => tensor_to_numpy(py, &t),
        }
    }

    /// Return the last logits row [V] for a causal LM.
    fn logits<'py>(&mut self, py: Python<'py>, tokens: Vec<usize>) -> PyResult<Bound<'py, PyAny>> {
        let logits = py.allow_threads(|| match &mut self.inner {
            PyLoadedModelInner::Cpu { model, .. } => {
                match model.forward(ModelInput::Tokens(tokens)) {
                    Ok(ModelOutput::Tensor(t)) => Ok(t),
                    Err(e) => Err(e),
                }
            }
            #[cfg(feature = "faer")]
            PyLoadedModelInner::Faer { model, .. } => {
                match model.forward(ModelInput::Tokens(tokens)) {
                    Ok(ModelOutput::Tensor(t)) => Ok(t),
                    Err(e) => Err(e),
                }
            }
            #[cfg(feature = "conversion-c")]
            PyLoadedModelInner::C { model, .. } => {
                match model.forward(ModelInput::Tokens(tokens)) {
                    Ok(ModelOutput::Tensor(t)) => Ok(t),
                    Err(e) => Err(e),
                }
            }
        });
        let logits = logits.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        let row = last_logits_row(&logits)?;
        Ok(PyArray::from_vec_bound(py, row.to_vec()).into_any())
    }

    #[pyo3(signature = (prompt_tokens, max_new_tokens, *, temperature=1.0, top_k=None, kv_cache=true, kv_cache_capacity=None))]
    fn generate_tokens(
        &mut self,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        kv_cache: bool,
        kv_cache_capacity: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let sampler = match top_k {
            Some(k) => Sampler::new(temperature).with_top_k(k),
            None => Sampler::new(temperature),
        };

        let cfg = GenerateConfig {
            max_new_tokens,
            kv_cache,
        };

        let result = match &mut self.inner {
            PyLoadedModelInner::Cpu { model, .. } => {
                let lm = model
                    .as_causal_lm()
                    .ok_or_else(|| PyValueError::new_err("model is not a causal language model"))?;
                if let Some(capacity) = kv_cache_capacity {
                    let mut gen = Generator::new_with_kv_cache_capacity(
                        lm,
                        &sampler,
                        &prompt_tokens,
                        cfg.kv_cache,
                        Some(capacity),
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                    for step in 0..cfg.max_new_tokens {
                        if step + 1 == cfg.max_new_tokens {
                            gen.step_final()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        } else {
                            gen.step()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        }
                    }
                    Ok(gen.into_tokens())
                } else {
                    gpt_rs::inference::generate::generate_tokens(lm, &prompt_tokens, &sampler, cfg)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
                }
            }
            #[cfg(feature = "faer")]
            PyLoadedModelInner::Faer { model, .. } => {
                let lm = model
                    .as_causal_lm()
                    .ok_or_else(|| PyValueError::new_err("model is not a causal language model"))?;
                if let Some(capacity) = kv_cache_capacity {
                    let mut gen = Generator::new_with_kv_cache_capacity(
                        lm,
                        &sampler,
                        &prompt_tokens,
                        cfg.kv_cache,
                        Some(capacity),
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                    for step in 0..cfg.max_new_tokens {
                        if step + 1 == cfg.max_new_tokens {
                            gen.step_final()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        } else {
                            gen.step()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        }
                    }
                    Ok(gen.into_tokens())
                } else {
                    gpt_rs::inference::generate::generate_tokens(lm, &prompt_tokens, &sampler, cfg)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
                }
            }
            #[cfg(feature = "conversion-c")]
            PyLoadedModelInner::C { model, .. } => {
                let lm = model
                    .as_causal_lm()
                    .ok_or_else(|| PyValueError::new_err("model is not a causal language model"))?;
                if let Some(capacity) = kv_cache_capacity {
                    let mut gen = Generator::new_with_kv_cache_capacity(
                        lm,
                        &sampler,
                        &prompt_tokens,
                        cfg.kv_cache,
                        Some(capacity),
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                    for step in 0..cfg.max_new_tokens {
                        if step + 1 == cfg.max_new_tokens {
                            gen.step_final()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        } else {
                            gen.step()
                                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
                        }
                    }
                    Ok(gen.into_tokens())
                } else {
                    gpt_rs::inference::generate::generate_tokens(lm, &prompt_tokens, &sampler, cfg)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
                }
            }
        }?;

        Ok(result)
    }

    /// Forward a vision model given a float32 NCHW input and return logits [N, C].
    fn forward_vision<'py>(
        &mut self,
        py: Python<'py>,
        input_nchw: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let out = match &mut self.inner {
            PyLoadedModelInner::Cpu { backend, model } => {
                let input = numpy_to_device_tensor(backend, input_nchw, 4, "vision")?;
                py.allow_threads(|| model.forward(ModelInput::Vision(input)))
            }
            #[cfg(feature = "faer")]
            PyLoadedModelInner::Faer { backend, model } => {
                let input = numpy_to_device_tensor(backend, input_nchw, 4, "vision")?;
                py.allow_threads(|| model.forward(ModelInput::Vision(input)))
            }
            #[cfg(feature = "conversion-c")]
            PyLoadedModelInner::C { backend, model } => {
                let input = numpy_to_device_tensor(backend, input_nchw, 4, "vision")?;
                py.allow_threads(|| model.forward(ModelInput::Vision(input)))
            }
        };

        let out = out.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        match out {
            ModelOutput::Tensor(t) => tensor_to_numpy(py, &t),
        }
    }

    fn backend(&self) -> &str {
        &self.backend_name
    }
}

#[pyfunction(signature = (checkpoint, *, backend=None))]
pub fn load_model(checkpoint: String, backend: Option<String>) -> PyResult<PyLoadedModel> {
    if let Some(name) = backend.as_deref() {
        crate::backend::set_backend(name)?;
    }

    let erased = crate::backend::create_current_backend()?;
    let backend_name = erased.backend_name().to_string();

    let path = checkpoint.clone();

    if let Some(backend) = registry::get_typed_backend::<CpuBackend>(erased.as_ref()) {
        let model = gpt_rs::runtime::load_model(Arc::clone(&backend), &path)
            .with_context(|| format!("failed to load checkpoint {path}"))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        return Ok(PyLoadedModel {
            backend_name,
            inner: PyLoadedModelInner::Cpu { backend, model },
        });
    }

    #[cfg(feature = "faer")]
    if let Some(backend) = registry::get_typed_backend::<FaerBackend>(erased.as_ref()) {
        let model = gpt_rs::runtime::load_model(Arc::clone(&backend), &path)
            .with_context(|| format!("failed to load checkpoint {path}"))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        return Ok(PyLoadedModel {
            backend_name,
            inner: PyLoadedModelInner::Faer { backend, model },
        });
    }

    #[cfg(feature = "conversion-c")]
    if let Some(backend) = registry::get_typed_backend::<CBackend>(erased.as_ref()) {
        let model = gpt_rs::runtime::load_model(Arc::clone(&backend), &path)
            .with_context(|| format!("failed to load checkpoint {path}"))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        return Ok(PyLoadedModel {
            backend_name,
            inner: PyLoadedModelInner::C { backend, model },
        });
    }

    Err(PyRuntimeError::new_err(format!(
        "unsupported backend '{backend_name}' for runtime.load_model (missing feature build?)",
    )))
}

#[pyfunction]
pub fn supported_model_kinds() -> PyResult<Vec<String>> {
    Ok(vec![
        "gpt".to_string(),
        "ministral".to_string(),
        "resnet34".to_string(),
        "mobilenet_v2".to_string(),
    ])
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
