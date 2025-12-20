use gpt_rs::backend::registry;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::checkpoint::CheckpointLoader;
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::model::Gpt;
use gpt_rs::ops::functional::DecodeKvCache;
use gpt_rs::tensor::{Shape, Tensor};
use numpy::{PyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type CpuBackend =
    gpt_rs_backend_ref_cpu::GenericCpuBackend<gpt_rs_backend_ref_cpu::NoopInterceptor>;

enum GptImpl {
    Cpu(Gpt<CpuBackend>),
    #[cfg(feature = "conversion-c")]
    C(Gpt<gpt_rs_backend_c::CBackend>),
    #[cfg(feature = "faer")]
    Faer(Gpt<gpt_rs_backend_faer::FaerPortableBackend>),
}

#[pyclass(name = "Gpt")]
pub struct PyGpt {
    inner: GptImpl,
}

fn last_logits_row(logits: &Tensor) -> PyResult<Vec<f32>> {
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
    Ok(data[start..start + vocab].to_vec())
}

fn generate_tokens_impl<B: PortableBackend + 'static>(
    model: &Gpt<B>,
    mut tokens: Vec<usize>,
    max_new_tokens: usize,
    temperature: f32,
    kv_cache: bool,
) -> PyResult<Vec<usize>> {
    if tokens.is_empty() {
        return Err(PyValueError::new_err("tokens must be non-empty"));
    }

    if max_new_tokens == 0 {
        return Ok(tokens);
    }

    let mut sampler = Sampler::new(temperature);
    // Note: top-k is not exposed yet; keep parity with gpt-rs-cli defaults.
    sampler.top_k = None;

    let mut caches: Option<Vec<Option<DecodeKvCache<B>>>> =
        kv_cache.then(|| vec![None; model.blocks.len()]);
    let mut processed_len = 0usize;

    let mut row = if let Some(caches_vec) = caches.as_mut() {
        let logits = model
            .forward_with_decode_cache(&tokens, 0, caches_vec)
            .map_err(|e| PyValueError::new_err(format!("forward_with_decode_cache failed: {e}")))?;
        processed_len = tokens.len();
        last_logits_row(&logits)?
    } else {
        let context_len = model.config.context_length.min(tokens.len());
        let start = tokens.len() - context_len;
        let logits = model
            .forward(&tokens[start..])
            .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
        last_logits_row(&logits)?
    };

    for step in 0..max_new_tokens {
        let vocab = row.len();
        let row_tensor = Tensor::from_vec(Shape::new([vocab]), row)
            .map_err(|e| PyValueError::new_err(format!("failed to wrap logits row: {e}")))?;
        let next = sampler.sample(&row_tensor);
        tokens.push(next);

        if step + 1 == max_new_tokens {
            break;
        }

        if let Some(caches_vec) = caches.as_mut() {
            if tokens.len() > model.config.context_length {
                let drop = tokens.len() - model.config.context_length;
                tokens.drain(0..drop);
                caches_vec.iter_mut().for_each(|slot| *slot = None);
                processed_len = 0;
            }

            let offset = processed_len.min(tokens.len());
            let chunk = &tokens[offset..];
            if chunk.is_empty() {
                break;
            }
            let logits = model
                .forward_with_decode_cache(chunk, offset, caches_vec)
                .map_err(|e| {
                    PyValueError::new_err(format!("forward_with_decode_cache failed: {e}"))
                })?;
            processed_len = tokens.len();
            row = last_logits_row(&logits)?;
        } else {
            let context_len = model.config.context_length.min(tokens.len());
            let start = tokens.len() - context_len;
            let logits = model
                .forward(&tokens[start..])
                .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
            row = last_logits_row(&logits)?;
        }
    }

    Ok(tokens)
}

#[pymethods]
impl PyGpt {
    /// Load a GPT checkpoint (gpt.rs `.bin` export).
    #[staticmethod]
    fn from_checkpoint(path: String) -> PyResult<Self> {
        let backend = crate::backend::create_current_backend()?;
        let backend_name = crate::backend::get_current_backend_name();

        let loaded = CheckpointLoader::load(&path)
            .map_err(|e| PyValueError::new_err(format!("failed to load checkpoint: {e}")))?;

        match backend_name.as_str() {
            "cpu" | "cpu-portable" => {
                let typed = registry::get_typed_backend::<CpuBackend>(backend.as_ref())
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected cpu"))?;
                let model = loaded
                    .into_model(typed)
                    .map_err(|e| PyValueError::new_err(format!("failed to build model: {e}")))?;
                Ok(PyGpt {
                    inner: GptImpl::Cpu(model),
                })
            }
            #[cfg(feature = "conversion-c")]
            "c" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_c::CBackend>(backend.as_ref())
                        .ok_or_else(|| {
                            PyValueError::new_err("backend type mismatch: expected c")
                        })?;
                let model = loaded
                    .into_model(typed)
                    .map_err(|e| PyValueError::new_err(format!("failed to build model: {e}")))?;
                Ok(PyGpt {
                    inner: GptImpl::C(model),
                })
            }
            #[cfg(feature = "faer")]
            "faer" | "faer-portable" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                        backend.as_ref(),
                    )
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected faer"))?;
                let model = loaded
                    .into_model(typed)
                    .map_err(|e| PyValueError::new_err(format!("failed to build model: {e}")))?;
                Ok(PyGpt {
                    inner: GptImpl::Faer(model),
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "backend '{}' is not supported for GPT models yet",
                backend_name
            ))),
        }
    }

    /// Return the last-token logits row for the provided token sequence.
    fn logits<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let row = match &self.inner {
            GptImpl::Cpu(model) => {
                let logits = model
                    .forward(&tokens)
                    .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
                last_logits_row(&logits)?
            }
            #[cfg(feature = "conversion-c")]
            GptImpl::C(model) => {
                let logits = model
                    .forward(&tokens)
                    .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
                last_logits_row(&logits)?
            }
            #[cfg(feature = "faer")]
            GptImpl::Faer(model) => {
                let logits = model
                    .forward(&tokens)
                    .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
                last_logits_row(&logits)?
            }
        };
        Ok(PyArray::from_vec_bound(py, row))
    }

    /// Generate up to `max_new_tokens` by sampling from the model.
    #[pyo3(signature = (tokens, max_new_tokens, temperature = 1.0, kv_cache = true))]
    fn generate_tokens(
        &self,
        tokens: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        kv_cache: bool,
    ) -> PyResult<Vec<usize>> {
        match &self.inner {
            GptImpl::Cpu(model) => {
                generate_tokens_impl(model, tokens, max_new_tokens, temperature, kv_cache)
            }
            #[cfg(feature = "conversion-c")]
            GptImpl::C(model) => {
                generate_tokens_impl(model, tokens, max_new_tokens, temperature, kv_cache)
            }
            #[cfg(feature = "faer")]
            GptImpl::Faer(model) => {
                generate_tokens_impl(model, tokens, max_new_tokens, temperature, kv_cache)
            }
        }
    }
}
