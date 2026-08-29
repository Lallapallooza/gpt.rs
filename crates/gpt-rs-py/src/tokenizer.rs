use gpt_rs::tokenizer::Tokenizer;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

/// Byte-level byte-pair encoding tokenizer. It reads GPT-2 flat configs and Hugging Face
/// `tokenizer.json` files.
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from JSON config file
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        let inner = Tokenizer::from_file(&path).map_err(|e| {
            if e.downcast_ref::<std::io::Error>().is_some() {
                PyIOError::new_err(format!("{e:#}"))
            } else {
                PyValueError::new_err(format!("{e:#}"))
            }
        })?;
        Ok(PyTokenizer { inner })
    }

    /// Encode text to token IDs
    fn encode(&self, text: String) -> PyResult<Vec<usize>> {
        self.inner
            .encode(&text)
            .map_err(|e| PyValueError::new_err(format!("{e:#}")))
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: Vec<usize>) -> String {
        self.inner.decode(&tokens)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}
