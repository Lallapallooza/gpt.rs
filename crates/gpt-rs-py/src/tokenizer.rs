use gpt_rs::tokenizer::{Tokenizer, TokenizerConfig};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::path::PathBuf;

/// GPT-style byte-pair encoding tokenizer
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load tokenizer from JSON config file
    #[staticmethod]
    fn from_file(path: String) -> PyResult<Self> {
        let path_buf = PathBuf::from(path);
        let file = std::fs::File::open(&path_buf)
            .map_err(|e| PyIOError::new_err(format!("failed to open tokenizer file: {}", e)))?;

        let config: TokenizerConfig = serde_json::from_reader(file).map_err(|e| {
            PyValueError::new_err(format!("failed to parse tokenizer config: {}", e))
        })?;

        Ok(PyTokenizer {
            inner: Tokenizer::from_config(config),
        })
    }

    /// Encode text to token IDs
    fn encode(&self, text: String) -> Vec<usize> {
        self.inner.encode(&text)
    }

    /// Decode token IDs back to text
    fn decode(&self, tokens: Vec<usize>) -> String {
        self.inner.decode(&tokens)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        // TODO: Add vocab_size() method to Tokenizer
        0
    }
}
