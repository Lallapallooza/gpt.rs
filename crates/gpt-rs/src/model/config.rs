use crate::tensor::DType;

/// How the runtime executes a loaded model, independent of the checkpoint.
#[derive(Debug, Clone, Default)]
pub struct ModelRuntimeConfig {
    /// Dtype that the projection layers cast their input to. `None` keeps the f32 input.
    pub matmul_input_dtype: Option<DType>,
}

/// What a model is built from. `kind` is the Hugging Face `model_type`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub kind: String,
    pub config: serde_json::Value,
    pub runtime: ModelRuntimeConfig,
    pub eos_token_ids: Vec<usize>,
}
