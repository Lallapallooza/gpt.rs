use crate::tensor::DType;

/// How the runtime executes a loaded model: what the checkpoint declares, unless the load options
/// set it.
#[derive(Debug, Clone, Default)]
pub struct ModelRuntimeConfig {
    /// Dtype that the projection layers cast their input to: bf16 for a checkpoint that declares
    /// bf16. On a loaded model, `None` keeps the f32 input. In the load options, `None` keeps the
    /// checkpoint's default.
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
