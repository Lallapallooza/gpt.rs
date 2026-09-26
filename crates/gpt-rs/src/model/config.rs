use crate::tensor::DType;

/// Weight streaming: parameters stay in the checkpoint and load each time a program uses them,
/// unless the optional cache keeps them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeightStreamingConfig {
    pub enabled: bool,
    pub device_budget_bytes: Option<u64>,
    pub device_weights_percent: Option<f32>,
    pub cache_budget_cap_bytes: Option<u64>,
    pub prefetch_layers: Option<usize>,
    pub small_param_persist_threshold: Option<u64>,
}

/// How the runtime executes a loaded model, independent of the checkpoint.
#[derive(Debug, Clone, Default)]
pub struct ModelRuntimeConfig {
    pub weight_streaming: WeightStreamingConfig,
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
