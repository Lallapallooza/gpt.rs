use crate::tensor::DType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WeightStreamingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_budget_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_weights_percent: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_budget_cap_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefetch_layers: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub small_param_persist_threshold: Option<u64>,
}

impl WeightStreamingConfig {
    fn is_disabled(&self) -> bool {
        !self.enabled
            && self.device_budget_bytes.is_none()
            && self.device_weights_percent.is_none()
            && self.cache_budget_cap_bytes.is_none()
            && self.prefetch_layers.is_none()
            && self.small_param_persist_threshold.is_none()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRuntimeConfig {
    #[serde(default, skip_serializing_if = "WeightStreamingConfig::is_disabled")]
    pub weight_streaming: WeightStreamingConfig,
    /// Dtype that every `nn::Linear` casts its input to. `None` keeps the f32 input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matmul_input_dtype: Option<DType>,
}

impl ModelRuntimeConfig {
    fn is_empty(&self) -> bool {
        self.weight_streaming.is_disabled() && self.matmul_input_dtype.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub kind: String,
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default, skip_serializing_if = "ModelRuntimeConfig::is_empty")]
    pub runtime: ModelRuntimeConfig,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eos_token_ids: Vec<usize>,
}

impl ModelConfig {
    pub fn new(kind: impl Into<String>, config: serde_json::Value) -> Self {
        Self::new_with_runtime(kind, config, ModelRuntimeConfig::default())
    }

    pub fn new_with_runtime(
        kind: impl Into<String>,
        config: serde_json::Value,
        runtime: ModelRuntimeConfig,
    ) -> Self {
        Self {
            kind: kind.into(),
            config,
            runtime,
            eos_token_ids: Vec::new(),
        }
    }
}
