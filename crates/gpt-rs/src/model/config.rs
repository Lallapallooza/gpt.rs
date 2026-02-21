use crate::ops::functional::FunctionalOverrides;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

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
    #[serde(default, skip_serializing_if = "FunctionalOverrides::is_empty")]
    pub functional_overrides: FunctionalOverrides,
    #[serde(default, skip_serializing_if = "WeightStreamingConfig::is_disabled")]
    pub weight_streaming: WeightStreamingConfig,
}

impl ModelRuntimeConfig {
    fn is_empty(&self) -> bool {
        self.functional_overrides.is_empty() && self.weight_streaming.is_disabled()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConfig {
    pub kind: String,
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default, skip_serializing_if = "ModelRuntimeConfig::is_empty")]
    pub runtime: ModelRuntimeConfig,
}

impl ModelConfig {
    pub fn new(kind: impl Into<String>, config: serde_json::Value) -> Self {
        Self {
            kind: kind.into(),
            config,
            runtime: ModelRuntimeConfig::default(),
        }
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
        }
    }
}

impl<'de> Deserialize<'de> for ModelConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;

        if let Some(kind) = value.get("kind").and_then(|v| v.as_str()) {
            let config = value
                .get("config")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let runtime = match value.get("runtime") {
                Some(runtime) if runtime.is_null() => ModelRuntimeConfig::default(),
                Some(runtime) => {
                    serde_json::from_value(runtime.clone()).map_err(D::Error::custom)?
                }
                None => ModelRuntimeConfig::default(),
            };
            return Ok(Self::new_with_runtime(kind.to_string(), config, runtime));
        }

        Err(D::Error::custom("invalid ModelConfig"))
    }
}
