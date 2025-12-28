use crate::ops::functional::FunctionalOverrides;
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub dropout: f32,
    #[serde(default)]
    pub functional_overrides: FunctionalOverrides,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4,
            dropout: 0.0,
            functional_overrides: FunctionalOverrides::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet34Config {
    pub num_classes: usize,
}

impl Default for ResNet34Config {
    fn default() -> Self {
        Self { num_classes: 1000 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileNetV2Config {
    pub num_classes: usize,
}

impl Default for MobileNetV2Config {
    fn default() -> Self {
        Self { num_classes: 1000 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelConfig {
    pub kind: String,
    #[serde(default)]
    pub config: serde_json::Value,
}

impl ModelConfig {
    pub fn new(kind: impl Into<String>, config: serde_json::Value) -> Self {
        Self {
            kind: kind.into(),
            config,
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
            return Ok(Self::new(kind.to_string(), config));
        }

        if let Ok(_legacy_gpt) = serde_json::from_value::<GptConfig>(value.clone()) {
            return Ok(Self::new("gpt".to_string(), value));
        }

        Err(D::Error::custom("invalid ModelConfig"))
    }
}
