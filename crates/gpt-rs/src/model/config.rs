use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

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

        if let Ok(_legacy_gpt) = serde_json::from_value::<super::gpt::GptConfig>(value.clone()) {
            return Ok(Self::new("gpt".to_string(), value));
        }

        Err(D::Error::custom("invalid ModelConfig"))
    }
}
