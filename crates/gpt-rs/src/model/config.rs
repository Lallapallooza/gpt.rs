use serde::{Deserialize, Serialize};

use crate::ops::functional::FunctionalOverrides;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
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

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
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
