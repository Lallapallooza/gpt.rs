use crate::backend::spec::PortableBackend;
use crate::inference::decoder::{CausalDecoder, DecoderConfig};
use crate::io::tensor_index::write_tensor_file;
use crate::model::ModelConfig;
use crate::module::{Module, ParamVisitor, TensorRole};
use crate::params::base_param_id;
use crate::tensor::DeviceTensor;
use anyhow::{Context, Result};
use std::path::Path;

pub struct CheckpointSaver;

impl CheckpointSaver {
    pub fn save<B: PortableBackend + 'static, C: DecoderConfig>(
        path: impl AsRef<Path>,
        model: &CausalDecoder<B, C>,
    ) -> Result<()> {
        let config = ModelConfig::new(C::KIND, serde_json::to_value(&model.config)?);
        let config_bytes = serde_json::to_vec(&config)?;
        let mut header = (config_bytes.len() as u32).to_le_bytes().to_vec();
        header.extend_from_slice(&config_bytes);

        let mut params = Vec::new();
        let mut export = |name: &str, _role: TensorRole, tensor: &DeviceTensor<B>| -> Result<()> {
            let host = tensor
                .to_host()
                .with_context(|| format!("failed to export checkpoint tensor '{name}'"))?;
            params.push((name.to_string(), base_param_id(name)?.0, host));
            Ok(())
        };
        model.visit_params(&mut ParamVisitor::new(&mut export))?;
        params.sort_by(|a, b| a.0.cmp(&b.0));
        let params: Vec<_> = params
            .iter()
            .map(|(name, id, host)| (name.as_str(), *id, host))
            .collect();
        write_tensor_file(path.as_ref(), super::loader::MAGIC, &header, &params, true)
    }
}
