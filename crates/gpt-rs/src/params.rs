use std::collections::HashSet;

use anyhow::{ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::module::{Module, ParamVisitorMut, TensorRole};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BaseParamId(pub u128);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ModelNamespaceId(pub u128);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParamKey(pub u128);

#[derive(Clone, Debug)]
pub struct BoundParam {
    pub name: String,
    pub role: TensorRole,
    pub base_id: BaseParamId,
    pub key: ParamKey,
}

pub trait ParamSource<B: PortableBackend + 'static>: Send + Sync {
    fn load(&self, base_id: BaseParamId) -> Result<B::TensorHandle>;
}

pub fn base_param_id(name: &str) -> Result<BaseParamId> {
    ensure!(name.is_ascii(), "param name must be ASCII, got '{name}'");
    let hash = blake3::hash(name.as_bytes());
    let raw: [u8; 16] = hash.as_bytes()[0..16]
        .try_into()
        .expect("blake3 hash prefix length mismatch");
    Ok(BaseParamId(u128::from_le_bytes(raw)))
}

pub fn param_key(namespace: ModelNamespaceId, base_id: BaseParamId) -> ParamKey {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"gpt-rs:param-key:v1");
    hasher.update(&namespace.0.to_le_bytes());
    hasher.update(&base_id.0.to_le_bytes());
    let hash = hasher.finalize();
    let raw: [u8; 16] = hash.as_bytes()[0..16]
        .try_into()
        .expect("blake3 hash prefix length mismatch");
    ParamKey(u128::from_le_bytes(raw))
}

pub fn bind_namespace<B: PortableBackend + 'static, M: Module<B>>(
    module: &mut M,
    namespace: ModelNamespaceId,
) -> Result<Vec<BoundParam>> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut bound: Vec<BoundParam> = Vec::new();

    let mut bind_one =
        |name: &str, role: TensorRole, tensor: &mut crate::tensor::DeviceTensor<B>| -> Result<()> {
            ensure!(
                seen.insert(name.to_string()),
                "duplicate parameter name '{name}'"
            );
            let base_id = base_param_id(name)?;
            let key = param_key(namespace, base_id);
            *tensor = tensor.as_param_with_id(key.0)?;
            bound.push(BoundParam {
                name: name.to_string(),
                role,
                base_id,
                key,
            });
            Ok(())
        };

    let mut visitor = ParamVisitorMut::new(&mut bind_one);

    module.visit_params_mut(&mut visitor)?;
    bound.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(bound)
}
