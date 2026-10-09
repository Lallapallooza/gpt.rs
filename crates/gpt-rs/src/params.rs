use anyhow::{ensure, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BaseParamId(pub u128);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ModelNamespaceId(pub u128);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ParamKey(pub u128);

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
