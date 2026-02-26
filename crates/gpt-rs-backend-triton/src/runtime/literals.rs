use gpt_rs::backend::hashing::{fnv1a_bytes, fnv1a_init};
use gpt_rs::backend::spec::Dimension;
use gpt_rs::profiling;

use super::*;

fn literal_cache_key(literal: &TensorLiteral) -> u64 {
    let mut hash = fnv1a_init();
    hash = fnv1a_bytes(hash, &[literal.spec.dtype as u8]);
    hash = fnv1a_bytes(hash, &(literal.spec.shape.rank() as u64).to_le_bytes());
    for dim in literal.spec.shape.dims() {
        match dim {
            Dimension::Static(value) => {
                hash = fnv1a_bytes(hash, &[0]);
                hash = fnv1a_bytes(hash, &(*value as u64).to_le_bytes());
            }
            Dimension::Dynamic(_) => {
                hash = fnv1a_bytes(hash, &[1]);
            }
        }
    }
    fnv1a_bytes(hash, literal.bytes.as_ref())
}

impl TritonExecutor {
    pub(super) fn materialize_literal(
        &self,
        driver: &Arc<CudaDriver>,
        literal: &TensorLiteral,
        output: Option<TritonTensor>,
    ) -> BackendResult<TritonTensor> {
        let key = literal_cache_key(literal);
        let cached = {
            let cache = lock_named(&self.literal_tensors, "literal_tensors")?;
            cache.get(&key).cloned()
        };

        let source = match cached {
            Some(tensor) => tensor,
            None => {
                profiling::cache_event("triton_backend.literal_cache_miss");
                let tensor = ops::literal_to_tensor(driver, literal, None)?;
                let mut cache = lock_named(&self.literal_tensors, "literal_tensors")?;
                cache.entry(key).or_insert_with(|| tensor.clone()).clone()
            }
        };

        match output {
            Some(dst) => {
                if dst.spec != literal.spec {
                    return Err(BackendError::execution(format!(
                        "literal output spec mismatch: expected {:?}, got {:?}",
                        literal.spec, dst.spec
                    )));
                }
                let bytes = literal.spec.byte_len().ok_or_else(|| {
                    BackendError::execution("cannot compute byte length for cached literal")
                })?;
                if bytes != source.buffer.bytes() || bytes != dst.buffer.bytes() {
                    return Err(BackendError::execution(
                        "cached literal byte length mismatch against destination",
                    ));
                }
                if dst.buffer.device_ptr() != source.buffer.device_ptr() {
                    driver.copy_device_to_device(
                        dst.buffer.device_ptr(),
                        source.buffer.device_ptr(),
                        bytes,
                    )?;
                }
                Ok(dst)
            }
            None => Ok(source),
        }
    }
}
