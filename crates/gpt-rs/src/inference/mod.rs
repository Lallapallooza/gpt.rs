pub mod generate;
pub mod sampler;

use anyhow::Result;

use crate::backend::spec::PortableBackend;
use crate::ops::functional::DecodeKvCache;
use crate::tensor::Tensor;

pub trait CausalLanguageModel<B: PortableBackend + 'static> {
    fn context_length(&self) -> usize;
    fn num_layers(&self) -> usize;

    fn forward(&self, tokens: &[usize]) -> Result<Tensor>;

    fn forward_with_decode_cache(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
    ) -> Result<Tensor>;

    fn forward_with_decode_cache_with_capacity(
        &self,
        tokens: &[usize],
        position_offset: usize,
        caches: &mut [Option<DecodeKvCache<B>>],
        capacity: usize,
    ) -> Result<Tensor>;
}
