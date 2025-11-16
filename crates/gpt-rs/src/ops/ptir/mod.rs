//! PTIR domain-specific language for constructing graph snippets.

pub mod axes;
pub mod graph;
pub mod tensor;

pub use axes::axes_iter;
pub use graph::{
    DotAttrs, DotDims, PtirGraph, PtirResults, PtirSession, PtirValue, SnippetEmitter, Tensor,
};
pub use tensor::{tensor, TensorPlaceholder};

use crate::backend::spec::{DType, PortableBackend};

/// Session trait exposing the subset of PTIR helpers required by functional kernels.
pub trait FunctionalSession<'ctx, 'gb, B: PortableBackend + 'static> {
    fn scalar(&self, value: f32) -> Tensor<'ctx, 'gb, B>;
    fn iota(&self, dims: Vec<usize>, axis: usize, dtype: DType) -> Tensor<'ctx, 'gb, B>;
    fn rng_uniform(&self, dims: Vec<usize>, dtype: DType) -> Tensor<'ctx, 'gb, B>;
}

impl<'ctx, 'gb, B: PortableBackend + 'static> FunctionalSession<'ctx, 'gb, B>
    for PtirSession<'ctx, 'gb, B>
{
    fn scalar(&self, value: f32) -> Tensor<'ctx, 'gb, B> {
        PtirSession::scalar(self, value)
    }

    fn iota(&self, dims: Vec<usize>, axis: usize, dtype: DType) -> Tensor<'ctx, 'gb, B> {
        PtirSession::iota(self, dims, axis, dtype)
    }

    fn rng_uniform(&self, dims: Vec<usize>, dtype: DType) -> Tensor<'ctx, 'gb, B> {
        PtirSession::rng_uniform(self, dims, dtype)
    }
}

/// Broadcasts a scalar literal to the provided shape within the PTIR DSL.
pub(crate) fn scalar_broadcast<'ctx, 'gb, B: PortableBackend + 'static>(
    session: &impl FunctionalSession<'ctx, 'gb, B>,
    value: f32,
    shape: &[usize],
) -> Tensor<'ctx, 'gb, B> {
    session.scalar(value).broadcast_to(shape.to_vec())
}

/// Converts flexible operands into PTIR tensors.
pub trait IntoTensor<'ctx, 'gb, B: PortableBackend + 'static> {
    fn into_tensor(self) -> Tensor<'ctx, 'gb, B>;
}

impl<'ctx, 'gb, B: PortableBackend + 'static> IntoTensor<'ctx, 'gb, B> for Tensor<'ctx, 'gb, B> {
    fn into_tensor(self) -> Tensor<'ctx, 'gb, B> {
        self
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> IntoTensor<'ctx, 'gb, B> for &Tensor<'ctx, 'gb, B> {
    fn into_tensor(self) -> Tensor<'ctx, 'gb, B> {
        *self
    }
}

/// Trait enabling [`sqrt`] to work with both tensors and scalars.
pub trait PtirSqrt {
    type Output;

    fn ptir_sqrt(self) -> Self::Output;
}

impl<'ctx, 'gb, B: PortableBackend + 'static> PtirSqrt for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn ptir_sqrt(self) -> Self::Output {
        self.sqrt()
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> PtirSqrt for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn ptir_sqrt(self) -> Self::Output {
        self.sqrt()
    }
}

impl PtirSqrt for f32 {
    type Output = f32;

    fn ptir_sqrt(self) -> Self::Output {
        self.sqrt()
    }
}

/// Computes `sqrt` for both PTIR tensors and scalar literals.
pub fn sqrt<T: PtirSqrt>(value: T) -> T::Output {
    value.ptir_sqrt()
}

/// Applies the error function elementwise to a PTIR tensor.
pub fn erf<'ctx, 'gb, B: PortableBackend + 'static>(
    value: impl IntoTensor<'ctx, 'gb, B>,
) -> Tensor<'ctx, 'gb, B> {
    value.into_tensor().erf()
}
