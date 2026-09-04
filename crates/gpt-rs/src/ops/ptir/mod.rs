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

/// Broadcasts a scalar literal to the provided shape within the PTIR DSL.
pub(crate) fn scalar_broadcast<'ctx, 'gb, B: PortableBackend + 'static>(
    session: &PtirSession<'ctx, 'gb, B>,
    value: f32,
    shape: &[usize],
) -> Tensor<'ctx, 'gb, B> {
    session.scalar(value).broadcast_to(shape.to_vec())
}

/// Builds an `[n, n]` lower-triangular selection: `inside` where `col <= row + diagonal`, and
/// `outside` elsewhere. With `inside = 1` and `outside = 0`, this is
/// `torch.tril(torch.ones(n, n), diagonal)`.
pub(crate) fn tril_mask<'ctx, 'gb, B: PortableBackend + 'static>(
    session: &PtirSession<'ctx, 'gb, B>,
    n: usize,
    diagonal: i64,
    inside: f32,
    outside: f32,
) -> Tensor<'ctx, 'gb, B> {
    // Row `i` and column `j` of the result read position `(i + r, j + c)` of a larger plane. So
    // `row >= col` in that plane means `i + diagonal >= j`.
    let (r, c) = (diagonal.max(0) as usize, (-diagonal).max(0) as usize);
    let plane = [n + r, n + c];
    let rows = session.iota(plane, 0, DType::Si32);
    let keep = rows.greater_equal(&session.iota(plane, 1, DType::Si32));
    Tensor::select(
        &keep,
        &scalar_broadcast(session, inside, &plane),
        &scalar_broadcast(session, outside, &plane),
    )
    .slice(vec![r, c], vec![n, n])
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
