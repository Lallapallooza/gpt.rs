//! Backend-agnostic functional operators built on top of device tensors.
//!
//! Modules provide reusable attention, normalization, embedding, and elementwise routines that
//! capture lazy graphs rather than executing eagerly. Callers can reach for these helpers or the
//! [`DeviceTensorOps`] extension trait to write concise forward
//! passes while staying compliant with backend restrictions.
//!
//! ## Backend Parameter Convention
//!
//! Functions annotated with `#[support_runtime_overload]` accept a `_backend: &B` parameter
//! as their first argument. **This parameter is required by the macro system** and is used for:
//!
//! 1. **Generic type resolution** - The macro generates a context struct that needs the backend type `B`
//! 2. **Registry lookup** - When a custom implementation is registered, the macro uses this to
//!    determine which backend's registry to consult
//!
//! The backend is **not directly used** in the function body because:
//! - It's already embedded in the `DeviceTensor<B>` arguments (accessible via `.backend()`)
//! - Validation ensures all tensors share the same backend instance
//!
//! **Example:**
//! ```rust,ignore
//! // The macro transforms this:
//! #[support_runtime_overload]
//! pub fn matmul<B: PortableBackend>(
//!     _backend: &B,  // ← Required by macro, used for type resolution
//!     a: &DeviceTensor<B>,
//!     b: &DeviceTensor<B>,
//! ) -> Result<DeviceTensor<B>>
//!
//! // Into wrapper code that checks registries and calls implementations
//! ```

#![deny(clippy::disallowed_methods, clippy::disallowed_types)]

pub mod activation;
pub mod attention;
pub(crate) mod common;
pub mod conv;
pub mod embedding;
pub mod linalg;
pub mod normalization;
pub mod pooling;
pub mod registry;
pub mod runtime;
pub mod shape;
pub mod stochastic;
pub mod tensor_ops;

pub use activation::*;
pub use attention::*;
pub use common::{CaptureIntoDeviceTensor, DeviceTensorOps};
pub use conv::*;
pub use embedding::*;
pub use linalg::*;
pub use normalization::*;
pub use pooling::*;
pub use registry::*;
pub use runtime::*;
pub use shape::*;
pub use stochastic::*;
pub use tensor_ops::*;

pub fn tensor_spec_from_device<B: crate::backend::spec::PortableBackend + 'static>(
    tensor: &crate::tensor::DeviceTensor<B>,
) -> crate::backend::spec::TensorSpec {
    common::tensor_spec_from_device(tensor)
}

pub fn resolve_graph_from_tensors<B: crate::backend::spec::PortableBackend + 'static>(
    tensors: &[&crate::tensor::DeviceTensor<B>],
) -> Option<std::sync::Arc<crate::ops::graph::GraphArena<B>>> {
    common::resolve_graph_from_tensors(tensors)
}
