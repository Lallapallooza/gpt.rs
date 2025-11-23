//! Portable activation helpers that operate on host tensors.
//!
//! These functions are placeholders for host-side fallbacks. They currently bubble up clear
//! errors so call sites know the backend implementation is missing.

use crate::backend::spec::PortableBackend;
use crate::tensor::Tensor;
use anyhow::{bail, Result};

/// Applies the GELU non-linearity using a host tensor fallback.
///
/// The portable backend does not yet provide a host implementation, so the function returns an
/// error explaining the missing support. Layers call into the functional graph-based GELU instead.
pub fn gelu(_backend: &impl PortableBackend, _x: &Tensor) -> Result<Tensor> {
    bail!("gelu is not available on the portable backend yet")
}
