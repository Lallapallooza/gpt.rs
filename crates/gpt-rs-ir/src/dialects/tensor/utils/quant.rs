//! Quantization helpers shared by the type rules, the semantics and the lowering of the
//! quantization ops.

use anyhow::{bail, ensure, Result};

use crate::ir::{Dim, Ranked};

/// The extent of the scales along an axis of `extent` elements in blocks of `block` elements: one
/// scale per block, the last of which may hold fewer.
pub fn scale_extent(extent: usize, block: usize) -> usize {
    extent.div_ceil(block)
}

/// The scale extents of a tensor of `dims` with `block`, [`scale_extent`] on each axis. An extent
/// known only during the run takes blocks of 1, whose scales have its extent.
pub fn scale_dims(dims: &[Dim], block: &[usize]) -> Result<Ranked<Dim>> {
    ensure!(
        block.len() == dims.len(),
        "block {block:?} must have one extent per axis of rank {}",
        dims.len()
    );
    let mut scales = Ranked::new();
    for (dim, &extent) in dims.iter().zip(block) {
        ensure!(extent > 0, "block {block:?} must be positive");
        scales.push(match dim {
            Dim::Static(size) => Dim::Static(scale_extent(*size, extent)),
            Dim::Dynamic(_) if extent == 1 => dim.clone(),
            Dim::Dynamic(_) => bail!("a block of {extent} needs a static extent, got {dim}"),
        });
    }
    Ok(scales)
}
