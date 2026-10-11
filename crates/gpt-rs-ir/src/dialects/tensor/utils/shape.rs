//! Shape helpers shared by the tensor ops' type rules, semantics and passes.

use anyhow::{ensure, Result};

/// Axis `axis` of a tensor of `rank`: `axis` itself, or `axis + rank` when it is negative and so
/// counts from the end. Fails outside `-rank..rank`.
pub fn axis_index(axis: i64, rank: usize) -> Result<usize> {
    let signed = rank as i64;
    ensure!(
        (-signed..signed).contains(&axis),
        "axis {axis} is out of range for rank {rank}"
    );
    Ok(axis.rem_euclid(signed) as usize)
}

/// Checks that a block of `size` fits in an axis of `extent`.
pub fn check_size(axis: usize, size: usize, extent: usize) -> Result<()> {
    ensure!(
        size <= extent,
        "size {size} exceeds extent {extent} on axis {axis}"
    );
    Ok(())
}

/// Checks that the window `start..start + size` lies within `0..extent`.
pub fn check_window(axis: usize, start: usize, size: usize, extent: usize) -> Result<()> {
    ensure!(
        start.checked_add(size).is_some_and(|end| end <= extent),
        "start {start} + size {size} exceeds extent {extent} on axis {axis}"
    );
    Ok(())
}

/// `values[axis]` for each of `axes`.
pub fn pick<T: Copy>(values: &[T], axes: &[usize]) -> Vec<T> {
    axes.iter().map(|&axis| values[axis]).collect()
}
