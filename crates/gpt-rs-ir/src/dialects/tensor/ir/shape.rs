//! Shape helpers shared by the tensor ops' type rules, semantics and passes.

use anyhow::{ensure, Result};
use smallvec::SmallVec;

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

/// Whether `perm` names every axis below `rank` once.
pub fn is_permutation(perm: &[usize], rank: usize) -> bool {
    let mut seen = SmallVec::<[bool; 8]>::from_elem(false, rank);
    perm.len() == rank
        && perm
            .iter()
            .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true))
}

/// The axes below `rank` that neither `a` nor `b` names, in order.
pub fn free_axes(rank: usize, a: &[usize], b: &[usize]) -> SmallVec<[usize; 4]> {
    (0..rank)
        .filter(|axis| !a.contains(axis) && !b.contains(axis))
        .collect()
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
