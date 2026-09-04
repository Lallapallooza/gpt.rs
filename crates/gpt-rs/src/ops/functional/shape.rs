//! Shape manipulation helpers captured into PTIR graphs.
//!
//! These routines expose common layout operations (reshape/transpose) to functional and model
//! code while keeping all work device-only (no host materialization).

use anyhow::{ensure, Result};

use crate::{capture, functional};

/// Reshapes a tensor while preserving element count.
#[functional]
pub fn reshape(x: &Tensor, dims: &[usize]) -> Result<Tensor> {
    ensure!(!dims.is_empty(), "{FUNCTIONAL}: dims must be non-empty");
    let (in_elems, out_elems) = (x.shape().num_elements(), dims.iter().product::<usize>());
    ensure!(
        in_elems == out_elems,
        "{FUNCTIONAL}: dims {dims:?} must hold the {in_elems} elements of x, got {out_elems}"
    );
    capture!(|x| x.reshape(dims))
}

/// Permutes tensor axes according to `perm`.
#[functional]
pub fn transpose(x: &Tensor, perm: &[usize]) -> Result<Tensor> {
    let rank = x.shape().rank();
    ensure!(
        perm.len() == rank,
        "{FUNCTIONAL}: perm {perm:?} must have one entry per axis of x (rank {rank})"
    );
    let mut seen = vec![false; rank];
    for &axis in perm {
        ensure!(
            axis < rank && !seen[axis],
            "{FUNCTIONAL}: perm {perm:?} must be a permutation of 0..{rank}"
        );
        seen[axis] = true;
    }
    capture!(|x| x.transpose(perm))
}
