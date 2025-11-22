//! Shape manipulation helpers captured into PTIR graphs.
//!
//! These routines expose common layout operations (reshape/transpose) to functional and model
//! code while keeping all work device-only (no host materialization).

use anyhow::{bail, ensure, Result};
use gpt_rs_macros::{capture_ptir, support_runtime_overload};

use crate::backend::spec::PortableBackend;
use crate::ops::functional::common::CaptureIntoDeviceTensor;
use crate::tensor::DeviceTensor;

/// Reshapes a tensor while preserving element count.
#[support_runtime_overload]
pub fn reshape<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    dims: &[usize],
) -> Result<DeviceTensor<B>> {
    ensure!(!dims.is_empty(), "reshape dims must be non-empty");
    let in_elems = x.shape().num_elements();
    let out_elems: usize = dims.iter().product();
    ensure!(
        in_elems == out_elems,
        "reshape element count mismatch: in {} vs out {}",
        in_elems,
        out_elems
    );

    capture_ptir!({ x }, |_session| Ok(x.reshape(dims).id()))?
        .into_device_tensor(x.requires_grad_flag())
}

/// Permutes tensor axes according to `perm`.
#[support_runtime_overload]
pub fn transpose<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    perm: &[usize],
) -> Result<DeviceTensor<B>> {
    let rank = x.shape().rank();
    if perm.len() != rank {
        bail!("transpose expects perm rank {}, got {}", rank, perm.len());
    }

    let mut seen = vec![false; rank];
    for &axis in perm {
        if axis >= rank {
            bail!("transpose axis {} out of range for rank {}", axis, rank);
        }
        if seen[axis] {
            bail!("transpose perm contains duplicate axis {}", axis);
        }
        seen[axis] = true;
    }

    capture_ptir!({ x }, |_session| Ok(x.transpose(perm).id()))?
        .into_device_tensor(x.requires_grad_flag())
}
