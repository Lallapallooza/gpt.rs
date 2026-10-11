//! Constructors of the layout ops.

use crate::dialects::tensor::ir::*;
use crate::types::{Dim, Ranked};

impl Reshape {
    /// A reshape to `shape`.
    pub fn new(shape: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        Self {
            shape: shape.into_iter().map(Into::into).collect(),
        }
    }
}

impl Transpose {
    /// A transpose by `perm`.
    pub fn new(perm: impl IntoIterator<Item = usize>) -> Self {
        Self {
            perm: perm.into_iter().collect(),
        }
    }
}

impl BroadcastTo {
    /// A broadcast to `shape`.
    pub fn new(shape: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        Self {
            shape: shape.into_iter().map(Into::into).collect(),
        }
    }

    /// The axis of an input of rank `rank` that result axis `axis` reads. The shape aligns the
    /// input with its trailing axes, so an axis that the broadcast adds reads none.
    pub fn source_axis(&self, axis: usize, rank: usize) -> Option<usize> {
        axis.checked_sub(self.shape.len() - rank)
    }
}

impl Slice {
    /// A slice with steps of 1.
    pub fn new(starts: impl AsRef<[usize]>, sizes: impl AsRef<[usize]>) -> Self {
        Self {
            starts: starts.as_ref().iter().copied().collect(),
            sizes: sizes.as_ref().iter().copied().collect(),
            strides: Ranked::new(),
        }
    }

    /// The step along `axis`.
    pub fn stride(&self, axis: usize) -> usize {
        self.strides.get(axis).copied().unwrap_or(1)
    }

    /// The extent of input that the slice spans along `axis`: from its first element to its last.
    pub fn span(&self, axis: usize) -> usize {
        match self.sizes[axis] {
            0 => 0,
            size => (size - 1) * self.stride(axis) + 1,
        }
    }
}
