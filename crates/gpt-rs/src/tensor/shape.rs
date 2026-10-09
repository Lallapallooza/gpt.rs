//! Lightweight wrapper for tensor shapes and dimension bookkeeping.

use smallvec::SmallVec;

/// Stores the logical dimensions of a tensor, inline up to rank 4.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}

impl Shape {
    /// Constructs a shape from `dims`. Empty `dims` is the shape of a scalar.
    pub fn new(dims: impl AsRef<[usize]>) -> Self {
        Shape {
            dims: SmallVec::from_slice(dims.as_ref()),
        }
    }

    /// Borrow the raw dimension slice for downstream calculations.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the rank (number of axes) of the shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Computes the total number of elements implied by the shape.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(dims: I) -> Self {
        Shape {
            dims: dims.into_iter().collect(),
        }
    }
}
