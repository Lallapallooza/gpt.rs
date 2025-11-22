//! Lightweight wrapper for tensor shapes and dimension bookkeeping.

/// Stores the logical dimensions of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Constructs a new shape from the provided dimensions.
    ///
    /// Panics if `dims` is empty, ensuring every tensor has at least one axis.
    pub fn new<D: Into<Vec<usize>>>(dims: D) -> Self {
        let dims = dims.into();
        assert!(!dims.is_empty(), "shape must have at least one dimension");
        Shape { dims }
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

    /// Asserts that another shape matches exactly, panicking on mismatch.
    pub fn assert_same(&self, other: &Shape) {
        assert_eq!(
            self.dims, other.dims,
            "shape mismatch: {:?} vs {:?}",
            self, other
        );
    }
}
