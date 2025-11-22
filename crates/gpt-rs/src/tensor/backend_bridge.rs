//! Helper utilities for translating frontend tensor metadata into backend representations.

use super::{dtype::DType, shape::Shape, spec_utils};
use crate::backend::spec::DType as BackendDType;
use crate::backend::spec::Shape as BackendShape;

/// Clones the backing shape as a backend descriptor.
pub(super) fn spec_shape_from_shape(shape: &Shape) -> BackendShape {
    spec_utils::backend_shape_from_shape(shape)
}

/// Maps the frontend dtype into the backend representation.
pub(super) fn to_backend_dtype(dtype: DType) -> BackendDType {
    spec_utils::backend_dtype(dtype)
}
