//! Shared shape/runtime helpers used across backends.

use crate::backend::spec::{DimSymbol, Dimension, Shape};

/// Returns static dimensions or maps the first dynamic dimension to caller-defined error type.
pub fn static_dims_or_error<E, F>(shape: &Shape, mut on_dynamic: F) -> Result<Vec<usize>, E>
where
    F: FnMut(&DimSymbol) -> E,
{
    let mut dims = Vec::with_capacity(shape.rank());
    for dim in shape.dims() {
        match dim {
            Dimension::Static(value) => dims.push(*value),
            Dimension::Dynamic(symbol) => return Err(on_dynamic(symbol)),
        }
    }
    Ok(dims)
}

/// Computes `product(dims)` with overflow checking.
pub fn checked_element_count_or_error<E, F>(dims: &[usize], mut on_overflow: F) -> Result<usize, E>
where
    F: FnMut() -> E,
{
    let mut count = 1usize;
    for dim in dims {
        count = count.checked_mul(*dim).ok_or_else(&mut on_overflow)?;
    }
    Ok(count)
}

/// Builds row-major contiguous strides with overflow checking.
pub fn contiguous_strides_or_error<E, F>(
    dims: &[usize],
    mut on_overflow: F,
) -> Result<Vec<usize>, E>
where
    F: FnMut() -> E,
{
    let mut strides = vec![0usize; dims.len()];
    let mut stride = 1usize;
    for axis in (0..dims.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(dims[axis])
            .ok_or_else(&mut on_overflow)?;
    }
    Ok(strides)
}
