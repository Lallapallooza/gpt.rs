//! Elementwise functionals beyond the single ops of [`DeviceTensor`](crate::DeviceTensor).

use crate::functional;

/// `min(max(x, min), max)`. Either bound is optional. Returns `x` itself when both bounds are
/// absent.
#[functional]
pub fn clamp(x: &Tensor, min: Option<&Tensor>, max: Option<&Tensor>) -> Tensor {
    let lower = match min {
        Some(min) => x.maximum(min),
        None => x.clone(),
    };
    match max {
        Some(max) => lower.minimum(max),
        None => lower,
    }
}
