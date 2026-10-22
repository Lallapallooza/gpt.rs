//! Picking tokens from logits on the device, so a decode step downloads one id instead of the
//! logits.

use crate::tensor::{DType, DeviceTensor, HostTensor};
use crate::{ensure_dtype, ensure_rank, functional};

/// Draws one index (`si32 [N]`) per row of `logits` (`[N, V]`) from
/// `softmax(logits / temperature)`, restricted to the `top_k` largest logits of the row when it is
/// set. The draw adds Gumbel noise from the generator `state` (`ui64 [2]`) to the scaled logits
/// and takes the largest (Gumbel-max), which needs no sort and no running sum. Returns the indices
/// and the state after the draw.
#[functional]
pub fn sample(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    state: &Tensor,
) -> (Tensor, Tensor) {
    ensure_rank!(logits, 2);
    ensure_dtype!(logits, F32);
    ensure_dtype!(state, Ui64);
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "{FUNCTIONAL}: temperature must be finite and positive, got {temperature}"
    );
    let (rows, vocab) = (logits.size(0), logits.size(1));
    let scaled = logits / temperature;
    let (candidates, top_indices) = match top_k {
        Some(count) if count < vocab => {
            assert!(count > 0, "{FUNCTIONAL}: top_k must be positive");
            let (values, indices) = scaled.top_k(count, -1);
            (values, Some(indices))
        }
        _ => (scaled, None),
    };
    let (uniform, next_state) = state.rng_uniform(candidates.shape().dims(), DType::F32);
    let gumbel = -(-uniform.log()).log();
    let choice = (candidates + gumbel).argmax(-1);
    let Some(indices) = top_indices else {
        return (choice, next_state);
    };
    // The chosen column of each row of `indices`, at `row * kept + choice` of its flat elements.
    let kept = indices.size(1);
    let row_offsets: Vec<i32> = (0..rows).map(|row| (row * kept) as i32).collect();
    let offsets = HostTensor::si32([rows], row_offsets).expect("one offset per row");
    let flat = DeviceTensor::constant(&choice, &offsets);
    let token = indices.reshape([rows * kept]).take(&(choice + flat));
    (token, next_state)
}
