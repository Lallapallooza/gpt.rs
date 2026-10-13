//! Semantics of the random ops.

use anyhow::{bail, Result};

use super::kernels::{from_f64, from_ints};
use crate::dialects::arith;
use crate::dialects::arith::math::{Draw, Drawn};
use crate::dialects::tensor::ir::*;
use crate::dialects::tensor::utils::random::uniform_bits;
use crate::host::{HostTensor, TensorData};
use crate::ir::{DType, EvalCx, Host, IntFormat, NoRegions, OwnedHost};

/// One block of Philox4x32-10: the four words of `counter` under `key`.
fn philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    const MULTIPLIERS: [u64; 2] = [0xD251_1F53, 0xCD9E_8D57];
    const KEY_STEPS: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];
    let (mut c, mut k) = (counter, key);
    for round in 0..10 {
        if round > 0 {
            k = [
                k[0].wrapping_add(KEY_STEPS[0]),
                k[1].wrapping_add(KEY_STEPS[1]),
            ];
        }
        let p0 = MULTIPLIERS[0] * u64::from(c[0]);
        let p1 = MULTIPLIERS[1] * u64::from(c[2]);
        c = [
            (p1 >> 32) as u32 ^ c[1] ^ k[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ c[3] ^ k[1],
            p0 as u32,
        ];
    }
    c
}

/// The first `count` words of the stream of the state `(key, counter)`, and the counter after the
/// blocks they take.
fn stream(key: u64, counter: u64, count: usize) -> (Vec<u32>, u64) {
    let key = [key as u32, (key >> 32) as u32];
    let blocks = count.div_ceil(4);
    let mut words = Vec::with_capacity(blocks * 4);
    for j in 0..blocks {
        let block = u128::from(counter) + j as u128;
        let words_of = [0, 32, 64, 96].map(|shift| (block >> shift) as u32);
        words.extend(philox(words_of, key));
    }
    words.truncate(count);
    (words, counter.wrapping_add(blocks as u64))
}

/// A uniform value on `[0, 1)` with `bits` significant bits, from the top bits of `words`, the
/// first word least significant.
fn unit(words: &[u32], bits: u32) -> f64 {
    let value = words
        .iter()
        .rev()
        .fold(0u64, |acc, &w| acc << 32 | u64::from(w));
    let total = 32 * words.len() as u32;
    (value >> (total - bits)) as f64 * 2f64.powi(-(bits as i32))
}

/// The `per` words of element `index` of the stream of the state `(key, counter)`: the words at
/// positions `index * per ..`, word `p` being word `p % 4` of block `counter + p / 4`.
pub(crate) fn element_words(key: u64, counter: u64, index: u64, per: usize) -> Vec<u32> {
    let key = [key as u32, (key >> 32) as u32];
    (0..per as u64)
        .map(|w| {
            let position = u128::from(index) * per as u128 + u128::from(w);
            let block = u128::from(counter) + position / 4;
            let words_of = [0, 32, 64, 96].map(|shift| (block >> shift) as u32);
            philox(words_of, key)[(position % 4) as usize]
        })
        .collect()
}

/// The integer that `words` give, the first word least significant, wrapped to `int`.
fn int_from_words(int: IntFormat, words: &[u32]) -> i128 {
    let bits = words
        .iter()
        .rev()
        .fold(0i128, |acc, &x| acc << 32 | i128::from(x));
    int.wrap(bits)
}

/// An integer takes the low bits of its words, and a float the top bits of its significand.
impl Draw for arith::RngUniform {
    fn words_per_element(dtype: DType) -> Result<usize> {
        Ok(match dtype.int_format() {
            Some(int) => (int.bits as usize).div_ceil(32),
            None => (uniform_bits(dtype)? as usize).div_ceil(32),
        })
    }

    fn draw(dtype: DType, words: &[u32]) -> Result<Drawn> {
        Ok(match dtype.int_format() {
            Some(int) => Drawn::Int(int_from_words(int, words)),
            None => Drawn::Float(unit(words, uniform_bits(dtype)?)),
        })
    }
}

/// Box-Muller on two uniforms, with `1 - u` in (0, 1] under the logarithm.
impl Draw for arith::RngNormal {
    fn words_per_element(dtype: DType) -> Result<usize> {
        Ok(2 * (uniform_bits(dtype)? as usize).div_ceil(32))
    }

    fn draw(dtype: DType, words: &[u32]) -> Result<Drawn> {
        let bits = uniform_bits(dtype)?;
        let per = words.len() / 2;
        let (u1, u2) = (1.0 - unit(&words[..per], bits), unit(&words[per..], bits));
        Ok(Drawn::Float(
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos(),
        ))
    }
}

/// The new state and the elements of `dims` in `dtype` that `T` draws from the state `state`.
pub(crate) fn draw_tensor<T: Draw>(
    state: &HostTensor,
    dtype: DType,
    dims: Vec<usize>,
) -> Result<(HostTensor, HostTensor)> {
    let TensorData::Ui64(state) = &state.data else {
        bail!("the state must be ui64, got {}", state.dtype());
    };
    let (key, counter) = (state[0], state[1]);
    let count: usize = dims.iter().product();
    let per = T::words_per_element(dtype)?;
    let (words, counter) = stream(key, counter, count * per);
    let drawn = words
        .chunks(per)
        .map(|element| T::draw(dtype, element))
        .collect::<Result<Vec<_>>>()?;
    let data = if dtype.int_format().is_some() {
        from_ints(
            dtype,
            drawn.into_iter().map(|d| match d {
                Drawn::Int(x) => x,
                Drawn::Float(_) => unreachable!("integers draw integers"),
            }),
        )?
    } else {
        from_f64(
            dtype,
            drawn.into_iter().map(|d| match d {
                Drawn::Float(x) => x,
                Drawn::Int(_) => unreachable!("floats draw floats"),
            }),
        )?
    };
    Ok((
        HostTensor::new(vec![2], TensorData::Ui64(vec![key, counter].into()))?,
        HostTensor::new(dims, data)?,
    ))
}

/// The semantics of every [`Rng`] op.
pub(crate) fn rng<T: Rng>(
    op: &T,
    ins: RngOperands<Host<'_>>,
    cx: &mut EvalCx<'_, NoRegions>,
) -> Result<RngResults<OwnedHost>> {
    let (new_state, output) = draw_tensor::<T::Scalar>(ins.state, op.dtype, cx.result_dims(1)?)?;
    Ok(RngResults { new_state, output })
}
