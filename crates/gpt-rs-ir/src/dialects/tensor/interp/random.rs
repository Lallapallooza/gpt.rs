//! Semantics of the ops in `ir/random.rs`.

use anyhow::{bail, Result};

use super::kernels::{from_f64, from_ints};
use crate::dialects::tensor::ir::{Distribution, Rng, RngOperands, RngResults};
use crate::host::{HostTensor, TensorData};
use crate::op::{Eval, EvalCx, Host, NoRegions, OwnedHost};
use crate::types::DType;

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

/// The bits of the significand of a float dtype that has uniform floats.
fn significand_bits(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F64 | DType::F32 | DType::F16 | DType::Bf16 => {
            Ok(super::kernels::significand_bits(dtype).expect("a float dtype"))
        }
        other => bail!("{other} has no uniform floats"),
    }
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

impl Eval for Rng {
    fn eval(
        &self,
        ins: RngOperands<Host<'_>>,
        cx: &mut EvalCx<'_, NoRegions>,
    ) -> Result<RngResults<OwnedHost>> {
        let TensorData::Ui64(state) = &ins.state.data else {
            bail!("the state must be ui64, got {}", ins.state.dtype());
        };
        let (key, counter) = (state[0], state[1]);
        let dims = cx.result_dims(1)?;
        let count: usize = dims.iter().product();
        let dtype = self.dtype;
        let (data, counter) = match (self.distribution, dtype.int_format()) {
            (Distribution::Uniform, Some(int)) => {
                // Each element takes the low bits of its words.
                let per = (int.bits as usize).div_ceil(32);
                let (words, counter) = stream(key, counter, count * per);
                let values = words.chunks(per).map(|w| {
                    let bits = w
                        .iter()
                        .rev()
                        .fold(0i128, |acc, &x| acc << 32 | i128::from(x));
                    int.wrap(bits)
                });
                (from_ints(dtype, values)?, counter)
            }
            (Distribution::Uniform, None) => {
                let bits = significand_bits(dtype)?;
                let per = (bits as usize).div_ceil(32);
                let (words, counter) = stream(key, counter, count * per);
                let values = words.chunks(per).map(|w| unit(w, bits));
                (from_f64(dtype, values)?, counter)
            }
            (Distribution::Normal, _) => {
                // Box-Muller on two uniforms, with `1 - u` in (0, 1] under the logarithm.
                let bits = significand_bits(dtype)?;
                let per = (bits as usize).div_ceil(32);
                let (words, counter) = stream(key, counter, count * 2 * per);
                let values = words.chunks(2 * per).map(|w| {
                    let (u1, u2) = (1.0 - unit(&w[..per], bits), unit(&w[per..], bits));
                    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
                });
                (from_f64(dtype, values)?, counter)
            }
        };
        Ok(RngResults {
            new_state: HostTensor::new(vec![2], TensorData::Ui64(vec![key, counter].into()))?,
            output: HostTensor::new(dims, data)?,
        })
    }
}
