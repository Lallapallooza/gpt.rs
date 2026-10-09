//! Ops that draw random numbers from an explicit generator state.

use anyhow::{ensure, Result};
use smallvec::SmallVec;

use super::class::{ensure_dtype_class, Class};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::types::{DType, Dim, TensorType};
use crate::Attr;

/// The distribution that random values follow.
#[derive(Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Distribution {
    /// Each value of an integer dtype equally likely, or a float on `[0, 1)`.
    Uniform,
    /// The standard normal distribution, of a float dtype.
    Normal,
}

/// The type of a generator state.
pub fn state_type() -> TensorType {
    TensorType::new(DType::Ui64, [2])
}

/// `tensor.rng_uniform`, `tensor.rng_normal`: a tensor of `shape` whose elements are random, and
/// the state after them.
///
/// - The ops are pure: one state gives the same values on every backend.
/// - A state, `ui64[2]`, is a key and a counter: `[key, counter]`. Its stream is the output of
///   Philox4x32-10 (Random123). Counter block `j` is the 128-bit number `counter + j`, split into
///   four 32-bit words from the least significant one up. The key splits into two words the same
///   way. Each block gives four 32-bit words. An op reads the first words of its stream and
///   returns `[key, counter + blocks read]`, the counter wrapping modulo `2^64`.
/// - Each element takes the next `ceil(bits / 32)` words, the first one least significant, where
///   `bits` is the width of `dtype`, or the significand of a uniform float.
/// - A uniform integer is the low `bits` of its words: each value of the dtype equally likely.
/// - A uniform float with a significand of `p` bits (`bf16`: 8, `f16`: 11, `f32`: 24, `f64`: 53)
///   is the top `p` bits of its words times `2^-p`: on `[0, 1)`, and exact in its dtype.
/// - A normal float takes two uniforms `u1` and `u2` of its dtype's precision, in that order, and
///   is `sqrt(-2 ln(1 - u1)) * cos(2 pi u2)`, rounded to its dtype.
/// - Uniform `ui32` values are the draws of stochastic rounding (`tensor.quantize`).
/// - An extension op: a backend MAY reject it as unimplemented.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.rng_{distribution}",
    operands(state: Tensor),
    results(new_state: Tensor, output: Tensor)
)]
pub struct Rng {
    pub distribution: Distribution,
    pub shape: SmallVec<[Dim; 4]>,
    pub dtype: DType,
}

impl OpDef for Rng {
    fn result_types(
        &self,
        ins: RngOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<RngResults<OwnedTypes>> {
        let state = state_type();
        ensure!(
            *ins.state == state,
            "the state must be {state}, got {}",
            ins.state
        );
        let classes: &[Class] = match self.distribution {
            Distribution::Uniform => &[Class::Int, Class::Float],
            Distribution::Normal => &[Class::Float],
        };
        ensure_dtype_class("dtype", self.dtype, classes)?;
        Ok(RngResults {
            new_state: state,
            output: TensorType {
                dtype: self.dtype,
                dims: self.shape.clone(),
            },
        })
    }
}
