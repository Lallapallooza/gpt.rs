//! Writing the draws of `arith.rng_uniform` as `arith`: the integer Philox block, then the element
//! it gives, bit for bit.

use anyhow::Result;

use crate::dialects::arith::math::Draw;
use crate::dialects::arith::{
    Add, And, Less, Mul, Or, RngUniform, RngUniformOp, ScalarBuilder, Select, ShiftLeft,
    ShiftRightLogical, Xor,
};
use crate::dialects::tensor::utils::random::{
    uniform_bits, BLOCK_WORDS, PHILOX_KEY_STEPS, PHILOX_MULTIPLIERS, PHILOX_ROUNDS,
};
use crate::ir::{DType, Value};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// `arith.rng_uniform` becomes its draw in `arith`: the Philox4x32-10 block that holds the
/// element's words of the stream, by integer ops on ui64 values of 32 bits each, the words picked
/// from it, and the element they give: the low bits of the words for an integer, wrapped to its
/// dtype, and for a float as many top bits of the words as its significand has, scaled into
/// `[0, 1)`. The words of one element lie in one block, since their count divides the block's.
#[pattern]
struct ExpandRngUniform;

impl Pattern for ExpandRngUniform {
    type Root = RngUniform;

    fn rewrite(&self, draw: RngUniformOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let dtype = draw.dtype;
        require!(
            let Ok(per) = RngUniform::words_per_element(dtype),
            "rng_uniform draws no {dtype} values"
        );
        let state = (draw.key().id(), draw.counter().id(), draw.index().id());
        let mut body = ScalarBuilder { builder: rewriter };
        let (bits, width) = match per {
            1 => {
                let [only] = element_words(&mut body, state);
                (only, 32)
            }
            2 => {
                let [low, high] = element_words(&mut body, state);
                let thirty_two = body.int(DType::Ui64, 32);
                let shifted = body.create(ShiftLeft, (high, thirty_two));
                (body.create(Or, (shifted, low)), 64)
            }
            _ => unreachable!("a uniform element takes one or two words"),
        };
        let value = uniform(&mut body, dtype, bits, width)
            .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        rewriter.replace_op(&draw, value);
        Ok(())
    }
}

/// The `PER` words of the element at `index` of the stream of `key` and `counter`, each in a ui64,
/// for `PER` a power of two that divides the block's words: words `index * PER ..` of the stream,
/// word `p` being word `p mod 4` of block `counter + p / 4`. Shifts of `index` give the block and
/// the place in it, exactly for every index.
fn element_words<const PER: usize>(
    body: &mut ScalarBuilder<Rewriter>,
    (key, counter, index): (Value, Value, Value),
) -> [Value; PER] {
    let element = body.cast(index, DType::Ui64);
    let block_shift = (BLOCK_WORDS / PER).trailing_zeros();
    let block_shift_value = body.int(DType::Ui64, i128::from(block_shift));
    let block_index = body.create(ShiftRightLogical, (element, block_shift_value));
    let low = body.create(Add, (counter, block_index));
    // The block's number is `counter + block_index` in 128 bits: the carry is its third word.
    let wrapped = body.create(Less, (low, counter));
    let one = body.int(DType::Ui64, 1);
    let zero = body.int(DType::Ui64, 0);
    let carry = body.create(Select, (wrapped, one, zero));
    let mask = body.int(DType::Ui64, i128::from(u32::MAX));
    let thirty_two = body.int(DType::Ui64, 32);
    let counter_words = [
        body.create(And, (low, mask)),
        body.create(ShiftRightLogical, (low, thirty_two)),
        carry,
        zero,
    ];
    let key_words = [
        body.create(And, (key, mask)),
        body.create(ShiftRightLogical, (key, thirty_two)),
    ];
    let block = philox(body, (counter_words, key_words), (mask, thirty_two));
    // The first word's place in the block: `index * PER mod 4`, from the low bits of `index`.
    let word_shift = body.int(DType::Ui64, i128::from(PER.trailing_zeros()));
    let first = body.create(ShiftLeft, (element, word_shift));
    let last = body.int(DType::Ui64, BLOCK_WORDS as i128 - 1);
    let offset = body.create(And, (first, last));
    std::array::from_fn(|taken| {
        let step = body.int(DType::Ui64, taken as i128);
        let position = body.create(Add, (offset, step));
        body.pick(position, &block)
    })
}

/// One block of Philox4x32-10 of the counter words `counter` under the key words `key`, each a
/// ui64 below `2^32`, with the constants `mask`, the low 32 bits, and `thirty_two`: the products
/// of 32 by 32 bits fit a ui64 whole.
fn philox(
    body: &mut ScalarBuilder<Rewriter>,
    (counter, key): ([Value; BLOCK_WORDS], [Value; 2]),
    (mask, thirty_two): (Value, Value),
) -> [Value; BLOCK_WORDS] {
    let multipliers =
        PHILOX_MULTIPLIERS.map(|multiplier| body.int(DType::Ui64, i128::from(multiplier)));
    let mut block = counter;
    for round in 0..PHILOX_ROUNDS {
        // Each round adds the steps to the key, modulo 2^32.
        let mut round_key = key;
        if round > 0 {
            for (key_word, step) in round_key.iter_mut().zip(PHILOX_KEY_STEPS) {
                let added = body.int(DType::Ui64, i128::from(step.wrapping_mul(round)));
                let sum = body.create(Add, (*key_word, added));
                *key_word = body.create(And, (sum, mask));
            }
        }
        let first_product = body.create(Mul, (multipliers[0], block[0]));
        let second_product = body.create(Mul, (multipliers[1], block[2]));
        let second_high = body.create(ShiftRightLogical, (second_product, thirty_two));
        let first_high = body.create(ShiftRightLogical, (first_product, thirty_two));
        let second_mixed = body.create(Xor, (second_high, block[1]));
        let first_mixed = body.create(Xor, (first_high, block[3]));
        block = [
            body.create(Xor, (second_mixed, round_key[0])),
            body.create(And, (second_product, mask)),
            body.create(Xor, (first_mixed, round_key[1])),
            body.create(And, (first_product, mask)),
        ];
    }
    block
}

/// The element of `dtype` that the low `width` bits of the ui64 `bits` give, with the stream's
/// first word least significant. An integer takes the low bits, as two's complement when it is
/// signed. A float takes as many top bits as its significand has, as an integer, and scales them
/// into `[0, 1)` by a power of two. It scales in f32 when the integer fits there, as it does for
/// every narrower float, and in f64 otherwise.
fn uniform(
    body: &mut ScalarBuilder<Rewriter>,
    dtype: DType,
    bits: Value,
    width: u32,
) -> Result<Value> {
    if let Some(int) = dtype.int_format() {
        let unsigned = DType::unsigned(int.bits as usize)?;
        let low = match int.bits {
            64 => bits,
            _ => {
                let low_mask = body.int(DType::Ui64, (1 << int.bits) - 1);
                let masked = body.create(And, (bits, low_mask));
                body.cast(masked, unsigned)
            }
        };
        return Ok(body.bitcast(low, dtype));
    }
    let significand_bits = uniform_bits(dtype)?;
    let drop = body.int(DType::Ui64, i128::from(width - significand_bits));
    let top = body.create(ShiftRightLogical, (bits, drop));
    let exact = match significand_bits <= DType::F32.significand_bits().expect("f32 is a float") {
        true => DType::F32,
        false => DType::F64,
    };
    let value = body.cast(top, exact);
    let scale = body.float(exact, 2f64.powi(-(significand_bits as i32)));
    let unit = body.create(Mul, (value, scale));
    Ok(body.cast(unit, dtype))
}

/// Writes each `arith.rng_uniform` as the `arith` of its draw: the integer Philox4x32-10 block that
/// holds the element's words of the stream, and the element they give. It keeps every bit, so a
/// backend that computes no draw of its own runs it, before it decides how to compute the ops.
#[pass(
    name = "expand-random",
    stats(expanded = "random draws written as arith")
)]
pub struct ExpandRandom;

fn populate_expand_random_patterns(set: &mut PatternSet) {
    set.add(ExpandRngUniform);
}

impl Pass for ExpandRandom {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_expand_random_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::EXPANDED)
    }
}
