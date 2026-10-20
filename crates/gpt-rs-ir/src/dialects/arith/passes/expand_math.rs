//! Writing `exp`, `tanh`, `log`, `sin`, `cos` and `erf` as `arith` polynomials. Each set of
//! coefficients states the interval it serves and its largest error there, measured in f64 on 2^22
//! points of the interval.

use std::f32::consts::{FRAC_1_SQRT_2, LOG2_E};
use std::marker::PhantomData;

use anyhow::Result;

use crate::dialects::arith::{
    Abs, Add, And, Cos, Div, Equal, Erf, Exp, GreaterEqual, Less, Log, Maximum, Minimum, Mul, Neg,
    NotEqual, Or, ScalarBuilder, Select, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical, Sin,
    Sub, Tanh, Unary,
};
use crate::dialects::tile::Numerics;
use crate::ir::{DType, Format, OpRef, Value};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// A function of an f32 that `expand-math` writes in `arith`.
trait Expanded: Unary {
    /// The function of the f32 `input`.
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value;
}

/// `arith.exp`, `tanh`, `log`, `sin`, `cos` or `erf` of an f32 becomes its polynomial in `arith`,
/// as the function's `Expanded` impl writes it, where the compilation allows the polynomials'
/// rounding.
#[pattern]
struct Expand<T> {
    numerics: Numerics,
    function: PhantomData<T>,
}

impl<T> Expand<T> {
    fn new(numerics: Numerics) -> Self {
        Expand {
            numerics,
            function: PhantomData,
        }
    }
}

impl<T: Expanded> Pattern for Expand<T> {
    type Root = T;

    fn rewrite(&self, root: OpRef<T>, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        require!(
            self.numerics.expand_math,
            "the compilation does not allow the polynomials' rounding"
        );
        let input = root.input().id();
        require!(
            rewriter.module().ty(input).as_scalar() == Some(DType::F32),
            "the polynomial is for f32"
        );
        let value = T::expand(&mut ScalarBuilder { builder: rewriter }, input);
        rewriter.replace_op(&root, value);
        Ok(())
    }
}

/// The f32 constant `value`.
fn constant(body: &mut ScalarBuilder<Rewriter>, value: f32) -> Value {
    body.float(DType::F32, f64::from(value))
}

/// `coefficients[0] + coefficients[1] x + ...` of `x`, the value `variable`, by Horner's rule from
/// the highest coefficient, in the dtype of `variable`, each coefficient a value of that dtype.
fn polynomial<C: Copy + Into<f64>>(
    body: &mut ScalarBuilder<Rewriter>,
    variable: Value,
    coefficients: &[C],
) -> Value {
    let dtype = body.dtype(variable);
    let (&highest, lower) = coefficients
        .split_last()
        .expect("a polynomial has a coefficient");
    let mut sum = body.float(dtype, highest.into());
    for &coefficient in lower.iter().rev() {
        let product = body.create(Mul, (sum, variable));
        let coefficient_value = body.float(dtype, coefficient.into());
        sum = body.create(Add, (product, coefficient_value));
    }
    sum
}

/// `ln 2` in two parts for `exp`: the first is 355/512, 0.693359375, of 9 bits, so its product by
/// any `n` of at most 150 in magnitude is exact in f32 (Cephes `expf`'s `C1` and `C2`).
const EXP_LN2_HIGH: f32 = 355.0 / 512.0;
const EXP_LN2_LOW: f32 = -2.121_944_4e-4;
/// `p` of `e^r = 1 + r + r^2 p(r)`, from its constant term: on `[-ln 2 / 2, ln 2 / 2]`, within
/// 2.7e-9 of `e^r` relative (Cephes `expf`).
const EXP_TERMS: [f32; 6] = [
    0.5,
    0.166_666_66,
    0.041_665_796,
    0.008_333_452,
    0.001_398_199_9,
    1.987_569_1e-4,
];

/// `2^n e^r` with `n = round(x / ln 2)` and `r = x - n ln 2`: `e^r` a polynomial of degree 7, and
/// `2^n` built in the exponent bits, in two halves so that a result below the normal range rounds
/// once. `x` is first clamped to `[-104, 89]`, where `e^x` underflows to 0 and overflows to
/// infinity, and a NaN passes through.
impl Expanded for Exp {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        let nan = body.is_nan(input);
        let zero = constant(body, 0.0);
        let number = body.create(Select, (nan, zero, input));
        let lowest = constant(body, -104.0);
        let above_lowest = body.create(Maximum, (number, lowest));
        let highest = constant(body, 89.0);
        let clamped = body.create(Minimum, (above_lowest, highest));

        // n = round(x / ln 2).
        let log2e = constant(body, LOG2_E);
        let scaled = body.create(Mul, (clamped, log2e));
        let whole = body.round_even(scaled);

        // r = x - n ln 2, with ln 2 in two parts whose first times n is exact.
        let ln2_high = constant(body, EXP_LN2_HIGH);
        let high_part = body.create(Mul, (whole, ln2_high));
        let reduced_high = body.create(Sub, (clamped, high_part));
        let ln2_low = constant(body, EXP_LN2_LOW);
        let low_part = body.create(Mul, (whole, ln2_low));
        let reduced = body.create(Sub, (reduced_high, low_part));

        // e^r = 1 + r + r^2 p(r).
        let inner = polynomial(body, reduced, &EXP_TERMS);
        let square = body.create(Mul, (reduced, reduced));
        let curvature = body.create(Mul, (inner, square));
        let linear = body.create(Add, (curvature, reduced));
        let one = constant(body, 1.0);
        let mantissa = body.create(Add, (linear, one));

        // 2^n = 2^half * 2^rest, each a normal f32 for n in [-150, 128].
        let exponent = body.cast(whole, DType::Si32);
        let one_bit = body.int(DType::Si32, 1);
        let half = body.create(ShiftRightArithmetic, (exponent, one_bit));
        let rest = body.create(Sub, (exponent, half));
        let half_scale = body.pow2(DType::F32, half);
        let rest_scale = body.pow2(DType::F32, rest);
        let partly_scaled = body.create(Mul, (mantissa, half_scale));
        let result = body.create(Mul, (partly_scaled, rest_scale));
        body.create(Select, (nan, input, result))
    }
}

/// `p` of `tanh(x) = x + x^3 p(x^2)`, from its constant term: on `[-0.625, 0.625]`, within
/// 8.6e-9 of `tanh(x)` relative (Cephes `tanhf`).
const TANH_TERMS: [f32; 5] = [
    -0.333_332_8,
    0.133_314_42,
    -0.053_739_715,
    0.020_639_088,
    -0.005_704_988_7,
];

/// Below 0.625 in magnitude, `x + x^3 p(x^2)` with `p` of degree 4, and above, `1 - 2 / (e^(2|x|)
/// + 1)` with the sign of `x`, the exponential as `arith.exp`'s polynomial.
impl Expanded for Tanh {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        let magnitude = body.create(Abs, input);
        let square = body.create(Mul, (input, input));
        let odd = polynomial(body, square, &TANH_TERMS);
        let cubic = body.create(Mul, (odd, square));
        let correction = body.create(Mul, (cubic, input));
        let small = body.create(Add, (correction, input));

        let doubled = body.create(Add, (magnitude, magnitude));
        let growth = Exp::expand(body, doubled);
        let one = constant(body, 1.0);
        let denominator = body.create(Add, (growth, one));
        let two = constant(body, 2.0);
        let fraction = body.create(Div, (two, denominator));
        let positive = body.create(Sub, (one, fraction));
        let negated = body.create(Neg, positive);
        let zero = constant(body, 0.0);
        let negative = body.create(Less, (input, zero));
        let large = body.create(Select, (negative, negated, positive));

        let threshold = constant(body, 0.625);
        let far = body.create(GreaterEqual, (magnitude, threshold));
        let result = body.create(Select, (far, large, small));
        // The polynomial gives +0 for -0, whose tanh is -0.
        let at_zero = body.create(Equal, (input, zero));
        body.create(Select, (at_zero, input, result))
    }
}

/// The parts of `ln 2` for `log`: the first is of 17 bits, so its product by any exponent, at most
/// 149 in magnitude, is exact (fdlibm `logf`'s `ln2_hi` and `ln2_lo`).
const LN2_HIGH: f32 = f32::from_bits(0x3f31_7180);
const LN2_LOW: f32 = f32::from_bits(0x3717_f7d1);
/// `R` of `log((1 + s) / (1 - s)) = 2s + s R(s^2)`, from its linear term: for `|s|` up to
/// `3 - 2 sqrt(2)`, within 5.0e-11 of `log((1 + s) / (1 - s)) / s - 2` (fdlibm `logf`).
const LOG_TERMS: [f32; 4] = [
    f32::from_bits(0x3f2a_aaaa),
    f32::from_bits(0x3ecc_ce13),
    f32::from_bits(0x3e91_e9ee),
    f32::from_bits(0x3e78_9e26),
];

/// `k ln 2 + log(m)` for `x = m 2^k` with `m` in `[1/sqrt(2), sqrt(2))`: `log(m)` from `s = (m -
/// 1) / (m + 1)` and a polynomial in `s^2` of degree 4, `ln 2` in two parts. A subnormal `x` is
/// scaled by `2^25` first. 0 gives -infinity, a negative number NaN, and infinity itself.
impl Expanded for Log {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        let format = Format::F32;
        // A subnormal scales by 2^25 into the normal range, and its exponent counts that.
        let bits = body.bitcast(input, DType::Ui32);
        let normal_bits = body.int(DType::Ui32, 1 << format.man_bits);
        let subnormal = body.create(Less, (bits, normal_bits));
        let scale = constant(body, 33_554_432.0);
        let scaled_up = body.create(Mul, (input, scale));
        let scaled = body.create(Select, (subnormal, scaled_up, input));
        let scale_exponent = body.int(DType::Si32, -25);
        let no_exponent = body.int(DType::Si32, 0);
        let exponent_shift = body.create(Select, (subnormal, scale_exponent, no_exponent));

        // x = m 2^k with m in [1/sqrt(2), sqrt(2)): the bits of 1/sqrt(2) move m's exponent to 0.
        let scaled_bits = body.bitcast(scaled, DType::Ui32);
        let inverse_sqrt2_bits = i128::from(FRAC_1_SQRT_2.to_bits());
        let to_one = body.int(DType::Ui32, i128::from(1f32.to_bits()) - inverse_sqrt2_bits);
        let shifted_bits = body.create(Add, (scaled_bits, to_one));
        let fraction_bits = body.int(DType::Ui32, i128::from(format.man_bits));
        let biased = body.create(ShiftRightLogical, (shifted_bits, fraction_bits));
        let biased_exponent = body.bitcast(biased, DType::Si32);
        let with_scale = body.create(Add, (exponent_shift, biased_exponent));
        let bias = body.int(DType::Si32, i128::from(format.bias));
        let exponent = body.create(Sub, (with_scale, bias));
        let fraction_mask = body.int(DType::Ui32, (1 << format.man_bits) - 1);
        let fraction = body.create(And, (shifted_bits, fraction_mask));
        let inverse_sqrt2 = body.int(DType::Ui32, inverse_sqrt2_bits);
        let mantissa_bits = body.create(Add, (fraction, inverse_sqrt2));
        let mantissa = body.bitcast(mantissa_bits, DType::F32);

        // log(m) = f - f^2/2 + s (f^2/2 + R(s^2)), with f = m - 1 and s = f / (2 + f).
        let one = constant(body, 1.0);
        let excess = body.create(Sub, (mantissa, one));
        let two = constant(body, 2.0);
        let denominator = body.create(Add, (two, excess));
        let ratio = body.create(Div, (excess, denominator));
        let ratio_square = body.create(Mul, (ratio, ratio));
        let fourth = body.create(Mul, (ratio_square, ratio_square));
        let [first, second, third, last] = LOG_TERMS.map(|term| constant(body, term));
        let even_inner = body.create(Mul, (fourth, last));
        let even_sum = body.create(Add, (second, even_inner));
        let even = body.create(Mul, (fourth, even_sum));
        let odd_inner = body.create(Mul, (fourth, third));
        let odd_sum = body.create(Add, (first, odd_inner));
        let odd = body.create(Mul, (ratio_square, odd_sum));
        let series = body.create(Add, (odd, even));
        let half = constant(body, 0.5);
        let half_excess = body.create(Mul, (half, excess));
        let half_square = body.create(Mul, (half_excess, excess));
        let inner = body.create(Add, (half_square, series));
        let correction = body.create(Mul, (ratio, inner));
        let exponent_value = body.cast(exponent, DType::F32);
        let ln2_low = constant(body, LN2_LOW);
        let low_part = body.create(Mul, (exponent_value, ln2_low));
        let with_low = body.create(Add, (correction, low_part));
        let less_half_square = body.create(Sub, (with_low, half_square));
        let with_excess = body.create(Add, (less_half_square, excess));
        let ln2_high = constant(body, LN2_HIGH);
        let high_part = body.create(Mul, (exponent_value, ln2_high));
        let logarithm = body.create(Add, (with_excess, high_part));

        // Infinity is its own log, a negative number or NaN gives NaN, and a zero -infinity.
        let infinity = constant(body, f32::INFINITY);
        let infinite = body.create(Equal, (input, infinity));
        let finite_or_infinite = body.create(Select, (infinite, input, logarithm));
        let zero = constant(body, 0.0);
        let negative = body.create(Less, (input, zero));
        let not_a_number = body.is_nan(input);
        let undefined = body.create(Or, (negative, not_a_number));
        let nan = constant(body, f32::NAN);
        let defined = body.create(Select, (undefined, nan, finite_or_infinite));
        let at_zero = body.create(Equal, (input, zero));
        let negative_infinity = constant(body, f32::NEG_INFINITY);
        body.create(Select, (at_zero, negative_infinity, defined))
    }
}

/// `2 / pi`, and `pi / 2` in two parts: the first of 25 bits, so its product by any `n` below
/// `2^28` is exact in f64 (fdlibm `__rem_pio2f`'s `invpio2`, `pio2_1` and `pio2_1t`).
const TWO_OVER_PI: f64 = f64::from_bits(0x3fe4_5f30_6dc9_c883);
const HALF_PI_HIGH: f64 = f64::from_bits(0x3ff9_21fb_5000_0000);
const HALF_PI_LOW: f64 = f64::from_bits(0x3e51_10b4_611a_6263);
/// The magnitude from which the reduction by `pi/2` in two parts no longer holds and the exact one
/// serves: `2^28 pi/2`, rounded to an f32.
const REDUCIBLE: f32 = f32::from_bits(0x4dc9_0fdb);
/// The bits of `2/pi` after the point, in words of 32, the highest first. Every window of 96 bits
/// that an f32 from `REDUCIBLE` on picks by its exponent lies inside these words.
const TWO_OVER_PI_WORDS: [u32; 7] = [
    0xa2f9_836e,
    0x4e44_1529,
    0xfc27_57d1,
    0xf534_ddc0,
    0xdb62_9599,
    0x3c43_9041,
    0xfe51_63ab,
];
/// `S` of `sin(y) = y + y^3 S(y^2)`, from its constant term: on `[-pi/4, pi/4]`, within 4.9e-12
/// of `sin(y) / y` (fdlibm `__kernel_sindf`).
const SINE_TERMS: [f64; 4] = [
    f64::from_bits(0xbfc5_5555_54cb_ac77),
    f64::from_bits(0x3f81_1110_896e_fbb2),
    f64::from_bits(0xbf2a_00f9_e2ca_e774),
    f64::from_bits(0x3ec6_cd87_8c3b_46a7),
];
/// `C` of `cos(y) = 1 + y^2 C(y^2)`, from its constant term: on `[-pi/4, pi/4]`, within
/// 5.4e-11 of `cos(y)` (fdlibm `__kernel_cosdf`).
const COSINE_TERMS: [f64; 4] = [
    f64::from_bits(0xbfdf_ffff_fd0c_5e81),
    f64::from_bits(0x3fa5_5553_e105_3a42),
    f64::from_bits(0xbf56_c087_e80f_1e27),
    f64::from_bits(0x3ef9_9342_e0ee_5069),
];

/// In f64, `y = x - n pi/2` with `n = round(x 2 / pi)`, then the sine or the cosine of `y`, by the
/// quarter turn `n mod 4`, each a polynomial of degree 9 or 8, rounded once to f32. Below `2^28
/// pi/2` in magnitude, about `4.2e8`, `y` comes from `pi/2` in two parts. From there on, `y` comes
/// exactly from the bits of `2/pi` (see [`large_reduction`]). A zero is its own sine, and the sine
/// of an infinity is NaN.
impl Expanded for Sin {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        quarter_turns(body, input, 0)
    }
}

/// `arith.sin`'s polynomials one quarter turn on: the cosine or the sine of the reduced `y`, by `(n
/// + 1) mod 4`.
impl Expanded for Cos {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        quarter_turns(body, input, 1)
    }
}

/// The quarter turns `n mod 4`, an si32, and the remainder `y = x - n pi/2` in f64 of an f32 `x`
/// of magnitude at least `REDUCIBLE`, with `n = round(x 2/pi)`, from the integer arithmetic of its
/// bits. `|x| = m 2^E` for its integer significand `m`, and the bits of `2/pi` before bit `E - 1`
/// give `|x| 2/pi` multiples of 4 alone, so `|x| 2/pi mod 4` is `m` times the 96 bits of `2/pi`
/// from that bit on, taken mod `2^96`, times `2^-94`. Of that product, the top two bits are the
/// quarter turns, and the 94 below them the fraction of a turn, which rounds to the nearest turn
/// and becomes radians in f64. The bits of `2/pi` past the window change `y` by less than
/// `2^-68`, and the remainder of an f32 from `REDUCIBLE` on is at least `2^-30` in magnitude: the
/// least, about `1.6e-9 = 2^-29.2` at `7.73e28`, among the remainders of all of them, each
/// computed exactly from the bits of `2/pi`.
fn large_reduction(body: &mut ScalarBuilder<Rewriter>, input: Value) -> (Value, Value) {
    let (significand, window) = significand_and_window(body, input);
    let product = product_mod_2_96(body, significand, &window);
    let (turns, radians) = nearest_turn(body, product);
    // x below zero turns the other way: -n mod 4, and -y.
    let four = body.int(DType::Ui64, 4);
    let three = body.int(DType::Ui64, 3);
    let opposite = body.create(Sub, (four, turns));
    let opposite_turns = body.create(And, (opposite, three));
    let zero = constant(body, 0.0);
    let negative = body.create(Less, (input, zero));
    let signed_turns = body.create(Select, (negative, opposite_turns, turns));
    let quarter = body.cast(signed_turns, DType::Si32);
    let negated = body.create(Neg, radians);
    let reduced = body.create(Select, (negative, negated, radians));
    (quarter, reduced)
}

/// The integer significand `m` of the f32 `input`'s magnitude, with its implicit bit, and the
/// window of 96 bits of `2/pi` from bit `E - 1` on, in three words, the highest first: `E`, the
/// power of two of `m`'s unit, is the exponent field less the bias and the fraction bits, so the
/// window starts at bit `E - 2` counted from 0.
fn significand_and_window(body: &mut ScalarBuilder<Rewriter>, input: Value) -> (Value, [Value; 3]) {
    let format = Format::F32;
    let magnitude = body.create(Abs, input);
    let raw = body.bitcast(magnitude, DType::Ui32);
    let bits = body.cast(raw, DType::Ui64);
    let fraction_bits = body.int(DType::Ui64, i128::from(format.man_bits));
    let exponent_field = body.create(ShiftRightLogical, (bits, fraction_bits));
    let exponent_mask = body.int(DType::Ui64, (1 << format.exp_bits) - 1);
    let exponent = body.create(And, (exponent_field, exponent_mask));
    let fraction_mask = body.int(DType::Ui64, (1 << format.man_bits) - 1);
    let fraction = body.create(And, (bits, fraction_mask));
    let implicit_one = body.int(DType::Ui64, 1 << format.man_bits);
    let significand = body.create(Or, (fraction, implicit_one));
    let window_bias = format.bias + format.man_bits as i32 + 2;
    let window_bias_value = body.int(DType::Ui64, i128::from(window_bias));
    let start = body.create(Sub, (exponent, window_bias_value));
    let five = body.int(DType::Ui64, 5);
    let first_word = body.create(ShiftRightLogical, (start, five));
    let bit_mask = body.int(DType::Ui64, 31);
    let shift = body.create(And, (start, bit_mask));
    let table = TWO_OVER_PI_WORDS.map(|word| body.int(DType::Ui64, i128::from(word)));
    let words: [Value; 4] =
        std::array::from_fn(|taken| body.pick(first_word, &table[taken..taken + 4]));
    let mask = body.int(DType::Ui64, i128::from(u32::MAX));
    let thirty_two = body.int(DType::Ui64, 32);
    let back = body.create(Sub, (thirty_two, shift));
    let mut join = |high: Value, low: Value| {
        let shifted = body.create(ShiftLeft, (high, thirty_two));
        let joined = body.create(Or, (shifted, low));
        let aligned = body.create(ShiftRightLogical, (joined, back));
        body.create(And, (aligned, mask))
    };
    let window = [
        join(words[0], words[1]),
        join(words[1], words[2]),
        join(words[2], words[3]),
    ];
    (significand, window)
}

/// `significand` times the 96 bits of `window`, mod `2^96`, as its three words of 32 bits, the
/// highest first: each product of 24 by 32 bits is exact in a ui64, and each word adds the carry
/// of the word below it.
fn product_mod_2_96(
    body: &mut ScalarBuilder<Rewriter>,
    significand: Value,
    window: &[Value; 3],
) -> [Value; 3] {
    let mask = body.int(DType::Ui64, i128::from(u32::MAX));
    let thirty_two = body.int(DType::Ui64, 32);
    let high_product = body.create(Mul, (significand, window[0]));
    let middle_product = body.create(Mul, (significand, window[1]));
    let low_product = body.create(Mul, (significand, window[2]));
    let lowest = body.create(And, (low_product, mask));
    let low_carry = body.create(ShiftRightLogical, (low_product, thirty_two));
    let middle_low = body.create(And, (middle_product, mask));
    let second_sum = body.create(Add, (low_carry, middle_low));
    let second = body.create(And, (second_sum, mask));
    let second_carry = body.create(ShiftRightLogical, (second_sum, thirty_two));
    let middle_high = body.create(ShiftRightLogical, (middle_product, thirty_two));
    let high_low = body.create(And, (high_product, mask));
    let carried = body.create(Add, (second_carry, middle_high));
    let top_sum = body.create(Add, (carried, high_low));
    let top = body.create(And, (top_sum, mask));
    [top, second, lowest]
}

/// The quarter turns of `|x| 2/pi`, rounded to the nearest and taken mod 4, and the rest in
/// radians in f64. `product` times `2^-94` is `|x| 2/pi mod 4`: its top two bits are the turns,
/// and the 94 bits below them are the fraction of a turn. When bit 93 is set, the turns round up,
/// and the rest is the fraction less one turn.
fn nearest_turn(
    body: &mut ScalarBuilder<Rewriter>,
    [top, second, lowest]: [Value; 3],
) -> (Value, Value) {
    let thirty = body.int(DType::Ui64, 30);
    let turns = body.create(ShiftRightLogical, (top, thirty));
    let twenty_nine = body.int(DType::Ui64, 29);
    let half_bit = body.create(ShiftRightLogical, (top, twenty_nine));
    let one = body.int(DType::Ui64, 1);
    let half = body.create(And, (half_bit, one));
    let rounds_up = body.create(Equal, (half, one));
    let fraction_top_mask = body.int(DType::Ui64, 0x3FFF_FFFF);
    let fraction_top = body.create(And, (top, fraction_top_mask));
    let thirty_two = body.int(DType::Ui64, 32);
    let fraction_top_shifted = body.create(ShiftLeft, (fraction_top, thirty_two));
    let fraction_high = body.create(Or, (fraction_top_shifted, second));
    // Rounded up, the low word borrows from the high one unless it is zero.
    let zero = body.int(DType::Ui64, 0);
    let borrows = body.create(NotEqual, (lowest, zero));
    let word_span = body.int(DType::Ui64, 1 << 32);
    let lowest_up = body.create(Sub, (word_span, lowest));
    let low_up = body.create(Select, (borrows, lowest_up, zero));
    let borrow = body.create(Select, (borrows, one, zero));
    let high_span = body.int(DType::Ui64, 1 << 62);
    let high_less = body.create(Sub, (high_span, fraction_high));
    let high_up = body.create(Sub, (high_less, borrow));
    let high_magnitude = body.create(Select, (rounds_up, high_up, fraction_high));
    let low_magnitude = body.create(Select, (rounds_up, low_up, lowest));
    let high_value = body.cast(high_magnitude, DType::F64);
    let high_scale = body.float(DType::F64, 2f64.powi(-62));
    let high_turn = body.create(Mul, (high_value, high_scale));
    let low_value = body.cast(low_magnitude, DType::F64);
    let low_scale = body.float(DType::F64, 2f64.powi(-94));
    let low_turn = body.create(Mul, (low_value, low_scale));
    let turn = body.create(Add, (high_turn, low_turn));
    let half_pi = body.float(DType::F64, std::f64::consts::FRAC_PI_2);
    let radians = body.create(Mul, (turn, half_pi));
    let below = body.create(Neg, radians);
    let reduced = body.create(Select, (rounds_up, below, radians));
    let rounded_turns = body.create(Add, (turns, half));
    let three = body.int(DType::Ui64, 3);
    let turns_mod_four = body.create(And, (rounded_turns, three));
    (turns_mod_four, reduced)
}

/// The sine of the f32 `input` turned `turns` quarter turns on: `sin(x)` for 0, `cos(x)` for 1.
fn quarter_turns(body: &mut ScalarBuilder<Rewriter>, input: Value, turns: i128) -> Value {
    // y = x - n pi/2, in f64, with n = round(x 2/pi).
    let wide = body.cast(input, DType::F64);
    let two_over_pi = body.float(DType::F64, TWO_OVER_PI);
    let scaled = body.create(Mul, (wide, two_over_pi));
    let whole = body.round_even(scaled);
    let half_pi_high = body.float(DType::F64, HALF_PI_HIGH);
    let high_part = body.create(Mul, (whole, half_pi_high));
    let reduced_high = body.create(Sub, (wide, high_part));
    let half_pi_low = body.float(DType::F64, HALF_PI_LOW);
    let low_part = body.create(Mul, (whole, half_pi_low));
    let reduced_near = body.create(Sub, (reduced_high, low_part));
    let quarter_near = body.cast(whole, DType::Si32);
    let (quarter_far, reduced_far) = large_reduction(body, input);
    let magnitude = body.create(Abs, input);
    let reducible = constant(body, REDUCIBLE);
    // A NaN compares false and keeps the reduction near zero, through which it stays a NaN.
    let far = body.create(GreaterEqual, (magnitude, reducible));
    let reduced = body.create(Select, (far, reduced_far, reduced_near));
    let quarter = body.create(Select, (far, quarter_far, quarter_near));

    // sin(y) = y + y^3 S(y^2) and cos(y) = 1 + y^2 C(y^2), each rounded once to f32.
    let square = body.create(Mul, (reduced, reduced));
    let fourth = body.create(Mul, (square, square));
    let [sine_first, sine_second, sine_third, sine_last] =
        SINE_TERMS.map(|term| body.float(DType::F64, term));
    let sine_high_inner = body.create(Mul, (square, sine_last));
    let sine_high = body.create(Add, (sine_third, sine_high_inner));
    let cube = body.create(Mul, (square, reduced));
    let sine_low_inner = body.create(Mul, (square, sine_second));
    let sine_low_sum = body.create(Add, (sine_first, sine_low_inner));
    let sine_low = body.create(Mul, (cube, sine_low_sum));
    let sine_start = body.create(Add, (reduced, sine_low));
    let cube_fourth = body.create(Mul, (cube, fourth));
    let sine_tail = body.create(Mul, (cube_fourth, sine_high));
    let sine_wide = body.create(Add, (sine_start, sine_tail));
    let sine = body.cast(sine_wide, DType::F32);
    let [cosine_first, cosine_second, cosine_third, cosine_last] =
        COSINE_TERMS.map(|term| body.float(DType::F64, term));
    let cosine_high_inner = body.create(Mul, (square, cosine_last));
    let cosine_high = body.create(Add, (cosine_third, cosine_high_inner));
    let one = body.float(DType::F64, 1.0);
    let cosine_low_inner = body.create(Mul, (square, cosine_first));
    let cosine_low = body.create(Add, (one, cosine_low_inner));
    let cosine_middle_inner = body.create(Mul, (fourth, cosine_second));
    let cosine_middle = body.create(Add, (cosine_low, cosine_middle_inner));
    let sixth = body.create(Mul, (fourth, square));
    let cosine_tail = body.create(Mul, (sixth, cosine_high));
    let cosine_wide = body.create(Add, (cosine_middle, cosine_tail));
    let cosine = body.cast(cosine_wide, DType::F32);

    // By the quarter turn: an odd one swaps the sine for the cosine, and the third and fourth
    // negate.
    let offset = body.int(DType::Si32, turns);
    let turned = body.create(Add, (quarter, offset));
    let one_bit = body.int(DType::Si32, 1);
    let odd_bit = body.create(And, (turned, one_bit));
    let odd = body.create(Equal, (odd_bit, one_bit));
    let two_bit = body.int(DType::Si32, 2);
    let half_turn_bit = body.create(And, (turned, two_bit));
    let half_turn = body.create(Equal, (half_turn_bit, two_bit));
    let swapped = body.create(Select, (odd, cosine, sine));
    let negated = body.create(Neg, swapped);
    let value = body.create(Select, (half_turn, negated, swapped));

    // An infinity gives NaN; a NaN stays one.
    let infinity = constant(body, f32::INFINITY);
    let infinite = body.create(Equal, (magnitude, infinity));
    let nan = constant(body, f32::NAN);
    let result = body.create(Select, (infinite, nan, value));
    if turns != 0 {
        return result;
    }
    // A zero is its own sine, -0 included.
    let zero = constant(body, 0.0);
    let at_zero = body.create(Equal, (input, zero));
    body.create(Select, (at_zero, input, result))
}

/// The base of `erf` near 1, which is `NEAR_BASE + P(|x| - 1) / Q(|x| - 1)`: `erf(1)` less the
/// constant term of `P`, `NEAR_NUMERATOR[0]` (fdlibm `erff`'s `erx`).
const NEAR_BASE: f32 = f32::from_bits(0x3f58_560b);
/// `8 * (2/sqrt(pi) - 1)`, the slope of `erf` at 0 less 1, by 8 (fdlibm `erff`'s `efx8`).
const TINY_SLOPE: f32 = f32::from_bits(0x3f83_75d4);
/// The rational functions of `erf`, numerators and denominators from their constant terms (fdlibm
/// `erff`'s `pp`, `qq`, `pa`, `qa`, `ra`, `sa`, `rb` and `sb`). Below 0.84375, `P/Q` of `x^2`,
/// within 3.3e-9 of `(erf(x) - x) / x`.
const SMALL_NUMERATOR: [f32; 5] = [
    f32::from_bits(0x3e03_75d4),
    f32::from_bits(0xbea6_6beb),
    f32::from_bits(0xbce9_528f),
    f32::from_bits(0xbbbd_1489),
    f32::from_bits(0xb7c7_56b1),
];
const SMALL_DENOMINATOR: [f32; 6] = [
    1.0,
    f32::from_bits(0x3ecb_bbce),
    f32::from_bits(0x3d85_2a63),
    f32::from_bits(0x3ba6_8116),
    f32::from_bits(0x390a_ee49),
    f32::from_bits(0xb684_e21a),
];
/// From 0.84375 to 1.25, `P/Q` of `|x| - 1`, within 3.4e-9 of `erf(|x|) - NEAR_BASE`.
const NEAR_NUMERATOR: [f32; 7] = [
    f32::from_bits(0xbb1a_cdc6),
    f32::from_bits(0x3ed4_6805),
    f32::from_bits(0xbebe_9208),
    f32::from_bits(0x3ea2_fe54),
    f32::from_bits(0xbde3_1cc2),
    f32::from_bits(0x3d11_51b3),
    f32::from_bits(0xbb0d_f9c0),
];
const NEAR_DENOMINATOR: [f32; 7] = [
    1.0,
    f32::from_bits(0x3dd9_f331),
    f32::from_bits(0x3f0a_5785),
    f32::from_bits(0x3d93_1ae7),
    f32::from_bits(0x3e01_3307),
    f32::from_bits(0x3c5f_6e13),
    f32::from_bits(0x3c44_5aa3),
];
/// From 1.25 to `1/0.35`, `R/S` of `1/x^2`, within 6.2e-9 of `log(x erfc(x)) + x^2 +
/// 0.5625`.
const MIDDLE_NUMERATOR: [f32; 8] = [
    f32::from_bits(0xbc21_a093),
    f32::from_bits(0xbf31_a0b7),
    f32::from_bits(0xc128_f022),
    f32::from_bits(0xc279_8057),
    f32::from_bits(0xc322_658c),
    f32::from_bits(0xc338_9ae7),
    f32::from_bits(0xc2a2_932b),
    f32::from_bits(0xc11d_077e),
];
const MIDDLE_DENOMINATOR: [f32; 9] = [
    1.0,
    f32::from_bits(0x419d_35ce),
    f32::from_bits(0x4309_a863),
    f32::from_bits(0x43d9_486f),
    f32::from_bits(0x4421_58c9),
    f32::from_bits(0x43d6_810b),
    f32::from_bits(0x42d9_451f),
    f32::from_bits(0x40d2_3f7c),
    f32::from_bits(0xbd77_7f97),
];
/// From `1/0.35` to 6, `R/S` of `1/x^2`, within 1.3e-9 of `log(x erfc(x)) + x^2 + 0.5625`.
const FAR_NUMERATOR: [f32; 7] = [
    f32::from_bits(0xbc21_a092),
    f32::from_bits(0xbf4c_9dd4),
    f32::from_bits(0xc18e_104b),
    f32::from_bits(0xc320_a2ea),
    f32::from_bits(0xc41f_6441),
    f32::from_bits(0xc480_230b),
    f32::from_bits(0xc3f1_c275),
];
const FAR_DENOMINATOR: [f32; 8] = [
    1.0,
    f32::from_bits(0x41f2_b459),
    f32::from_bits(0x43a2_e571),
    f32::from_bits(0x44c0_1759),
    f32::from_bits(0x4547_fdbb),
    f32::from_bits(0x451f_90ce),
    f32::from_bits(0x43ed_43a7),
    f32::from_bits(0xc1b3_8712),
];

/// `numerator(x) / denominator(x)` of `x`, the value `variable`.
fn rational(
    body: &mut ScalarBuilder<Rewriter>,
    variable: Value,
    numerator: &[f32],
    denominator: &[f32],
) -> Value {
    let top = polynomial(body, variable, numerator);
    let bottom = polynomial(body, variable, denominator);
    body.create(Div, (top, bottom))
}

/// `erf` by the magnitude of `x`, with the sign of `x`:
/// - below `2^-28`, `2 x / sqrt(pi)`;
/// - below 0.84375, `x` plus `x` times a rational function of `x^2`;
/// - below 1.25, a rational function of `|x| - 1`;
/// - below 6, `1 - e^(-z^2 - 0.5625) e^((z - |x|)(z + |x|) + R(1/x^2)) / |x|`, with `z` the
///   magnitude cut to 11 significant bits, `R` a rational function of two pieces split at
///   `1/0.35`, and the exponentials as `arith.exp`'s polynomial;
/// - from 6 on, 1.
impl Expanded for Erf {
    fn expand(body: &mut ScalarBuilder<Rewriter>, input: Value) -> Value {
        let magnitude = body.create(Abs, input);
        // Below 2^-28: x (1 + 2/sqrt(pi) - 1), scaled by 8 so a subnormal keeps its bits.
        let eighth = constant(body, 0.125);
        let eight = constant(body, 8.0);
        let tiny_slope = constant(body, TINY_SLOPE);
        let eight_x = body.create(Mul, (eight, input));
        let slope_x = body.create(Mul, (tiny_slope, input));
        let tiny_sum = body.create(Add, (eight_x, slope_x));
        let tiny = body.create(Mul, (eighth, tiny_sum));
        // Below 0.84375: x + x P(x^2) / Q(x^2).
        let square = body.create(Mul, (input, input));
        let small_ratio = rational(body, square, &SMALL_NUMERATOR, &SMALL_DENOMINATOR);
        let small_correction = body.create(Mul, (input, small_ratio));
        let small = body.create(Add, (input, small_correction));
        // Below 1.25: NEAR_BASE + P(|x| - 1) / Q(|x| - 1), as 1 less its complement.
        let one = constant(body, 1.0);
        let past_one = body.create(Sub, (magnitude, one));
        let near_ratio = rational(body, past_one, &NEAR_NUMERATOR, &NEAR_DENOMINATOR);
        let near_base = constant(body, NEAR_BASE);
        let complement_base = body.create(Sub, (one, near_base));
        let near_complement = body.create(Sub, (complement_base, near_ratio));
        let near = body.create(Sub, (one, near_complement));
        // Below 6: 1 - exp(-z^2 - 0.5625) exp((z - |x|)(z + |x|) + R(1/x^2)) / |x|.
        let magnitude_square = body.create(Mul, (magnitude, magnitude));
        let inverse_square = body.create(Div, (one, magnitude_square));
        let middle_end = constant(body, f32::from_bits(0x4036_db6d));
        let middle = body.create(Less, (magnitude, middle_end));
        let middle_top = polynomial(body, inverse_square, &MIDDLE_NUMERATOR);
        let far_top = polynomial(body, inverse_square, &FAR_NUMERATOR);
        let top = body.create(Select, (middle, middle_top, far_top));
        let middle_bottom = polynomial(body, inverse_square, &MIDDLE_DENOMINATOR);
        let far_bottom = polynomial(body, inverse_square, &FAR_DENOMINATOR);
        let bottom = body.create(Select, (middle, middle_bottom, far_bottom));
        let magnitude_bits = body.bitcast(magnitude, DType::Ui32);
        let upper_mask = body.int(DType::Ui32, 0xffff_e000);
        let upper_bits = body.create(And, (magnitude_bits, upper_mask));
        let upper = body.bitcast(upper_bits, DType::F32);
        let negated_upper = body.create(Neg, upper);
        let upper_square = body.create(Mul, (negated_upper, upper));
        let shift = constant(body, 0.5625);
        let first_power = body.create(Sub, (upper_square, shift));
        let first = Exp::expand(body, first_power);
        let below = body.create(Sub, (upper, magnitude));
        let above = body.create(Add, (upper, magnitude));
        let product = body.create(Mul, (below, above));
        let tail_ratio = body.create(Div, (top, bottom));
        let second_power = body.create(Add, (product, tail_ratio));
        let second = Exp::expand(body, second_power);
        let exponentials = body.create(Mul, (first, second));
        let tail = body.create(Div, (exponentials, magnitude));
        let far = body.create(Sub, (one, tail));
        // From 6 on, 1; then the sign of x.
        let near_end = constant(body, 1.25);
        let is_near = body.create(Less, (magnitude, near_end));
        let far_end = constant(body, 6.0);
        let is_far = body.create(Less, (magnitude, far_end));
        let far_or_one = body.create(Select, (is_far, far, one));
        let unsigned = body.create(Select, (is_near, near, far_or_one));
        let negated = body.create(Neg, unsigned);
        let zero = constant(body, 0.0);
        let negative = body.create(Less, (input, zero));
        let signed = body.create(Select, (negative, negated, unsigned));
        let tiny_end = constant(body, f32::from_bits(0x3180_0000));
        let is_tiny = body.create(Less, (magnitude, tiny_end));
        let small_end = constant(body, f32::from_bits(0x3f58_0000));
        let is_small = body.create(Less, (magnitude, small_end));
        let tiny_or_small = body.create(Select, (is_tiny, tiny, small));
        let value = body.create(Select, (is_small, tiny_or_small, signed));
        let not_a_number = body.is_nan(input);
        body.create(Select, (not_a_number, input, value))
    }
}

/// Writes `arith.exp`, `tanh`, `log`, `sin`, `cos` and `erf` of f32 scalars as polynomials in
/// `arith`, for a target that has no lane-wise form of these functions, where the compilation
/// allows their rounding (`numerics = <expand_math = true>`). The sine and the cosine compute in
/// f64 and on ui64 words, so the target needs f64 and 64-bit integer lanes.
///
/// Measured on every f32 input against the interpreter's math functions, the `libm` crate's f64
/// functions rounded to f32: each expansion is within 1 ulp of its reference and gives its
/// infinities and NaNs. The polynomials round otherwise than the reference, so this is a pass of
/// its own, not part of `canonicalize`.
#[pass(
    name = "expand-math",
    stats(expanded = "math functions written as polynomials")
)]
#[derive(Default)]
pub struct ExpandMath {
    /// The rounding changes the compilation allows: the pass writes the polynomials only where
    /// they name `expand_math`.
    pub numerics: Numerics,
}

fn populate_expand_math_patterns(set: &mut PatternSet, numerics: Numerics) {
    set.add(Expand::<Exp>::new(numerics));
    set.add(Expand::<Tanh>::new(numerics));
    set.add(Expand::<Log>::new(numerics));
    set.add(Expand::<Sin>::new(numerics));
    set.add(Expand::<Cos>::new(numerics));
    set.add(Expand::<Erf>::new(numerics));
}

impl Pass for ExpandMath {
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_expand_math_patterns(&mut patterns, self.numerics);
        Rewrites {
            patterns,
            ..Rewrites::default()
        }
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::EXPANDED)
    }
}
