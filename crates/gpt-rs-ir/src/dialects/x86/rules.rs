//! The C forms of the `isa` ops on AVX-512 registers, and what each needs.

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::{arith, isa};
use crate::ir::{DType, InOp, Inst, Module, Type, Value, VectorType};
use crate::targets::{Feature, Prefetch};
use crate::Op;

/// AVX-512 Foundation.
const AVX512F: &[Feature] = &[Feature::Avx512f];
/// The doubleword and quadword instructions, with Foundation.
const AVX512DQ: &[Feature] = &[Feature::Avx512f, Feature::Avx512dq];
/// The byte and word instructions, with Foundation.
const AVX512BW: &[Feature] = &[Feature::Avx512f, Feature::Avx512bw];
/// The byte and word instructions on 128- and 256-bit registers.
const AVX512BW_VL: &[Feature] = &[Feature::Avx512f, Feature::Avx512bw, Feature::Avx512vl];

/// The C of an `isa` op on one register of its group, and what it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Form {
    /// The C expression or statement.
    pub text: String,
    /// The extensions its intrinsics need.
    pub features: &'static [Feature],
    /// The mask registers it holds besides its operands and its result.
    pub scratch_masks: usize,
    /// The vector registers it holds besides its operands and its result, as clang compiles it.
    pub scratch_vectors: usize,
    /// The operands, by position, that its instruction can read from memory, one at a time.
    pub memory_operands: &'static [usize],
}

impl Form {
    fn new(text: impl Into<String>, features: &'static [Feature]) -> Self {
        Self {
            text: text.into(),
            features,
            scratch_masks: 0,
            scratch_vectors: 0,
            memory_operands: &[],
        }
    }

    fn with_scratch(mut self, masks: usize, vectors: usize) -> Self {
        self.scratch_masks = masks;
        self.scratch_vectors = vectors;
        self
    }
}

/// The C type of one register of `lanes` lanes of `dtype`.
pub fn register_type(dtype: DType, lanes: usize) -> Result<&'static str> {
    if dtype == DType::I1 {
        return mask_type(lanes);
    }
    Ok(match (dtype, dtype.bits() * lanes) {
        (DType::Si4 | DType::Ui4, 64) => "__m128i",
        (DType::F32, 512) => "__m512",
        (DType::F64, 512) => "__m512d",
        (_, 512) if dtype.is_integer() => "__m512i",
        (DType::Bf16 | DType::F16 | DType::F8E4M3 | DType::F8E5M2, bits) => match Width::of(bits) {
            Some(width) => width.integer_type(),
            None => bail!("x86 holds no register of {lanes} lanes of {dtype}, {bits} bits"),
        },
        (_, bits) => {
            bail!("x86 computes in 512-bit registers, not {lanes} lanes of {dtype}, {bits} bits")
        }
    })
}

/// The C type of a mask of `lanes` flags.
pub fn mask_type(lanes: usize) -> Result<&'static str> {
    Ok(match lanes {
        8 => "__mmask8",
        16 => "__mmask16",
        32 => "__mmask32",
        64 => "__mmask64",
        other => bail!("x86 has no mask of {other} lanes"),
    })
}

/// The extensions that a mask of `lanes` flags needs: masks of 32 and 64 flags are the byte and
/// word instructions'.
fn mask_features(lanes: usize) -> &'static [Feature] {
    match lanes > 16 {
        true => AVX512BW,
        false => AVX512F,
    }
}

/// The class of lanes a C form is written for, with the names it is spelled with.
enum Class {
    /// f32 or f64: `$P` is `ps` or `pd`.
    Float(&'static str),
    /// An integer: `$I` is `epi32` or `epi64`, and `$M` that or `epu32`, `epu64` for unsigned
    /// lanes; with its width and whether it is signed.
    Int(&'static str, &'static str, usize, bool),
    /// `i1` lanes, in a mask of type `$K`, of its lanes.
    Mask(&'static str, usize),
    /// Lanes of a float narrower than f32, held as their bits, which x86 moves and converts but
    /// computes on only in f32: `$H` is the intrinsics' prefix (`_mm`, `_mm256`, `_mm512`) and
    /// `$S` the register's suffix (`si128`, ...) at the register's width, and `$E` the lanes'
    /// suffix, `epi16` or `epi8`; with the extensions of the masked moves of such lanes.
    Narrow(Width, &'static str, &'static [Feature]),
    /// 16 lanes of a 4-bit integer, two to a byte from the lower bits up, in the lower 64 bits of
    /// a 128-bit register, which x86 loads and widens to 32 bits but computes on in no other way.
    Nibbles,
}

impl Class {
    fn of(dtype: DType, lanes: usize) -> Result<Self> {
        Ok(match dtype {
            DType::F32 => Class::Float("ps"),
            DType::F64 => Class::Float("pd"),
            DType::I1 => Class::Mask(mask_type(lanes)?, lanes),
            DType::Si32 => Class::Int("epi32", "epi32", 32, true),
            DType::Ui32 => Class::Int("epi32", "epu32", 32, false),
            DType::Si64 => Class::Int("epi64", "epi64", 64, true),
            DType::Ui64 => Class::Int("epi64", "epu64", 64, false),
            DType::Si4 | DType::Ui4 => {
                ensure!(
                    lanes == 16,
                    "x86 holds {dtype} 16 lanes to a register, not {lanes}"
                );
                Class::Nibbles
            }
            DType::Bf16 | DType::F16 | DType::F8E4M3 | DType::F8E5M2 => {
                let element = match dtype.bits() {
                    16 => "epi16",
                    _ => "epi8",
                };
                let width = Width::of(dtype.bits() * lanes)
                    .ok_or_else(|| anyhow!("x86 has no register of {lanes} lanes of {dtype}"))?;
                Class::Narrow(width, element, width.byte_word_features())
            }
            other => bail!("x86 has no vector form for {other} lanes"),
        })
    }

    /// The extensions of the moves of this class in memory and between registers.
    fn move_features(&self) -> &'static [Feature] {
        match self {
            Class::Narrow(.., features) => features,
            _ => AVX512F,
        }
    }

    /// `template` spelled for this class.
    fn spell(&self, template: &str) -> String {
        match self {
            Class::Float(suffix) => template.replace("$P", suffix),
            Class::Int(int, ordered, ..) => template.replace("$I", int).replace("$M", ordered),
            Class::Mask(ty, _) => template.replace("$K", ty),
            Class::Narrow(width, element, _) => template
                .replace("$H", width.prefix())
                .replace("$S", width.register_suffix())
                .replace("$E", element),
            Class::Nibbles => template.to_string(),
        }
    }
}

/// The width of a vector register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Width {
    Xmm,
    Ymm,
    Zmm,
}

impl Width {
    fn of(bits: usize) -> Option<Width> {
        Some(match bits {
            128 => Width::Xmm,
            256 => Width::Ymm,
            512 => Width::Zmm,
            _ => return None,
        })
    }

    /// The prefix of the intrinsics on registers of this width.
    fn prefix(self) -> &'static str {
        match self {
            Width::Xmm => "_mm",
            Width::Ymm => "_mm256",
            Width::Zmm => "_mm512",
        }
    }

    /// The suffix of the whole-register loads and stores of this width.
    fn register_suffix(self) -> &'static str {
        match self {
            Width::Xmm => "si128",
            Width::Ymm => "si256",
            Width::Zmm => "si512",
        }
    }

    /// The C type of an integer register of this width.
    fn integer_type(self) -> &'static str {
        match self {
            Width::Xmm => "__m128i",
            Width::Ymm => "__m256i",
            Width::Zmm => "__m512i",
        }
    }

    /// The extensions of the masked moves of byte and word lanes at this width.
    fn byte_word_features(self) -> &'static [Feature] {
        match self {
            Width::Zmm => AVX512BW,
            Width::Xmm | Width::Ymm => AVX512BW_VL,
        }
    }
}

/// Whether `op` is a `T`.
fn is<T: InOp>(op: &Op) -> bool {
    op.get::<T>().is_some()
}

/// The C form of one lane-wise op on one class of lanes: on lanes of `bits` bits only, when it
/// names a width, holding `scratch_vectors` vector registers besides its operands and its result.
struct Row {
    op: fn(&Op) -> bool,
    bits: Option<usize>,
    text: &'static str,
    features: &'static [Feature],
    scratch_vectors: usize,
}

const fn row(op: fn(&Op) -> bool, text: &'static str, features: &'static [Feature]) -> Row {
    Row {
        op,
        bits: None,
        text,
        features,
        scratch_vectors: 0,
    }
}

impl Row {
    /// The row, whose form holds `vectors` vector registers besides its operands and its result.
    const fn holding(self, vectors: usize) -> Row {
        Row {
            scratch_vectors: vectors,
            ..self
        }
    }
}

/// The forms of the lane-wise ops on float lanes.
const FLOAT_ROWS: &[Row] = &[
    row(
        is::<isa::Neg>,
        "_mm512_xor_$P($0, _mm512_set1_$P(-0.0))",
        AVX512DQ,
    ),
    row(is::<isa::Abs>, "_mm512_abs_$P($0)", AVX512F),
    // The dividend 1.0 is in a register.
    row(
        is::<isa::Rsqrt>,
        "_mm512_div_$P(_mm512_set1_$P(1.0), _mm512_sqrt_$P($0))",
        AVX512F,
    )
    .holding(1),
    row(
        is::<isa::Reciprocal>,
        "_mm512_div_$P(_mm512_set1_$P(1.0), $0)",
        AVX512F,
    )
    .holding(1),
    row(is::<isa::Add>, "_mm512_add_$P($0, $1)", AVX512F),
    row(is::<isa::Sub>, "_mm512_sub_$P($0, $1)", AVX512F),
    row(is::<isa::Mul>, "_mm512_mul_$P($0, $1)", AVX512F),
    row(is::<isa::Div>, "_mm512_div_$P($0, $1)", AVX512F),
    row(
        is::<isa::Less>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_LT_OQ)",
        AVX512F,
    ),
    row(
        is::<isa::LessEqual>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_LE_OQ)",
        AVX512F,
    ),
    row(
        is::<isa::Equal>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_EQ_OQ)",
        AVX512F,
    ),
    row(
        is::<isa::GreaterEqual>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_GE_OQ)",
        AVX512F,
    ),
    row(
        is::<isa::Greater>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_GT_OQ)",
        AVX512F,
    ),
    row(
        is::<isa::NotEqual>,
        "_mm512_cmp_$P_mask($0, $1, _CMP_NEQ_UQ)",
        AVX512F,
    ),
    row(
        is::<isa::Select>,
        "_mm512_mask_blend_$P($0, $2, $1)",
        AVX512F,
    ),
    row(is::<isa::Fma>, "_mm512_fmadd_$P($0, $1, $2)", AVX512F),
];

/// The forms of the lane-wise ops on integer lanes, which wrap.
const INT_ROWS: &[Row] = &[
    // The zero is in a register.
    row(
        is::<isa::Neg>,
        "_mm512_sub_$I(_mm512_setzero_si512(), $0)",
        AVX512F,
    )
    .holding(1),
    row(is::<isa::Add>, "_mm512_add_$I($0, $1)", AVX512F),
    row(is::<isa::Sub>, "_mm512_sub_$I($0, $1)", AVX512F),
    Row {
        bits: Some(32),
        ..row(is::<isa::Mul>, "_mm512_mullo_$I($0, $1)", AVX512F)
    },
    Row {
        bits: Some(64),
        ..row(is::<isa::Mul>, "_mm512_mullo_$I($0, $1)", AVX512DQ)
    },
    row(is::<isa::Maximum>, "_mm512_max_$M($0, $1)", AVX512F),
    row(is::<isa::Minimum>, "_mm512_min_$M($0, $1)", AVX512F),
    row(
        is::<isa::Not>,
        "_mm512_xor_si512($0, _mm512_set1_$I(-1))",
        AVX512F,
    ),
    row(is::<isa::And>, "_mm512_and_si512($0, $1)", AVX512F),
    row(is::<isa::Or>, "_mm512_or_si512($0, $1)", AVX512F),
    row(is::<isa::Xor>, "_mm512_xor_si512($0, $1)", AVX512F),
    row(is::<isa::ShiftLeft>, "_mm512_sllv_$I($0, $1)", AVX512F),
    row(
        is::<isa::ShiftRightLogical>,
        "_mm512_srlv_$I($0, $1)",
        AVX512F,
    ),
    row(
        is::<isa::ShiftRightArithmetic>,
        "_mm512_srav_$I($0, $1)",
        AVX512F,
    ),
    row(
        is::<isa::Less>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_LT)",
        AVX512F,
    ),
    row(
        is::<isa::LessEqual>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_LE)",
        AVX512F,
    ),
    row(
        is::<isa::Equal>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_EQ)",
        AVX512F,
    ),
    row(
        is::<isa::GreaterEqual>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_NLT)",
        AVX512F,
    ),
    row(
        is::<isa::Greater>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_NLE)",
        AVX512F,
    ),
    row(
        is::<isa::NotEqual>,
        "_mm512_cmp_$M_mask($0, $1, _MM_CMPINT_NE)",
        AVX512F,
    ),
    row(
        is::<isa::Select>,
        "_mm512_mask_blend_$I($0, $2, $1)",
        AVX512F,
    ),
];

/// The forms of the lane-wise ops on `i1` lanes, held in masks: C operators on the mask, whose
/// type needs the extensions of its width.
const MASK_ROWS: &[Row] = &[
    row(is::<isa::Not>, "(($K)~$0)", AVX512F),
    row(is::<isa::And>, "(($K)($0 & $1))", AVX512F),
    row(is::<isa::Or>, "(($K)($0 | $1))", AVX512F),
    row(is::<isa::Xor>, "(($K)($0 ^ $1))", AVX512F),
    row(is::<isa::Equal>, "(($K)~($0 ^ $1))", AVX512F),
    row(is::<isa::NotEqual>, "(($K)($0 ^ $1))", AVX512F),
    row(is::<isa::Select>, "(($K)(($0 & $1) | (~$0 & $2)))", AVX512F),
];

/// The form of the lane-wise op of `inst`: an `isa` op that applies an `arith` op to each lane,
/// `isa.select`, `isa.fma`, `isa.cast` or `isa.bitcast`.
pub fn lanewise(module: &Module, inst: Inst) -> Result<Form> {
    let op = module.op(inst);
    let operands = module.operands(inst);
    let lanes = vector_type(module, module.results(inst)[0])?.lanes();
    // A select takes its mask first; its lanes are those of the values.
    let lanes_of = match op.get::<isa::Select>() {
        Some(_) => operands[1],
        None => operands[0],
    };
    let dtype = vector_type(module, lanes_of)?.dtype;
    let mut constants = [None, None];
    for (constant, &operand) in constants.iter_mut().zip(operands) {
        *constant = constant_lanes(module, operand);
    }
    lanewise_form(op, dtype, lanes, constants)
}

/// The form of the lane-wise `isa` op `op` on registers of `lanes` lanes of `dtype`, the dtype of
/// its first operand, or of its values for `isa.select`, whose first two operands hold `constants`
/// in every lane where they are splats of constants.
fn lanewise_form(op: &Op, dtype: DType, lanes: usize, constants: [Option<f64>; 2]) -> Result<Form> {
    if let Some(cast) = op.get::<isa::Cast>() {
        return cast_form(dtype, cast.to, lanes);
    }
    if let Some(bitcast) = op.get::<isa::Bitcast>() {
        return Ok(Form::new(bitcast_form(dtype, bitcast.to), AVX512F));
    }
    let class = Class::of(dtype, lanes)?;
    if let Some(form) = extremum_form(op, &class, constants) {
        return Ok(form);
    }
    if let Class::Nibbles = class {
        bail!(
            "{} of {dtype} lanes has no AVX-512 form: x86 widens {dtype} to 32 bits to compute",
            op.name()
        );
    }
    if let Class::Narrow(.., features) = class {
        ensure!(
            is::<isa::Select>(op),
            "{} of {dtype} lanes has no AVX-512 form: x86 computes on {dtype} in f32",
            op.name()
        );
        return Ok(Form::new(
            class.spell("$H_mask_blend_$E($0, $2, $1)"),
            features,
        ));
    }
    let (rows, bits) = match class {
        Class::Float(_) => (FLOAT_ROWS, None),
        Class::Int(_, _, bits, signed) => {
            if is::<isa::Abs>(op) {
                return Ok(match signed {
                    true => Form::new(class.spell("_mm512_abs_$I($0)"), AVX512F),
                    false => Form::new("$0", AVX512F),
                });
            }
            (INT_ROWS, Some(bits))
        }
        Class::Mask(..) => (MASK_ROWS, None),
        Class::Narrow(..) => unreachable!("narrow lanes take only select"),
        Class::Nibbles => unreachable!("4-bit lanes take no lane-wise op"),
    };
    let found = rows
        .iter()
        .find(|row| (row.op)(op) && (row.bits.is_none() || row.bits == bits))
        .ok_or_else(|| {
            anyhow!(
                "{} of {dtype} lanes has no AVX-512 form{}",
                op.name(),
                match dtype {
                    DType::F32 => "; expand-math writes the math functions of f32 as polynomials",
                    _ => "",
                }
            )
        })?;
    let features = match class {
        Class::Mask(_, mask_lanes) => mask_features(mask_lanes),
        _ => found.features,
    };
    Ok(Form {
        memory_operands: memory_operands(op, dtype),
        ..Form::new(class.spell(found.text), features).with_scratch(0, found.scratch_vectors)
    })
}

/// The operands of the lane-wise op `op` on lanes of `dtype` that its instruction can read from
/// memory, one at a time, where they hold one constant in every lane: AVX-512 broadcasts the last
/// source of an instruction from one element of 32 or 64 bits, and the C compiler takes either
/// source of an op that commutes or compares there, and any of an fma's. A float maximum or
/// minimum takes a constant in a register.
fn memory_operands(op: &Op, dtype: DType) -> &'static [usize] {
    if dtype == DType::I1 || !matches!(dtype.bits(), 32 | 64) {
        return &[];
    }
    if is::<isa::Fma>(op) {
        return &[0, 1, 2];
    }
    if dtype.is_float() && (is::<isa::Maximum>(op) || is::<isa::Minimum>(op)) {
        return &[];
    }
    let either: [fn(&Op) -> bool; 13] = [
        is::<isa::Add>,
        is::<isa::Mul>,
        is::<isa::Maximum>,
        is::<isa::Minimum>,
        is::<isa::And>,
        is::<isa::Or>,
        is::<isa::Xor>,
        is::<isa::Less>,
        is::<isa::LessEqual>,
        is::<isa::Equal>,
        is::<isa::GreaterEqual>,
        is::<isa::Greater>,
        is::<isa::NotEqual>,
    ];
    if either.iter().any(|is_op| is_op(op)) {
        return &[0, 1];
    }
    let last: [fn(&Op) -> bool; 6] = [
        is::<isa::Sub>,
        is::<isa::Div>,
        is::<isa::ShiftLeft>,
        is::<isa::ShiftRightLogical>,
        is::<isa::ShiftRightArithmetic>,
        is::<isa::Select>,
    ];
    match last.iter().any(|is_op| is_op(op)) {
        true => &[1],
        false => &[],
    }
}

/// The form of `isa.maximum` or `isa.minimum` of float lanes, which give a NaN for a NaN operand.
/// Against a constant `c` that is not a NaN, it is one comparison and a blend:
/// `select(x <= c, c, x)` for `maximum`, with `x < c` against `-0`, which `+0` passes, and
/// `select(x >= c, c, x)` for `minimum`, with `x > c` against `+0`. Otherwise it is `vrangeps`,
/// which orders the zeros as they do and drops a quiet NaN, with the sum of the operands, a NaN,
/// where either is one. None for any other op or lanes.
fn extremum_form(op: &Op, class: &Class, constants: [Option<f64>; 2]) -> Option<Form> {
    let Class::Float(_) = class else {
        return None;
    };
    let (range, passed, below, passed_zero_negative) =
        match (is::<isa::Maximum>(op), is::<isa::Minimum>(op)) {
            (true, _) => ("0x05", "_CMP_LE_OQ", "_CMP_LT_OQ", true),
            (_, true) => ("0x04", "_CMP_GE_OQ", "_CMP_GT_OQ", false),
            _ => return None,
        };
    let against = match constants {
        [_, Some(constant)] => Some(("$0", "$1", constant)),
        [Some(constant), None] => Some(("$1", "$0", constant)),
        [None, None] => None,
    };
    if let Some((varying, fixed, constant)) = against.filter(|(.., constant)| !constant.is_nan()) {
        let passed_zero = constant == 0.0 && constant.is_sign_negative() == passed_zero_negative;
        let compare = match passed_zero {
            true => below,
            false => passed,
        };
        let text = format!(
            "_mm512_mask_blend_$P(_mm512_cmp_$P_mask({varying}, {fixed}, {compare}), {varying}, {fixed})"
        );
        return Some(Form::new(class.spell(&text), AVX512F).with_scratch(1, 0));
    }
    // The range is held while the comparison runs.
    let text = format!(
        "_mm512_mask_add_$P(_mm512_range_$P($0, $1, {range}), \
         _mm512_cmp_$P_mask($0, $1, _CMP_UNORD_Q), $0, $1)"
    );
    Some(Form::new(class.spell(&text), AVX512DQ).with_scratch(1, 1))
}

/// The value in every lane of the registers `value`, when it is a splat of a float constant,
/// repeated or not.
fn constant_lanes(module: &Module, value: Value) -> Option<f64> {
    let inst = module.def_inst(value)?;
    let op = module.op(inst);
    if op.get::<isa::Expand>().is_some() {
        return constant_lanes(module, module.operands(inst)[0]);
    }
    op.get::<isa::Splat>()?;
    let scalar = module.def_inst(module.operands(inst)[0])?;
    let constant = module.op(scalar).get::<arith::Constant>()?;
    constant.value.scalar().as_f64()
}

/// The form of `isa.cast` of `lanes` lanes of `from` to `to`.
fn cast_form(from: DType, to: DType, lanes: usize) -> Result<Form> {
    let narrow = |dtype: DType| matches!(Class::of(dtype, lanes), Ok(Class::Narrow(..)));
    if narrow(from) || narrow(to) {
        ensure!(
            lanes == 16,
            "x86 converts between {from} and {to} 16 lanes at a time, not {lanes}"
        );
    }
    if let Some(form) = half_cast_form(from, to) {
        return Ok(form);
    }
    if let Some((widened, scratch_vectors)) = nibbles_widened(from) {
        let text = match to {
            DType::Si32 | DType::Ui32 => widened,
            DType::F32 => format!("_mm512_cvtepi32_ps({widened})"),
            _ => bail!("x86 has no vector cast of {from} to {to}"),
        };
        return Ok(Form::new(text, AVX512F).with_scratch(0, scratch_vectors));
    }
    // Every narrow float is an f32, so a cast through f32 rounds once, as the cast itself does.
    if narrow(from) && to != DType::F32 {
        let widen = half_cast_form(from, DType::F32)
            .ok_or_else(|| anyhow!("x86 has no vector cast of {from} to f32"))?;
        let rest = cast_form(DType::F32, to, lanes)?;
        let features = match rest
            .features
            .iter()
            .all(|feature| widen.features.contains(feature))
        {
            true => widen.features,
            false => rest.features,
        };
        // The widened lanes are the operand of the rest, so it holds the vectors of either.
        let scratch_vectors = widen.scratch_vectors.max(rest.scratch_vectors);
        return Ok(Form::new(rest.text.replace("$0", &widen.text), features)
            .with_scratch(widen.scratch_masks + rest.scratch_masks, scratch_vectors));
    }
    let (text, scratch_vectors) = match (from, to) {
        _ if from == to => ("$0", 0),
        // Truncated toward zero, then saturated above, and NaN gives 0: the conversion gives the
        // least value for both. It holds a mask for the lanes above the range and one for NaN, and
        // the truncated lanes and the zero while it compares.
        (DType::F32, DType::Si32) => {
            return Ok(Form::new(
                "_mm512_mask_mov_epi32(_mm512_mask_mov_epi32(_mm512_cvttps_epi32($0), \
                 _mm512_cmp_ps_mask($0, _mm512_set1_ps(2147483648.0f), _CMP_GE_OQ), \
                 _mm512_set1_epi32(0x7fffffff)), _mm512_cmp_ps_mask($0, $0, _CMP_UNORD_Q), \
                 _mm512_setzero_si512())",
                AVX512F,
            )
            .with_scratch(2, 2));
        }
        (DType::Si32, DType::F32) => ("_mm512_cvtepi32_ps($0)", 0),
        (DType::I1, DType::F32) => ("_mm512_maskz_mov_ps($0, _mm512_set1_ps(1.0f))", 0),
        (DType::I1, DType::Si32 | DType::Ui32) => {
            ("_mm512_maskz_mov_epi32($0, _mm512_set1_epi32(1))", 0)
        }
        // The zero is in a register.
        (DType::F32, DType::I1) => (
            "_mm512_cmp_ps_mask($0, _mm512_setzero_ps(), _CMP_NEQ_UQ)",
            1,
        ),
        (DType::Si32 | DType::Ui32, DType::I1) => ("_mm512_test_epi32_mask($0, $0)", 0),
        // The zero is in a register.
        (DType::Si32, DType::Ui32) => ("_mm512_max_epi32($0, _mm512_setzero_si512())", 1),
        (DType::Ui32, DType::Si32) => ("_mm512_min_epu32($0, _mm512_set1_epi32(0x7fffffff))", 0),
        _ => bail!("x86 has no vector cast of {from} to {to}"),
    };
    Ok(Form::new(text, AVX512F).with_scratch(0, scratch_vectors))
}

/// 16 lanes of the 4-bit integer `dtype` widened to 32 bits, exactly: the lower and the upper 4
/// bits of each byte apart, interleaved into the 16 bytes of the lanes in order, each widened, and
/// for signed lanes `(x ^ 8) - 8`; and the vector registers it holds besides its operand and its
/// result: the upper bits, and for signed lanes the 8. None for any other dtype.
fn nibbles_widened(dtype: DType) -> Option<(String, usize)> {
    let unsigned = "_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(\
        _mm_and_si128($0, _mm_set1_epi8(15)), \
        _mm_and_si128(_mm_srli_epi16($0, 4), _mm_set1_epi8(15))))";
    Some(match dtype {
        DType::Ui4 => (unsigned.to_string(), 1),
        DType::Si4 => (
            format!(
                "_mm512_sub_epi32(_mm512_xor_si512({unsigned}, _mm512_set1_epi32(8)), \
                 _mm512_set1_epi32(8))"
            ),
            2,
        ),
        _ => return None,
    })
}

/// The form of `isa.cast` between 16 lanes of f32, in a 512-bit register, and of a narrower float,
/// in a 256- or 128-bit one, as `tensor.cast` converts: exactly to f32, and to bf16 or f16 rounded
/// to nearest-even, NaN staying NaN. A bf16 is the upper half of an f32, so the rounding adds
/// `0x7fff` and the lowest kept bit before it drops the lower half; it holds a mask for the NaN
/// lanes, and the upper halves, the rounded lanes and a zero, which the NaN test compares with.
/// An e4m3 code takes its magnitude, its sign and its normal and subnormal values at once.
fn half_cast_form(from: DType, to: DType) -> Option<Form> {
    Some(match (from, to) {
        (DType::Bf16, DType::F32) => Form::new(
            "_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32($0), 16))",
            AVX512F,
        ),
        (DType::F32, DType::Bf16) => Form::new(
            "_mm512_cvtepi32_epi16(_mm512_mask_mov_epi32(\
             _mm512_srli_epi32(_mm512_add_epi32(_mm512_castps_si512($0), \
             _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(\
             _mm512_srli_epi32(_mm512_castps_si512($0), 16), _mm512_set1_epi32(1)))), 16), \
             _mm512_cmp_ps_mask($0, $0, _CMP_UNORD_Q), _mm512_or_si512(\
             _mm512_srli_epi32(_mm512_castps_si512($0), 16), _mm512_set1_epi32(0x40))))",
            AVX512F,
        )
        .with_scratch(1, 3),
        (DType::F16, DType::F32) => Form::new("_mm512_cvtph_ps($0)", AVX512F),
        (DType::F8E4M3, DType::F32) => {
            Form::new("gptrs_f32x16_of_f8e4m3($0)", AVX512F).with_scratch(0, 4)
        }
        // An e5m2 code is the upper byte of the f16 of its value.
        (DType::F8E5M2, DType::F32) => Form::new(
            "_mm512_cvtph_ps(_mm256_slli_epi16(_mm256_cvtepu8_epi16($0), 8))",
            AVX512BW_VL,
        ),
        (DType::F32, DType::F16) => Form::new(
            "_mm512_cvtps_ph($0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)",
            AVX512F,
        ),
        _ => return None,
    })
}

/// The form of `isa.bitcast` of lanes of `from` to `to`, of one width.
fn bitcast_form(from: DType, to: DType) -> &'static str {
    match (from, to) {
        (DType::F32, _) => "_mm512_castps_si512($0)",
        (_, DType::F32) => "_mm512_castsi512_ps($0)",
        (DType::F64, _) => "_mm512_castpd_si512($0)",
        (_, DType::F64) => "_mm512_castsi512_pd($0)",
        _ => "$0",
    }
}

/// The form of `isa.splat` of the scalar `$0` into a register of `lanes` lanes of `dtype`.
pub fn splat(dtype: DType, lanes: usize) -> Result<Form> {
    let class = Class::of(dtype, lanes)?;
    let template = match class {
        Class::Float(_) => "_mm512_set1_$P($0)",
        Class::Int(..) => "_mm512_set1_$I($0)",
        Class::Mask(..) => "(($K)(0 - (uint64_t)($0 != 0)))",
        Class::Narrow(..) => "$H_set1_$E($0)",
        Class::Nibbles => "_mm_set1_epi8((char)((($0) & 15) * 17))",
    };
    let features = match class {
        Class::Mask(_, mask_lanes) => mask_features(mask_lanes),
        _ => AVX512F,
    };
    Ok(Form::new(class.spell(template), features))
}

/// Where a memory op reaches its elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// In memory, every lane inside the buffer.
    Memory,
    /// In memory, with a pad for each lane outside the buffer.
    Padded,
    /// In the register variables of a `reg` buffer.
    Registers,
}

/// The form of the memory op `op` at `place`, on registers of `lanes` lanes of `dtype`.
pub fn memory(op: &Op, place: Place, dtype: DType, lanes: usize) -> Result<Form> {
    let class = Class::of(dtype, lanes)?;
    let (forms, features) = match place {
        Place::Memory => MemoryForms::in_memory(&class)?,
        Place::Padded => (MemoryForms::padded(&class)?, class.move_features()),
        Place::Registers => (MemoryForms::in_registers(&class)?, class.move_features()),
    };
    let (text, scratch_vectors) = forms.pick(op);
    if text.is_empty() {
        let at = match place {
            Place::Memory => "in memory",
            Place::Padded => "that pads",
            Place::Registers => "of a reg buffer",
        };
        bail!("x86 has no {} of {dtype} lanes {at}", op.name());
    }
    Ok(Form::new(class.spell(text), features).with_scratch(0, scratch_vectors))
}

/// The memory ops, in the order of the forms of [`MemoryForms`].
const MEMORY_OPS: [fn(&Op) -> bool; 6] = [
    is::<isa::Load>,
    is::<isa::MaskedLoad>,
    is::<isa::Broadcast>,
    is::<isa::MaskedBroadcast>,
    is::<isa::Store>,
    is::<isa::MaskedStore>,
];

/// The forms of the six memory ops on one class of lanes in one place, in the order of
/// [`MEMORY_OPS`], and the vector registers each holds besides its operands and its result.
struct MemoryForms {
    texts: [String; 6],
    scratch_vectors: [usize; 6],
}

impl MemoryForms {
    fn new(forms: [&str; 6]) -> Self {
        Self {
            texts: forms.map(str::to_string),
            scratch_vectors: [0; 6],
        }
    }

    /// The forms on lanes of `class` in memory, and the extensions they need.
    fn in_memory(class: &Class) -> Result<(Self, &'static [Feature])> {
        Ok(match class {
            Class::Float(_) => (
                Self::new([
                    "_mm512_loadu_$P($A)",
                    "_mm512_maskz_loadu_$P($Q, $A)",
                    "_mm512_set1_$P(*($A))",
                    "_mm512_maskz_mov_$P($Q, $B)",
                    "_mm512_storeu_$P($A, $V)",
                    "_mm512_mask_storeu_$P($A, $Q, $V)",
                ]),
                AVX512F,
            ),
            Class::Int(..) => (
                Self::new([
                    "_mm512_loadu_si512($A)",
                    "_mm512_maskz_loadu_$I($Q, $A)",
                    "_mm512_set1_$I(*($A))",
                    "_mm512_maskz_mov_$I($Q, $B)",
                    "_mm512_storeu_si512($A, $V)",
                    "_mm512_mask_storeu_$I($A, $Q, $V)",
                ]),
                AVX512F,
            ),
            Class::Mask(_, lanes) => Self::of_bytes(*lanes)?,
            Class::Nibbles => (
                Self::new(["_mm_loadl_epi64((const __m128i *)($A))", "", "", "", "", ""]),
                AVX512F,
            ),
            Class::Narrow(.., features) => (
                Self::new([
                    "$H_loadu_$S((const void *)($A))",
                    "$H_maskz_loadu_$E($Q, (const void *)($A))",
                    "$H_set1_$E(*($A))",
                    "$H_maskz_mov_$E($Q, $B)",
                    "$H_storeu_$S((void *)($A), $V)",
                    "$H_mask_storeu_$E((void *)($A), $Q, $V)",
                ]),
                features,
            ),
        })
    }

    /// The forms on `i1` lanes, a byte each in memory, held in a mask of `lanes` lanes, and the
    /// extensions they need: the byte instructions, on 128- and 256-bit registers below 64 lanes.
    fn of_bytes(lanes: usize) -> Result<(Self, &'static [Feature])> {
        let (prefix, load, store, features) = match lanes {
            8 => (
                "_mm",
                "_mm_loadl_epi64((const __m128i *)($A))",
                "_mm_storel_epi64((__m128i *)($A), $X)",
                AVX512BW_VL,
            ),
            16 => (
                "_mm",
                "_mm_loadu_si128((const void *)($A))",
                "_mm_storeu_si128((void *)($A), $X)",
                AVX512BW_VL,
            ),
            32 => (
                "_mm256",
                "_mm256_loadu_si256((const void *)($A))",
                "_mm256_storeu_si256((void *)($A), $X)",
                AVX512BW_VL,
            ),
            64 => (
                "_mm512",
                "_mm512_loadu_si512((const void *)($A))",
                "_mm512_storeu_si512((void *)($A), $X)",
                AVX512BW,
            ),
            other => bail!("x86 has no mask of {other} lanes"),
        };
        let ones = format!("{prefix}_set1_epi8(1)");
        let bytes = format!("{prefix}_maskz_mov_epi8($V, {ones})");
        // The loads and the stores move the flags through a register of bytes.
        let forms = Self {
            texts: [
                format!("(($K){prefix}_test_epi8_mask({load}, {ones}))"),
                format!(
                    "(($K){prefix}_test_epi8_mask({prefix}_maskz_loadu_epi8($Q, (const void *)($A)), {ones}))"
                ),
                "(($K)(0 - (uint64_t)(*($A) != 0)))".to_string(),
                "(($K)((0 - (uint64_t)(*($A) != 0)) & $Q))".to_string(),
                store.replace("$X", &bytes),
                format!("{prefix}_mask_storeu_epi8((void *)($A), $Q, {bytes})"),
            ],
            scratch_vectors: [1, 1, 0, 0, 1, 1],
        };
        Ok((forms, features))
    }

    /// The forms on the register variables of a `reg` buffer of lanes of `class`.
    fn in_registers(class: &Class) -> Result<Self> {
        Ok(match class {
            Class::Float(_) => Self::new([
                "$R",
                "_mm512_maskz_mov_$P($Q, $R)",
                "",
                "",
                "$R = $V",
                "$R = _mm512_mask_mov_$P($R, $Q, $V)",
            ]),
            Class::Int(..) => Self::new([
                "$R",
                "_mm512_maskz_mov_$I($Q, $R)",
                "",
                "",
                "$R = $V",
                "$R = _mm512_mask_mov_$I($R, $Q, $V)",
            ]),
            Class::Narrow(..) => Self::new([
                "$R",
                "$H_maskz_mov_$E($Q, $R)",
                "",
                "",
                "$R = $V",
                "$R = $H_mask_mov_$E($R, $Q, $V)",
            ]),
            Class::Mask(..) => bail!("a reg buffer holds no i1 lanes"),
            Class::Nibbles => bail!("a reg buffer holds no 4-bit lanes"),
        })
    }

    /// The forms of the reads of lanes of `class` in memory that pad: `$D` is the pad and `$F`
    /// the flags of the lanes inside the buffer, which for a broadcast are all set or none.
    fn padded(class: &Class) -> Result<Self> {
        Ok(match class {
            Class::Float(_) => Self::new([
                "_mm512_mask_loadu_$P(_mm512_set1_$P($D), $F, $A)",
                "_mm512_mask_loadu_$P(_mm512_maskz_mov_$P($Q, _mm512_set1_$P($D)), ($Q) & ($F), $A)",
                "$B",
                "_mm512_maskz_mov_$P($Q, $B)",
                "",
                "",
            ]),
            Class::Int(..) => Self::new([
                "_mm512_mask_loadu_$I(_mm512_set1_$I($D), $F, $A)",
                "_mm512_mask_loadu_$I(_mm512_maskz_mov_$I($Q, _mm512_set1_$I($D)), ($Q) & ($F), $A)",
                "$B",
                "_mm512_maskz_mov_$I($Q, $B)",
                "",
                "",
            ]),
            Class::Narrow(..) => Self::new([
                "$H_mask_loadu_$E($H_set1_$E($D), $F, (const void *)($A))",
                "$H_mask_loadu_$E($H_maskz_mov_$E($Q, $H_set1_$E($D)), ($Q) & ($F), (const void *)($A))",
                "$B",
                "$H_maskz_mov_$E($Q, $B)",
                "",
                "",
            ]),
            Class::Mask(..) => bail!("x86 has no padded read of i1 lanes"),
            Class::Nibbles => bail!("x86 has no padded read of 4-bit lanes"),
        })
    }

    /// The form of the memory op `op` and the vector registers it holds: an empty form where it
    /// has none.
    fn pick(&self, op: &Op) -> (&str, usize) {
        for (position, is_op) in MEMORY_OPS.iter().enumerate() {
            if is_op(op) {
                return (&self.texts[position], self.scratch_vectors[position]);
            }
        }
        ("", 0)
    }
}

/// The element of `dtype` at `address` in every lane of a register of `lanes` lanes where
/// `condition` holds, else `fill`: a masked scalar load, which reads nothing where the condition
/// fails, so the C has no branch.
pub fn broadcast_one(
    dtype: DType,
    lanes: usize,
    address: &str,
    condition: &str,
    fill: &str,
) -> Result<String> {
    let flag = format!("(__mmask8)({condition})");
    if let Class::Narrow(width, element, _) = Class::of(dtype, lanes)? {
        let prefix = width.prefix();
        let one = match element {
            "epi16" => "w",
            _ => "b",
        };
        return Ok(format!(
            "{prefix}_broadcast{one}_{element}(_mm_mask_loadu_{element}(_mm_set1_{element}({fill}), {flag}, (const void *)({address})))"
        ));
    }
    Ok(match dtype {
        DType::F32 => format!(
            "_mm512_broadcastss_ps(_mm_mask_load_ss(_mm_set_ss({fill}), {flag}, (const float *)({address})))"
        ),
        DType::F64 => format!(
            "_mm512_broadcastsd_pd(_mm_mask_load_sd(_mm_set_sd({fill}), {flag}, (const double *)({address})))"
        ),
        DType::Si32 | DType::Ui32 => format!(
            "_mm512_broadcastd_epi32(_mm_castps_si128(_mm_mask_load_ss(_mm_castsi128_ps(_mm_cvtsi32_si128({fill})), {flag}, (const float *)({address}))))"
        ),
        DType::Si64 | DType::Ui64 => format!(
            "_mm512_broadcastq_epi64(_mm_castpd_si128(_mm_mask_load_sd(_mm_castsi128_pd(_mm_cvtsi64_si128({fill})), {flag}, (const double *)({address}))))"
        ),
        other => bail!("x86 has no broadcast of one {other} element"),
    })
}

/// The constant of `_mm_prefetch` for the hint `hint` that a target's memory names.
pub fn cache_hint(hint: Prefetch) -> &'static str {
    match hint {
        Prefetch::T0 => "_MM_HINT_T0",
        Prefetch::T1 => "_MM_HINT_T1",
        Prefetch::T2 => "_MM_HINT_T2",
        Prefetch::Nta => "_MM_HINT_NTA",
    }
}

fn vector_type(module: &Module, value: Value) -> Result<VectorType> {
    match module.ty(value) {
        Type::Vector(vector) => Ok(vector.clone()),
        other => bail!("{other} is not a group of registers"),
    }
}

/// The rules of the C that the x86 backend writes for `isa` code: AVX-512 intrinsics.
pub struct Avx512;

/// What the code of `form` needs besides its operands and its result.
fn needs(form: Form) -> isa::Needs {
    isa::Needs {
        features: form.features,
        scratch_masks: form.scratch_masks,
        scratch_vectors: form.scratch_vectors,
        memory_operands: form.memory_operands,
    }
}

impl isa::Rules for Avx512 {
    fn lanewise(&self, op: &Op, dtype: DType, lanes: usize) -> Result<isa::Needs> {
        register_type(dtype, lanes)?;
        Ok(needs(lanewise_form(op, dtype, lanes, [None, None])?))
    }

    fn holds(&self, dtype: DType, lanes: usize) -> Result<isa::Needs> {
        register_type(dtype, lanes)?;
        let (_, features) = MemoryForms::in_memory(&Class::of(dtype, lanes)?)?;
        Ok(isa::Needs {
            features,
            ..isa::Needs::default()
        })
    }

    fn needs(&self, module: &Module, inst: Inst) -> Result<isa::Needs> {
        let op = module.op(inst);
        if op.get::<isa::Prefetch>().is_some() {
            return Ok(isa::Needs::default());
        }
        if let Some(access) = isa::Access::of(module, inst) {
            let (dtype, lanes) = (access.vector.dtype, access.vector.lanes());
            register_type(dtype, lanes)?;
            if isa::in_registers(module, access.buffer) {
                memory(op, Place::Registers, dtype, lanes)?;
            }
            let place = match access.pad {
                Some(_) => Place::Padded,
                None => Place::Memory,
            };
            return Ok(needs(memory(op, place, dtype, lanes)?));
        }
        let ty = vector_type(module, module.results(inst)[0])?;
        let lanes = ty.lanes();
        register_type(ty.dtype, lanes)?;
        if op.get::<isa::Splat>().is_some() {
            return Ok(needs(splat(ty.dtype, lanes)?));
        }
        if op.get::<isa::Expand>().is_some() {
            return Ok(isa::Needs::default());
        }
        if op.get::<isa::TileMask>().is_some() {
            return Ok(isa::Needs {
                features: mask_features(lanes),
                ..isa::Needs::default()
            });
        }
        Ok(needs(lanewise(module, inst)?))
    }
}
