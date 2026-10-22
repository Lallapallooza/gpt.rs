//! Random inputs, and the interpreter's results of a module before and after its passes.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use gpt_rs_ir::dialects::func::{Func, ParamId};
use gpt_rs_ir::host::{HostTensor, TensorData};
use gpt_rs_ir::interp;
use gpt_rs_ir::interp::HostValue;
use gpt_rs_ir::ir::{e8m0_decode, DType, Dim, Format, IntFormat, Module, Type};
use half::{bf16, f16};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The extent a run gives each dynamic dimension. Random integers stay below it, so they index
/// any axis of that extent.
const DYNAMIC_EXTENT: usize = 3;

/// The extents a run gives `dims`.
pub fn run_dims(dims: &[Dim]) -> Vec<usize> {
    dims.iter()
        .map(|dim| match dim {
            Dim::Static(extent) => *extent,
            Dim::Dynamic(_) => DYNAMIC_EXTENT,
        })
        .collect()
}

/// Signed zeros, infinities and NaNs: the quiet NaN, its negative and a NaN with a payload. Their
/// count is prime, as [`rotation`] needs.
const FLOATS: [f32; 7] = [
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -f32::NAN,
    f32::from_bits(0x7fe0_0001),
];

/// 0, 1, -1 and the extremes of an integer dtype. Their count is prime, as [`rotation`] needs.
fn ints(int: IntFormat) -> [i128; 5] {
    [0, 1, int.wrap(-1), int.min(), int.max()]
}

/// The values that start each input: the ones rewrites most often get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specials {
    /// `FLOATS` for floats, and the values of `ints` for integers.
    All,
    /// Only the finite ones: real numbers, for rewrites that hold in real arithmetic.
    Finite,
    /// No special values: random ones only.
    None,
}

impl Specials {
    fn floats(self) -> &'static [f32] {
        match self {
            Specials::All => &FLOATS,
            Specials::Finite => &[0.0, -0.0],
            Specials::None => &[],
        }
    }

    fn ints(self, int: IntFormat) -> Vec<i128> {
        match self {
            Specials::All | Specials::Finite => ints(int).to_vec(),
            Specials::None => Vec::new(),
        }
    }
}

/// The number of rounds of inputs a test runs.
pub const ROUNDS: usize = FLOATS.len() * FLOATS.len();

/// How far input `input` rotates the specials in round `round`: by `a + d * input` in round
/// `a + 7 * d`, where 7 is the count of `FLOATS`. Each list of specials has a prime count of at
/// most 7. So for any two inputs and any ordered pair of specials, some round starts the two
/// inputs with that pair, unless the inputs' positions differ by a multiple of the list's count.
pub fn rotation(input: usize, round: usize) -> usize {
    round % FLOATS.len() + round / FLOATS.len() * input
}

/// The special value at element `index` of an input rotated by `offset`, if that element holds one.
fn special<T: Copy>(specials: &[T], index: usize, offset: usize) -> Option<T> {
    (index < specials.len()).then(|| specials[(index + offset) % specials.len()])
}

/// Element `index` of a float input whose specials are rotated by `offset`: its special value, or
/// a random one.
fn float_element(specials: &[f32], index: usize, offset: usize, rng: &mut StdRng) -> f32 {
    special(specials, index, offset).unwrap_or_else(|| rng.gen_range(-4.0f32..4.0))
}

/// A random value of type `ty`. A tensor's first elements are the `specials`, rotated by `offset`.
/// Random values fill the rest.
pub fn random_value(
    ty: &Type,
    offset: usize,
    specials: Specials,
    rng: &mut StdRng,
) -> Result<HostValue, String> {
    let Type::Tensor(tensor) = ty else {
        return Err(format!("no random inputs for {ty}"));
    };
    let dims = run_dims(&tensor.dims);
    let count: usize = dims.iter().product();
    let floats = specials.floats();
    let dtype = tensor.dtype;
    let data = match dtype {
        DType::F64 => TensorData::F64(
            (0..count)
                .map(|element| f64::from(float_element(floats, element, offset, rng)))
                .collect(),
        ),
        DType::F32 => TensorData::F32(
            (0..count)
                .map(|element| float_element(floats, element, offset, rng))
                .collect(),
        ),
        DType::F16 => TensorData::F16(
            (0..count)
                .map(|element| f16::from_f32(float_element(floats, element, offset, rng)))
                .collect(),
        ),
        DType::Bf16 => TensorData::Bf16(
            (0..count)
                .map(|element| bf16::from_f32(float_element(floats, element, offset, rng)))
                .collect(),
        ),
        DType::I1 => TensorData::I1((0..count).map(|_| rng.gen_range(0..2)).collect()),
        _ => match dtype.int_format() {
            // Small integers index any axis of a run.
            Some(int) => {
                let int_specials = specials.ints(int);
                integers(dtype, count, |element| {
                    special(&int_specials, element, offset)
                        .unwrap_or_else(|| rng.gen_range(0..DYNAMIC_EXTENT as i128))
                })?
            }
            // Codes of narrow floats and scales: every one, NaN among them, is a value.
            None => {
                let bits = dtype.bits() as u32;
                integers(dtype, count, |_| i128::from(rng.gen::<u8>() & mask(bits)))?
            }
        },
    };
    HostTensor::new(dims, data)
        .map(HostValue::Tensor)
        .map_err(|err| err.to_string())
}

/// A byte with its low `bits` bits set.
fn mask(bits: u32) -> u8 {
    (1u16 << bits.min(8)).wrapping_sub(1) as u8
}

/// `count` elements of `dtype`, element `i` with the bits of `value(i)`.
fn integers(
    dtype: DType,
    count: usize,
    mut value: impl FnMut(usize) -> i128,
) -> Result<TensorData, String> {
    let mut data = TensorData::zeroed(dtype, count).map_err(|err| err.to_string())?;
    match &mut data {
        TensorData::Packed(packed) => {
            for element in 0..count {
                packed.set(element, value(element) as u8);
            }
        }
        data => {
            let size = dtype.size_in_bytes().expect("a whole-byte dtype");
            for (position, element) in data.bytes_mut().chunks_exact_mut(size).enumerate() {
                element.copy_from_slice(&value(position).to_le_bytes()[..size]);
            }
        }
    }
    Ok(data)
}

/// Whether `lhs` and `rhs` have the same bits, except that any NaN equals any NaN: the IR leaves
/// the sign and payload of a NaN unspecified. `-0.0` and `0.0` differ.
fn same_float(lhs: f64, rhs: f64) -> bool {
    lhs.to_bits() == rhs.to_bits() || lhs.is_nan() && rhs.is_nan()
}

fn same_f32(lhs: f32, rhs: f32) -> bool {
    same_float(lhs.into(), rhs.into())
}

/// The value of every element of a float or `e8m0` tensor, exactly.
fn float_values(data: &TensorData) -> Option<Vec<f64>> {
    let decode = |format: Format, codes: &[u8]| {
        codes
            .iter()
            .map(|&code| format.decode(code.into()))
            .collect()
    };
    Some(match data {
        TensorData::F64(values) => values.to_vec(),
        TensorData::F32(values) => values.iter().map(|&value| f64::from(value)).collect(),
        TensorData::F16(values) => values.iter().map(|value| value.to_f64()).collect(),
        TensorData::Bf16(values) => values.iter().map(|value| value.to_f64()).collect(),
        TensorData::F8E4M3(values) => values
            .iter()
            .map(|value| f64::from(value.to_f32()))
            .collect(),
        TensorData::F8E5M2(codes) => decode(Format::F8E5M2, codes),
        TensorData::F8E4M3Fnuz(codes) => decode(Format::F8E4M3FNUZ, codes),
        TensorData::F8E5M2Fnuz(codes) => decode(Format::F8E5M2FNUZ, codes),
        TensorData::E8M0(codes) => codes.iter().map(|&code| e8m0_decode(code)).collect(),
        TensorData::Packed(packed) => {
            let format = Format::of(packed.dtype())?;
            (0..packed.len())
                .map(|element| format.decode(packed.get(element).into()))
                .collect()
        }
        _ => return None,
    })
}

/// Whether `lhs` and `rhs` are tensors of one shape and dtype whose elements compare equal as
/// `same_float` compares floats, and bit for bit otherwise.
pub fn bits_eq(lhs: &HostValue, rhs: &HostValue) -> bool {
    let (Some(lhs_tensor), Some(rhs_tensor)) = (lhs.as_tensor(), rhs.as_tensor()) else {
        return false;
    };
    if lhs_tensor.dims != rhs_tensor.dims || lhs_tensor.dtype() != rhs_tensor.dtype() {
        return false;
    }
    match (
        float_values(&lhs_tensor.data),
        float_values(&rhs_tensor.data),
    ) {
        (Some(lhs_values), Some(rhs_values)) => lhs_values
            .iter()
            .zip(&rhs_values)
            .all(|(&lhs_value, &rhs_value)| same_float(lhs_value, rhs_value)),
        _ => match (&lhs_tensor.data, &rhs_tensor.data) {
            (TensorData::Packed(lhs_packed), TensorData::Packed(rhs_packed)) => (0..lhs_packed
                .len())
                .all(|element| lhs_packed.get(element) == rhs_packed.get(element)),
            (lhs_data, rhs_data) => lhs_data.bytes() == rhs_data.bytes(),
        },
    }
}

/// The inputs of one run: a value per parameter, and the run-time inputs in order.
#[derive(Default)]
struct Inputs {
    params: HashMap<ParamId, HostValue>,
    run_time: Vec<HostValue>,
}

fn funcs(module: &Module) -> Vec<Func> {
    let mut funcs = Vec::new();
    for func in module.funcs() {
        funcs.push(func.attrs().clone());
    }
    funcs
}

/// Random inputs for `func`, starting with `specials`, and a value for every other parameter
/// that a function of `module` reads.
fn random_inputs(
    module: &Module,
    func: &Func,
    seed: u64,
    round: usize,
    specials: Specials,
) -> Result<Inputs, String> {
    let mut rng = StdRng::seed_from_u64(seed ^ round as u64);
    let mut inputs = Inputs::default();
    for (index, ty) in func.args.iter().enumerate() {
        let value = random_value(ty, rotation(index, round), specials, &mut rng)?;
        match func.param(index) {
            Some(id) => {
                inputs.params.entry(id).or_insert(value);
            }
            None => inputs.run_time.push(value),
        }
    }
    for other in funcs(module) {
        for (index, ty) in other.args.iter().enumerate() {
            let Some(Entry::Vacant(slot)) = other.param(index).map(|id| inputs.params.entry(id))
            else {
                continue;
            };
            let value = random_value(ty, rotation(index, round), specials, &mut rng)?;
            slot.insert(value);
        }
    }
    Ok(inputs)
}

/// The arguments of `func`: the value in `params` of each parameter it reads, and the `run_time`
/// inputs in order.
fn args_of(
    func: &Func,
    params: &HashMap<ParamId, HostValue>,
    run_time: &[HostValue],
) -> anyhow::Result<Vec<HostValue>> {
    let mut run_time_inputs = run_time.iter();
    let mut args = Vec::with_capacity(func.args.len());
    for index in 0..func.args.len() {
        let arg = match func.param(index) {
            Some(id) => params.get(&id).cloned().ok_or_else(|| {
                anyhow::anyhow!("@{}: no value for parameter {}", func.name, id.0)
            })?,
            None => run_time_inputs
                .next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("@{}: too few run-time inputs", func.name))?,
        };
        args.push(arg);
    }
    Ok(args)
}

/// Runs the function `name` of `module` on `inputs`, after the module's load-time functions.
fn run_bound(module: &Module, name: &str, inputs: &Inputs) -> anyhow::Result<Vec<HostValue>> {
    let mut params = inputs.params.clone();
    let funcs = funcs(module);
    for func in funcs.iter().filter(|func| func.is_load_time()) {
        let results = interp::run(
            module,
            &func.name,
            args_of(func, &params, &inputs.run_time)?,
        )?;
        params.extend(func.result_params.iter().copied().zip(results));
    }
    let func = funcs
        .iter()
        .find(|func| func.name == name)
        .ok_or_else(|| anyhow::anyhow!("no function @{name}"))?;
    interp::run(module, name, args_of(func, &params, &inputs.run_time)?)
}

/// One round's inputs for `func` and its results in `module`: inputs starting with `specials`, or
/// plain random ones where the function cannot run on those.
fn run_inputs(
    module: &Module,
    func: &Func,
    seed: u64,
    round: usize,
    specials: Specials,
) -> Result<(Inputs, Vec<HostValue>), String> {
    let mut last = String::new();
    for tried_specials in [specials, Specials::None] {
        let inputs = random_inputs(module, func, seed, round, tried_specials)?;
        match run_bound(module, &func.name, &inputs) {
            Ok(results) => return Ok((inputs, results)),
            Err(err) => last = format!("{err:#}"),
        }
    }
    Err(format!("@{}: the input does not run: {last}", func.name))
}

/// How a rewrite may change the bits of a result, as the classes of a case's
/// `// ROUNDING: <classes>: <reason>` line name it. Every class holds in real arithmetic, so the
/// case's inputs are finite. With several classes, the loosest holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rounding {
    /// Bit for bit on finite inputs: `skip_pad`, a skipped product of the pad, which is 0 only
    /// where the weight is finite.
    Exact,
    /// Element by element, within this many units in the last place: `poly`, a polynomial in
    /// place of a math function, within 1, as `expand-math` measures its expansions against the
    /// interpreter's math functions.
    Ulps(u32),
    /// A sum whose terms round differently: in another order (`reassoc`), with each product rounded
    /// once with the sum (`contract`), through Winograd's transforms (`winograd`), or with a
    /// scale rounded into each term (`fold_scale`). Its error scales with its terms, not with its
    /// value, which cancels near 0, so each f32 element is within 1e-5 times the largest
    /// magnitude of its tensor, plus 1e-6.
    Sums,
}

/// The rounding of each class that a `ROUNDING:` line can name.
const ROUNDING_CLASSES: [(&str, Rounding); 6] = [
    ("skip_pad", Rounding::Exact),
    ("poly", Rounding::Ulps(1)),
    ("reassoc", Rounding::Sums),
    ("contract", Rounding::Sums),
    ("winograd", Rounding::Sums),
    ("fold_scale", Rounding::Sums),
];

impl Rounding {
    /// The rounding that the directive `line`, `<classes>: <reason>`, names.
    pub fn of_directive(line: &str) -> Result<Rounding, String> {
        let Some((classes, reason)) = line.split_once(':') else {
            return Err(format!(
                "`ROUNDING: {line}` names no class: write `<classes>: <reason>`"
            ));
        };
        if reason.trim().is_empty() {
            return Err(format!("`ROUNDING: {line}` gives no reason"));
        }
        let mut loosest = Rounding::Exact;
        for class in classes.split(',') {
            let name = class.trim();
            let Some(&(_, rounding)) = ROUNDING_CLASSES.iter().find(|(known, _)| *known == name)
            else {
                let mut known = Vec::new();
                for (class_name, _) in ROUNDING_CLASSES {
                    known.push(class_name);
                }
                return Err(format!(
                    "no rounding class `{name}`: the classes are {}",
                    known.join(", ")
                ));
            };
            loosest = loosest.max(rounding);
        }
        Ok(loosest)
    }

    /// Whether `got` equals `want` up to this rounding. Floats other than f32 compare as
    /// [`bits_eq`] does, and NaNs and infinities stay in their places.
    fn holds(self, want: &HostValue, got: &HostValue) -> bool {
        let (HostValue::Tensor(want_tensor), HostValue::Tensor(got_tensor)) = (want, got) else {
            return bits_eq(want, got);
        };
        let (TensorData::F32(want_values), TensorData::F32(got_values)) =
            (&want_tensor.data, &got_tensor.data)
        else {
            return bits_eq(want, got);
        };
        if want_tensor.dims != got_tensor.dims || want_values.len() != got_values.len() {
            return false;
        }
        let mut largest = 0f32;
        for value in want_values.iter().filter(|value| value.is_finite()) {
            largest = largest.max(value.abs());
        }
        for (&expected, &computed) in want_values.iter().zip(got_values.iter()) {
            if !expected.is_finite() || !computed.is_finite() {
                if !same_f32(expected, computed) {
                    return false;
                }
                continue;
            }
            let near = match self {
                Rounding::Exact => same_f32(expected, computed),
                Rounding::Ulps(ulps) => ulp_distance(expected, computed) <= u64::from(ulps),
                Rounding::Sums => (expected - computed).abs() <= 1e-5 * largest + 1e-6,
            };
            if !near {
                return false;
            }
        }
        true
    }
}

/// How many f32 values lie from `lhs` up to `rhs`, or down: 0 for `0.0` and `-0.0`.
fn ulp_distance(lhs: f32, rhs: f32) -> u64 {
    // The bits of a float, read as a sign and a magnitude, order the floats once the negative
    // ones count down from zero.
    let ordered = |value: f32| {
        let bits = i64::from(value.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    ordered(lhs).abs_diff(ordered(rhs))
}

/// Runs every run-time function of `before` and `after` on the same random inputs and compares
/// their results: as [`bits_eq`] does, or up to `rounding` on finite inputs.
pub fn interp_differential(
    before: &Module,
    after: &Module,
    seed: u64,
    rounding: Option<Rounding>,
) -> Result<(), String> {
    let equal = |expected: &HostValue, computed: &HostValue| match rounding {
        Some(allowed) => allowed.holds(expected, computed),
        None => bits_eq(expected, computed),
    };
    let specials = match rounding {
        Some(_) => Specials::Finite,
        None => Specials::All,
    };
    for func in funcs(before).iter().filter(|func| !func.is_load_time()) {
        for round in 0..ROUNDS {
            let (inputs, want) = run_inputs(before, func, seed, round, specials)?;
            let got = run_bound(after, &func.name, &inputs)
                .map_err(|err| format!("@{}: the output does not run: {err:#}", func.name))?;
            if want.len() != got.len()
                || want
                    .iter()
                    .zip(&got)
                    .any(|(want_value, got_value)| !equal(want_value, got_value))
            {
                return Err(format!(
                    "@{}: results differ after the passes (round {round})\nbefore: {want:?}\nafter: {got:?}",
                    func.name
                ));
            }
        }
    }
    Ok(())
}
