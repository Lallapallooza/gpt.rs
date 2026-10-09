//! Random inputs, and the interpreter's results of a module before and after its passes.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use gpt_rs_ir::dialects::func::{Func, ParamId};
use gpt_rs_ir::host::{HostTensor, HostValue, TensorData};
use gpt_rs_ir::interp;
use gpt_rs_ir::minifloat::{e8m0_decode, Format};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::types::{DType, Dim, IntFormat, Type};
use half::{bf16, f16};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The extent a run gives each dynamic dimension. Random integers stay below it, so they index
/// any axis of that extent.
const DYNAMIC_EXTENT: usize = 3;

/// The extents a run gives `dims`.
pub fn run_dims(dims: &[Dim]) -> Vec<usize> {
    dims.iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
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
    let float = |i: usize, rng: &mut StdRng| -> f32 {
        special(specials.floats(), i, offset).unwrap_or_else(|| rng.gen_range(-4.0f32..4.0))
    };
    let dtype = tensor.dtype;
    let data = match dtype {
        DType::F64 => TensorData::F64((0..count).map(|i| f64::from(float(i, rng))).collect()),
        DType::F32 => TensorData::F32((0..count).map(|i| float(i, rng)).collect()),
        DType::F16 => TensorData::F16((0..count).map(|i| f16::from_f32(float(i, rng))).collect()),
        DType::Bf16 => {
            TensorData::Bf16((0..count).map(|i| bf16::from_f32(float(i, rng))).collect())
        }
        DType::I1 => TensorData::I1((0..count).map(|_| rng.gen_range(0..2)).collect()),
        _ => match dtype.int_format() {
            // Small integers index any axis of a run.
            Some(int) => {
                let specials = specials.ints(int);
                integers(dtype, count, |i| {
                    special(&specials, i, offset)
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
            for i in 0..count {
                packed.set(i, value(i) as u8);
            }
        }
        data => {
            let size = dtype.size_in_bytes().expect("a whole-byte dtype");
            for (i, element) in data.bytes_mut().chunks_exact_mut(size).enumerate() {
                element.copy_from_slice(&value(i).to_le_bytes()[..size]);
            }
        }
    }
    Ok(data)
}

/// Whether `a` and `b` have the same bits, except that any NaN equals any NaN: the IR leaves the
/// sign and payload of a NaN unspecified. `-0.0` and `0.0` differ.
fn same_float(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
}

fn same_f32(a: f32, b: f32) -> bool {
    same_float(a.into(), b.into())
}

/// The value of every element of a float or `e8m0` tensor, exactly.
fn float_values(data: &TensorData) -> Option<Vec<f64>> {
    let codes = |format: Format, v: &[u8]| v.iter().map(|&c| format.decode(c.into())).collect();
    Some(match data {
        TensorData::F64(v) => v.to_vec(),
        TensorData::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        TensorData::F16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::Bf16(v) => v.iter().map(|x| x.to_f64()).collect(),
        TensorData::F8E4M3(v) => v.iter().map(|x| f64::from(x.to_f32())).collect(),
        TensorData::F8E5M2(v) => codes(Format::F8E5M2, v),
        TensorData::F8E4M3Fnuz(v) => codes(Format::F8E4M3FNUZ, v),
        TensorData::F8E5M2Fnuz(v) => codes(Format::F8E5M2FNUZ, v),
        TensorData::E8M0(v) => v.iter().map(|&c| e8m0_decode(c)).collect(),
        TensorData::Packed(packed) => {
            let format = Format::of(packed.dtype())?;
            (0..packed.len())
                .map(|i| format.decode(packed.get(i).into()))
                .collect()
        }
        _ => return None,
    })
}

/// Whether `a` and `b` are tensors of one shape and dtype whose elements compare equal as
/// `same_float` compares floats, and bit for bit otherwise.
pub fn bits_eq(a: &HostValue, b: &HostValue) -> bool {
    let (Some(a), Some(b)) = (a.as_tensor(), b.as_tensor()) else {
        return false;
    };
    if a.dims != b.dims || a.dtype() != b.dtype() {
        return false;
    }
    match (float_values(&a.data), float_values(&b.data)) {
        (Some(x), Some(y)) => x.iter().zip(&y).all(|(&p, &q)| same_float(p, q)),
        _ => match (&a.data, &b.data) {
            (TensorData::Packed(x), TensorData::Packed(y)) => {
                (0..x.len()).all(|i| x.get(i) == y.get(i))
            }
            (x, y) => x.bytes() == y.bytes(),
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
    module
        .funcs()
        .into_iter()
        .map(|f| module.op(f).as_func().expect("a function").clone())
        .collect()
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

/// Runs the function `name` of `module` on `inputs`, after the module's load-time functions.
fn run_bound(module: &Module, name: &str, inputs: &Inputs) -> anyhow::Result<Vec<HostValue>> {
    let mut params = inputs.params.clone();
    let funcs = funcs(module);
    let args_of =
        |func: &Func, params: &HashMap<ParamId, HostValue>| -> anyhow::Result<Vec<HostValue>> {
            let mut run_time = inputs.run_time.iter();
            (0..func.args.len())
                .map(|index| match func.param(index) {
                    Some(id) => params.get(&id).cloned().ok_or_else(|| {
                        anyhow::anyhow!("@{}: no value for parameter {}", func.name, id.0)
                    }),
                    None => run_time
                        .next()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("@{}: too few run-time inputs", func.name)),
                })
                .collect()
        };
    for func in funcs.iter().filter(|f| f.is_load_time()) {
        let results = interp::run(module, &func.name, args_of(func, &params)?)?;
        params.extend(func.result_params.iter().copied().zip(results));
    }
    let func = funcs
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| anyhow::anyhow!("no function @{name}"))?;
    interp::run(module, name, args_of(func, &params)?)
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
    for specials in [specials, Specials::None] {
        let inputs = random_inputs(module, func, seed, round, specials)?;
        match run_bound(module, &func.name, &inputs) {
            Ok(results) => return Ok((inputs, results)),
            Err(err) => last = format!("{err:#}"),
        }
    }
    Err(format!("@{}: the input does not run: {last}", func.name))
}

/// Whether `got` equals `want` up to rounding: f32 elements within `1e-5 * max(|got|, |want|) +
/// 1e-6` of each other, NaNs and infinities in the same places, and everything else as
/// [`bits_eq`] compares it.
fn close(want: &HostValue, got: &HostValue) -> bool {
    let (HostValue::Tensor(a), HostValue::Tensor(b)) = (want, got) else {
        return bits_eq(want, got);
    };
    let (TensorData::F32(x), TensorData::F32(y)) = (&a.data, &b.data) else {
        return bits_eq(want, got);
    };
    let near = |&(x, y): &(&f32, &f32)| {
        if !x.is_finite() || !y.is_finite() {
            return same_f32(*x, *y);
        }
        (x - y).abs() <= 1e-5 * x.abs().max(y.abs()) + 1e-6
    };
    a.dims == b.dims && x.len() == y.len() && x.iter().zip(y.iter()).all(|p| near(&p))
}

/// Runs every run-time function of `before` and `after` on the same random inputs and compares
/// their results: as [`bits_eq`] does, or up to rounding with `rounding`.
pub fn interp_differential(
    before: &Module,
    after: &Module,
    seed: u64,
    rounding: bool,
) -> Result<(), String> {
    let equal = |a: &HostValue, b: &HostValue| if rounding { close(a, b) } else { bits_eq(a, b) };
    let specials = if rounding {
        Specials::Finite
    } else {
        Specials::All
    };
    for func in funcs(before).iter().filter(|f| !f.is_load_time()) {
        for round in 0..ROUNDS {
            let (inputs, want) = run_inputs(before, func, seed, round, specials)?;
            let got = run_bound(after, &func.name, &inputs)
                .map_err(|err| format!("@{}: the output does not run: {err:#}", func.name))?;
            if want.len() != got.len() || want.iter().zip(&got).any(|(a, b)| !equal(a, b)) {
                return Err(format!(
                    "@{}: results differ after the passes (round {round})\nbefore: {want:?}\nafter: {got:?}",
                    func.name
                ));
            }
        }
    }
    Ok(())
}
