use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use gpt_rs::backend::PortableBackend;
use gpt_rs::nn::layers::{Embedding, LayerNorm, Linear, RmsNorm, RmsNormConfig};
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use tch::Tensor as TchTensor;

use super::harness;
pub use crate::{load_layer, panic_message};

pub const ATOL: f64 = 5e-4;
pub const RTOL: f64 = 1e-4;

static PARITY_CONFIG: OnceLock<ParityConfig> = OnceLock::new();
thread_local! {
    static PARITY_CONTEXT: RefCell<Option<ParityContext>> = const { RefCell::new(None) };
}

#[derive(Clone)]
struct ParityContext {
    backend: String,
    test_name: String,
}

#[derive(Clone, Default, Deserialize)]
struct ParityConfig {
    #[serde(default)]
    default: Option<ToleranceConfig>,
    #[serde(default)]
    rules: Vec<ToleranceRule>,
}

#[derive(Clone, Default, Deserialize)]
struct ToleranceRule {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    test: Option<String>,
    #[serde(default)]
    atol: Option<f64>,
    #[serde(default)]
    rtol: Option<f64>,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct ToleranceConfig {
    #[serde(default)]
    atol: Option<f64>,
    #[serde(default)]
    rtol: Option<f64>,
}

#[derive(Clone, Copy)]
struct ResolvedTolerance {
    atol: f64,
    rtol: f64,
}

pub struct ParityContextGuard {
    prev: Option<ParityContext>,
}

pub fn set_parity_context(backend: &str, test_name: &str) -> ParityContextGuard {
    PARITY_CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        let prev = slot.take();
        *slot = Some(ParityContext {
            backend: backend.to_string(),
            test_name: test_name.to_string(),
        });
        ParityContextGuard { prev }
    })
}

impl Drop for ParityContextGuard {
    fn drop(&mut self) {
        PARITY_CONTEXT.with(|slot| {
            *slot.borrow_mut() = self.prev.take();
        });
    }
}

fn current_context() -> Option<ParityContext> {
    PARITY_CONTEXT.with(|slot| slot.borrow().clone())
}

fn config_path() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../../configs/torch_parity.json")
}

fn load_parity_config() -> ParityConfig {
    let path = config_path();
    if !path.exists() {
        return ParityConfig::default();
    }
    let contents = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read torch parity config {}: {err}",
            path.display()
        )
    });
    serde_json::from_str(&contents).unwrap_or_else(|err| {
        panic!(
            "failed to parse torch parity config {}: {err}",
            path.display()
        )
    })
}

fn parity_config() -> &'static ParityConfig {
    PARITY_CONFIG.get_or_init(load_parity_config)
}

fn matches_pattern(value: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return value == pattern;
    }
    let parts: Vec<&str> = pattern.split('*').filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        return true;
    }
    let mut offset = 0usize;
    let mut start_index = 0usize;
    if !pattern.starts_with('*') {
        let first = parts[0];
        if !value.starts_with(first) {
            return false;
        }
        offset = first.len();
        start_index = 1;
    }
    for part in &parts[start_index..] {
        if let Some(found) = value[offset..].find(part) {
            offset += found + part.len();
        } else {
            return false;
        }
    }
    if !pattern.ends_with('*') {
        let last = parts.last().unwrap_or(&"");
        return value.ends_with(last);
    }
    true
}

fn apply_tolerance(target: &mut ResolvedTolerance, override_cfg: &ToleranceConfig) {
    if let Some(atol) = override_cfg.atol {
        target.atol = atol;
    }
    if let Some(rtol) = override_cfg.rtol {
        target.rtol = rtol;
    }
}

fn resolve_tolerance() -> ResolvedTolerance {
    let config = parity_config();
    let mut resolved = ResolvedTolerance {
        atol: ATOL,
        rtol: RTOL,
    };
    if let Some(defaults) = config.default {
        apply_tolerance(&mut resolved, &defaults);
    }
    let Some(context) = current_context() else {
        return resolved;
    };
    let mut backend_rule: Option<ToleranceConfig> = None;
    let mut test_rule: Option<ToleranceConfig> = None;
    let mut backend_test_rule: Option<ToleranceConfig> = None;

    for rule in &config.rules {
        let backend_match = rule
            .backend
            .as_deref()
            .map(|pattern| matches_pattern(&context.backend, pattern))
            .unwrap_or(false);
        let test_match = rule
            .test
            .as_deref()
            .map(|pattern| matches_pattern(&context.test_name, pattern))
            .unwrap_or(false);
        match (backend_match, test_match) {
            (true, true) => {
                backend_test_rule = Some(ToleranceConfig {
                    atol: rule.atol,
                    rtol: rule.rtol,
                });
            }
            (false, true) => {
                test_rule = Some(ToleranceConfig {
                    atol: rule.atol,
                    rtol: rule.rtol,
                });
            }
            (true, false) => {
                backend_rule = Some(ToleranceConfig {
                    atol: rule.atol,
                    rtol: rule.rtol,
                });
            }
            _ => {}
        }
    }

    if let Some(rule) = backend_rule {
        apply_tolerance(&mut resolved, &rule);
    }
    if let Some(rule) = test_rule {
        apply_tolerance(&mut resolved, &rule);
    }
    if let Some(rule) = backend_test_rule {
        apply_tolerance(&mut resolved, &rule);
    }

    resolved
}

pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

pub fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

pub fn random_vec_range(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
    let span = hi - lo;
    (0..len).map(|_| lo + rng.gen::<f32>() * span).collect()
}

pub fn random_vec_nonzero(rng: &mut StdRng, len: usize, min_abs: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            let mut v = rng.gen::<f32>() * 2.0 - 1.0;
            if v.abs() < min_abs {
                v = if v.is_sign_negative() {
                    -min_abs
                } else {
                    min_abs
                };
            }
            v
        })
        .collect()
}

/// NaNs of both signs and with a payload, both infinities, both zeros and two finite values. The
/// checks compare zeros without their sign, because Torch's `maximum` and `minimum` do not order
/// -0.0 below 0.0.
pub const SPECIAL_VALUES: [f32; 9] = [
    f32::NAN,
    -f32::NAN,
    f32::from_bits(0x7fc0_1234),
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    1.5,
    -2.0,
];

pub fn special_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| SPECIAL_VALUES[rng.gen_range(0..SPECIAL_VALUES.len())])
        .collect()
}

pub fn const_vec(len: usize, value: f32) -> Vec<f32> {
    vec![value; len]
}

pub fn tensor_from_vec(shape: &[usize], data: Vec<f32>) -> HostTensor {
    HostTensor::f32(shape, data).unwrap()
}

pub fn device_tensor_from_data<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
) -> DeviceTensor<B> {
    DeviceTensor::from_host(Arc::clone(backend), tensor_from_vec(shape, data.to_vec())).unwrap()
}

/// The token positions `positions` as an `si32` tensor.
pub fn positions_tensor<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    positions: std::ops::Range<usize>,
) -> DeviceTensor<B> {
    let positions: Vec<i32> = positions.map(|p| p as i32).collect();
    let host = HostTensor::si32([positions.len()], positions).unwrap();
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}

pub fn tch_tensor_from_vec(shape: &[usize], data: &[f32]) -> TchTensor {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    TchTensor::from_slice(data).reshape(dims.as_slice())
}

pub fn tensor_to_vec(tensor: &TchTensor) -> Vec<f32> {
    let len = tensor.numel();
    let mut data = vec![0f32; len];
    tensor.copy_data(&mut data, len);
    data
}

pub fn to_host_vec<B: PortableBackend + 'static>(tensor: &DeviceTensor<B>) -> Vec<f32> {
    tensor.to_host().unwrap().as_f32().unwrap().to_vec()
}

pub fn assert_close(expected: &[f32], actual: &[f32]) {
    let resolved = resolve_tolerance();
    assert_close_tol(expected, actual, resolved.atol, resolved.rtol);
}

/// NaN and the infinities match only themselves, so a NaN of any sign or payload must appear
/// exactly where `expected` has one.
pub fn assert_close_tol(expected: &[f32], actual: &[f32], atol: f64, rtol: f64) {
    assert_eq!(expected.len(), actual.len());
    for (idx, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        if !e.is_finite() || !a.is_finite() {
            assert!(
                e == a || (e.is_nan() && a.is_nan()),
                "value mismatch at index {idx}: expected {e}, actual {a}"
            );
            continue;
        }
        let diff = (e as f64 - a as f64).abs();
        let thresh = atol + rtol * e.abs().max(a.abs()) as f64;
        assert!(
            diff <= thresh,
            "value mismatch at index {idx}: expected {e}, actual {a}, diff {diff}, thresh {thresh}"
        );
    }
}

pub fn timed_torch<T, F: FnOnce() -> T>(f: F) -> T {
    harness::timed_torch(f)
}

pub fn timed_gpt<T, F: FnOnce() -> T>(f: F) -> T {
    harness::timed_gpt(f)
}

/// Host weights of a `Linear`: an `[out, in]` weight and an optional bias.
pub struct HostLinear {
    in_features: usize,
    out_features: usize,
    weight: HostTensor,
    bias: Option<HostTensor>,
}

impl HostLinear {
    pub fn random(rng: &mut StdRng, in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = tensor_from_vec(
            &[out_features, in_features],
            random_vec(rng, out_features * in_features),
        );
        let bias = bias.then(|| tensor_from_vec(&[out_features], random_vec(rng, out_features)));
        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    pub fn tensors(&self, prefix: &str) -> Vec<(String, HostTensor)> {
        let weight = (format!("{prefix}.weight"), self.weight.clone());
        let bias = self
            .bias
            .as_ref()
            .map(|bias| (format!("{prefix}.bias"), bias.clone()));
        std::iter::once(weight).chain(bias).collect()
    }

    pub fn layer<B: PortableBackend + 'static>(&self, backend: &Arc<B>) -> Linear<B> {
        load_layer(backend, self.tensors("proj"), |p| {
            p.scoped("proj", Linear::new)
        })
        .unwrap()
    }

    pub fn torch(&self, input: &TchTensor) -> TchTensor {
        let weight = tch_tensor_from_vec(
            &[self.out_features, self.in_features],
            self.weight.as_f32().unwrap(),
        );
        let bias = self
            .bias
            .as_ref()
            .map(|b| tch_tensor_from_vec(&[self.out_features], b.as_f32().unwrap()));
        input.linear(&weight, bias.as_ref())
    }
}

/// Host weights of a `Conv1D`: an `[in, out]` weight and a bias.
pub struct HostConv1D {
    in_features: usize,
    out_features: usize,
    weight: HostTensor,
    bias: HostTensor,
}

impl HostConv1D {
    pub fn random(rng: &mut StdRng, in_features: usize, out_features: usize) -> Self {
        let weight = tensor_from_vec(
            &[in_features, out_features],
            random_vec(rng, in_features * out_features),
        );
        let bias = tensor_from_vec(&[out_features], random_vec(rng, out_features));
        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    pub fn tensors(&self, prefix: &str) -> Vec<(String, HostTensor)> {
        vec![
            (format!("{prefix}.weight"), self.weight.clone()),
            (format!("{prefix}.bias"), self.bias.clone()),
        ]
    }

    pub fn torch(&self, input: &TchTensor) -> TchTensor {
        let weight = tch_tensor_from_vec(
            &[self.in_features, self.out_features],
            self.weight.as_f32().unwrap(),
        );
        let bias = tch_tensor_from_vec(&[self.out_features], self.bias.as_f32().unwrap());
        bias.addmm(input, &weight)
    }
}

pub fn embedding_layer<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    weight: HostTensor,
) -> Embedding<B> {
    load_layer(backend, [("embed.weight".to_string(), weight)], |p| {
        p.scoped("embed", Embedding::new)
    })
    .unwrap()
}

pub fn layer_norm_layer<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    weight: HostTensor,
    bias: HostTensor,
    eps: f32,
) -> LayerNorm<B> {
    let tensors = [
        ("norm.weight".to_string(), weight),
        ("norm.bias".to_string(), bias),
    ];
    load_layer(backend, tensors, |p| {
        p.scoped("norm", |p| LayerNorm::new(p, eps))
    })
    .unwrap()
}

pub fn rms_norm_layer<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    weight: HostTensor,
    config: RmsNormConfig,
) -> RmsNorm<B> {
    load_layer(backend, [("norm.weight".to_string(), weight)], |p| {
        p.scoped("norm", |p| RmsNorm::new(p, config))
    })
    .unwrap()
}
