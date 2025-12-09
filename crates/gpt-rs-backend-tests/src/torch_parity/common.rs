use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use tch::Tensor as TchTensor;

use super::harness;

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

pub fn const_vec(len: usize, value: f32) -> Vec<f32> {
    vec![value; len]
}

pub fn tensor_from_vec(shape: &[usize], data: Vec<f32>) -> Tensor {
    Tensor::from_vec(Shape::new(shape.to_vec()), data).unwrap()
}

pub fn device_tensor_from_data<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    data: &[f32],
) -> DeviceTensor<B> {
    DeviceTensor::from_host(Arc::clone(backend), tensor_from_vec(shape, data.to_vec())).unwrap()
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
    tensor.to_host().unwrap().data().to_vec()
}

pub fn assert_close(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    let resolved = resolve_tolerance();
    let atol = resolved.atol;
    let rtol = resolved.rtol;
    for (idx, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e as f64 - a as f64).abs();
        let thresh = atol + rtol * e.abs().max(a.abs()) as f64;
        assert!(
            diff <= thresh,
            "value mismatch at index {idx}: expected {e}, actual {a}, diff {diff}, thresh {thresh}"
        );
    }
}

pub fn assert_close_tol(expected: &[f32], actual: &[f32], atol: f64, rtol: f64) {
    assert_eq!(expected.len(), actual.len());
    for (idx, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
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
