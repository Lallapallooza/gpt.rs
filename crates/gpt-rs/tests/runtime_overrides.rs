use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional::registry::{FunctionalPolicy, FunctionalRegistryEntry};
use gpt_rs::ops::functional::{
    self, build_registry, FunctionalOverrides, MatmulContext, MatmulEntry, MatmulImplementation,
};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use serde_json::json;

fn cpu_backend() -> Arc<CpuPortableBackend> {
    Arc::new(CpuPortableBackend::new())
}

static FAST_CALLS: AtomicUsize = AtomicUsize::new(0);
static SLOW_CALLS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn acquire_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test mutex poisoned")
}

fn tensor_from_data<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    data: &[f32],
) -> Result<DeviceTensor<B>> {
    let host = Tensor::from_vec(Shape::new([2, 2]), data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

fn overrides_from_str<B: PortableBackend + 'static>(raw: &str) -> FunctionalOverrides {
    let key = MatmulEntry::<B>::key().as_str();
    serde_json::from_value(json!({ key: raw })).expect("override config should deserialize")
}

fn reset_counters() {
    FAST_CALLS.store(0, Ordering::SeqCst);
    SLOW_CALLS.store(0, Ordering::SeqCst);
}

fn naive_matmul<B: PortableBackend + 'static>(
    ctx: MatmulContext<'_, B>,
) -> Result<DeviceTensor<B>> {
    let lhs = ctx.a.to_host()?;
    let rhs = ctx.b.to_host()?;

    let dims_lhs = ctx.a.shape().dims();
    let dims_rhs = ctx.b.shape().dims();
    assert_eq!(dims_lhs.len(), 2);
    assert_eq!(dims_rhs.len(), 2);
    assert_eq!(dims_lhs[1], dims_rhs[0]);

    let m = dims_lhs[0];
    let k = dims_lhs[1];
    let n = dims_rhs[1];

    let mut out = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += lhs.data()[row * k + kk] * rhs.data()[kk * n + col];
            }
            out[row * n + col] = acc;
        }
    }

    let requires_grad = ctx.a.requires_grad_flag() || ctx.b.requires_grad_flag();
    let tensor = Tensor::from_vec(Shape::new([m, n]), out)?.requires_grad(requires_grad);
    DeviceTensor::from_host(ctx.a.backend(), tensor)
}

fn fast_matmul<B: PortableBackend + 'static>(ctx: MatmulContext<'_, B>) -> Result<DeviceTensor<B>> {
    FAST_CALLS.fetch_add(1, Ordering::SeqCst);
    naive_matmul(ctx)
}

fn slow_matmul<B: PortableBackend + 'static>(ctx: MatmulContext<'_, B>) -> Result<DeviceTensor<B>> {
    SLOW_CALLS.fetch_add(1, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(5));
    naive_matmul(ctx)
}

fn sample_inputs<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) -> Result<(DeviceTensor<B>, DeviceTensor<B>)> {
    let lhs = tensor_from_data(backend, &[1.0, 2.0, 3.0, 4.0])?;
    let rhs = tensor_from_data(backend, &[5.0, 6.0, 7.0, 8.0])?;
    Ok((lhs, rhs))
}

#[test]
fn force_override_prefers_named_impl() {
    let backend = cpu_backend();
    let _guard = acquire_lock();
    reset_counters();
    let overrides = overrides_from_str::<CpuPortableBackend>("force=matmul_slow");
    assert!(matches!(
        overrides.policy(MatmulEntry::<CpuPortableBackend>::key()),
        FunctionalPolicy::Force { .. }
    ));
    let registry = build_registry::<CpuPortableBackend>(&overrides);

    let fast_impl = MatmulImplementation::<CpuPortableBackend>::new(
        "matmul_a_b",
        fast_matmul::<CpuPortableBackend>,
    );
    let slow_impl = MatmulImplementation::<CpuPortableBackend>::new(
        "matmul_slow",
        slow_matmul::<CpuPortableBackend>,
    );

    registry.register::<MatmulEntry<CpuPortableBackend>, _>(fast_impl);
    registry.register::<MatmulEntry<CpuPortableBackend>, _>(slow_impl);

    let (lhs, rhs) = sample_inputs(&backend).unwrap();

    let result = functional::with_registry(Arc::clone(&registry), || {
        functional::matmul(backend.as_ref(), &lhs, &rhs)
    })
    .unwrap();

    let host = result.to_host().unwrap();
    assert_eq!(FAST_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(SLOW_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(host.data(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn benchmark_override_caches_best_candidate() {
    let backend = cpu_backend();
    let _guard = acquire_lock();
    reset_counters();
    let overrides = overrides_from_str::<CpuPortableBackend>("benchmark(cache=2)");
    assert!(matches!(
        overrides.policy(MatmulEntry::<CpuPortableBackend>::key()),
        FunctionalPolicy::Benchmark { .. }
    ));
    let registry = build_registry::<CpuPortableBackend>(&overrides);

    let fast_impl = MatmulImplementation::<CpuPortableBackend>::new(
        "matmul_a_b",
        fast_matmul::<CpuPortableBackend>,
    );
    let slow_impl = MatmulImplementation::<CpuPortableBackend>::new(
        "matmul_slow",
        slow_matmul::<CpuPortableBackend>,
    );

    registry.register::<MatmulEntry<CpuPortableBackend>, _>(fast_impl);
    registry.register::<MatmulEntry<CpuPortableBackend>, _>(slow_impl);

    let (lhs, rhs) = sample_inputs(&backend).unwrap();

    let first = functional::with_registry(Arc::clone(&registry), || {
        functional::matmul(backend.as_ref(), &lhs, &rhs)
    })
    .unwrap();
    let second = functional::with_registry(Arc::clone(&registry), || {
        functional::matmul(backend.as_ref(), &lhs, &rhs)
    })
    .unwrap();

    let first_host = first.to_host().unwrap();
    let second_host = second.to_host().unwrap();
    assert_eq!(first_host.data(), second_host.data());
    assert_eq!(first_host.data(), &[19.0, 22.0, 43.0, 50.0]);

    assert_eq!(FAST_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(SLOW_CALLS.load(Ordering::SeqCst), 1);
}
