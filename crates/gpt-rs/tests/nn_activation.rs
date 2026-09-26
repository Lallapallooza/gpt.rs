use std::sync::Arc;

use gpt_rs::module::Layer;
use gpt_rs::nn::layers::ActivationFunction;
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

/// Hand-computed values at -1, 1, and 7. Exact GELU and tanh GELU differ by 1.5e-4 at 1, and ReLU
/// and ReLU6 differ at 7, so the 1e-5 tolerance tells each pair apart.
#[test]
fn each_activation_function_applies_its_own_functional() {
    let backend = Arc::new(CpuPortableBackend::new());
    let x = Tensor::from_vec(Shape::new([3]), vec![-1.0, 1.0, 7.0]).unwrap();
    let x = DeviceTensor::from_host(Arc::clone(&backend), x).unwrap();
    for (activation, expected) in [
        (ActivationFunction::Gelu, [-0.158_655_25, 0.841_344_8, 7.0]),
        (
            ActivationFunction::GeluTanh,
            [-0.158_808_01, 0.841_192, 7.0],
        ),
        (ActivationFunction::Relu, [0.0, 1.0, 7.0]),
        (ActivationFunction::Relu6, [0.0, 1.0, 6.0]),
    ] {
        let y = activation.layer().call(&x).unwrap().to_host().unwrap();
        let close = y
            .data()
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= 1e-5);
        assert!(
            close,
            "{activation:?}: {:?}, expected {expected:?}",
            y.data()
        );
    }
}
