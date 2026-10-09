use std::sync::Arc;

use anyhow::Result;
use gpt_rs::nn::LayerNorm;
use gpt_rs::tensor::HostTensor;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::load_layer;

#[test]
fn optional_sublayer_needs_all_or_none_of_its_tensors() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let norm = |names: &[&str]| {
        let tensors = names.iter().map(|name| {
            (
                name.to_string(),
                HostTensor::f32([4], vec![1.0; 4]).unwrap(),
            )
        });
        load_layer(&backend, tensors, |p| {
            p.optional(|p| p.scoped("norm", |p| LayerNorm::new(p, 1e-5)))
        })
    };
    assert!(norm(&[])?.is_none());
    assert!(norm(&["norm.weight", "norm.bias"])?.is_some());
    let Err(err) = norm(&["norm.weight"]) else {
        panic!("a partly present sublayer is an error");
    };
    assert_eq!(err.to_string(), "missing tensor 'norm.bias' in checkpoint");
    Ok(())
}
