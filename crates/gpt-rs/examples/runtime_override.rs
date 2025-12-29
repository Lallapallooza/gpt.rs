use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional::{
    self, build_registry, with_registry, FunctionalOverrides, MatmulContext, MatmulEntry,
    MatmulImplementation,
};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use serde_json::json;

fn main() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let overrides: FunctionalOverrides = serde_json::from_value(json!({
        "gpt_rs::ops::functional::matmul": "force=custom_matmul"
    }))?;
    let registry = build_registry::<CpuPortableBackend>(&overrides);

    registry.register::<MatmulEntry<CpuPortableBackend>, _>(MatmulImplementation::<
        CpuPortableBackend,
    >::new(
        "custom_matmul",
        custom_matmul::<CpuPortableBackend>,
    ));

    let lhs = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let rhs = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    with_registry(Arc::clone(&registry), || {
        let output = functional::matmul(backend.as_ref(), &lhs, &rhs)?;
        let host = output.to_host()?;
        println!("custom override result: {:?}", host.data());
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}

fn custom_matmul<B: PortableBackend + 'static>(
    ctx: MatmulContext<'_, B>,
) -> Result<DeviceTensor<B>> {
    println!("running custom matmul for shape {:?}", ctx.a.shape().dims());
    naive_matmul(ctx)
}

fn tensor_from_data<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    data: &[f32],
) -> Result<DeviceTensor<B>> {
    let tensor = Tensor::from_vec(Shape::new([2, 2]), data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), tensor)
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

    let tensor = Tensor::from_vec(Shape::new([m, n]), out)?;
    DeviceTensor::from_host(ctx.a.backend(), tensor)
}
