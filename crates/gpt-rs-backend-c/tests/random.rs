//! The C backend draws the stream that the spec defines for `tensor.rng_uniform`: Philox4x32-10,
//! checked against Random123's known answer.

use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::PortableBackend;
use gpt_rs::tensor::HostTensor;
use gpt_rs_backend_c::CBackend;
use gpt_rs_ir::host::TensorData;
use gpt_rs_ir::module::Module;

#[test]
fn rng_uniform_draws_the_philox_known_answer() -> Result<()> {
    let module = Module::from_text(
        "func.func @main(%0: tensor<2xui64>) -> (tensor<2xui64>, tensor<4xf32>) {
  %1, %2 = tensor.rng_uniform(%0) {shape = [4], dtype = f32} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xf32>)
  func.return(%1, %2) : (tensor<2xui64>, tensor<4xf32>) -> ()
}",
    )?;
    let backend = CBackend::new();
    let exe = backend.compile(&Arc::new(module), "main")?;
    let state = HostTensor::new(vec![2], TensorData::Ui64(vec![0, 0].into()))?;
    let out = backend.run(&exe, &[backend.upload(state)?])?;
    // Random123's known answer for Philox4x32-10 with counter and key 0.
    let words = [0x6627_e8d5u32, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];
    // A uniform f32 is the top 24 bits of its word over 2^24, and one block moves the counter on
    // by one.
    let expected: Vec<f32> = words
        .iter()
        .map(|w| (w >> 8) as f32 / 16_777_216.0)
        .collect();
    assert_eq!(backend.download(&out[1])?.as_f32()?, expected);
    assert_eq!(
        backend.download(&out[0])?.data,
        TensorData::Ui64(vec![0, 1].into())
    );
    Ok(())
}
