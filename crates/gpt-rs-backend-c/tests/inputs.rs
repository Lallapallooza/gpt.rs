//! What a run of a compiled program reads from its inputs.

use std::cell::UnsafeCell;
use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs_backend_c::CBackend;
use gpt_rs_ir::host::{ExternalBytes, HostTensor};
use gpt_rs_ir::ir::{DType, Module, TensorType};

/// f32 values that the test rewrites between runs, at the same address. A memory-mapped file
/// changes the same way when another file is mapped where one was unmapped.
struct Rewritable(UnsafeCell<Vec<f32>>);

// SAFETY: the test writes the values only between runs, while no reference to them exists.
unsafe impl Sync for Rewritable {}

impl Rewritable {
    /// # Safety
    /// No reference to the values may exist: no run that reads them is in progress.
    unsafe fn write(&self, values: &[f32]) {
        unsafe { (*self.0.get()).copy_from_slice(values) }
    }
}

impl AsRef<[u8]> for Rewritable {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: no write is in progress while a run reads the values, and f32 has no padding.
        let values = unsafe { (*self.0.get()).as_slice() };
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
    }
}

/// Multiples of 0.25 in [-1, 1.5]. Their products and the sums of up to 2^14 of them are exact in
/// f32, so every summation order gives the same result.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + seed * 13) % 11) as f32 * 0.25 - 1.0)
        .collect()
}

#[test]
fn each_run_reads_the_values_its_inputs_hold() {
    const N: usize = 96;
    let text = format!(
        "func.func @main(%0: tensor<{N}x{N}xf32>, %1: tensor<{N}x{N}xf32>) -> tensor<{N}x{N}xf32> {{
  %2 = tensor.dot_general(%0, %1) {{contract_lhs = [1], contract_rhs = [0]}} : (tensor<{N}x{N}xf32>, tensor<{N}x{N}xf32>) -> tensor<{N}x{N}xf32>
  func.return(%2) : (tensor<{N}x{N}xf32>) -> ()
}}"
    );
    let module = Arc::new(Module::from_text(&text).expect("the text parses"));
    let backend = CBackend::new();
    let main = backend.compile(&module, "main").expect("@main compiles");
    let buffers = [0, 1].map(|_| Arc::new(Rewritable(UnsafeCell::new(vec![0.0; N * N]))));
    let inputs = buffers.clone().map(|buffer| {
        let bytes = ExternalBytes::new(buffer, 0, N * N * 4).expect("a view of the buffer");
        let view = HostTensor::view(&TensorType::new(DType::F32, [N, N]), bytes);
        backend
            .upload(view.expect("a view of the buffer"))
            .expect("an input tensor")
    });
    for run in 0..4 {
        let (a, b) = (values(N * N, run), values(N * N, run + 100));
        // SAFETY: no run is in progress.
        unsafe {
            buffers[0].write(&a);
            buffers[1].write(&b);
        }
        let out = backend.run(&main, inputs.to_vec()).expect("the run");
        let out = backend.download(&out[0]).expect("the result");
        let got = out.as_f32().expect("f32 results");
        let want: Vec<f32> = (0..N * N)
            .map(|ij| (0..N).map(|p| a[ij / N * N + p] * b[p * N + ij % N]).sum())
            .collect();
        let wrong = got
            .iter()
            .zip(&want)
            .filter(|(got, want)| got != want)
            .count();
        assert_eq!(
            wrong,
            0,
            "run {run}: {wrong} of {} outputs differ",
            want.len()
        );
    }
}
