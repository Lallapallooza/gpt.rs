//! The kernel that the C program of a real `nn::Linear` calls, in the C source that
//! `generate_c_module` writes (`gpt-rs-cli convert`). The C backend's passes have filetests under
//! `filetests/backend/c/`.

use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::Linear;
use gpt_rs::ops::graph::{context::with_default_arena, GraphArena};
use gpt_rs::tensor::{DType, DeviceTensor, HostTensor};
use gpt_rs_backend_c::{generate_c_module, CBackend};
use gpt_rs_backend_tests::recording_backend::RecordingBackend;
use gpt_rs_backend_tests::{load_layer, tensor_as};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::Pipeline;
use gpt_rs_ir::types::TensorType;
use std::sync::Arc;

/// Records the programs the graph arena runs, which it compiles with the C backend's pipeline.
#[derive(Default)]
struct RecordingCBackend(RecordingBackend);

impl PortableBackend for RecordingCBackend {
    type TensorHandle = ();
    type Executable = usize;

    fn backend_name(&self) -> &str {
        "recording-c"
    }

    fn pipeline(&self) -> Pipeline {
        CBackend::new().pipeline()
    }

    fn params(&self) -> &ParamStore<()> {
        self.0.params()
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> anyhow::Result<usize> {
        self.0.compile(module, func)
    }

    fn run(&self, exe: &usize, args: &[()]) -> anyhow::Result<Vec<()>> {
        self.0.run(exe, args)
    }

    fn upload(&self, host: HostTensor) -> anyhow::Result<()> {
        self.0.upload(host)
    }

    fn download(&self, tensor: &()) -> anyhow::Result<HostTensor> {
        self.0.download(tensor)
    }
}

/// Runs an `nn::Linear` whose `[3, 4]` f8e4m3 weight has a one-element scale of dtype `scale`,
/// with inputs cast to `input` when it is set. Returns the module the graph arena compiled for it
/// and the name of its function. A model runs its layers inside one default arena, so the test does
/// too.
fn optimized_fp8_linear(input: Option<DType>, scale: DType) -> (Arc<Module>, String) {
    let backend = Arc::new(RecordingCBackend::default());
    let weight = HostTensor::from_le_bytes(&TensorType::new(DType::F8E4M3, [3, 4]), &[0x38; 12]);
    let tensors = [
        ("weight".to_string(), weight.expect("f8e4m3 weight")),
        (
            "weight_scale_inv".to_string(),
            tensor_as(&[], &[0.5], scale),
        ),
    ];
    let mut layer = load_layer(&backend, tensors, Linear::new).expect("the layer loads");
    layer.input_dtype = input;
    let x = tensor_as(&[2, 4], &[0.5; 8], DType::F32);
    let x = DeviceTensor::from_host(Arc::clone(&backend), x).expect("upload");
    with_default_arena(GraphArena::new(Arc::clone(&backend)), || {
        let out = layer.call(&x);
        out.materialize().expect("materialize linear");
    });
    backend.0.recorded_or_panic()
}

/// The C program reads an FP8 weight and its scale as stored, as `docs/backends/c.md` documents:
/// it calls the kernel that takes them in place. A change in how `nn::Linear` dequantizes
/// the weight, or in how C selects kernels, would widen the weight in memory with the numbers still
/// right, which Torch parity and `eval.py` do not see.
#[test]
fn c_linear_reads_fp8_weights_and_their_scale_as_stored() {
    let kernels = [
        (
            None,
            DType::Bf16,
            "gpt_rs_c_linear_nt_f32_f8e4m3_bf16_scale",
        ),
        (None, DType::F32, "gpt_rs_c_linear_nt_f32_f8e4m3_f32_scale"),
        (
            Some(DType::Bf16),
            DType::Bf16,
            "gpt_rs_c_linear_nt_bf16_f8e4m3_bf16_scale",
        ),
        (
            Some(DType::Bf16),
            DType::F32,
            "gpt_rs_c_linear_nt_bf16_f8e4m3_f32_scale",
        ),
    ];
    for (input, scale, kernel) in kernels {
        let (module, func) = optimized_fp8_linear(input, scale);
        let source = generate_c_module(&module, &func).expect("the C source");
        assert!(
            source.contains(&format!("{kernel}((const ")),
            "{kernel} not called:\n{source}"
        );
    }
}
