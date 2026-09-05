use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Result};
use gpt_rs::checkpoint::{CheckpointReader, CheckpointSaver};
use gpt_rs::io::tensor_archive::TensorArchive;
use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::module::{Module, ParamVisitor, TensorRole};
use gpt_rs::nn::ActivationFunction;
use gpt_rs::runtime::{load_model_with_options, LoadOptions, LoadedModel, ModelInput, ModelOutput};
use gpt_rs::tensor::{DType, DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::tensor_as;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn unique_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    path.push(format!("{prefix}_{nanos}.{ext}"));
    path
}

struct TempFile {
    path: PathBuf,
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn assert_tensor_bytes_eq(a: &Tensor, b: &Tensor) {
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.shape().dims(), b.shape().dims());
    assert_eq!(a.to_le_bytes(), b.to_le_bytes());
}

fn small_gpt(backend: &Arc<CpuPortableBackend>) -> Result<Gpt<CpuPortableBackend>> {
    let config = GptConfig {
        vocab_size: 32,
        n_positions: 16,
        n_embd: 8,
        n_layer: 1,
        n_head: 2,
        n_inner: Some(16),
        layer_norm_epsilon: 1e-5,
        activation_function: ActivationFunction::GeluTanh,
    };
    Gpt::random(config, Arc::clone(backend), &mut StdRng::seed_from_u64(11))
}

#[test]
fn checkpoint_reader_and_runtime_loader_work_for_small_gpt() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let model = small_gpt(&backend)?;

    let tokens = vec![1usize, 2, 3, 4];
    let baseline = model.forward(&tokens)?;

    let checkpoint = TempFile {
        path: unique_path("gpt_rs_checkpoint_test", "bin"),
    };

    CheckpointSaver::save(&checkpoint.path, &model)?;

    let reader = CheckpointReader::open(&checkpoint.path)?;
    assert_eq!(reader.config().kind, "gpt");
    assert!(reader.entries().iter().all(|entry| entry.offset % 64 == 0));

    let mut expected = Vec::new();
    let mut collect = |name: &str, _: TensorRole, tensor: &DeviceTensor<CpuPortableBackend>| {
        expected.push((name.to_string(), tensor.to_host()?));
        Ok(())
    };
    model.visit_params(&mut ParamVisitor::new(&mut collect))?;

    for (name, expected_tensor) in &expected {
        let from_name = reader.get(name)?;
        assert_tensor_bytes_eq(&from_name, expected_tensor);
        let from_id = reader.get_by_base_id(gpt_rs::params::base_param_id(name)?)?;
        assert_tensor_bytes_eq(&from_id, expected_tensor);
    }

    let mut loaded = gpt_rs::runtime::load_model(Arc::clone(&backend), &checkpoint.path)?;
    let output = loaded.forward(ModelInput::Tokens(tokens))?;
    let ModelOutput::Tensor(loaded_out) = output;

    ensure!(
        loaded_out.shape().dims() == baseline.shape().dims(),
        "loaded output shape mismatch"
    );
    let a = loaded_out.data();
    let b = baseline.data();
    ensure!(a.len() == b.len(), "loaded output length mismatch");
    for (&x, &y) in a.iter().zip(b.iter()) {
        ensure!((x - y).abs() < 1e-6, "loaded output differs from baseline");
    }

    Ok(())
}

#[test]
fn load_option_matmul_input_dtype_rounds_projection_inputs() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let checkpoint = TempFile {
        path: unique_path("gpt_rs_checkpoint_bf16_inputs", "bin"),
    };
    CheckpointSaver::save(&checkpoint.path, &small_gpt(&backend)?)?;

    let logits = |matmul_input_dtype| -> Result<Vec<f32>> {
        let options = LoadOptions {
            namespace: None,
            matmul_input_dtype,
        };
        let mut model = load_model_with_options(Arc::clone(&backend), &checkpoint.path, options)?;
        let ModelOutput::Tensor(out) = model.forward(ModelInput::Tokens(vec![1, 2, 3, 4]))?;
        Ok(out.data().to_vec())
    };
    let (f32_inputs, bf16_inputs) = (logits(None)?, logits(Some(DType::BF16))?);
    ensure!(
        f32_inputs != bf16_inputs,
        "bf16 inputs left the logits unchanged"
    );
    let scale = f32_inputs.iter().fold(0f32, |m, v| m.max(v.abs()));
    for (a, b) in f32_inputs.iter().zip(&bf16_inputs) {
        ensure!(
            (a - b).abs() <= 1e-2 * scale,
            "logit {a} differs from {b} by more than bf16 rounding"
        );
    }
    Ok(())
}

#[test]
fn truncated_or_corrupted_checkpoints_are_rejected() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let checkpoint = TempFile {
        path: unique_path("gpt_rs_checkpoint_truncated", "bin"),
    };
    CheckpointSaver::save(&checkpoint.path, &small_gpt(&backend)?)?;
    let bytes = fs::read(&checkpoint.path)?;

    let broken = TempFile {
        path: unique_path("gpt_rs_checkpoint_broken", "bin"),
    };
    fs::write(&broken.path, &bytes[..bytes.len() - 1])?;
    let err = CheckpointReader::open(&broken.path)
        .err()
        .expect("a truncated payload must be rejected");
    assert!(format!("{err:#}").contains("exceeds file size"), "{err:#}");

    let mut corrupted = bytes.clone();
    corrupted[0] = b'X';
    fs::write(&broken.path, &corrupted)?;
    let err = CheckpointReader::open(&broken.path)
        .err()
        .expect("a bad magic number must be rejected");
    assert!(format!("{err:#}").contains("header"), "{err:#}");
    Ok(())
}

#[test]
fn tensor_archive_round_trips_every_dtype() -> Result<()> {
    let values = [1.0f32, -2.5, 3.25, 0.0, 1e-3, -7.0];
    let tensors: HashMap<String, Tensor> = [
        ("a.f32", tensor_as(&[2, 3], &values, DType::F32)),
        ("b.bf16", tensor_as(&[3], &values[..3], DType::BF16)),
        ("c.f16", tensor_as(&[5], &values[..5], DType::F16)),
        ("d.i32", Tensor::from_i32(Shape::new([2]), vec![-4, 9])?),
    ]
    .into_iter()
    .map(|(name, tensor)| (name.to_string(), tensor))
    .collect();
    let archive = TempFile {
        path: unique_path("gpt_rs_tensor_archive", "bin"),
    };
    TensorArchive::save(&archive.path, &tensors)?;
    let loaded = TensorArchive::load(&archive.path)?;
    assert_eq!(loaded.len(), tensors.len());
    for (name, tensor) in &tensors {
        assert_tensor_bytes_eq(&loaded[name], tensor);
    }
    Ok(())
}
