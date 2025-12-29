use std::fs;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Result};
use gpt_rs::checkpoint::{CheckpointReader, CheckpointSaver};
use gpt_rs::model::{Gpt, GptConfig};
use gpt_rs::runtime::{ModelInput, ModelOutput};
use gpt_rs::tensor::Tensor;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
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

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u128(reader: &mut impl Read) -> Result<u128> {
    let mut buf = [0u8; 16];
    reader.read_exact(&mut buf)?;
    Ok(u128::from_le_bytes(buf))
}

fn assert_tensor_bytes_eq(a: &Tensor, b: &Tensor) {
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.shape().dims(), b.shape().dims());
    assert_eq!(a.to_literal().bytes.as_ref(), b.to_literal().bytes.as_ref());
}

fn patch_checkpoint_reserved_byte(path: &Path, new_value: u8) -> Result<()> {
    let mut bytes = fs::read(path)?;
    let (index_start, index_len) = {
        let mut cursor = Cursor::new(bytes.as_slice());

        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        ensure!(&magic == b"GPTRSCHK", "invalid checkpoint magic header");

        let version = read_u32(&mut cursor)?;
        ensure!(version == 2, "unsupported checkpoint version {version}");

        let config_len = read_u32(&mut cursor)? as u64;
        cursor.seek(SeekFrom::Current(i64::try_from(config_len)?))?;

        let index_len = read_u32(&mut cursor)? as u64;
        let index_start = cursor.position() as usize;
        (index_start, index_len)
    };
    let index_end = index_start
        .checked_add(usize::try_from(index_len)?)
        .expect("index length overflow");

    let index_bytes = bytes
        .get_mut(index_start..index_end)
        .expect("index slice within file");
    let mut index = Cursor::new(&mut *index_bytes);

    let tensor_count = read_u32(&mut index)? as usize;
    ensure!(
        tensor_count > 0,
        "checkpoint must contain at least one tensor"
    );

    // Patch the first entry's reserved byte (the legacy requires-grad flag).
    let name_len = read_u32(&mut index)? as u64;
    index.seek(SeekFrom::Current(i64::try_from(name_len)?))?;
    let _base_id = read_u128(&mut index)?;
    let rank = read_u32(&mut index)? as u64;
    index.seek(SeekFrom::Current(i64::try_from(rank * 8)?))?;
    let _dtype_tag = read_u32(&mut index)?;
    let reserved_pos = index.position() as usize;

    index_bytes[reserved_pos] = new_value;
    fs::write(path, bytes)?;
    Ok(())
}

#[test]
fn checkpoint_reader_and_runtime_loader_work_for_small_gpt() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let mut rng = StdRng::seed_from_u64(11);
    let config = GptConfig {
        vocab_size: 32,
        context_length: 16,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 2,
        mlp_ratio: 2,
        dropout: 0.0,
    };
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng)?;

    let tokens = vec![1usize, 2, 3, 4];
    let baseline = model.forward(&tokens)?;

    let checkpoint = TempFile {
        path: unique_path("gpt_rs_checkpoint_test", "bin"),
    };

    CheckpointSaver::save(&checkpoint.path, &model)?;

    let mut reader = CheckpointReader::open(&checkpoint.path)?;
    assert_eq!(reader.config().kind, "gpt");

    let mut expected = Vec::new();
    model.for_each_parameter(|name, tensor| {
        expected.push((name.to_string(), tensor.to_host()?));
        Ok(())
    })?;

    for (name, expected_tensor) in &expected {
        let from_name = reader.get(name)?;
        assert_tensor_bytes_eq(&from_name, expected_tensor);
        let from_id = reader.get_by_base_id(gpt_rs::params::base_param_id(name)?)?;
        assert_tensor_bytes_eq(&from_id, expected_tensor);
    }

    patch_checkpoint_reserved_byte(&checkpoint.path, 1)?;
    let mut reader = CheckpointReader::open(&checkpoint.path)?;
    let from_disk = reader.get("tok_embeddings.weight")?;
    let expected_tok = expected
        .iter()
        .find(|(name, _)| name == "tok_embeddings.weight")
        .expect("tok_embeddings.weight saved")
        .1
        .clone();
    assert_tensor_bytes_eq(&from_disk, &expected_tok);

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
