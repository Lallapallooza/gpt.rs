use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use gpt_rs::backend::param_resolver::ParamResolver;
use gpt_rs::backend::spec::{Instruction, PortableBackend, Program, TensorInit};
use gpt_rs::model::config::{ModelRuntimeConfig, WeightStreamingConfig};
use gpt_rs::model::{Gpt, GptConfig, ModelConfig};
use gpt_rs::params::base_param_id;
use gpt_rs::runtime::{load_model, LoadedModel, ModelInput};
use gpt_rs::tensor::Tensor;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

const MAGIC: &[u8; 8] = b"GPTRSCHK";
const VERSION_V2: u32 = 2;

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

#[derive(Clone)]
struct CountingMaterializeBackend {
    inner: Arc<CpuPortableBackend>,
    materialize_calls: Arc<AtomicUsize>,
}

impl CountingMaterializeBackend {
    fn new() -> Self {
        Self {
            inner: Arc::new(CpuPortableBackend::new()),
            materialize_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn materialize_calls(&self) -> usize {
        self.materialize_calls.load(Ordering::SeqCst)
    }
}

impl PortableBackend for CountingMaterializeBackend {
    type TensorHandle = <CpuPortableBackend as PortableBackend>::TensorHandle;

    fn backend_name(&self) -> &str {
        "cpu-counting-materialize"
    }

    fn param_resolver(&self) -> Option<Arc<dyn ParamResolver<Handle = Self::TensorHandle>>> {
        self.inner.param_resolver()
    }

    fn materialize(
        &self,
        init: TensorInit,
    ) -> gpt_rs::backend::spec::BackendResult<Self::TensorHandle> {
        self.materialize_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.materialize(init)
    }

    fn to_literal(
        &self,
        tensor: &Self::TensorHandle,
    ) -> gpt_rs::backend::spec::BackendResult<gpt_rs::backend::spec::TensorLiteral> {
        self.inner.to_literal(tensor)
    }

    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> gpt_rs::backend::spec::BackendResult<Vec<Self::TensorHandle>> {
        self.inner.execute_instruction(instruction, inputs)
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> gpt_rs::backend::spec::BackendResult<Vec<Self::TensorHandle>> {
        self.inner.run_program(program, entry_inputs)
    }
}

struct IndexEntry {
    name: String,
    base_id: u128,
    dims: Vec<u64>,
    dtype_tag: u32,
    offset_rel: u64,
    len: u64,
}

fn push_u32(dst: &mut Vec<u8>, value: u32) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(dst: &mut Vec<u8>, value: u64) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn push_u128(dst: &mut Vec<u8>, value: u128) {
    dst.extend_from_slice(&value.to_le_bytes());
}

fn build_index_bytes(entries: &[IndexEntry], data_start: u64) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    push_u32(&mut out, entries.len() as u32);
    for entry in entries {
        let name_bytes = entry.name.as_bytes();
        push_u32(&mut out, name_bytes.len() as u32);
        out.extend_from_slice(name_bytes);
        push_u128(&mut out, entry.base_id);

        push_u32(&mut out, entry.dims.len() as u32);
        for dim in &entry.dims {
            push_u64(&mut out, *dim);
        }

        push_u32(&mut out, entry.dtype_tag);
        out.push(0u8);

        let offset = data_start
            .checked_add(entry.offset_rel)
            .context("checkpoint offset overflow")?;
        push_u64(&mut out, offset);
        push_u64(&mut out, entry.len);
    }
    Ok(out)
}

fn save_checkpoint_with_runtime(
    path: &Path,
    model: &Gpt<CpuPortableBackend>,
    runtime: ModelRuntimeConfig,
) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION_V2.to_le_bytes())?;

    let config = ModelConfig::new_with_runtime(
        "gpt",
        serde_json::to_value(&model.config).context("serialize gpt config")?,
        runtime,
    );
    let config_bytes = serde_json::to_vec(&config)?;
    writer.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&config_bytes)?;

    let mut params: Vec<(String, Tensor)> = Vec::new();
    model.for_each_parameter(|name, tensor| {
        let host = tensor
            .to_host()
            .with_context(|| format!("failed to export checkpoint tensor '{name}'"))?;
        params.push((name.to_string(), host));
        Ok(())
    })?;
    params.sort_by(|(a, _), (b, _)| a.cmp(b));

    let mut entries = Vec::with_capacity(params.len());
    let mut running_offset = 0u64;
    for (name, tensor) in &params {
        let base_id = base_param_id(name)?.0;
        let dims: Vec<u64> = tensor.shape().dims().iter().map(|d| *d as u64).collect();
        let dtype = tensor.dtype();
        let len = match dtype {
            gpt_rs::tensor::DType::F32 => (tensor.data().len() * dtype.size_in_bytes()) as u64,
            gpt_rs::tensor::DType::I32 => (tensor.data_i32().len() * dtype.size_in_bytes()) as u64,
            gpt_rs::tensor::DType::F16 | gpt_rs::tensor::DType::BF16 => {
                anyhow::bail!("checkpoint dtype {:?} is not supported yet", dtype)
            }
        };
        entries.push(IndexEntry {
            name: name.clone(),
            base_id,
            dims,
            dtype_tag: dtype.tag(),
            offset_rel: running_offset,
            len,
        });
        running_offset = running_offset
            .checked_add(len)
            .context("checkpoint data offset overflow")?;
    }

    let index_rel = build_index_bytes(&entries, 0)?;
    ensure!(
        index_rel.len() <= u32::MAX as usize,
        "checkpoint index too large"
    );
    let index_len = index_rel.len() as u32;

    let data_start = (MAGIC.len() + 4 + 4 + config_bytes.len() + 4 + index_len as usize) as u64;
    let index_abs = build_index_bytes(&entries, data_start)?;
    ensure!(
        index_abs.len() == index_rel.len(),
        "checkpoint index length mismatch after offset fixup"
    );

    writer.write_all(&index_len.to_le_bytes())?;
    writer.write_all(&index_abs)?;

    for (_name, tensor) in params {
        match tensor.dtype() {
            gpt_rs::tensor::DType::F32 => {
                for &value in tensor.data() {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            gpt_rs::tensor::DType::I32 => {
                for &value in tensor.data_i32() {
                    writer.write_all(&value.to_le_bytes())?;
                }
            }
            gpt_rs::tensor::DType::F16 | gpt_rs::tensor::DType::BF16 => {
                anyhow::bail!("checkpoint dtype {:?} is not supported yet", tensor.dtype())
            }
        }
    }

    writer.flush()?;
    Ok(())
}

fn build_small_gpt() -> Result<Gpt<CpuPortableBackend>> {
    let backend = Arc::new(CpuPortableBackend::new());
    let mut rng = StdRng::seed_from_u64(7);
    let config = GptConfig {
        vocab_size: 32,
        context_length: 16,
        embed_dim: 8,
        num_layers: 1,
        num_heads: 2,
        mlp_ratio: 2,
        dropout: 0.0,
    };
    Gpt::random(config, backend, &mut rng)
}

fn run_case(weight_streaming: WeightStreamingConfig) -> Result<(usize, usize)> {
    let model = build_small_gpt()?;
    let runtime = ModelRuntimeConfig {
        weight_streaming,
        ..ModelRuntimeConfig::default()
    };

    let checkpoint = TempFile {
        path: unique_path("gpt_rs_runtime_streaming_case", "bin"),
    };
    save_checkpoint_with_runtime(&checkpoint.path, &model, runtime)?;

    let backend = Arc::new(CountingMaterializeBackend::new());
    let mut loaded = load_model(Arc::clone(&backend), &checkpoint.path)?;
    let tokens = vec![1usize, 2, 3, 4];

    let first = forward_delta(loaded.as_mut(), &backend, &tokens)?;
    let second = forward_delta(loaded.as_mut(), &backend, &tokens)?;
    Ok((first, second))
}

fn forward_delta(
    model: &mut dyn LoadedModel<CountingMaterializeBackend>,
    backend: &Arc<CountingMaterializeBackend>,
    tokens: &[usize],
) -> Result<usize> {
    let before = backend.materialize_calls();
    let _ = model.forward(ModelInput::Tokens(tokens.to_vec()))?;
    let after = backend.materialize_calls();
    Ok(after.saturating_sub(before))
}

#[test]
fn default_runtime_avoids_streaming_reload_penalty_across_forwards() -> Result<()> {
    let (first, second) = run_case(WeightStreamingConfig::default())?;
    ensure!(
        second <= first,
        "expected default runtime to avoid growing materialization cost across forwards (first={first}, second={second})"
    );
    Ok(())
}

#[test]
fn streaming_without_budget_reloads_more_than_default_runtime() -> Result<()> {
    let (_default_first, default_second) = run_case(WeightStreamingConfig::default())?;

    let no_cache_streaming = WeightStreamingConfig {
        enabled: true,
        ..WeightStreamingConfig::default()
    };
    let (stream_first, stream_second) = run_case(no_cache_streaming)?;

    ensure!(
        stream_second >= stream_first / 2,
        "expected streamed second forward to remain expensive without cache budget (first={stream_first}, second={stream_second})"
    );
    ensure!(
        stream_second > default_second,
        "expected streamed second forward ({stream_second}) to exceed default second forward ({default_second})"
    );
    Ok(())
}

#[test]
fn budget_and_host_cap_change_streaming_reload_behavior() -> Result<()> {
    let no_cache_streaming = WeightStreamingConfig {
        enabled: true,
        ..WeightStreamingConfig::default()
    };
    let (_no_cache_first, no_cache_second) = run_case(no_cache_streaming)?;

    let large_budget_streaming = WeightStreamingConfig {
        enabled: true,
        device_budget_bytes: Some(1_000_000_000),
        ..WeightStreamingConfig::default()
    };
    let (_large_first, large_second) = run_case(large_budget_streaming)?;

    let host_capped_streaming = WeightStreamingConfig {
        enabled: true,
        device_budget_bytes: Some(1_000_000_000),
        cache_budget_cap_bytes: Some(1),
        ..WeightStreamingConfig::default()
    };
    let (_host_first, host_second) = run_case(host_capped_streaming)?;

    ensure!(
        large_second < no_cache_second,
        "expected large device budget to reduce second-forward reloads (large={large_second}, no-cache={no_cache_second})"
    );
    ensure!(
        host_second >= large_second,
        "expected tiny host budget to prevent large-budget caching gains (host={host_second}, large={large_second})"
    );
    Ok(())
}

#[test]
fn prefetch_layers_reduce_streaming_first_forward_materialization() -> Result<()> {
    let no_prefetch_streaming = WeightStreamingConfig {
        enabled: true,
        ..WeightStreamingConfig::default()
    };
    let (no_prefetch_first, no_prefetch_second) = run_case(no_prefetch_streaming)?;

    let prefetch_streaming = WeightStreamingConfig {
        enabled: true,
        prefetch_layers: Some(1),
        ..WeightStreamingConfig::default()
    };
    let (prefetch_first, prefetch_second) = run_case(prefetch_streaming)?;

    ensure!(
        prefetch_first < no_prefetch_first,
        "expected prefetch_layers to reduce first-forward materialization calls (prefetch={prefetch_first}, baseline={no_prefetch_first})"
    );
    ensure!(
        prefetch_second <= no_prefetch_second,
        "expected prefetch_layers to not increase second-forward materialization calls (prefetch={prefetch_second}, baseline={no_prefetch_second})"
    );
    Ok(())
}
