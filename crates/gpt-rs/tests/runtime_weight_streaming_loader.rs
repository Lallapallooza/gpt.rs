mod checkpoint_support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{ensure, Result};
use gpt_rs::backend::param_resolver::ParamResolver;
use gpt_rs::backend::spec::{Instruction, PortableBackend, Program, TensorInit};
use gpt_rs::model::config::{ModelRuntimeConfig, WeightStreamingConfig};
use gpt_rs::runtime::{load_model_with_options, LoadOptions, LoadedModel, ModelInput};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

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

fn run_case(weight_streaming: WeightStreamingConfig) -> Result<(usize, usize)> {
    let checkpoint = checkpoint_support::gpt2()?;
    let backend = Arc::new(CountingMaterializeBackend::new());
    let options = LoadOptions {
        runtime: ModelRuntimeConfig {
            weight_streaming,
            ..ModelRuntimeConfig::default()
        },
        ..LoadOptions::default()
    };
    let mut loaded = load_model_with_options(Arc::clone(&backend), &checkpoint.path, options)?;
    let tokens = vec![1usize, 2, 3, 4];

    let first = forward_delta(&mut loaded, &backend, &tokens)?;
    let second = forward_delta(&mut loaded, &backend, &tokens)?;
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
