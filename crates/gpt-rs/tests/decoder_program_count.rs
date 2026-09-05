//! A causal decoder runs every prefill chunk and every decode step as exactly one backend
//! program, including the computations that read only parameters.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::spec::Program;
use gpt_rs::inference::decoder::PREFILL_CHUNK;
use gpt_rs::inference::CausalLanguageModel;
use gpt_rs::model::qwen3_5::Qwen35LayerType;
use gpt_rs::model::{Gpt, GptConfig, Qwen35, Qwen35Config};
use gpt_rs::nn::{ActivationFunction, RopeParameters, RopeScaling};
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Default)]
struct ExecutionCounter {
    programs: AtomicUsize,
}

impl ExecutionTraceSink for ExecutionCounter {
    fn before_program(&self, _context: &ProgramContext, _program: &Program) {
        self.programs.fetch_add(1, Ordering::Relaxed);
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

impl ExecutionCounter {
    /// Programs executed while running `f`.
    fn count<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<usize> {
        let before = self.programs.load(Ordering::Relaxed);
        f()?;
        Ok(self.programs.load(Ordering::Relaxed) - before)
    }
}

const CONTEXT: usize = PREFILL_CHUNK + 64;

fn qwen3_5(backend: &Arc<CpuPortableBackend>) -> Result<Qwen35<CpuPortableBackend>> {
    let config = Qwen35Config {
        vocab_size: 64,
        max_position_embeddings: CONTEXT,
        hidden_size: 16,
        layer_types: vec![
            Qwen35LayerType::LinearAttention,
            Qwen35LayerType::FullAttention,
        ],
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 8,
        attention_bias: false,
        rope_parameters: RopeParameters {
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5,
            scaling: RopeScaling::None,
        },
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
    };
    Qwen35::random(config, Arc::clone(backend), &mut StdRng::seed_from_u64(0))
}

fn gpt(backend: &Arc<CpuPortableBackend>) -> Result<Gpt<CpuPortableBackend>> {
    let config = GptConfig {
        vocab_size: 64,
        n_positions: CONTEXT,
        n_embd: 16,
        n_layer: 2,
        n_head: 2,
        n_inner: None,
        layer_norm_epsilon: 1e-5,
        activation_function: ActivationFunction::GeluTanh,
    };
    Gpt::random(config, Arc::clone(backend), &mut StdRng::seed_from_u64(0))
}

/// Executions of a two-chunk full-sequence forward, of a two-chunk prefill and of each following
/// single-token decode step.
fn executions(
    counter: &ExecutionCounter,
    model: &dyn CausalLanguageModel<CpuPortableBackend>,
) -> Result<(usize, usize, Vec<usize>)> {
    let prompt: Vec<usize> = (0..PREFILL_CHUNK + 8).map(|i| i % 64).collect();
    let forward = counter.count(|| model.forward(&prompt))?;
    let mut caches = vec![None; model.num_layers()];
    let capacity = Some(CONTEXT);
    let prefill =
        counter.count(|| model.forward_with_decode_cache(&prompt, 0, &mut caches, capacity))?;
    let decode = (0..3)
        .map(|step| {
            let position = prompt.len() + step;
            counter.count(|| {
                model.forward_with_decode_cache(&[step + 1], position, &mut caches, capacity)
            })
        })
        .collect::<Result<_>>()?;
    Ok((forward, prefill, decode))
}

#[test]
fn decoder_runs_one_program_per_prefill_chunk_and_decode_step() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let counter = Arc::new(ExecutionCounter::default());
    let _trace = trace::install_global_sink(Arc::clone(&counter) as Arc<dyn ExecutionTraceSink>);

    let qwen = qwen3_5(&backend)?;
    let gpt = gpt(&backend)?;
    let models: [(&str, &dyn CausalLanguageModel<CpuPortableBackend>); 2] =
        [("qwen3_5", &qwen), ("gpt2", &gpt)];
    for (name, model) in models {
        let (forward, prefill, decode) = executions(&counter, model)?;
        assert_eq!(forward, 2, "{name}: programs run by a two-chunk forward");
        assert_eq!(prefill, 2, "{name}: programs run by a two-chunk prefill");
        assert_eq!(decode, [1, 1, 1], "{name}: programs run per decode step");
    }
    Ok(())
}
