//! A causal decoder runs every prefill chunk and every decode step as exactly one backend
//! program, including the computations that read only parameters. Decode steps after the first
//! reuse the compiled program.

mod checkpoint_support;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpt_rs::backend::spec::Program;
use gpt_rs::inference::decoder::PREFILL_CHUNK;
use gpt_rs::inference::CausalLanguageModel;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::runtime::load_model;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

use checkpoint_support::CONTEXT;

/// Whether each backend program run reused a compiled program.
#[derive(Default)]
struct ProgramRuns {
    cache_hits: Mutex<Vec<bool>>,
}

impl ExecutionTraceSink for ProgramRuns {
    fn before_program(&self, context: &ProgramContext, _program: &Program) {
        let hit = context.cache.program_cache_hit;
        self.cache_hits.lock().unwrap().push(hit);
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

impl ProgramRuns {
    /// Whether each program that `f` runs reused a compiled program.
    fn record<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<Vec<bool>> {
        let before = self.cache_hits.lock().unwrap().len();
        f()?;
        Ok(self.cache_hits.lock().unwrap()[before..].to_vec())
    }
}

#[test]
fn decoder_runs_one_program_per_prefill_chunk_and_decode_step() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let runs = Arc::new(ProgramRuns::default());
    let _trace = trace::install_global_sink(Arc::clone(&runs) as Arc<dyn ExecutionTraceSink>);

    for (name, checkpoint) in [
        ("gpt2", checkpoint_support::gpt2()?),
        ("qwen3_5", checkpoint_support::qwen3_5()?),
    ] {
        let loaded = load_model(Arc::clone(&backend), &checkpoint.path)?;
        let model: &dyn CausalLanguageModel<CpuPortableBackend> =
            loaded.as_causal_lm().expect("a decoder is a causal LM");

        let prompt: Vec<usize> = (0..PREFILL_CHUNK + 8).map(|i| i % 64).collect();
        let forward = runs.record(|| model.forward(&prompt))?;
        assert_eq!(
            forward.len(),
            2,
            "{name}: programs run by a two-chunk forward"
        );

        let mut caches = vec![None; model.num_layers()];
        let capacity = Some(CONTEXT);
        let prefill =
            runs.record(|| model.forward_with_decode_cache(&prompt, 0, &mut caches, capacity))?;
        assert_eq!(
            prefill.len(),
            2,
            "{name}: programs run by a two-chunk prefill"
        );

        for step in 0..3 {
            let position = prompt.len() + step;
            let decode = runs.record(|| {
                model.forward_with_decode_cache(&[step + 1], position, &mut caches, capacity)
            })?;
            assert_eq!(
                decode.len(),
                1,
                "{name}: programs run by decode step {step}"
            );
            assert!(
                step == 0 || decode[0],
                "{name}: decode step {step} compiled its program again"
            );
        }
    }
    Ok(())
}
