//! A causal decoder runs every prefill chunk and every decode step as exactly one backend
//! program, including the computations that read only parameters. Decode steps after the first
//! reuse the compiled program.

mod checkpoint_support;

use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpt_rs::gpt_rs_ir::ir::Module;
use gpt_rs::inference::decoder::PREFILL_CHUNK;
use gpt_rs::inference::generate::{Generator, Sampling};
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::runtime::load_model;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

/// Whether each backend program run reused a compiled program.
#[derive(Default)]
struct ProgramRuns {
    cache_hits: Mutex<Vec<bool>>,
}

impl ExecutionTraceSink for ProgramRuns {
    fn before_program(&self, context: &ProgramContext, _module: &Module) {
        let hit = context.program_cache_hit;
        self.cache_hits.lock().unwrap().push(hit);
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

impl ProgramRuns {
    /// What `f` returns, and whether each program that it runs reused a compiled program.
    fn record<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<(T, Vec<bool>)> {
        let before = self.cache_hits.lock().unwrap().len();
        let value = f()?;
        Ok((value, self.cache_hits.lock().unwrap()[before..].to_vec()))
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
        let model = loaded.causal_lm()?;

        let prompt: Vec<usize> = (0..PREFILL_CHUNK + 8).map(|i| i % 64).collect();
        let (_, forward) = runs.record(|| model.forward(&prompt))?;
        assert_eq!(
            forward.len(),
            2,
            "{name}: programs run by a two-chunk forward"
        );

        // A random pick with top-k records the most ops into each step.
        let sampling = Sampling::random(0.8, 7).with_top_k(5);
        let (mut generator, prefill) =
            runs.record(|| Generator::new(model, &prompt, 4, &sampling))?;
        assert_eq!(
            prefill.len(),
            2,
            "{name}: programs run by a two-chunk prefill"
        );
        let (_, first) = runs.record(|| generator.step())?;
        assert!(
            first.is_empty(),
            "{name}: the prefill picks the first token"
        );
        for step in 1..4 {
            let (_, decode) = runs.record(|| generator.step())?;
            assert_eq!(
                decode.len(),
                1,
                "{name}: programs run by decode step {step}"
            );
            assert!(
                step == 1 || decode[0],
                "{name}: decode step {step} compiled its program again"
            );
        }
    }
    Ok(())
}
