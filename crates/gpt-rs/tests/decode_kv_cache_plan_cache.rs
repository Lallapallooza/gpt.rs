use std::sync::{Arc, Mutex};

use gpt_rs::backend::spec::Program;
use gpt_rs::inference::generate::Generator;
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::model::{Gpt, ModelConfig};
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct ContextSink {
    contexts: Mutex<Vec<ProgramContext>>,
}

impl ExecutionTraceSink for ContextSink {
    fn before_program(&self, context: &ProgramContext, _program: &Program) {
        self.contexts.lock().unwrap().push(context.clone());
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

#[test]
fn decode_kv_cache_reuses_program_cache_in_lazy_mode() -> anyhow::Result<()> {
    let backend = Arc::new(CpuPortableBackend::default());
    let mut rng = StdRng::seed_from_u64(0);
    let config = ModelConfig {
        vocab_size: 32,
        context_length: 32,
        embed_dim: 32,
        num_layers: 2,
        num_heads: 4,
        mlp_ratio: 2,
        dropout: 0.0,
        functional_overrides: Default::default(),
    };
    let model = Gpt::random(config, Arc::clone(&backend), &mut rng)?;
    let sampler = Sampler::new(0.0);

    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let sink_for_install: Arc<dyn ExecutionTraceSink> = sink.clone();
    let _trace_guard = trace::install_global_sink(sink_for_install);

    // Prompt length 4 ensures the decode cache bucket stays stable (8) after the first step,
    // so subsequent decode steps should reuse the cached program.
    let prompt = vec![1usize, 2, 3, 4];
    let mut gen = Generator::new(&model, &sampler, &prompt, true)?;
    gen.step()?;
    gen.step()?;
    gen.step()?;

    let contexts = sink.contexts.lock().unwrap().clone();
    assert!(
        !contexts.is_empty(),
        "expected at least one backend execution to be traced"
    );

    assert!(
        contexts
            .iter()
            .any(|ctx| ctx.cache.program_cache_hit && !ctx.cache.plan_cache_hit),
        "expected at least one program cache hit during decoding, got none"
    );

    Ok(())
}
