use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::backend::optimizer::{FunctionPass, OptimizeContext, Optimizer, PassResult};
use crate::backend::passes::{
    BroadcastCanonicalizationPass, CastCanonicalizationPass, CommonSubexpressionEliminationPass,
    DeadCodeEliminationPass, ElementwiseSimplificationPass, ParamOnlyFoldToParamPass,
    ReshapeCanonicalizationPass, SliceCanonicalizationPass, TransposeCanonicalizationPass,
};
use crate::backend::spec::{Function, PortableBackend};

pub enum Step<B: PortableBackend + 'static> {
    Pass(Arc<dyn FunctionPass<B>>),
    FixedPoint {
        max_iters: usize,
        steps: Vec<Step<B>>,
    },
}

pub struct PipelineBuilder<B: PortableBackend + 'static> {
    steps: Vec<Step<B>>,
}

impl<B: PortableBackend + 'static> PipelineBuilder<B> {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn pass(&mut self, pass: Arc<dyn FunctionPass<B>>) {
        self.steps.push(Step::Pass(pass));
    }

    pub fn fixed_point<F>(&mut self, max_iters: usize, build: F)
    where
        F: FnOnce(&mut PipelineBuilder<B>),
    {
        let mut inner = PipelineBuilder::new();
        build(&mut inner);
        self.steps.push(Step::FixedPoint {
            max_iters: max_iters.max(1),
            steps: inner.steps,
        });
    }

    pub fn finish(self) -> Vec<Step<B>> {
        self.steps
    }
}

impl<B: PortableBackend + 'static> Default for PipelineBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait BackendPipeline<B: PortableBackend + 'static>: Send + Sync {
    fn populate_pre(&self, _p: &mut PipelineBuilder<B>) {}
    fn populate_legalize(&self, _p: &mut PipelineBuilder<B>) {}
    fn populate_fuse(&self, _p: &mut PipelineBuilder<B>) {}
    fn populate_cleanup(&self, _p: &mut PipelineBuilder<B>) {}
}

pub struct PipelineOptimizer<B: PortableBackend + 'static> {
    steps: Vec<Step<B>>,
    log_stats: bool,
    run_counter: AtomicUsize,
}

impl<B: PortableBackend + 'static> PipelineOptimizer<B> {
    pub fn new(backend_pipeline: Option<Arc<dyn BackendPipeline<B>>>) -> Self {
        let mut builder = PipelineBuilder::new();

        let pre_iters = std::env::var("GPTRS_OPT_PRE_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2);
        let post_iters = std::env::var("GPTRS_OPT_POST_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4);

        builder.fixed_point(pre_iters, |p| {
            p.pass(Arc::new(BroadcastCanonicalizationPass::default()));
            p.pass(Arc::new(ReshapeCanonicalizationPass::default()));
            p.pass(Arc::new(TransposeCanonicalizationPass::default()));
            p.pass(Arc::new(SliceCanonicalizationPass::default()));
            p.pass(Arc::new(CastCanonicalizationPass::default()));
            p.pass(Arc::new(ElementwiseSimplificationPass::default()));
        });

        if let Some(pipeline) = backend_pipeline.as_ref() {
            pipeline.populate_pre(&mut builder);
            pipeline.populate_legalize(&mut builder);
        }

        builder.pass(Arc::new(ParamOnlyFoldToParamPass));

        if let Some(pipeline) = backend_pipeline.as_ref() {
            pipeline.populate_fuse(&mut builder);
        }

        builder.fixed_point(post_iters, |p| {
            p.pass(Arc::new(BroadcastCanonicalizationPass::default()));
            p.pass(Arc::new(ReshapeCanonicalizationPass::default()));
            p.pass(Arc::new(TransposeCanonicalizationPass::default()));
            p.pass(Arc::new(SliceCanonicalizationPass::default()));
            p.pass(Arc::new(CastCanonicalizationPass::default()));
            p.pass(Arc::new(ElementwiseSimplificationPass::default()));
            p.pass(Arc::new(CommonSubexpressionEliminationPass));
            p.pass(Arc::new(DeadCodeEliminationPass));
        });

        if let Some(pipeline) = backend_pipeline.as_ref() {
            pipeline.populate_cleanup(&mut builder);
        }

        Self {
            steps: builder.finish(),
            log_stats: std::env::var("GPTRS_PASS_STATS").is_ok(),
            run_counter: AtomicUsize::new(0),
        }
    }
}

impl<B: PortableBackend + 'static> Optimizer<B> for PipelineOptimizer<B> {
    fn optimize(&self, function: &mut Function, cx: &mut OptimizeContext<B>) -> PassResult {
        let run_id = if self.log_stats {
            Some(self.run_counter.fetch_add(1, Ordering::Relaxed))
        } else {
            None
        };

        let mut result = PassResult::default();
        run_steps(
            &self.steps,
            function,
            cx,
            run_id,
            &mut result,
            self.log_stats,
        );
        result
    }
}

fn run_steps<B: PortableBackend + 'static>(
    steps: &[Step<B>],
    function: &mut Function,
    cx: &mut OptimizeContext<B>,
    run_id: Option<usize>,
    totals: &mut PassResult,
    log_stats: bool,
) -> bool {
    let mut changed_any = false;
    for step in steps {
        match step {
            Step::Pass(pass) => {
                let _scope = crate::profiling::compile_pass_scope(pass.name());
                let stats = pass.run(function, cx);
                changed_any |= stats.changed;
                *totals = totals.merge(stats);
                if log_stats {
                    log_pass_stats(pass.name(), function, run_id, stats);
                }
            }
            Step::FixedPoint { max_iters, steps } => {
                let mut iter = 0usize;
                loop {
                    if iter >= *max_iters {
                        break;
                    }
                    iter += 1;
                    let mut local = PassResult::default();
                    let changed = run_steps(steps, function, cx, run_id, &mut local, log_stats);
                    *totals = totals.merge(local);
                    changed_any |= changed;
                    if !changed {
                        break;
                    }
                }
            }
        }
    }
    changed_any
}

fn log_pass_stats(name: &str, function: &Function, run_id: Option<usize>, stats: PassResult) {
    let func_len = function.body.len();
    let rid = run_id.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
    println!(
        "[optimizer] run={} func={} pass={} changed={} rewrites={} erased={} iters={} body_len={}",
        rid,
        function.name,
        name,
        stats.changed,
        stats.rewrites_applied,
        stats.erased_insts,
        stats.iterations,
        func_len
    );
}
