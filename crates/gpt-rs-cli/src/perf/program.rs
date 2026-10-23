//! `perf ptir`: a PTIR program that a backend's passes have lowered as far as one of them, timed on
//! that backend. Such a program is a pass's dump that `--dump-dir` writes, as it is or edited by
//! hand, such as a schedule written into the dump after `schedule`.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Args as ClapArgs;
use gpt_rs::backend::PortableBackend;
use gpt_rs_backends::{BackendOptions, BackendVisitor};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::{DType, Module, TensorType, Type};
use gpt_rs_ir::passes::{run_pass, PruneParams};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::median;

#[derive(ClapArgs)]
pub struct PtirArgs {
    /// The PTIR text program.
    #[arg(long)]
    ptir: PathBuf,

    /// The pass of the backend's pipeline that the program has been through: the passes after its
    /// first run lower the program.
    #[arg(long)]
    after: String,

    /// The backend that lowers and runs the program.
    #[arg(long, default_value = "x86")]
    backend: String,

    /// Runs before timing.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Timed runs, at most.
    #[arg(long, default_value_t = 200)]
    iters: usize,

    /// Time the program for at most this long, after its warmup.
    #[arg(long, value_name = "MS", default_value_t = 1000)]
    budget_ms: u64,
}

/// Lowers the program of `args` with the passes of its backend after `args.after`, runs its
/// load-time functions on random parameters, then times its run-time function on random
/// arguments, and prints the median and the fastest run.
pub fn run(args: &PtirArgs, options: &BackendOptions) -> Result<()> {
    let path = &args.ptir;
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let module = Module::from_text(&text).with_context(|| format!("{}", path.display()))?;
    let timed =
        gpt_rs_backends::with_backend(&args.backend, options, TimeProgram { module, args })??;
    println!(
        "{}: {:.1} us, fastest {:.1} us, over {} runs",
        path.display(),
        timed.median_us,
        timed.fastest_us,
        timed.runs
    );
    Ok(())
}

/// The times of the runs of a program.
struct Timed {
    median_us: f64,
    fastest_us: f64,
    runs: usize,
}

/// Lowers and times `module` on the backend it visits.
struct TimeProgram<'a> {
    module: Module,
    args: &'a PtirArgs,
}

impl BackendVisitor for TimeProgram<'_> {
    type Output = Result<Timed>;

    fn visit<B: PortableBackend + 'static>(self, backend: Arc<B>) -> Result<Timed> {
        let mut lowered = self.module;
        let pipeline = backend.pipeline();
        let passes = pipeline.passes();
        let Some(done) = passes
            .iter()
            .position(|pass| pass.name() == self.args.after)
        else {
            bail!(
                "the {} backend runs no pass `{}`",
                backend.backend_name(),
                self.args.after
            );
        };
        for pass in &passes[done + 1..] {
            run_pass(pass.as_ref(), &mut lowered)?;
        }
        run_pass(&PruneParams, &mut lowered)?;
        let module = Arc::new(lowered);
        let mut rng = StdRng::seed_from_u64(0);
        let mut params: HashMap<u128, B::TensorHandle> = HashMap::new();
        let mut run_time = Vec::new();
        for func in module.funcs() {
            if !func.is_load_time() {
                run_time.push(func.name.clone());
                continue;
            }
            let mut leaves = Vec::with_capacity(func.args.len());
            for (index, ty) in func.args.iter().enumerate() {
                let id = func.param(index).ok_or_else(|| {
                    anyhow!("@{} reads an argument that is no parameter", func.name)
                })?;
                leaves.push(param(&*backend, &mut params, &mut rng, id.0, ty)?);
            }
            let exe = backend.compile(&module, &func.name)?;
            let results = backend.run(&exe, leaves)?;
            for (id, result) in func.result_params.iter().zip(results) {
                params.insert(id.0, result);
            }
        }
        let [name] = &run_time[..] else {
            bail!(
                "the program has {} run-time functions, not one",
                run_time.len()
            );
        };
        let func = module.func(name).expect("a function of the module");
        let mut inputs = Vec::with_capacity(func.args.len());
        for (index, ty) in func.args.iter().enumerate() {
            let input = match func.param(index) {
                Some(id) => param(&*backend, &mut params, &mut rng, id.0, ty)?,
                None => backend.upload(random(&mut rng, ty)?)?,
            };
            inputs.push(input);
        }
        let exe = backend.compile(&module, name)?;
        for _ in 0..self.args.warmup {
            backend.run(&exe, inputs.clone())?;
        }
        let budget = Duration::from_millis(self.args.budget_ms);
        let started = Instant::now();
        let mut runs = Vec::new();
        while runs.len() < self.args.iters.max(1) && (runs.is_empty() || started.elapsed() < budget)
        {
            let given = inputs.clone();
            let start = Instant::now();
            backend.run(&exe, given)?;
            runs.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let fastest_us = runs.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(Timed {
            median_us: median(&mut runs),
            fastest_us,
            runs: runs.len(),
        })
    }
}

/// The handle of parameter `id` of type `ty`: one a load-time function computed, or random values
/// uploaded once.
fn param<B: PortableBackend>(
    backend: &B,
    params: &mut HashMap<u128, B::TensorHandle>,
    rng: &mut StdRng,
    id: u128,
    ty: &Type,
) -> Result<B::TensorHandle> {
    if let Some(handle) = params.get(&id) {
        return Ok(handle.clone());
    }
    let handle = backend.upload(random(rng, ty)?)?;
    params.insert(id, handle.clone());
    Ok(handle)
}

/// A tensor of type `ty`: for f32, values from [0.25, 1), which no op of a model turns into
/// infinities or subnormals that would change the timing. A tensor of any other dtype holds zeros.
fn random(rng: &mut StdRng, ty: &Type) -> Result<HostTensor> {
    let tensor: &TensorType = ty
        .as_tensor()
        .ok_or_else(|| anyhow!("an argument of type {ty} is no tensor"))?;
    let dims = tensor
        .static_dims()
        .ok_or_else(|| anyhow!("an argument of type {ty} has extents unknown before the run"))?;
    if tensor.dtype != DType::F32 {
        return HostTensor::zeros(tensor);
    }
    let count = dims.iter().product();
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(rng.gen_range(0.25..1.0));
    }
    HostTensor::f32(dims, values)
}
