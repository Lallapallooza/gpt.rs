//! `gpt-rs-cli perf`: the ops, layers and blocks of GPT-2, ResNet-18 and MobileNetV2 at the shapes
//! those models run them ([`cases`]), timed on backends, and profiled down to the lines of
//! generated C and of Rust that each sample comes from ([`profile`]). `perf ptir` times a lowered
//! PTIR program ([`program`]).

mod cases;
mod profile;
mod program;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use gpt_rs::backend::PortableBackend;
use gpt_rs::ops::graph::context::push_default_arena;
use gpt_rs::ops::graph::GraphArena;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, PassRun, ProgramContext, ProgramStats};
use gpt_rs::tensor::DeviceTensor;
use gpt_rs_backends::{BackendOptions, BackendVisitor};
use gpt_rs_ir::ir::Module;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use self::cases::Case;

#[derive(ClapArgs)]
pub struct PerfArgs {
    #[command(subcommand)]
    command: PerfCommand,
}

#[derive(Subcommand)]
enum PerfCommand {
    /// List the cases that the filters or names pick, or every case.
    List(Selection),
    /// Time cases on backends.
    Run(RunArgs),
    /// Run one case under `perf record` and attribute its samples: to the lines of generated C,
    /// the ops they come from at each level, and the lines of Rust that recorded them.
    Profile(profile::ProfileArgs),
    /// Time a PTIR program that a backend's passes lowered as far as one of them, such as a
    /// pass's dump edited by hand: the passes after it lower the program, its load-time functions
    /// run on random parameters, and its run-time function on random arguments.
    Ptir(program::PtirArgs),
}

/// Which cases to run: those that the filters or the names pick, or every one.
#[derive(ClapArgs, Clone)]
pub struct Selection {
    /// Pick the cases whose names contain any of these, such as `resnet18/layer4` or `.block`.
    #[arg(long = "filter", value_name = "TEXT")]
    filters: Vec<String>,

    /// Pick the case of this full name.
    #[arg(long = "case", value_name = "NAME")]
    names: Vec<String>,
}

impl Selection {
    fn cases(&self) -> Vec<Case> {
        let everything = self.filters.is_empty() && self.names.is_empty();
        cases::catalog()
            .into_iter()
            .filter(|case| {
                everything
                    || self.names.contains(&case.name)
                    || self
                        .filters
                        .iter()
                        .any(|filter| case.name.contains(filter.as_str()))
            })
            .collect()
    }
}

#[derive(ClapArgs, Clone)]
pub struct RunArgs {
    #[command(flatten)]
    selection: Selection,

    /// The backends to time each case on. The others compare with the first.
    #[arg(long, value_delimiter = ',', default_value = "x86,c")]
    backends: Vec<String>,

    /// Runs before timing, the first of which compiles.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Timed runs of each case, at most.
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// Time each case on a backend for at most this long, after its warmup.
    #[arg(long, value_name = "MS", default_value_t = 1000)]
    budget_ms: u64,

    /// Also write every measurement to this JSON file.
    #[arg(long, value_name = "PATH")]
    json: Option<PathBuf>,
}

/// Runs the perf command `args`, on backends made as `options` say.
pub fn run(args: PerfArgs, options: &BackendOptions) -> Result<()> {
    match args.command {
        PerfCommand::List(selection) => {
            for case in selection.cases() {
                println!("{:<56} {:?}", case.name, case.kind);
            }
            Ok(())
        }
        PerfCommand::Run(run) => run_cases(&run, options),
        PerfCommand::Profile(profile) => profile::run(profile),
        PerfCommand::Ptir(program) => program::run(&program, options),
    }
}

/// What one case did on one backend.
#[derive(Debug, Clone, Serialize)]
pub struct Measurement {
    pub case: String,
    pub model: String,
    pub backend: String,
    /// How long the backend ran the case's programs, per timed run, in microseconds: the median
    /// and the fastest.
    pub run_us: f64,
    pub run_min_us: f64,
    /// How long a whole run took, recording and planning included, in microseconds: the median.
    pub call_us: f64,
    pub iterations: usize,
    /// Programs the backend ran per run.
    pub programs: usize,
    /// The arithmetic of the case's program, and the bytes its ops move one by one.
    pub flops: u64,
    pub op_bytes: u64,
    /// The bytes of the program's arguments and results: what any implementation reads and
    /// writes at least.
    pub io_bytes: u64,
    pub error: Option<String>,
}

impl Measurement {
    fn gflops(&self) -> f64 {
        self.flops as f64 / (self.run_us * 1e3)
    }

    fn gbytes(&self) -> f64 {
        self.io_bytes as f64 / (self.run_us * 1e3)
    }
}

fn run_cases(args: &RunArgs, options: &BackendOptions) -> Result<()> {
    let cases = args.selection.cases();
    if cases.is_empty() {
        bail!("no case matches the filters");
    }
    let mut all = Vec::new();
    for backend in &args.backends {
        eprintln!("timing {} cases on {backend}", cases.len());
        let measured = gpt_rs_backends::with_backend(
            backend,
            options,
            TimeCases {
                cases: &cases,
                args,
                backend,
            },
        )??;
        all.extend(measured);
    }
    print_table(&cases, &args.backends, &all);
    if let Some(path) = &args.json {
        std::fs::write(path, serde_json::to_string_pretty(&all)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// Times `cases` on the backend it visits.
struct TimeCases<'a> {
    cases: &'a [Case],
    args: &'a RunArgs,
    backend: &'a str,
}

impl BackendVisitor for TimeCases<'_> {
    type Output = Result<Vec<Measurement>>;

    fn visit<B: PortableBackend + 'static>(self, backend: Arc<B>) -> Result<Vec<Measurement>> {
        let timing = Arc::new(Timing {
            inner: trace::current_sink(),
            state: Mutex::default(),
        });
        let _sink = trace::install_global_sink(timing.clone());
        let mut measurements = Vec::with_capacity(self.cases.len());
        for case in self.cases {
            let measured = time_case(&backend, case, self.args, &timing);
            let measurement = measured.unwrap_or_else(|err| Measurement {
                case: case.name.clone(),
                model: case.model.to_string(),
                backend: self.backend.to_string(),
                run_us: f64::NAN,
                run_min_us: f64::NAN,
                call_us: f64::NAN,
                iterations: 0,
                programs: 0,
                flops: 0,
                op_bytes: 0,
                io_bytes: 0,
                error: Some(format!("{err:#}")),
            });
            if let Some(err) = &measurement.error {
                eprintln!("  {}: {err}", case.name);
            }
            measurements.push(Measurement {
                backend: self.backend.to_string(),
                ..measurement
            });
        }
        Ok(measurements)
    }
}

/// Runs `case` on `backend`: its warmup, then timed runs until `args` says to stop.
fn time_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    case: &Case,
    args: &RunArgs,
    timing: &Timing,
) -> Result<Measurement> {
    let mut rng = StdRng::seed_from_u64(seed(&case.name));
    let workload = case.kind.prepare(backend, &mut rng)?;
    timing.reset();
    for _ in 0..args.warmup.max(1) {
        run_once(backend, &workload)?;
        timing.take_runs();
    }
    let work = timing.work();
    let budget = Duration::from_millis(args.budget_ms);
    let started = Instant::now();
    let (mut runs, mut calls, mut programs) = (Vec::new(), Vec::new(), 0);
    while runs.len() < args.iters.max(1) && (runs.is_empty() || started.elapsed() < budget) {
        let call = run_once(backend, &workload)?;
        let ran = timing.take_runs();
        programs = ran.len();
        runs.push(ran.iter().sum::<Duration>().as_secs_f64() * 1e6);
        calls.push(call.as_secs_f64() * 1e6);
    }
    let fastest = runs.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(Measurement {
        case: case.name.clone(),
        model: case.model.to_string(),
        backend: String::new(),
        run_us: median(&mut runs),
        run_min_us: fastest,
        call_us: median(&mut calls),
        iterations: runs.len(),
        programs,
        flops: work.flops,
        op_bytes: work.op_bytes,
        io_bytes: work.io_bytes,
        error: None,
    })
}

/// Records the case once more in a fresh arena, as a model's forward does, and materializes its
/// outputs. Gives the time it took.
fn run_once<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    workload: &cases::Workload<B>,
) -> Result<Duration> {
    let _arena = push_default_arena(GraphArena::new(Arc::clone(backend)));
    let start = Instant::now();
    let outputs = workload();
    let refs: Vec<&DeviceTensor<B>> = outputs.iter().collect();
    DeviceTensor::materialize_many(&refs)?;
    Ok(start.elapsed())
}

fn seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    match values.len() {
        0 => f64::NAN,
        len if len % 2 == 1 => values[len / 2],
        len => (values[len / 2 - 1] + values[len / 2]) / 2.0,
    }
}

/// The arithmetic and the traffic of a case's program.
#[derive(Debug, Clone, Copy, Default)]
struct Work {
    flops: u64,
    op_bytes: u64,
    io_bytes: u64,
}

impl Work {
    /// The work of the run-time function `@captured` of `module`.
    fn of(module: &Module) -> Self {
        let Some(func) = module.func("captured") else {
            return Self::default();
        };
        let cost = module
            .cost_summary(func.body().id(), |_| String::new())
            .total;
        let io_bytes = func
            .args
            .iter()
            .chain(&func.results)
            .filter_map(gpt_rs_ir::analysis::bytes)
            .sum();
        Self {
            flops: cost.flops,
            op_bytes: cost.bytes(),
            io_bytes,
        }
    }
}

#[derive(Default)]
struct TimingState {
    /// The duration of each program the backend ran since the last take.
    runs: Vec<Duration>,
    /// The work of the first program the backend ran since the last reset.
    work: Option<Work>,
}

/// A trace sink that times the programs the backend runs, and hands every event on to the sink
/// installed before it, such as the one `--dump-dir` installs.
struct Timing {
    inner: Option<Arc<dyn ExecutionTraceSink>>,
    state: Mutex<TimingState>,
}

impl Timing {
    fn reset(&self) {
        *self.state.lock().expect("timing poisoned") = TimingState::default();
    }

    fn take_runs(&self) -> Vec<Duration> {
        std::mem::take(&mut self.state.lock().expect("timing poisoned").runs)
    }

    fn work(&self) -> Work {
        self.state
            .lock()
            .expect("timing poisoned")
            .work
            .unwrap_or_default()
    }
}

impl ExecutionTraceSink for Timing {
    fn before_program(&self, context: &ProgramContext, module: &Module) {
        self.state
            .lock()
            .expect("timing poisoned")
            .work
            .get_or_insert_with(|| Work::of(module));
        if let Some(inner) = &self.inner {
            inner.before_program(context, module);
        }
    }

    fn after_program(&self, context: &ProgramContext, stats: &ProgramStats) {
        self.state
            .lock()
            .expect("timing poisoned")
            .runs
            .push(stats.duration);
        if let Some(inner) = &self.inner {
            inner.after_program(context, stats);
        }
    }

    fn on_pass(&self, pass: &PassRun, module: &Module) {
        if let Some(inner) = &self.inner {
            inner.on_pass(pass, module);
        }
    }

    fn on_artifact(&self, name: &str, contents: &str) {
        if let Some(inner) = &self.inner {
            inner.on_artifact(name, contents);
        }
    }
}

/// One row per case: each backend's median run time, then the first backend's time against each
/// other backend's, and the first backend's rates.
fn print_table(cases: &[Case], backends: &[String], all: &[Measurement]) {
    let by_key: BTreeMap<(&str, &str), &Measurement> = all
        .iter()
        .map(|measurement| {
            let key = (measurement.case.as_str(), measurement.backend.as_str());
            (key, measurement)
        })
        .collect();
    let first = &backends[0];
    let mut header = format!("{:<52}", "case");
    for backend in backends {
        header.push_str(&format!(" {:>11}", format!("{backend} us")));
    }
    for other in &backends[1..] {
        header.push_str(&format!(" {:>9}", format!("{first}/{other}")));
    }
    header.push_str(&format!(" {:>9} {:>9}", "GFLOP/s", "GB/s"));
    println!("{header}");
    for case in cases {
        let mut row = format!("{:<52}", case.name);
        for backend in backends {
            let cell = match by_key.get(&(case.name.as_str(), backend.as_str())) {
                Some(measurement) if measurement.error.is_none() => {
                    format!("{:.1}", measurement.run_us)
                }
                _ => "error".to_string(),
            };
            row.push_str(&format!(" {cell:>11}"));
        }
        let lead = by_key.get(&(case.name.as_str(), first.as_str()));
        for other in &backends[1..] {
            let ratio = match (lead, by_key.get(&(case.name.as_str(), other.as_str()))) {
                (Some(ours), Some(theirs)) if ours.error.is_none() && theirs.error.is_none() => {
                    format!("{:.2}", ours.run_us / theirs.run_us)
                }
                _ => "-".to_string(),
            };
            row.push_str(&format!(" {ratio:>9}"));
        }
        match lead {
            Some(measurement) if measurement.error.is_none() => row.push_str(&format!(
                " {:>9.1} {:>9.1}",
                measurement.gflops(),
                measurement.gbytes()
            )),
            _ => row.push_str(&format!(" {:>9} {:>9}", "-", "-")),
        }
        println!("{row}");
    }
}
