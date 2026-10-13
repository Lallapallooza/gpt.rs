mod ir;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{env, fs};

use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use gpt_rs::inference::generate::{Generator, Sampling};
use gpt_rs::io::safetensors;
use gpt_rs::model::config::ModelRuntimeConfig;
use gpt_rs::ops::trace::{self, FileTraceOptions, FileTraceSink, ProgramDumpFilter};
use gpt_rs::profiling;
use gpt_rs::runtime::{LoadOptions, LoadedModel, ModelHandle};
use gpt_rs::tensor::HostTensor;
use gpt_rs::tokenizer::{StreamDecoder, Tokenizer};
use gpt_rs_ir::dialects::func::Func;
use gpt_rs_ir::ir::Module;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

mod benchmark;

use self::benchmark::compile_freeze::{self, CompileFreezeGate};

/// Loads the model directory `checkpoint` onto the backend `backend`.
fn load_checkpoint(backend: &str, checkpoint: &Path, load: &LoadArgs) -> Result<ModelHandle> {
    let options = LoadOptions {
        runtime: ModelRuntimeConfig {
            matmul_input_dtype: load.matmul_input_dtype,
        },
        ..LoadOptions::default()
    };
    gpt_rs_backends::load_model(backend, checkpoint, options)
        .with_context(|| format!("failed to load checkpoint {}", checkpoint.display()))
}

/// Loads `tokenizer`, or else the `tokenizer.json` of the model directory `checkpoint`.
fn load_tokenizer(checkpoint: &Path, tokenizer: Option<&Path>) -> Result<Tokenizer> {
    let path = tokenizer.map_or_else(|| checkpoint.join("tokenizer.json"), Path::to_path_buf);
    Tokenizer::from_file(&path)
        .with_context(|| format!("failed to load tokenizer {}", path.display()))
}

fn install_dump_sink(path: &Path, dump_mode: DumpMode) -> Result<trace::TraceGuard> {
    let options = FileTraceOptions {
        dump_filter: match dump_mode {
            DumpMode::All => ProgramDumpFilter::AllExecutions,
            DumpMode::Compile => ProgramDumpFilter::CompileOnly,
        },
        ..Default::default()
    };

    let sink = FileTraceSink::with_options(path.to_path_buf(), options)
        .with_context(|| format!("failed to prepare PTIR dump directory {}", path.display()))?;
    Ok(trace::install_global_sink(Arc::new(sink)))
}

#[derive(Parser)]
#[command(version, about = "gpt.rs model runner")]
struct Args {
    #[command(subcommand)]
    command: Command,

    #[arg(
        long,
        value_name = "NAME",
        global = true,
        help = "Backend to use (overrides $GPTRS_BACKEND) [default: faer]"
    )]
    backend: Option<String>,

    #[arg(
        long,
        value_name = "DIR",
        global = true,
        help = "Dump PTIR programs and metadata to the specified directory"
    )]
    dump_dir: Option<PathBuf>,

    #[arg(
        long,
        value_enum,
        global = true,
        default_value = "all",
        help = "Controls which PTIR executions are dumped (all, compile)"
    )]
    dump_mode: DumpMode,

    #[arg(long, global = true, help = "Print gpt-rs profiler tables to stderr")]
    profile: bool,

    #[arg(
        long,
        global = true,
        help = "Show full backend profiling rows without compact filtering"
    )]
    profile_full: bool,

    #[arg(
        long,
        value_name = "DIR",
        global = true,
        help = "Write profiling artifacts to the specified directory"
    )]
    profile_out: Option<PathBuf>,

    #[arg(
        long,
        global = true,
        help = "Write Chrome trace timeline artifact (timeline.chrome_trace.json; requires --profile-out)"
    )]
    profile_timeline: bool,

    #[arg(
        long,
        global = true,
        help = "Write flamegraph artifacts (flamegraph.folded and flamegraph.speedscope.json; requires --profile-out)"
    )]
    profile_flamegraph: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DumpMode {
    #[value(name = "all")]
    All,
    #[value(name = "compile")]
    Compile,
}

#[derive(Clone, Debug, Default)]
struct ProfileArtifactConfig {
    out_dir: Option<PathBuf>,
    write_timeline: bool,
    write_flamegraph: bool,
}

impl ProfileArtifactConfig {
    fn trace_enabled(&self) -> bool {
        self.write_timeline || self.write_flamegraph
    }

    fn artifacts_enabled(&self) -> bool {
        self.out_dir.is_some()
    }
}

struct TraceEnableGuard;

impl Drop for TraceEnableGuard {
    fn drop(&mut self) {
        profiling::trace_disable();
    }
}

#[derive(Subcommand)]
enum Command {
    /// Causal text generation for models that implement `CausalLanguageModel`.
    Generate(GenerateArgs),

    /// Dedicated benchmark modes for causal generation.
    Benchmark(BenchmarkArgs),

    /// Run a model forward pass once (tokens or vision input).
    Forward(ForwardArgs),

    /// Write the C source that the C backend generates for a PTIR program.
    Convert(ConvertArgs),

    /// Print what the functions of a PTIR program cost, op by op.
    IrStats(ir::IrStatsArgs),

    /// Print how the ops of two versions of a PTIR program and their cost differ.
    IrDiff(ir::IrDiffArgs),
}

/// Checkpoint loading options shared by every command that loads a model.
#[derive(ClapArgs, Clone)]
struct LoadArgs {
    /// Input dtype of the projection matmuls (f32 or bf16). The default is bf16 for a checkpoint
    /// that declares bf16, else f32.
    #[arg(long, value_name = "DTYPE")]
    matmul_input_dtype: Option<gpt_rs::DType>,
}

/// Model, prompt and sampling options of the causal generation commands.
#[derive(ClapArgs, Clone)]
struct CausalArgs {
    /// Hugging Face model directory.
    #[arg(long)]
    checkpoint: PathBuf,

    #[command(flatten)]
    load: LoadArgs,

    /// Tokenizer (`tokenizer.json`) [default: tokenizer.json in the model directory].
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, default_value = "Hello")]
    prompt: String,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = false)]
    greedy: bool,

    #[arg(long)]
    top_k: Option<usize>,

    /// Seed of the random draws [default: a fresh seed for each run].
    #[arg(long)]
    seed: Option<u64>,
}

/// A loaded causal LM with its encoded prompt and sampling.
struct CausalSession {
    model: ModelHandle,
    tokenizer: Tokenizer,
    prompt_tokens: Vec<usize>,
    sampling: Sampling,
}

impl CausalArgs {
    fn session(&self, backend: &str) -> Result<CausalSession> {
        let model = load_checkpoint(backend, &self.checkpoint, &self.load)?;
        model.causal_lm()?;
        let tokenizer = load_tokenizer(&self.checkpoint, self.tokenizer.as_deref())?;
        let prompt_tokens = tokenizer.encode(&self.prompt)?;
        ensure!(
            !prompt_tokens.is_empty(),
            "prompt produced an empty token sequence"
        );
        let sampling = if self.greedy {
            Sampling::greedy()
        } else {
            let seed = self.seed.unwrap_or_else(rand::random);
            let sampling = Sampling::random(self.temperature, seed);
            match self.top_k {
                Some(k) => sampling.with_top_k(k),
                None => sampling,
            }
        };
        Ok(CausalSession {
            model,
            tokenizer,
            prompt_tokens,
            sampling,
        })
    }
}

impl CausalSession {
    /// Prefills the prompt to generate up to `max_new_tokens` after it.
    fn generator(&self, max_new_tokens: usize) -> Result<Generator<'_>> {
        let model = self.model.causal_lm()?;
        Generator::new(model, &self.prompt_tokens, max_new_tokens, &self.sampling)
    }
}

#[derive(ClapArgs, Clone)]
struct GenerateArgs {
    #[command(flatten)]
    causal: CausalArgs,

    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    #[arg(
        long,
        default_value_t = 0,
        value_name = "N",
        help = "Exclude the first N generated tokens from timing/profiling (warmup)"
    )]
    warmup_tokens: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Keep generating past the model's end-of-sequence tokens"
    )]
    ignore_eos: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BenchmarkMode {
    #[value(name = "prefill")]
    Prefill,
    #[value(name = "first-token")]
    FirstToken,
    #[value(name = "decode-steady")]
    DecodeSteady,
}

#[derive(ClapArgs, Clone)]
struct BenchmarkArgs {
    #[command(flatten)]
    causal: CausalArgs,

    #[arg(
        long,
        value_enum,
        default_value = "decode-steady",
        help = "Benchmark mode: prefill | first-token | decode-steady"
    )]
    mode: BenchmarkMode,

    #[arg(
        long,
        default_value_t = 130,
        help = "Number of generated tokens for decode-steady mode"
    )]
    max_tokens: usize,

    #[arg(
        long,
        default_value_t = 2,
        value_name = "N",
        help = "Warmup tokens excluded from decode-steady timing: the first comes from the prefill, \
                the second compiles the decode step"
    )]
    warmup_tokens: usize,

    #[arg(
        long,
        default_value_t = 16,
        value_name = "N",
        help = "Measured decode token threshold after which compile activity is considered an anomaly"
    )]
    compile_freeze_after: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Fail benchmark when compile activity occurs after --compile-freeze-after"
    )]
    strict_compile_freeze: bool,
}

#[derive(ClapArgs, Clone)]
struct VisionInputArgs {
    /// Safetensors file that holds the input tensor (`--input-key`).
    #[arg(long)]
    input: Option<PathBuf>,

    #[arg(long, default_value = "input")]
    input_key: String,

    /// Generate a deterministic random input when `--input` is not provided.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long, default_value_t = 1)]
    batch: usize,

    #[arg(long, default_value_t = 224)]
    image_size: usize,
}

#[derive(ClapArgs, Clone)]
struct ForwardArgs {
    /// Hugging Face model directory.
    #[arg(long)]
    checkpoint: PathBuf,

    #[command(flatten)]
    load: LoadArgs,

    /// Tokenizer (`tokenizer.json`) for `--prompt` [default: tokenizer.json in the model
    /// directory].
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Text prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Comma-separated token ids, e.g. `--tokens 1,2,3`.
    #[arg(long, value_delimiter = ',')]
    tokens: Vec<usize>,

    #[command(flatten)]
    vision: VisionInputArgs,

    /// Safetensors file to write the logits to (tensor "logits").
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(ClapArgs, Clone)]
struct ConvertArgs {
    /// Input PTIR text module, such as a `program.ptir` that `--dump-dir` writes.
    #[arg(long)]
    ptir: PathBuf,

    /// Output directory for the C source, `program.c`.
    #[arg(long)]
    out: PathBuf,
}

fn load_vision_input_tensor(args: &VisionInputArgs) -> Result<HostTensor> {
    if let Some(path) = args.input.as_ref() {
        let tensors = safetensors::load(path)?;
        let input = tensors.get(&args.input_key).ok_or_else(|| {
            anyhow!(
                "input file {} has no tensor '{}'",
                path.display(),
                args.input_key
            )
        })?;
        return Ok(input.clone());
    }

    ensure!(
        args.batch > 0,
        "--batch must be > 0 when generating a random input",
    );
    ensure!(
        args.image_size > 0,
        "--image-size must be > 0 when generating a random input",
    );

    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);
    let dims = [args.batch, 3, args.image_size, args.image_size];
    HostTensor::f32(dims, standard_normal(dims.iter().product(), &mut rng))
}

/// `len` samples of the standard normal distribution, by the Box-Muller transform.
fn standard_normal(len: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut values = Vec::with_capacity(len + 1);
    while values.len() < len {
        let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.gen::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values.extend([r * theta.cos(), r * theta.sin()]);
    }
    values.truncate(len);
    values
}

fn main() -> Result<()> {
    let args = Args::parse();
    let artifact_config = ProfileArtifactConfig {
        out_dir: args.profile_out.clone(),
        write_timeline: args.profile_timeline,
        write_flamegraph: args.profile_flamegraph,
    };
    if artifact_config.trace_enabled() && artifact_config.out_dir.is_none() {
        bail!("--profile-timeline and --profile-flamegraph require --profile-out <DIR>");
    }
    let profile_enabled = args.profile || args.profile_full || artifact_config.artifacts_enabled();
    let _trace_guard = args
        .dump_dir
        .as_ref()
        .map(|dir| install_dump_sink(dir, args.dump_mode))
        .transpose()?;
    let _profile_trace_guard = if profile_enabled && artifact_config.trace_enabled() {
        profiling::trace_reset();
        profiling::trace_enable();
        Some(TraceEnableGuard)
    } else {
        None
    };

    if profile_enabled {
        profiling::reset();
    }

    let command = match args.command {
        Command::Convert(convert) => return run_conversion(convert),
        Command::IrStats(stats) => return ir::run_stats(stats),
        Command::IrDiff(diff) => return ir::run_diff(diff),
        command => command,
    };

    let backend = args
        .backend
        .or_else(|| env::var("GPTRS_BACKEND").ok())
        .unwrap_or_else(|| gpt_rs_backends::DEFAULT.to_string());
    let profile_consumed = run(
        backend.trim(),
        command,
        profile_enabled,
        args.profile_full,
        &artifact_config,
    )?;

    if profile_enabled && !profile_consumed {
        let options = profiling::ProfileFormatOptions {
            profile_full: args.profile_full,
            ..profiling::ProfileFormatOptions::default()
        };
        if let Some(snapshot) = profiling::take_snapshot(false, &options) {
            eprintln!("{}", snapshot.formatted);
            write_profile_artifacts(&snapshot, &artifact_config)?;
        } else {
            eprintln!(
                "profiling enabled but no report available; rebuild gpt-rs with profiler support"
            );
        }
    }

    Ok(())
}

/// Runs `command` on the backend `backend`. Returns whether the command printed the profile.
fn run(
    backend: &str,
    command: Command,
    profile: bool,
    profile_full: bool,
    artifact_config: &ProfileArtifactConfig,
) -> Result<bool> {
    // Keep profiler instrumentation compiled in, but suspend runtime collection unless
    // the user explicitly asked for a report via `--profile`.
    let _suspend_profiler = if profile {
        None
    } else {
        Some(profiling::suspend())
    };

    match command {
        Command::Generate(args) => {
            run_generate(backend, &args, profile)?;
            Ok(false)
        }
        Command::Benchmark(args) => {
            run_benchmark(backend, &args, profile, profile_full, artifact_config)
        }
        Command::Forward(args) => {
            run_forward(backend, &args, profile)?;
            Ok(false)
        }
        Command::Convert(_) | Command::IrStats(_) | Command::IrDiff(_) => {
            unreachable!("main runs the commands that load no model")
        }
    }
}

fn run_generate(backend: &str, args: &GenerateArgs, profile: bool) -> Result<()> {
    ensure!(args.max_tokens > 0, "--max-tokens must be > 0");
    let session = args.causal.session(backend)?;
    let stop_tokens = if args.ignore_eos {
        &[][..]
    } else {
        session.model.eos_token_ids()
    };
    let warmup = args.warmup_tokens.min(args.max_tokens);

    if profile && warmup == 0 {
        profiling::reset();
    }
    let prefill_t0 = Instant::now();
    let mut generator = session
        .generator(args.max_tokens)?
        .with_stop_tokens(stop_tokens);
    let prefill_elapsed = prefill_t0.elapsed();

    let mut decoder = StreamDecoder::new();
    let mut stdout = io::BufWriter::with_capacity(STREAM_FLUSH_BYTES, io::stdout().lock());
    let (mut generated, mut decode_steps, mut decode_elapsed) = (0usize, 0usize, Duration::ZERO);
    for step in 0.. {
        if profile && warmup > 0 && step == warmup {
            profiling::reset();
        }
        let t0 = Instant::now();
        let token = generator.step()?;
        // The prefill picked the first token. Every later step runs the forward pass of the
        // token before it.
        if step > 0 && step >= warmup {
            decode_elapsed += t0.elapsed();
            decode_steps += 1;
        }
        if stop_tokens.contains(&token) {
            break;
        }
        generated += 1;
        let piece = decoder.push(&session.tokenizer, token);
        stdout.write_all(piece.as_bytes())?;
        if generated.is_multiple_of(STREAM_FLUSH_TOKENS) || piece.contains('\n') {
            stdout.flush()?;
        }
        if generator.is_finished() {
            break;
        }
    }
    stdout.write_all(decoder.finish().as_bytes())?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;

    eprintln!(
        "Prompt: {} tokens, prefill {:.2?}",
        session.prompt_tokens.len(),
        prefill_elapsed
    );
    eprintln!(
        "Generated {generated} tokens. {decode_steps} decode steps after {warmup} warmup tokens took {decode_elapsed:.2?} ({:.2} tokens/sec)",
        decode_steps as f64 / decode_elapsed.as_secs_f64().max(1e-9),
    );
    Ok(())
}

/// `generate` flushes stdout every `STREAM_FLUSH_TOKENS` tokens, at each newline, and when the
/// buffer holds `STREAM_FLUSH_BYTES` bytes. Streamed text stays visible without one write per
/// token.
const STREAM_FLUSH_TOKENS: usize = 16;
const STREAM_FLUSH_BYTES: usize = 4096;

fn run_benchmark(
    backend: &str,
    args: &BenchmarkArgs,
    profile: bool,
    profile_full: bool,
    artifact_config: &ProfileArtifactConfig,
) -> Result<bool> {
    ensure!(args.max_tokens > 0, "--max-tokens must be > 0");
    let session = args.causal.session(backend)?;
    let prompt_tokens = &session.prompt_tokens;

    match args.mode {
        BenchmarkMode::Prefill => {
            if profile {
                profiling::reset();
            }
            let elapsed = {
                let _scope = profile.then(|| profiling::backend_scope("benchmark.prefill_total"));
                let t0 = Instant::now();
                let _generator = session.generator(args.max_tokens)?;
                t0.elapsed()
            };
            eprintln!(
                "Benchmark prefill latency: prompt_tokens={} elapsed={:.2?}",
                prompt_tokens.len(),
                elapsed
            );

            if profile {
                emit_profile_with_attribution(
                    elapsed,
                    1.0,
                    Some("run"),
                    "prefill",
                    profile_full,
                    artifact_config,
                    None,
                )?;
                return Ok(true);
            }
            Ok(false)
        }
        BenchmarkMode::FirstToken => {
            if profile {
                profiling::reset();
            }
            let (token, elapsed) = {
                let _scope =
                    profile.then(|| profiling::backend_scope("benchmark.first_token_total"));
                let t0 = Instant::now();
                let mut generator = session.generator(args.max_tokens)?;
                let token = generator.step()?;
                (token, t0.elapsed())
            };
            eprintln!(
                "Benchmark first-token latency: prompt_tokens={} token={} elapsed={:.2?} ({:.2} tokens/sec)",
                prompt_tokens.len(),
                token,
                elapsed,
                1.0f64 / elapsed.as_secs_f64().max(1e-9)
            );

            if profile {
                emit_profile_with_attribution(
                    elapsed,
                    1.0,
                    Some("run"),
                    "first-token",
                    profile_full,
                    artifact_config,
                    None,
                )?;
                return Ok(true);
            }
            Ok(false)
        }
        BenchmarkMode::DecodeSteady => {
            ensure!(
                args.max_tokens > 0,
                "--max-tokens must be > 0 for decode-steady benchmark"
            );
            let warmup = args.warmup_tokens.min(args.max_tokens);
            let measured_steps = args.max_tokens.saturating_sub(warmup);
            ensure!(
                measured_steps > 0,
                "decode-steady requires at least one measured token; got max_tokens={} warmup_tokens={}",
                args.max_tokens,
                args.warmup_tokens
            );

            let mut generator = session.generator(args.max_tokens)?;

            for _ in 0..warmup {
                generator.step()?;
            }

            if profile {
                profiling::reset();
            }
            let freeze_capture_token = args.compile_freeze_after.min(measured_steps);
            let mut compile_baseline_ms = if profile && freeze_capture_token == 0 {
                profiling::take_snapshot(false, &profiling::ProfileFormatOptions::default())
                    .and_then(|snapshot| {
                        compile_freeze::compile_ms_from_report_json(&snapshot.report_json)
                    })
            } else {
                None
            };
            let elapsed = {
                let _scope =
                    profile.then(|| profiling::backend_scope("benchmark.decode_steady_total"));
                let t0 = Instant::now();
                for step in 0..measured_steps {
                    generator.step()?;
                    if profile && step + 1 == freeze_capture_token {
                        let options = profiling::ProfileFormatOptions::default();
                        compile_baseline_ms =
                            profiling::take_snapshot(false, &options).and_then(|snapshot| {
                                compile_freeze::compile_ms_from_report_json(&snapshot.report_json)
                            });
                    }
                }
                t0.elapsed()
            };
            let tps = (measured_steps as f64) / elapsed.as_secs_f64().max(1e-9);
            eprintln!(
                "Benchmark decode-steady: measured_tokens={} warmup_tokens={} elapsed={:.2?} ({:.2} tokens/sec)",
                measured_steps,
                warmup,
                elapsed,
                tps
            );

            if profile {
                emit_profile_with_attribution(
                    elapsed,
                    measured_steps as f64,
                    Some("token"),
                    "decode-steady",
                    profile_full,
                    artifact_config,
                    Some(CompileFreezeGate {
                        measured_token_threshold: args.compile_freeze_after,
                        strict: args.strict_compile_freeze,
                        baseline_compile_ms: compile_baseline_ms,
                    }),
                )?;
                return Ok(true);
            }
            Ok(false)
        }
    }
}

fn emit_profile_with_attribution(
    elapsed: Duration,
    measured_units: f64,
    unit_label: Option<&str>,
    label: &str,
    profile_full: bool,
    artifact_config: &ProfileArtifactConfig,
    compile_freeze_gate: Option<CompileFreezeGate>,
) -> Result<()> {
    let format_options = profiling::ProfileFormatOptions {
        measured_units: Some(measured_units).filter(|units| *units > 0.0),
        unit_label: unit_label.map(str::to_string),
        profile_full,
        ..profiling::ProfileFormatOptions::default()
    };
    if let Some(snapshot) = profiling::take_snapshot(false, &format_options) {
        let report_json = snapshot.report_json.as_str();
        if let Some(gate) = compile_freeze_gate {
            match compile_freeze::evaluate_compile_freeze(report_json, gate) {
                Some(observation) if observation.has_anomaly() => {
                    eprintln!(
                        "Compile freeze check: compile activity detected after measured token {} ({:.3} ms total)",
                        gate.measured_token_threshold,
                        observation.compile_ms_after_threshold
                    );
                    if gate.strict {
                        bail!(
                            "strict compile freeze violation: compile activity ({:.3} ms) occurred after measured token {}",
                            observation.compile_ms_after_threshold,
                            gate.measured_token_threshold
                        );
                    }
                }
                Some(_) => {
                    eprintln!(
                        "Compile freeze check: no compile activity detected after measured token {}",
                        gate.measured_token_threshold
                    );
                }
                None => {
                    eprintln!(
                        "Compile freeze check: unavailable (missing profiler baseline or compile tables)"
                    );
                }
            }
        }
        if let Some(component_ms) = measured_component_ms_from_report_json(report_json) {
            let wall_ms = elapsed.as_secs_f64() * 1_000.0;
            let coverage = if wall_ms > 0.0 {
                (component_ms / wall_ms) * 100.0
            } else {
                0.0
            };
            let clamped = coverage.clamp(0.0, 100.0);
            let per_unit_ms = if measured_units > 0.0 {
                wall_ms / measured_units
            } else {
                wall_ms
            };
            eprintln!(
                "Attribution ({label}): {:.1}% measured ({:.3} ms profiled / {:.3} ms wall, {:.3} ms/unit)",
                clamped, component_ms, wall_ms, per_unit_ms
            );
            if clamped < 90.0 {
                eprintln!(
                    "Attribution warning: measured coverage is below 90% (current {:.1}%).",
                    clamped
                );
            }
        } else {
            eprintln!("Attribution ({label}): unavailable (failed to parse profiler report).");
        }
        eprintln!("{}", snapshot.formatted);
        write_profile_artifacts(&snapshot, artifact_config)?;
    } else {
        eprintln!(
            "profiling enabled but no report available; rebuild gpt-rs with profiler support"
        );
    }
    Ok(())
}

fn write_profile_artifacts(
    snapshot: &profiling::ProfileSnapshot,
    artifact_config: &ProfileArtifactConfig,
) -> Result<()> {
    let Some(out_dir) = artifact_config.out_dir.as_ref() else {
        return Ok(());
    };
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create profile output dir {}", out_dir.display()))?;

    let jsonl_path = out_dir.join("profile.jsonl");
    fs::write(&jsonl_path, snapshot.profile_jsonl.as_bytes())
        .with_context(|| format!("failed to write {}", jsonl_path.display()))?;

    if artifact_config.write_timeline || artifact_config.write_flamegraph {
        let Some(trace_json) = profiling::take_trace_json(false) else {
            return Err(anyhow!(
                "trace export requested but no trace events were captured; ensure profiler support is enabled"
            ));
        };
        if artifact_config.write_timeline {
            let timeline_path = out_dir.join("timeline.chrome_trace.json");
            fs::write(&timeline_path, trace_json.as_bytes())
                .with_context(|| format!("failed to write {}", timeline_path.display()))?;
        }
        if artifact_config.write_flamegraph {
            let folded =
                profiling::chrome_trace_to_folded(trace_json.as_str()).ok_or_else(|| {
                    anyhow!("failed to convert Chrome trace into folded flamegraph format")
                })?;
            let folded_path = out_dir.join("flamegraph.folded");
            fs::write(&folded_path, folded.as_bytes())
                .with_context(|| format!("failed to write {}", folded_path.display()))?;

            let speedscope = profiling::chrome_trace_to_speedscope(trace_json.as_str())
                .ok_or_else(|| anyhow!("failed to convert Chrome trace into speedscope format"))?;
            let speedscope_path = out_dir.join("flamegraph.speedscope.json");
            fs::write(&speedscope_path, speedscope.as_bytes())
                .with_context(|| format!("failed to write {}", speedscope_path.display()))?;
        }
    }
    Ok(())
}

fn measured_component_ms_from_report_json(report_json: &str) -> Option<f64> {
    let parsed: serde_json::Value = serde_json::from_str(report_json).ok()?;
    let sections = parsed.get("sections")?.as_array()?;
    let mut total = 0.0f64;

    for section in sections {
        let tables = section.get("tables")?;
        for key in [
            "layers",
            "functionals",
            "backend",
            "compilation",
            "compile_passes",
        ] {
            if let Some(rows) = tables.get(key).and_then(|v| v.as_array()) {
                for row in rows {
                    total += row.get("excl_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                }
            }
        }
    }
    Some(total)
}

/// Writes the C source that the C backend generates for the run-time function of a PTIR text
/// module.
fn run_conversion(args: ConvertArgs) -> Result<()> {
    let path = &args.ptir;
    let src =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let module = Module::from_text(&src).with_context(|| format!("{}", path.display()))?;
    gpt_rs_ir::ir::verify(&module).map_err(|err| anyhow!("{}: {err}", path.display()))?;
    let run_time: Vec<String> = module
        .funcs()
        .into_iter()
        .filter_map(|f| module.op(f).get::<Func>().filter(|f| !f.is_load_time()))
        .map(|f| f.name.clone())
        .collect();
    let [func] = &run_time[..] else {
        bail!(
            "{}: convert takes one run-time function, found {}",
            path.display(),
            run_time.len()
        );
    };
    let source = c_source(&module, func)?;
    fs::create_dir_all(&args.out)
        .with_context(|| format!("failed to create output dir {}", args.out.display()))?;
    let out_path = args.out.join("program.c");
    fs::write(&out_path, source)
        .with_context(|| format!("failed to write {}", out_path.display()))?;
    eprintln!(
        "Converted PTIR '{}' to {}",
        path.display(),
        out_path.display()
    );
    Ok(())
}

#[cfg(feature = "conversion-c")]
fn c_source(module: &Module, func: &str) -> Result<String> {
    gpt_rs_backend_c::generate_c_module(module, func)
}

#[cfg(not(feature = "conversion-c"))]
fn c_source(_module: &Module, _func: &str) -> Result<String> {
    bail!("convert requires building gpt-rs-cli with --features conversion-c")
}

fn run_forward(backend: &str, args: &ForwardArgs, profile: bool) -> Result<()> {
    let model = load_checkpoint(backend, &args.checkpoint, &args.load)?;

    let is_text = args.prompt.is_some() || !args.tokens.is_empty();
    if args.prompt.is_some() && !args.tokens.is_empty() {
        bail!("use either --prompt or --tokens, not both");
    }
    if args.tokenizer.is_some() && args.prompt.is_none() {
        bail!("--tokenizer is only valid together with --prompt");
    }
    if is_text && args.vision.input.is_some() {
        bail!("cannot combine text input (--prompt/--tokens) with vision input (--input)");
    }

    let tokens = match args.prompt.as_ref() {
        Some(prompt) => {
            let tokenizer = load_tokenizer(&args.checkpoint, args.tokenizer.as_deref())?;
            let tokens = tokenizer.encode(prompt)?;
            ensure!(
                !tokens.is_empty(),
                "prompt produced an empty token sequence"
            );
            tokens
        }
        None => args.tokens.clone(),
    };
    let pixels = (!is_text)
        .then(|| load_vision_input_tensor(&args.vision))
        .transpose()?;

    if profile {
        profiling::reset();
    }
    let t0 = Instant::now();
    let host = match pixels {
        Some(pixels) => model.vision()?.forward(pixels)?,
        None => model.causal_lm()?.forward(&tokens)?,
    };
    let elapsed = t0.elapsed();

    let model_kind = model.kind();
    let shape = host.dims.as_slice();
    println!("model={} output_shape={:?}", model_kind, shape);

    if shape.len() == 2 && shape[0] > 0 && shape[1] > 0 {
        let rows = shape[0];
        let cols = shape[1];
        let row = if is_text { rows - 1 } else { 0 };
        let start = row * cols;
        let logits = &host.as_f32().unwrap()[start..start + cols];
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best = i;
                best_val = v;
            }
        }
        println!("top1={} logit={:.6}", best, best_val);
    }

    eprintln!("wall_s={:.3}", elapsed.as_secs_f64());

    if let Some(out) = args.out.as_ref() {
        safetensors::save(out, [("logits", &host)])
            .with_context(|| "failed to write output tensors")?;
    }

    Ok(())
}
