//! Compiling the C of a function with clang, once per source: libraries are kept in the process
//! and in a directory of the user's on disk, keyed by the source, the compiler and its flags. A
//! library that spills a whole vector or a mask, the registers that the register model of its code
//! counts ([`gpt_rs_ir::dialects::isa::RegisterCode`]), is an error. Clang's verbose assembly
//! marks each spill in a comment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{anyhow, ensure, Context, Result};
use gpt_rs::backend::hashing::FingerprintHasher;
use gpt_rs_ir::targets::{create_private_dir, feature_name, user_cache_dir, Arch, TargetDesc};
use libloading::Library;

use crate::emit::LineMap;

/// The libraries this process has loaded, by fingerprint.
static LOADED: LazyLock<Mutex<HashMap<u64, Arc<Library>>>> = LazyLock::new(Default::default);

/// One lock per fingerprint being compiled, so two threads never compile one source at once.
static GATES: LazyLock<Mutex<HashMap<u64, Arc<Mutex<()>>>>> = LazyLock::new(Default::default);

/// The compiler and the flags it builds a library for a target with.
struct Compiler {
    path: String,
    version: String,
    flags: Vec<String>,
}

impl Compiler {
    /// `$CC`, or `cc`, for `target`, asked for its version once per process and set of flags: no
    /// fast math and no contraction, so each `arith` op rounds as the IR says.
    fn for_target(target: &TargetDesc) -> Arc<Self> {
        static COMPILERS: LazyLock<Mutex<HashMap<Vec<String>, Arc<Compiler>>>> =
            LazyLock::new(Default::default);
        let mut compilers = COMPILERS.lock().expect("compilers poisoned");
        let compiler = compilers
            .entry(target_flags(target))
            .or_insert_with(|| Arc::new(Self::new(target)));
        Arc::clone(compiler)
    }

    fn new(target: &TargetDesc) -> Self {
        let path = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let version = Command::new(&path)
            .arg("--version")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
            .unwrap_or_default();
        // `-g` changes no code; it lets a profiler map samples to lines of the C. The C is
        // vectorized as the schedules of its dispatches say, so the compiler vectorizes nothing
        // itself. It turns none of the C's loops into a library call (`memset`), across which it
        // would save registers.
        // A constant stays a memory operand of each op that reads it, as the register model of
        // the C counts it, instead of a register that the compiler hoists out of the loop and
        // holds through it (`-disable-machine-licm`), and the code computes in the order of the
        // C, which the register model counts, instead of one that starts every load of a tile
        // first and holds them all (`-enable-misched=false`). OpenMP runs the parallel loops.
        let common = [
            "-O3",
            "-g",
            "-fverbose-asm",
            "-fopenmp",
            "-ffp-contract=off",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-builtin",
            "-mllvm",
            "-disable-machine-licm",
            "-mllvm",
            "-enable-misched=false",
            "-fPIC",
        ]
        .map(str::to_string);
        let flags = [common.as_slice(), &target_flags(target)].concat();
        Self {
            path,
            version,
            flags,
        }
    }

    /// Whether the compiler is clang, whose verbose assembly marks each spill and which takes the
    /// `-mllvm` flags.
    fn is_clang(&self) -> bool {
        self.version.contains("clang")
    }

    fn fingerprint(&self, source: &str) -> u64 {
        let mut hasher = FingerprintHasher::new();
        hasher.write(&source);
        hasher.write(&self.path);
        hasher.write(&self.version);
        hasher.write(&self.flags);
        hasher.finish()
    }
}

/// The flags that let the compiler use what `target` has: `-march` its microarchitecture or
/// x86-64 level, which also tunes for it, and `-m` each feature it claims, which the level may
/// leave out.
fn target_flags(target: &TargetDesc) -> Vec<String> {
    let mut flags = vec![format!("-march={}", target.isa.cc_march)];
    for &feature in &target.isa.features {
        if feature.arch() == Arch::X86_64 {
            flags.push(format!("-m{}", feature_name(feature)));
        }
    }
    flags
}

/// The directory of compiled libraries: `$GPTRS_X86_CACHE_DIR`, or `gpt-rs/x86` in the user's
/// cache directory, which only the user can read and write.
fn cache_dir() -> Result<PathBuf> {
    match std::env::var("GPTRS_X86_CACHE_DIR") {
        Ok(dir) if !dir.trim().is_empty() => Ok(PathBuf::from(dir.trim())),
        _ => user_cache_dir("x86"),
    }
}

/// The library of `source`, compiled for `target`. The C, `map`, where each of its lines comes
/// from, as JSON in `map_json`, and the assembly are kept beside it: `program_<fingerprint>.c`,
/// `program_<fingerprint>.map.json` and `program_<fingerprint>.s`. Fails when the library spills a
/// vector or a mask register, naming the op each spill comes from. A library on disk is one that a
/// compile of the same source placed whole, by renaming it into the user's own directory.
pub(crate) fn library(
    source: &str,
    map: &LineMap,
    map_json: &str,
    target: &TargetDesc,
) -> Result<Arc<Library>> {
    let compiler = Compiler::for_target(target);
    ensure!(
        compiler.is_clang(),
        "the x86 backend reads the spill comments of clang's verbose assembly, and $CC is `{}`, \
         which is not clang: set CC to clang",
        compiler.path
    );
    let fingerprint = compiler.fingerprint(source);
    if let Some(found) = loaded(fingerprint) {
        return Ok(found);
    }
    let gate = Arc::clone(
        GATES
            .lock()
            .expect("compile gates poisoned")
            .entry(fingerprint)
            .or_default(),
    );
    let _gate = gate.lock().expect("compile gate poisoned");
    if let Some(found) = loaded(fingerprint) {
        return Ok(found);
    }
    let dir = cache_dir()?;
    create_private_dir(&dir)?;
    let source_path = dir.join(format!("program_{fingerprint:016x}.c"));
    let library_path = dir.join(format!("libgpt_rs_x86_{fingerprint:016x}.so"));
    let map_path = dir.join(format!("program_{fingerprint:016x}.map.json"));
    if !map_path.exists() {
        std::fs::write(&map_path, map_json)
            .with_context(|| format!("writing {}", map_path.display()))?;
    }
    if !library_path.exists() {
        compile(&compiler, source, &source_path, &library_path, map)?;
    }
    // SAFETY: the library is the C of `source` that this backend wrote and compiled, with no
    // constructors, placed whole in the user's directory.
    let opened = unsafe { Library::new(&library_path) }
        .with_context(|| format!("loading {}", library_path.display()))?;
    let library = Arc::new(opened);
    LOADED
        .lock()
        .expect("loaded libraries poisoned")
        .insert(fingerprint, Arc::clone(&library));
    GATES
        .lock()
        .expect("compile gates poisoned")
        .remove(&fingerprint);
    Ok(library)
}

fn loaded(fingerprint: u64) -> Option<Arc<Library>> {
    LOADED
        .lock()
        .expect("loaded libraries poisoned")
        .get(&fingerprint)
        .cloned()
}

/// Compiles `source` into `library` from `source_path`, which the debug info names, so a
/// profiler finds the lines of the C: first to assembly, kept beside it as `program_<hash>.s`,
/// then to the library. Each file is written under a name of its own and renamed into place, so
/// no reader sees half a file, and a library that spills a vector or a mask register is never
/// placed.
fn compile(
    compiler: &Compiler,
    source: &str,
    source_path: &Path,
    library: &Path,
    map: &LineMap,
) -> Result<()> {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = format!(
        "{}.{}",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    );
    let source_tmp = source_path.with_extension(format!("{nonce}.c"));
    let library_tmp = library.with_extension(format!("{nonce}.so"));
    let assembly = source_path.with_extension("s");
    let assembly_tmp = source_path.with_extension(format!("{nonce}.s"));
    std::fs::write(&source_tmp, source)?;
    std::fs::rename(&source_tmp, source_path)?;
    run(
        compiler,
        Command::new(&compiler.path)
            .args(&compiler.flags)
            .arg("-S")
            .arg("-o")
            .arg(&assembly_tmp)
            .arg(source_path),
        source_path,
    )?;
    // Data in the debug info may hold bytes that are not UTF-8; no instruction does.
    let assembly_bytes = std::fs::read(&assembly_tmp)
        .with_context(|| format!("reading {}", assembly_tmp.display()))?;
    let assembly_text = String::from_utf8_lossy(&assembly_bytes);
    std::fs::rename(&assembly_tmp, &assembly)?;
    // Verbose assembly opens each function with this comment; without the comments, no spill
    // would show.
    ensure!(
        assembly_text.contains("-- Begin function"),
        "the assembly of {} has no comments, so its spills cannot be checked: the x86 backend \
         reads the spill comments of clang's verbose assembly",
        source_path.display()
    );
    let mut problems = Vec::new();
    for spill in register_spills(&assembly_text) {
        problems.push(spill.describe(map));
    }
    ensure!(
        problems.is_empty(),
        "{} spills registers; its assembly is {}:\n{}",
        source_path.display(),
        assembly.display(),
        problems.join("\n")
    );
    run(
        compiler,
        Command::new(&compiler.path)
            .args(["-shared", "-fPIC", "-fopenmp", "-o"])
            .arg(&library_tmp)
            .arg(&assembly),
        source_path,
    )?;
    std::fs::rename(&library_tmp, library)
        .map_err(|err| anyhow!("placing {}: {err}", library.display()))
}

/// Runs `command`, the compiler on the C at `source_path`, and fails with its messages.
fn run(compiler: &Compiler, command: &mut Command, source_path: &Path) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("running {}", compiler.path))?;
    ensure!(
        output.status.success(),
        "{} failed on {}:\n{}",
        compiler.path,
        source_path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

/// Where code of the C spills: its function and line, and what it moves to the stack and back.
struct Spill {
    function: String,
    line: u32,
    what: String,
}

impl Spill {
    /// The spill, the op its line comes from and the line of code that recorded that op.
    fn describe(&self, map: &LineMap) -> String {
        let line_origin = map.lines.iter().find(|origin| origin.line == self.line);
        let from = match line_origin {
            Some(origin) => format!(
                " from {} #{}{}{}",
                origin.op,
                origin.number,
                origin
                    .scope
                    .as_ref()
                    .map_or(String::new(), |scope| format!(" in {scope}")),
                origin
                    .source
                    .as_ref()
                    .map_or(String::new(), |source| format!(" recorded at {source}")),
            ),
            None => String::new(),
        };
        format!(
            "  line {} in {}{from}: {}",
            self.line, self.function, self.what
        )
    }
}

/// The spills and reloads in `assembly` of a whole vector or a mask, which clang marks `Spill` and
/// `Reload`, each with the line of C its `.loc` directive gives.
fn register_spills(assembly: &str) -> Vec<Spill> {
    let mut found = Vec::new();
    let mut function = String::new();
    let mut line = 0;
    for text in assembly.lines() {
        // A function starts at its label in the first column: `gptrs_run_<stem>:`, or that of a
        // parallel loop that OpenMP outlines from it.
        let before_colon = text.split_once(':').map(|(head, _)| head);
        if let Some(label) = before_colon.filter(|head| head.starts_with("gptrs_")) {
            function = label.to_string();
            continue;
        }
        let instruction = text.trim_start();
        if let Some(location) = instruction.strip_prefix(".loc") {
            line = location
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(line);
            continue;
        }
        let Some((code, comment)) = instruction.split_once('#') else {
            continue;
        };
        let spills = comment.contains("Spill") || comment.contains("Reload");
        if spills && moves_vector(code) {
            found.push(Spill {
                function: function.clone(),
                line,
                what: format!("{} ({})", code.trim(), comment.trim()),
            });
        }
    }
    found
}

/// Whether the spill or reload `code` moves a whole vector or a mask. A scalar float, which lives
/// in the low lane of an `xmm` register, moves with `movss` or `movsd`.
fn moves_vector(code: &str) -> bool {
    let mnemonic = code.split_whitespace().next().unwrap_or_default();
    let scalar = mnemonic.ends_with("ss") || mnemonic.ends_with("sd");
    let registers = ["%xmm", "%ymm", "%zmm", "%k"];
    !scalar && registers.iter().any(|register| code.contains(register))
}
