//! `perf profile`: runs one case under `perf record`, then attributes its samples. The library's
//! debug info maps a sample in the code that the x86 backend generated to a line of C. The line map
//! beside that C then gives the op the line comes from and the passes that made that op. It also
//! gives the line of Rust and the layer that recorded the op. Other samples go to their symbols.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use clap::Args as ClapArgs;
use gpt_rs_backends::x86::{LineMap, LineOrigin};

#[derive(ClapArgs)]
pub struct ProfileArgs {
    /// The case, by its full name, as `gpt-rs-cli perf list` prints it.
    case: String,

    /// The backend to run it on.
    #[arg(long, default_value = "x86")]
    backend: String,

    /// Timed runs of the case, at most.
    #[arg(long, default_value_t = 400)]
    iters: usize,

    /// Run the case for at most this long, after its warmup.
    #[arg(long, value_name = "MS", default_value_t = 4000)]
    budget_ms: u64,

    /// Samples per second.
    #[arg(long, default_value_t = 4999)]
    frequency: u32,

    /// Rows of each table.
    #[arg(long, default_value_t = 12)]
    top: usize,

    /// The directory of the recording.
    #[arg(long, value_name = "DIR", default_value = "target/perf")]
    out: PathBuf,
}

pub fn run(args: ProfileArgs) -> Result<()> {
    std::fs::create_dir_all(&args.out)?;
    let data = args.out.join(format!("{}.data", file_name(&args.case)));
    record(&args, &data)?;
    let samples = samples(&data)?;
    ensure!(!samples.is_empty(), "perf recorded no samples");
    let attribution = attribute(&samples)?;
    attribution.print(args.top);
    println!("\nrecording: {}", data.display());
    Ok(())
}

fn file_name(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// Runs the case under `perf record`, writing the samples to `data`.
fn record(args: &ProfileArgs, data: &Path) -> Result<()> {
    let exe = std::env::current_exe().context("finding this program")?;
    let output = Command::new("perf")
        .args(["record", "-m", "8", "-F", &args.frequency.to_string(), "-o"])
        .arg(data)
        .arg("--")
        .arg(exe)
        .args([
            "perf",
            "run",
            "--case",
            &args.case,
            "--backends",
            &args.backend,
        ])
        .args(["--warmup", "2", "--iters", &args.iters.to_string()])
        .args(["--budget-ms", &args.budget_ms.to_string()])
        .output()
        .context("running perf record: is perf installed?")?;
    if !output.status.success() {
        bail!(
            "perf record failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // The run's table, and its errors, such as a case the backend cannot compile, which the run
    // indents under the backend's name.
    eprint!("{}", String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.lines().filter(|line| line.starts_with("  ")) {
        eprintln!("{line}");
    }
    Ok(())
}

/// One sample: the object it is in, its offset there, and its symbol.
struct Sample {
    dso: String,
    offset: u64,
    symbol: String,
}

/// The samples of `data`, from `perf script`.
fn samples(data: &Path) -> Result<Vec<Sample>> {
    let output = Command::new("perf")
        .args(["script", "-F", "ip,sym,dso,dsoff", "-i"])
        .arg(data)
        .output()
        .context("running perf script")?;
    ensure!(
        output.status.success(),
        "perf script failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    let mut samples = Vec::new();
    for line in text.lines() {
        // `<ip> <symbol> (<dso>+0x<offset>)`, the symbol and the offset when perf knows them.
        let trimmed = line.trim();
        let Some(open) = trimmed.rfind(" (") else {
            continue;
        };
        let place = trimmed[open + 2..].trim_end_matches(')');
        let (dso, offset) = match place.rsplit_once("+0x") {
            Some((dso, offset)) => (dso, u64::from_str_radix(offset, 16).unwrap_or(0)),
            None => (place, 0),
        };
        let printed_symbol = trimmed[..open]
            .split_once(' ')
            .map_or("", |(_, symbol)| symbol)
            .trim();
        let symbol = printed_symbol
            .split_once("+0x")
            .map_or(printed_symbol, |(name, _)| name);
        samples.push(Sample {
            dso: dso.to_string(),
            offset,
            symbol: symbol.to_string(),
        });
    }
    Ok(samples)
}

/// A line of generated C and the op that wrote it. Line 0 is code the compiler gave no line.
#[derive(Clone)]
struct CLine {
    function: String,
    file: String,
    line: u32,
    text: String,
    origin: Option<LineOrigin>,
}

impl CLine {
    /// What the line is, for a table: its op, or its function.
    fn what(&self) -> String {
        match &self.origin {
            Some(origin) => format!("{} #{}", origin.op, origin.number),
            None if self.line == 0 => format!("(no line) in {}", self.function),
            None => format!("(no op) in {}", self.function),
        }
    }
}

/// The samples, where they fell.
struct Attribution {
    total: usize,
    /// Samples in generated code, by line of C.
    lines: Vec<(CLine, usize)>,
    /// Samples elsewhere, by object and symbol.
    other: BTreeMap<(String, String), usize>,
}

fn attribute(samples: &[Sample]) -> Result<Attribution> {
    let mut by_library: HashMap<&str, Vec<u64>> = HashMap::new();
    let mut other: BTreeMap<(String, String), usize> = BTreeMap::new();
    for sample in samples {
        if is_generated(&sample.dso) {
            by_library
                .entry(&sample.dso)
                .or_default()
                .push(sample.offset);
        } else {
            let object = Path::new(&sample.dso)
                .file_name()
                .map_or(sample.dso.clone(), |name| {
                    name.to_string_lossy().into_owned()
                });
            *other.entry((object, sample.symbol.clone())).or_default() += 1;
        }
    }
    let mut counts: HashMap<Place, usize> = HashMap::new();
    let mut maps = SourceMaps::default();
    for (library, offsets) in &by_library {
        for place in addr2line(library, offsets)? {
            *counts.entry(place).or_default() += 1;
        }
    }
    let mut lines = Vec::with_capacity(counts.len());
    for (place, count) in counts {
        lines.push((maps.line(place), count));
    }
    lines.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    Ok(Attribution {
        total: samples.len(),
        lines,
        other,
    })
}

/// Whether `dso` is a library the x86 backend compiled.
fn is_generated(dso: &str) -> bool {
    Path::new(dso)
        .file_name()
        .is_some_and(|name| name.to_string_lossy().starts_with("libgpt_rs_x86_"))
}

/// The function, file and line of a sample in generated code.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
struct Place {
    function: String,
    file: String,
    line: u32,
}

/// The place of each offset of `library`, in order, from its debug info.
fn addr2line(library: &str, offsets: &[u64]) -> Result<Vec<Place>> {
    let mut unique: Vec<u64> = offsets.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let mut places = HashMap::new();
    for chunk in unique.chunks(4096) {
        let output = Command::new("addr2line")
            .args(["-f", "-e"])
            .arg(library)
            .args(chunk.iter().map(|offset| format!("{offset:#x}")))
            .output()
            .context("running addr2line")?;
        ensure!(output.status.success(), "addr2line failed on {library}");
        let text = String::from_utf8_lossy(&output.stdout);
        let mut lines = text.lines();
        for &offset in chunk {
            let function = lines.next().unwrap_or("?").to_string();
            // `file:line`, perhaps with ` (discriminator n)`, and `?` for no line.
            let place_line = lines.next().unwrap_or("?:0");
            let place = place_line.split(" (").next().unwrap_or(place_line);
            let (file, line) = place.rsplit_once(':').unwrap_or((place, "0"));
            places.insert(
                offset,
                Place {
                    function,
                    file: file.to_string(),
                    line: line.parse().unwrap_or(0),
                },
            );
        }
    }
    Ok(offsets
        .iter()
        .map(|offset| places.get(offset).cloned().unwrap_or_default())
        .collect())
}

/// The C files and line maps read so far.
#[derive(Default)]
struct SourceMaps {
    files: HashMap<String, (Vec<String>, HashMap<u32, LineOrigin>)>,
}

impl SourceMaps {
    fn line(&mut self, place: Place) -> CLine {
        let Place {
            function,
            file,
            line,
        } = place;
        let (lines, origins) = self.files.entry(file.clone()).or_insert_with(|| {
            let text = std::fs::read_to_string(&file).unwrap_or_default();
            let map: LineMap = file
                .strip_suffix(".c")
                .and_then(|stem| std::fs::read_to_string(format!("{stem}.map.json")).ok())
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default();
            let origins = map
                .lines
                .into_iter()
                .map(|origin| (origin.line, origin))
                .collect();
            (text.lines().map(str::to_string).collect(), origins)
        });
        let line_index = line.checked_sub(1).map(|index| index as usize);
        let text = line_index
            .and_then(|index| lines.get(index))
            .map(|text| text.trim().to_string())
            .unwrap_or_default();
        let origin = origins.get(&line).cloned();
        CLine {
            function,
            file,
            line,
            text,
            origin,
        }
    }
}

impl Attribution {
    fn generated(&self) -> usize {
        self.lines.iter().map(|(_, count)| count).sum()
    }

    fn percent(&self, count: usize) -> f64 {
        100.0 * count as f64 / self.total as f64
    }

    /// `count` as a share of the samples in generated code.
    fn share(&self, count: usize) -> f64 {
        100.0 * count as f64 / self.generated().max(1) as f64
    }

    fn print(&self, top: usize) {
        let generated = self.generated();
        println!(
            "{} samples: {:.1}% in generated code, {:.1}% elsewhere",
            self.total,
            self.percent(generated),
            self.percent(self.total - generated)
        );

        println!("\nThe tables of generated code give shares of its samples.");
        self.print_groups("generated code by function", top, |line| {
            Some(line.function.clone())
        });

        println!("\nhottest lines of C");
        for (line, count) in self.lines.iter().take(top) {
            let file = Path::new(&line.file)
                .file_name()
                .map_or(line.file.clone(), |name| {
                    name.to_string_lossy().into_owned()
                });
            let statement: String = match line.text.is_empty() {
                true => line.what(),
                false => line.text.chars().take(100).collect(),
            };
            println!(
                "{:>6.2}%  {file}:{}  {statement}",
                self.share(*count),
                line.line
            );
        }

        self.print_groups("by op of the lowered program", top, |line| {
            Some(line.what())
        });
        for pass in ["loop-to-isa", "tile-to-loop", "tensor-to-tile"] {
            self.print_groups(&format!("by the op {pass} made it from"), top, |line| {
                let origin = line.origin.as_ref()?;
                origin
                    .trail
                    .iter()
                    .find(|step| step.starts_with(&format!("{pass}(")))
                    .cloned()
            });
        }
        self.print_groups("by line of Rust and layer", top, |line| {
            let origin = line.origin.as_ref()?;
            Some(format!(
                "{} in {}",
                origin.source.as_deref().unwrap_or("?"),
                origin.scope.as_deref().unwrap_or("?")
            ))
        });

        println!("\nwhere the hottest lines come from");
        for (line, count) in self.lines.iter().take(3) {
            let Some(origin) = &line.origin else {
                continue;
            };
            println!(
                "{:.2}%  line {}: {}",
                self.share(*count),
                line.line,
                line.text
            );
            println!(
                "    op #{} {} in {}",
                origin.number, origin.op, origin.function
            );
            for step in origin.trail.iter().rev() {
                println!("    made by {step}");
            }
            println!(
                "    recorded at {} in {}",
                origin.source.as_deref().unwrap_or("?"),
                origin.scope.as_deref().unwrap_or("?")
            );
        }

        println!("\nelsewhere");
        let mut other: Vec<_> = self.other.iter().collect();
        other.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
        for ((object, symbol), count) in other.into_iter().take(top) {
            let truncated: String = symbol.chars().take(90).collect();
            println!("{:>6.2}%  {object}  {truncated}", self.percent(*count));
        }
    }

    /// The samples in generated code, grouped by the key `key` gives each line.
    fn print_groups(&self, title: &str, top: usize, key: impl Fn(&CLine) -> Option<String>) {
        let mut groups: HashMap<String, usize> = HashMap::new();
        for (line, count) in &self.lines {
            let group = key(line).unwrap_or_else(|| line.what());
            *groups.entry(group).or_default() += count;
        }
        let mut ranked: Vec<_> = groups.into_iter().collect();
        ranked.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        println!("\n{title}");
        for (group, count) in ranked.into_iter().take(top) {
            println!("{:>6.2}%  {group}", self.share(count));
        }
    }
}
