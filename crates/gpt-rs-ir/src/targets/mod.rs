//! The hardware a module compiles for: its instruction set and registers, its execution levels and
//! memories, its measured clock and bandwidth, and what its instructions cost. The passes of `tile`
//! read it to choose the schedule of each dispatch, and the lowerings below `tile` read the facts
//! they realize, such as the widths of the vector registers or the buffer space of a held copy.
//!
//! A target is one of:
//! - a preset, a file of `targets/` named by its stem: `x86-64-v4`, `znver5`. Each describes one
//!   part, which its description names. Its header gives the sources of its facts. A preset
//!   named for an x86-64 level describes a part of that level;
//! - `host`, the machine this process runs on, probed once and kept in a cache file;
//! - a path to a target file, `my-cpu.toml`. A target file may start from another target and state
//!   only what differs: with `base = "znver5"`, a preset, or the path of a target file relative
//!   to it, its tables merge into those of the base key by key, and any other value it states,
//!   an array of tables included, replaces the base's.
//!
//! A target states what its instructions cost, one row per op, vector width and dtype. A row for
//! one dtype refines the row for every dtype. A pass that needs a row the target does not state
//! fails and names it.
//!
//! A target file lists the execution levels from the outermost, `[[level]]`, each with its units
//! and, when probed, the CPUs of each unit. It lists the memories from the innermost,
//! `[[memory]]`: the registers first, then caches and scratch, then main memory. A memory states an
//! execution level when each unit of that level has one of its own, as the registers do. Each cache
//! and scratch states its size and the buffer space of a copy held for it. A cache also states the
//! hint of a prefetch into it. Main memory states the alignment of the buffers it holds.
//!
//! A dispatch names its target as `"<name>@<fingerprint>"`. The fingerprint is eight hex digits of
//! a hash of the facts that the dispatch's schedule was chosen for. So a dispatch and a target that
//! changed since never meet silently.
//!
//! A program states its target where it starts, and resolves it once: the backend holds the one
//! [`Target`], and hands it to every pass that needs one.

mod presets;
mod probe;

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

pub use self::presets::PRESETS;
pub use self::probe::probe_host;
use crate::dialects::{isa, x86};
use crate::ir::{DType, Level, Space};
use crate::Op;

/// What the compiler knows of the hardware it compiles for. Its text form is TOML, as the files of
/// `targets/` hold it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetDesc {
    /// The name that resolves it: a preset's name, `host`, or the path of its file.
    #[serde(skip)]
    pub name: String,
    /// What it describes, in one line, which its fingerprint leaves out.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The cycles per second of one busy core, measured by the probe. A preset leaves it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_hz: Option<f64>,
    pub isa: Isa,
    /// The execution levels, outermost first: units of work that run at once, each made of units
    /// of the level after it. A dispatch's parallel loops run on the units of the innermost.
    #[serde(rename = "level")]
    pub levels: Vec<ExecutionLevel>,
    /// The memories, innermost first: the registers, then each memory that holds what the one
    /// before it does not.
    #[serde(rename = "memory")]
    pub memories: Vec<Memory>,
    /// Read bandwidth measured by the probe. A preset leaves it out.
    #[serde(default, rename = "bandwidth", skip_serializing_if = "Vec::is_empty")]
    pub bandwidth: Vec<Bandwidth>,
    /// What instructions cost on the microarchitecture, measured or as its vendor documents them,
    /// one row per op, vector width and dtype.
    #[serde(default, rename = "cost", skip_serializing_if = "Vec::is_empty")]
    pub costs: Vec<InstructionCost>,
}

/// The instruction set: what the vector registers hold and which extensions the code may use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Isa {
    pub arch: Arch,
    pub features: Vec<Feature>,
    /// The widths in bits of the vectors its registers hold, ascending.
    pub vector_bits: Vec<u32>,
    pub vector_registers: u32,
    /// The predicate registers that can mask a vector op: on x86, `k1` to `k7`, since an op reads
    /// `k0` as no mask. With none, masks live in vector registers.
    #[serde(default)]
    pub write_masks: u32,
    /// The name a C compiler's `-march` gives the instruction set.
    pub cc_march: String,
    /// The bytes of a page of the memory the code addresses, where the target states them. A cache
    /// whose ways hold no more than a page each places a line in the set that its address within
    /// the page names. A schedule keeps no more of the lines it reads in one set than the cache's
    /// ways but one. A target without a page states no sets: a schedule then counts only the bytes
    /// each cache holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<u32>,
}

/// An instruction set architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Arch {
    #[serde(rename = "x86_64")]
    X86_64,
    #[serde(rename = "aarch64")]
    Aarch64,
}

/// An extension of an instruction set, by the name compilers give it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Feature {
    #[serde(rename = "sse4.2")]
    Sse42,
    #[serde(rename = "popcnt")]
    Popcnt,
    #[serde(rename = "avx")]
    Avx,
    #[serde(rename = "avx2")]
    Avx2,
    #[serde(rename = "fma")]
    Fma,
    #[serde(rename = "f16c")]
    F16c,
    #[serde(rename = "bmi2")]
    Bmi2,
    #[serde(rename = "avx512f")]
    Avx512f,
    #[serde(rename = "avx512bw")]
    Avx512bw,
    #[serde(rename = "avx512cd")]
    Avx512cd,
    #[serde(rename = "avx512dq")]
    Avx512dq,
    #[serde(rename = "avx512vl")]
    Avx512vl,
    #[serde(rename = "avx512bf16")]
    Avx512bf16,
    #[serde(rename = "avx512vnni")]
    Avx512vnni,
    #[serde(rename = "avx512vbmi")]
    Avx512vbmi,
    #[serde(rename = "avx512vbmi2")]
    Avx512vbmi2,
    #[serde(rename = "avx512ifma")]
    Avx512ifma,
    #[serde(rename = "avx512bitalg")]
    Avx512bitalg,
    #[serde(rename = "avx512vpopcntdq")]
    Avx512vpopcntdq,
    #[serde(rename = "avx512fp16")]
    Avx512fp16,
    #[serde(rename = "avxvnni")]
    Avxvnni,
    #[serde(rename = "gfni")]
    Gfni,
    #[serde(rename = "vaes")]
    Vaes,
    #[serde(rename = "vpclmulqdq")]
    Vpclmulqdq,
    #[serde(rename = "neon")]
    Neon,
}

impl Feature {
    /// The instruction set the extension belongs to.
    pub fn arch(self) -> Arch {
        match self {
            Feature::Neon => Arch::Aarch64,
            _ => Arch::X86_64,
        }
    }
}

/// An execution level of a target: units of work that run at once, which a `workers` mode shares
/// work among.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionLevel {
    /// The name a schedule gives it: `core`.
    pub name: String,
    /// Its units in all.
    pub units: u32,
    /// The logical CPUs of each unit, when probed: `0-7,16-23`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpus: Vec<String>,
}

impl ExecutionLevel {
    /// The logical CPUs of each of its units, when the target lists them.
    pub fn unit_cpus(&self) -> Result<Vec<Vec<usize>>> {
        let mut units = Vec::with_capacity(self.cpus.len());
        for list in &self.cpus {
            units.push(cpu_list(list)?);
        }
        Ok(units)
    }
}

/// The CPUs of a list in the form of Linux's sysfs: `0-7,16-23`.
pub fn cpu_list(text: &str) -> Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        let (first, last) = match part.split_once('-') {
            Some((first, last)) => (first.parse::<usize>()?, last.parse::<usize>()?),
            None => {
                let cpu = part.parse::<usize>()?;
                (cpu, cpu)
            }
        };
        cpus.extend(first..=last);
    }
    Ok(cpus)
}

/// A memory of a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    /// The name a schedule gives it: `l1`.
    pub name: String,
    pub kind: MemoryKind,
    /// Its size in bytes: the size of one unit's own memory, when it states a level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// The execution level whose units each have their own: `core` for a core's own cache. None
    /// for a memory that the whole target shares.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    /// The bytes of a line of a cache, which every cache states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// The ways of each set of a cache, which every cache states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ways: Option<u32>,
    /// The buffer space of a copy held for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<Space>,
    /// The hint of a prefetch that asks it for a line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefetch: Option<Prefetch>,
    /// What moves a copy into it: the code's loads and stores, where the target states nothing.
    #[serde(default, skip_serializing_if = "Fill::is_loads")]
    pub fill: Fill,
    /// The bytes, a power of two, that each buffer of a launch in main memory starts at a multiple
    /// of: the arenas and the allocations of the buffer form. Only main memory states it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment: Option<u64>,
}

/// What moves a copy into a memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Fill {
    /// The code's loads and stores, through the registers.
    #[default]
    Loads,
    /// An engine of the memory's own, which moves the copy beside the code: a DMA engine.
    Dma,
}

impl Fill {
    /// Whether the code's loads and stores move a copy.
    pub fn is_loads(&self) -> bool {
        *self == Fill::Loads
    }
}

/// The hint of an x86 prefetch, which names the caches it asks for a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Prefetch {
    /// `prefetcht0`: every level of the caches.
    T0,
    /// `prefetcht1`: the L2 and the levels after it.
    T1,
    /// `prefetcht2`: the L3 and the levels after it.
    T2,
    /// `prefetchnta`: a place near the core for data read once, where it pollutes the caches least.
    Nta,
}

/// What a memory is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// The registers of a unit, which hold the register tile of a block.
    Registers,
    /// A cache, which holds the lines of the memory after it that the code reads.
    Cache,
    /// Memory that the code fills and reads by itself.
    Scratch,
    /// The memory that holds every buffer of a launch.
    Main,
}

/// Read bandwidth from one memory as cores are added: `domains` units of the outermost execution
/// level each run the counts of `cores_per_domain` units of the innermost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bandwidth {
    /// The memory read, by name.
    pub memory: String,
    pub domains: u32,
    pub cores_per_domain: Vec<u32>,
    /// Bytes per second, one per entry of `cores_per_domain`.
    pub read_bytes_per_s: Vec<f64>,
}

/// What one instruction costs on the microarchitecture: an op of `isa` at a vector width, for one
/// dtype or all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionCost {
    /// The op: `isa.add`.
    pub op: String,
    pub width: u32,
    #[serde(default, with = "dtype_text", skip_serializing_if = "Option::is_none")]
    pub dtype: Option<DType>,
    /// Cycles from its operands to its result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<f64>,
    /// Instructions a core starts per cycle.
    pub issue: f64,
}

/// The dtype of a cost row in its text form, `f32`, read when the target loads.
mod dtype_text {
    use std::str::FromStr;

    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::ir::DType;

    pub fn serialize<S: Serializer>(dtype: &Option<DType>, out: S) -> Result<S::Ok, S::Error> {
        match dtype {
            Some(named) => out.serialize_str(&named.to_string()),
            None => out.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(text: D) -> Result<Option<DType>, D::Error> {
        let Some(name) = Option::<String>::deserialize(text)? else {
            return Ok(None);
        };
        DType::from_str(&name).map(Some).map_err(D::Error::custom)
    }
}

/// The TOML table of the target file at `path`: what it states, over what its `base` states.
/// `chain` holds the canonical paths of the files that start from it. The file at `path` must not
/// be one of them.
fn file_table(path: &Path, chain: &mut Vec<PathBuf>) -> Result<toml::Table> {
    let canonical = std::fs::canonicalize(path)
        .with_context(|| format!("reading the target file {}", path.display()))?;
    if let Some(start) = chain.iter().position(|known| *known == canonical) {
        let mut cycle = Vec::with_capacity(chain.len() - start + 1);
        for file in &chain[start..] {
            cycle.push(file.display().to_string());
        }
        cycle.push(canonical.display().to_string());
        bail!(
            "the target files start from one another: {}",
            cycle.join(" -> ")
        );
    }
    let text = std::fs::read_to_string(&canonical)
        .with_context(|| format!("reading the target file {}", path.display()))?;
    let mut table: toml::Table = text
        .parse()
        .with_context(|| format!("the target file {}", path.display()))?;
    let Some(base_value) = table.remove("base") else {
        return Ok(table);
    };
    let base = base_value
        .as_str()
        .ok_or_else(|| anyhow!("{}: `base` names a preset or a target file", path.display()))?;
    let mut merged = match presets::preset(base) {
        Some(preset) => preset.parse()?,
        None => {
            chain.push(canonical);
            let beside = path.parent().unwrap_or(Path::new("")).join(base);
            file_table(&beside, chain)?
        }
    };
    merge_tables(&mut merged, table);
    Ok(merged)
}

/// `base` with what `over` states: tables merged key by key, and any other value replaced.
fn merge_tables(base: &mut toml::Table, over: toml::Table) {
    for (key, value) in over {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(inner)), toml::Value::Table(over_inner)) => {
                merge_tables(inner, over_inner)
            }
            (_, over_value) => {
                base.insert(key, over_value);
            }
        }
    }
}

/// The fingerprint of a target's facts ([`TargetDesc::fingerprint`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint(pub u32);

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:08x}", self.0)
    }
}

impl TargetDesc {
    /// The target `name` names: a preset, `host`, or a path to a target file.
    pub fn resolve(name: &str) -> Result<TargetDesc> {
        if let Some(text) = presets::preset(name) {
            return TargetDesc::from_toml(name, text)
                .with_context(|| format!("the preset `{name}`"));
        }
        if name == "host" {
            return probe::host();
        }
        if name.ends_with(".toml") || name.contains('/') {
            let table = file_table(Path::new(name), &mut Vec::new())?;
            return TargetDesc::from_table(name, table)
                .with_context(|| format!("the target {name}"));
        }
        bail!(
            "no target `{name}`: name a preset ({}), `host`, or a target file",
            presets::names().join(", ")
        )
    }

    /// The target of the TOML text `text`, named `name`, once its facts agree with each other.
    pub fn from_toml(name: &str, text: &str) -> Result<TargetDesc> {
        TargetDesc::from_table(name, text.parse()?)
    }

    /// The target of the TOML table `table`, named `name`, once its facts agree with each other.
    fn from_table(name: &str, table: toml::Table) -> Result<TargetDesc> {
        let mut desc: TargetDesc = table.try_into()?;
        desc.name = name.to_string();
        desc.check()?;
        Ok(desc)
    }

    /// Its TOML text, as a target file states it.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("a target serializes")
    }

    /// The fingerprint of its facts: the first 32 bits of the BLAKE3 hash of their TOML text in
    /// one form, the fields in the order of the struct, the features in order, and the
    /// description left out, so that files that state the same facts share it.
    pub fn fingerprint(&self) -> Fingerprint {
        let mut facts = self.clone();
        facts.description.clear();
        facts.isa.features.sort();
        facts.isa.features.dedup();
        let hash = blake3::hash(facts.to_toml().as_bytes());
        let bytes = hash.as_bytes();
        Fingerprint(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// The bytes per cycle that each of `units` units of the parallel level reads from `memory`
    /// when all of them read it at once, from the measured clock and bandwidth. The units that
    /// share one copy of the memory split the bandwidth measured for the largest reader count up
    /// to their number.
    pub fn read_bytes_per_cycle(&self, memory: &Memory, units: usize) -> Option<f64> {
        let clock = self.clock_hz?;
        let readers_of_one = self.units_sharing(memory).min(units).max(1);
        let mut measured: Option<(usize, f64)> = None;
        for curve in &self.bandwidth {
            if curve.memory != memory.name {
                continue;
            }
            for (index, &cores) in curve.cores_per_domain.iter().enumerate() {
                let readers = curve.domains as usize * cores as usize;
                let Some(&rate) = curve.read_bytes_per_s.get(index) else {
                    continue;
                };
                let more = measured.is_none_or(|(most, _)| readers > most);
                if readers <= readers_of_one && more {
                    measured = Some((readers, rate));
                }
            }
        }
        let (_, rate) = measured?;
        Some(rate / readers_of_one as f64 / clock)
    }

    /// The bytes that each buffer of a launch starts at a multiple of, as main memory states them.
    /// Fails when the target states no main memory or no alignment of it.
    pub fn buffer_alignment(&self) -> Result<u64> {
        let main = self
            .memories
            .iter()
            .find(|memory| memory.kind == MemoryKind::Main)
            .ok_or_else(|| anyhow!("{} states no main memory, which holds buffers", self.name))?;
        main.alignment.ok_or_else(|| {
            anyhow!(
                "{} states no alignment of the buffers in its main memory, {}",
                self.name,
                main.name
            )
        })
    }

    /// The widest vector its registers hold, in bits.
    pub fn max_vector_bits(&self) -> u32 {
        *self
            .isa
            .vector_bits
            .last()
            .expect("a target has vector widths")
    }

    /// The memory named `level`, when the target has it.
    pub fn memory(&self, level: &Level) -> Option<&Memory> {
        self.memories
            .iter()
            .find(|memory| memory.name == level.as_str())
    }

    /// The memory that a copy can be held for, named `level`: a cache or scratch.
    pub fn holding_memory(&self, level: &Level) -> Option<&Memory> {
        let memory = self.memory(level)?;
        matches!(memory.kind, MemoryKind::Cache | MemoryKind::Scratch).then_some(memory)
    }

    /// The execution level named `level`, when the target has it.
    pub fn level(&self, level: &Level) -> Option<&ExecutionLevel> {
        self.levels
            .iter()
            .find(|entry| entry.name == level.as_str())
    }

    /// The units of the execution level `level`, when the target has it.
    pub fn units(&self, level: &Level) -> Option<usize> {
        Some(self.level(level)?.units as usize)
    }

    /// The units of the parallel level that share one of `memory`: those of one unit of its
    /// level, or all of them for a memory that the whole target shares.
    pub fn units_sharing(&self, memory: &Memory) -> usize {
        let parallel = self.units(&self.parallel_level()).unwrap_or(1);
        let owners = memory
            .level
            .as_ref()
            .and_then(|level| self.units(&Level::new(level)))
            .unwrap_or(1);
        parallel.div_ceil(owners).max(1)
    }

    /// The execution level whose units run the work of a dispatch's parallel loops: its innermost.
    pub fn parallel_level(&self) -> Level {
        let innermost = self.levels.last().expect("a target has execution levels");
        Level::new(&innermost.name)
    }

    /// How the target computes `isa` code: the rules of the code that gpt.rs writes for its
    /// instruction set, when it has them: AVX-512 on an x86 target with `avx512f`.
    pub fn isa_rules(&self) -> Option<&'static dyn isa::Rules> {
        match self.isa.arch {
            Arch::X86_64 if self.has_feature(Feature::Avx512f) => Some(&x86::Avx512),
            Arch::X86_64 | Arch::Aarch64 => None,
        }
    }

    /// Whether the target computes the lane-wise `isa` ops `ops`, each with the dtype of its lanes,
    /// on registers of `lanes` lanes, and holds such registers of each of `dtypes`: one of its
    /// vector widths holds them, in all its bits or the lower ones, the rules of its architecture
    /// have code for each, and it has the extensions that code needs. Fails with the first that it
    /// lacks.
    pub fn computes_vectors(
        &self,
        ops: &[(Op, DType)],
        dtypes: &[DType],
        lanes: usize,
    ) -> Result<()> {
        let rules = self.isa_rules().ok_or_else(|| {
            anyhow!(
                "{} computes no vectors: gpt.rs has no rules for its instruction set",
                self.name
            )
        })?;
        for &dtype in dtypes {
            let bits = (lanes * dtype.bits()) as u32;
            let held = self.isa.vector_bits.iter().any(|&width| width >= bits);
            ensure!(
                dtype == DType::I1 || held,
                "{lanes} lanes of {dtype} take {bits} bits, and {} has registers of {:?} bits",
                self.name,
                self.isa.vector_bits
            );
            let needs = rules.holds(dtype, lanes)?;
            self.has_needs(needs, &format!("a register of {lanes} lanes of {dtype}"))?;
        }
        for (op, dtype) in ops {
            let needs = rules.lanewise(op, *dtype, lanes)?;
            self.has_needs(needs, &format!("{} on {lanes} lanes of {dtype}", op.name()))?;
        }
        Ok(())
    }

    /// Fails with the first extension in `needs` that the target lacks. `what` names the code in
    /// the message.
    pub fn has_needs(&self, needs: isa::Needs, what: &str) -> Result<()> {
        for &feature in needs.features {
            ensure!(
                self.has_feature(feature),
                "{what} takes {}, which {} lacks",
                feature_name(feature),
                self.name
            );
        }
        Ok(())
    }

    /// Whether the target computes `lhs * rhs + acc` rounded once, `isa.fma`: x86 with FMA3, and
    /// every AArch64 core with NEON.
    pub fn has_fma(&self) -> bool {
        self.has_feature(Feature::Fma) || self.has_feature(Feature::Neon)
    }

    pub fn has_feature(&self, feature: Feature) -> bool {
        self.isa.features.contains(&feature)
    }

    /// The cost row of the op named `op` at `width` bits for `dtype`, or for every dtype.
    pub fn cost_of(&self, op: &str, width: u32, dtype: DType) -> Option<&InstructionCost> {
        let mut generic = None;
        for cost in &self.costs {
            if cost.op != op || cost.width != width {
                continue;
            }
            match cost.dtype {
                Some(row_dtype) if row_dtype == dtype => return Some(cost),
                Some(_) => {}
                None => generic = Some(cost),
            }
        }
        generic
    }

    /// Fails with the first fact that disagrees with the others.
    pub fn check(&self) -> Result<()> {
        self.check_isa()?;
        self.check_levels()?;
        self.check_memories()?;
        self.check_bandwidth()?;
        self.check_costs()
    }

    /// Fails unless the vector widths ascend in powers of two from 64, there are vector
    /// registers, and every feature extends the ISA's architecture.
    fn check_isa(&self) -> Result<()> {
        let isa = &self.isa;
        ensure!(!isa.vector_bits.is_empty(), "the ISA has no vector widths");
        for pair in isa.vector_bits.windows(2) {
            ensure!(pair[0] < pair[1], "the vector widths are not ascending");
        }
        for &bits in &isa.vector_bits {
            ensure!(
                bits >= 64 && bits.is_power_of_two(),
                "a vector of {bits} bits is not a power of two from 64"
            );
        }
        ensure!(isa.vector_registers > 0, "the ISA has no vector registers");
        for &feature in &isa.features {
            ensure!(
                feature.arch() == isa.arch,
                "{} is not an extension of {}",
                feature_name(feature),
                arch_name(isa.arch)
            );
        }
        Ok(())
    }

    /// Fails unless the target has execution levels of distinct names, each with units that the
    /// units of the level before it share evenly, and the CPUs of each unit or none.
    fn check_levels(&self) -> Result<()> {
        ensure!(!self.levels.is_empty(), "a target has execution levels");
        for (position, level) in self.levels.iter().enumerate() {
            let named = self.levels[..position]
                .iter()
                .any(|other| other.name == level.name);
            ensure!(!named, "two execution levels are named {}", level.name);
            ensure!(level.units > 0, "level {} has no units", level.name);
            if let Some(outer) = position.checked_sub(1).map(|before| &self.levels[before]) {
                ensure!(
                    level.units % outer.units == 0,
                    "level {} has {} units, which the {} units of level {} do not share evenly",
                    level.name,
                    level.units,
                    outer.units,
                    outer.name
                );
            }
            ensure!(
                level.cpus.is_empty() || level.cpus.len() == level.units as usize,
                "level {} lists the CPUs of {} of its {} units",
                level.name,
                level.cpus.len(),
                level.units
            );
            level
                .unit_cpus()
                .with_context(|| format!("the CPUs of level {}", level.name))?;
        }
        Ok(())
    }

    /// Fails unless the memories meet each of these:
    /// - their names are distinct;
    /// - the registers come first, and main memory, when the target states it, last;
    /// - each memory with a level names one of the target's;
    /// - each cache is one that [`check_cache`] takes;
    /// - only scratch has an engine that fills it;
    /// - only main memory aligns buffers, to a power of two bytes.
    fn check_memories(&self) -> Result<()> {
        let first = self.memories.first().map(|memory| memory.kind);
        ensure!(
            first == Some(MemoryKind::Registers),
            "a target's first memory is its registers"
        );
        for (position, memory) in self.memories.iter().enumerate() {
            let name = &memory.name;
            let named = self.memories[..position]
                .iter()
                .any(|other| other.name == *name);
            ensure!(!named, "two memories are named {name}");
            let last = position + 1 == self.memories.len();
            ensure!(
                (position == 0) == (memory.kind == MemoryKind::Registers)
                    && (memory.kind != MemoryKind::Main || last),
                "memory {name} is out of order: the registers come first and main memory last"
            );
            if let Some(level) = &memory.level {
                ensure!(
                    self.levels.iter().any(|entry| entry.name == *level),
                    "memory {name} belongs to the units of level {level}, which the target does \
                     not have"
                );
            }
            if memory.kind == MemoryKind::Cache {
                check_cache(memory)?;
            }
            ensure!(
                memory.fill.is_loads() || memory.kind == MemoryKind::Scratch,
                "memory {name} is filled by an engine of its own, which only scratch is"
            );
            if let Some(alignment) = memory.alignment {
                ensure!(
                    memory.kind == MemoryKind::Main,
                    "memory {name} states an alignment of buffers, which only main memory holds"
                );
                ensure!(
                    alignment.is_power_of_two(),
                    "memory {name} aligns buffers to {alignment} bytes, not a power of two"
                );
            }
        }
        Ok(())
    }

    /// Fails unless the measured clock is positive, and each bandwidth curve reads a memory of the
    /// target, runs on units the outermost level has, measures counts of cores that ascend from
    /// one, and a positive rate for each count.
    fn check_bandwidth(&self) -> Result<()> {
        if let Some(clock) = self.clock_hz {
            ensure!(clock > 0.0, "a clock of {clock} Hz is not positive");
        }
        let outer_units = self.levels.first().map_or(0, |level| level.units);
        for curve in &self.bandwidth {
            let memory = &curve.memory;
            ensure!(
                self.memories.iter().any(|known| known.name == *memory),
                "a bandwidth curve reads {memory}, a memory the target does not have"
            );
            ensure!(
                curve.domains >= 1 && curve.domains <= outer_units,
                "a bandwidth curve of {memory} runs on {} units of the outermost level, of \
                 {outer_units}",
                curve.domains
            );
            ensure!(
                !curve.cores_per_domain.is_empty(),
                "a bandwidth curve of {memory} measures no count of cores"
            );
            let mut fewer = 0;
            for &cores in &curve.cores_per_domain {
                ensure!(
                    cores > fewer,
                    "the counts of cores {:?} of a bandwidth curve of {memory} do not ascend from \
                     one",
                    curve.cores_per_domain
                );
                fewer = cores;
            }
            ensure!(
                curve.cores_per_domain.len() == curve.read_bytes_per_s.len(),
                "a bandwidth curve of {memory} has {} counts of cores and {} rates",
                curve.cores_per_domain.len(),
                curve.read_bytes_per_s.len()
            );
            for &rate in &curve.read_bytes_per_s {
                ensure!(
                    rate > 0.0,
                    "a bandwidth curve of {memory} reads at {rate} bytes per second, which is not \
                     positive"
                );
            }
        }
        Ok(())
    }

    /// Fails unless each cost names an op of `isa` and a vector width the ISA has, and issues
    /// instructions.
    fn check_costs(&self) -> Result<()> {
        let ops = isa::IsaOps::infos();
        for cost in &self.costs {
            ensure!(
                ops.iter().any(|info| info.name == cost.op),
                "a cost names {}, which is not an op of isa",
                cost.op
            );
            ensure!(
                self.isa.vector_bits.contains(&cost.width),
                "the cost of {} is for {}-bit vectors, which the ISA does not have",
                cost.op,
                cost.width
            );
            ensure!(
                cost.issue > 0.0,
                "{} issues no instruction a cycle",
                cost.op
            );
        }
        Ok(())
    }
}

/// Fails unless the cache `memory` states its size, its line and its ways, holds bytes, has ways,
/// lines of a power of two bytes, and whole sets of its ways and lines.
fn check_cache(memory: &Memory) -> Result<()> {
    let name = &memory.name;
    let bytes = memory
        .bytes
        .ok_or_else(|| anyhow!("cache {name} states no size"))?;
    let line = memory
        .line
        .ok_or_else(|| anyhow!("cache {name} states no line"))?;
    let ways = memory
        .ways
        .ok_or_else(|| anyhow!("cache {name} states no ways"))?;
    ensure!(bytes > 0, "cache {name} holds no bytes");
    ensure!(ways > 0, "cache {name} has no ways");
    ensure!(
        line.is_power_of_two(),
        "the {line}-byte lines of cache {name} are not a power of two bytes"
    );
    ensure!(
        bytes % (u64::from(ways) * u64::from(line)) == 0,
        "the {bytes} bytes of cache {name} are not whole sets of {ways} ways of {line}-byte lines"
    );
    Ok(())
}

/// The name compilers give `feature`.
pub fn feature_name(feature: Feature) -> String {
    toml::Value::try_from(feature)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .expect("a feature has a name")
}

fn arch_name(arch: Arch) -> &'static str {
    match arch {
        Arch::X86_64 => "x86_64",
        Arch::Aarch64 => "aarch64",
    }
}

/// How a dispatch names its target: the name that resolves it, and the fingerprint of the facts its
/// schedule was chosen for. Its text form is a string, `"<name>@<fingerprint>"`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = String)]
pub struct TargetRef {
    pub name: String,
    pub fingerprint: Fingerprint,
}

impl TargetRef {
    /// The target it names, as [`TargetRef::resolve`] gives it, read once per process: a
    /// reference names its facts by their fingerprint, so a later read gives the same ones.
    pub fn target(&self) -> Result<Target> {
        static RESOLVED: OnceLock<Mutex<HashMap<TargetRef, Target>>> = OnceLock::new();
        let resolved = RESOLVED.get_or_init(Mutex::default);
        let known = resolved
            .lock()
            .expect("no thread panics holding the targets")
            .get(self)
            .cloned();
        if let Some(target) = known {
            return Ok(target);
        }
        let target = Target::new(self.resolve()?);
        resolved
            .lock()
            .expect("no thread panics holding the targets")
            .insert(self.clone(), target.clone());
        Ok(target)
    }

    /// The target it names, which must still have the facts its fingerprint hashes.
    pub fn resolve(&self) -> Result<TargetDesc> {
        let desc = TargetDesc::resolve(&self.name)?;
        let now = desc.fingerprint();
        ensure!(
            now == self.fingerprint,
            "the target {name}@{} is now {name}@{now}: the schedule was chosen for other facts",
            self.fingerprint,
            name = self.name
        );
        Ok(desc)
    }
}

impl From<&TargetRef> for String {
    fn from(target: &TargetRef) -> String {
        format!("{}@{}", target.name, target.fingerprint)
    }
}

impl TryFrom<String> for TargetRef {
    type Error = anyhow::Error;

    fn try_from(text: String) -> Result<TargetRef> {
        let (name, hex) = text
            .rsplit_once('@')
            .ok_or_else(|| anyhow!("a target reference is `name@fingerprint`, not `{text}`"))?;
        ensure!(
            !name.is_empty() && hex.len() == 8,
            "a target reference is `name@` and 8 hex digits, not `{text}`"
        );
        let fingerprint = u32::from_str_radix(hex, 16)
            .map_err(|_| anyhow!("`{hex}` is not a fingerprint of 8 hex digits"))?;
        Ok(TargetRef {
            name: name.to_string(),
            fingerprint: Fingerprint(fingerprint),
        })
    }
}

/// A target as passes take it: the description that a name resolved to, which every pass of a
/// pipeline shares, with the reference that names it. As a pass option its text form is the name,
/// `{target = "znver5"}`, which parsing resolves. Two are equal when they have the same reference.
#[derive(crate::Attr, Debug, Clone)]
#[attr(via = String)]
pub struct Target(Arc<Named>);

/// A target's description and its reference, computed once.
#[derive(Debug)]
struct Named {
    desc: TargetDesc,
    reference: TargetRef,
}

impl Target {
    pub fn new(desc: TargetDesc) -> Target {
        let reference = TargetRef {
            name: desc.name.clone(),
            fingerprint: desc.fingerprint(),
        };
        Target(Arc::new(Named { desc, reference }))
    }

    /// The target that `name` names: a preset, `host`, or a path to a target file.
    pub fn resolve(name: &str) -> Result<Target> {
        TargetDesc::resolve(name).map(Target::new)
    }

    /// How a dispatch names it.
    pub fn reference(&self) -> &TargetRef {
        &self.0.reference
    }
}

impl std::ops::Deref for Target {
    type Target = TargetDesc;

    fn deref(&self) -> &TargetDesc {
        &self.0.desc
    }
}

impl PartialEq for Target {
    fn eq(&self, other: &Target) -> bool {
        self.0.reference == other.0.reference
    }
}

impl Eq for Target {}

impl Hash for Target {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.reference.hash(state);
    }
}

impl From<&Target> for String {
    fn from(target: &Target) -> String {
        target.name.clone()
    }
}

impl TryFrom<String> for Target {
    type Error = anyhow::Error;

    /// The target that `name` names, which fails with the whole chain of what went wrong in one
    /// message.
    fn try_from(name: String) -> Result<Target> {
        Target::resolve(&name).map_err(|err| anyhow!("{err:#}"))
    }
}

/// The target of the pass `pass`, which fails without one.
pub fn required<'t>(target: Option<&'t Target>, pass: &str) -> Result<&'t Target> {
    target.ok_or_else(|| {
        anyhow!(
            "`{pass}` needs a target: the pipeline gives the backend's, and a filetest names one, \
             `{{target = \"znver5\"}}`"
        )
    })
}

/// The directory `gpt-rs/<name>` in the user's cache directory, `$XDG_CACHE_HOME` or else
/// `~/.cache`.
pub fn user_cache_dir(name: &str) -> Result<PathBuf> {
    let cache = match std::env::var("XDG_CACHE_HOME") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => {
            let home = std::env::var("HOME")
                .map_err(|_| anyhow!("neither XDG_CACHE_HOME nor HOME is set"))?;
            Path::new(&home).join(".cache")
        }
    };
    Ok(cache.join("gpt-rs").join(name))
}

/// Creates `dir` and the directories above it that are missing, each readable and writable by
/// the user alone on a host with Unix permissions.
pub fn create_private_dir(dir: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder
        .create(dir)
        .with_context(|| format!("creating {}", dir.display()))
}
