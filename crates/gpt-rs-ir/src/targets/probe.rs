//! The host probe: the instruction set from CPUID, the cores and caches from Linux sysfs, and the
//! clock and read bandwidth measured with threads pinned to cores. A probe takes a few seconds, so
//! `host` keeps its result in a cache file per machine: in `$GPTRS_TARGET_CACHE_DIR`, or in
//! `gpt-rs/targets` of the user's cache directory. The file keeps the fingerprint of the host the
//! same from one process to the next. Probing again measures anew. The costs of the instructions
//! are those of the preset of the CPU's microarchitecture, which the probe does not measure. The
//! probe knows only the preset of the Zen 5 desktop parts, and fails on any other CPU.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context, Result};

use super::{create_private_dir, user_cache_dir, TargetDesc};

/// The host's target, read once per process: from the cache file when this machine has one, else
/// probed and cached.
pub(super) fn host() -> Result<TargetDesc> {
    static HOST: OnceLock<Result<TargetDesc, String>> = OnceLock::new();
    let host = HOST.get_or_init(|| cached_or_probed().map_err(|err| format!("{err:#}")));
    host.clone().map_err(|message| anyhow!(message))
}

/// The host's target from the cache file when this machine has one, else probed and cached.
fn cached_or_probed() -> Result<TargetDesc> {
    let path = host_cache_path()?;
    let Ok(text) = std::fs::read_to_string(&path) else {
        let (desc, _) = probe_host()?;
        return Ok(desc);
    };
    TargetDesc::from_toml("host", &text).with_context(|| {
        format!(
            "the host target cached in {} does not read; delete it to probe again",
            path.display()
        )
    })
}

/// The cache file of this machine's target: one per CPU model, CPU count, instruction set and
/// facts of the preset it takes its costs from.
fn host_cache_path() -> Result<PathBuf> {
    let dir = match std::env::var("GPTRS_TARGET_CACHE_DIR") {
        Ok(value) if !value.trim().is_empty() => PathBuf::from(value.trim()),
        _ => user_cache_dir("targets")?,
    };
    let machine = machine::key()?;
    Ok(dir.join(format!("host-{machine}.toml")))
}

/// Probes this machine and writes its target to the cache file that `host` reads, whole: into a
/// file of this process, then renamed over the cache file. Gives the target and the cache file.
pub fn probe_host() -> Result<(TargetDesc, PathBuf)> {
    let desc = machine::probe()?;
    let path = host_cache_path()?;
    let dir = path.parent().expect("the cache file is in a directory");
    create_private_dir(dir)?;
    let partial = path.with_extension(format!("toml.{}", std::process::id()));
    std::fs::write(&partial, desc.to_toml())
        .with_context(|| format!("writing the host target to {}", partial.display()))?;
    std::fs::rename(&partial, &path)
        .with_context(|| format!("moving the host target to {}", path.display()))?;
    Ok((desc, path))
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
mod machine {
    use super::*;

    pub(super) fn key() -> Result<String> {
        bail!("the host probe reads x86 CPUID and Linux sysfs; name a preset or a target file")
    }

    pub(super) fn probe() -> Result<TargetDesc> {
        bail!("the host probe reads x86 CPUID and Linux sysfs; name a preset or a target file")
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod machine {
    use std::arch::x86_64::__cpuid;
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, ensure};

    use super::*;
    use crate::ir::Space;
    use crate::targets::{
        cpu_list, Arch, Bandwidth, ExecutionLevel, Feature, Fill, Isa, Memory, MemoryKind, Prefetch,
    };

    /// What tells two machines apart without measuring: the CPU model, the CPU count and the
    /// instruction set, with the facts of the preset whose costs the probe takes, so that a cached
    /// target taken from a preset that changed since is probed again.
    pub(super) fn key() -> Result<String> {
        let preset = microarchitecture()?;
        let known = TargetDesc::resolve(preset)
            .with_context(|| format!("the costs of this CPU, `{preset}`"))?
            .fingerprint();
        let text = format!(
            "{}|{}|{:?}|{preset}@{known}",
            model_name()?,
            online_cpus()?.len(),
            isa(preset)?
        );
        let hash = blake3::hash(text.as_bytes());
        Ok(hash.to_hex()[..16].to_string())
    }

    pub(super) fn probe() -> Result<TargetDesc> {
        let preset = microarchitecture()?;
        let known = TargetDesc::resolve(preset)
            .with_context(|| format!("the costs of this CPU, `{preset}`"))?;
        // The buffers of a launch start at the alignment the preset states.
        let topology = topology(known.buffer_alignment()?)?;
        let bandwidth = measure_bandwidth(&topology)?;
        let clock_hz = measure_clock(&topology)?;
        let desc = TargetDesc {
            name: "host".to_string(),
            description: model_name()?,
            clock_hz: Some(clock_hz),
            isa: isa(preset)?,
            levels: topology.levels,
            memories: topology.memories,
            bandwidth,
            costs: known.costs,
        };
        desc.check()?;
        Ok(desc)
    }

    /// The model name of the first CPU, from `/proc/cpuinfo`.
    fn model_name() -> Result<String> {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").context("reading /proc/cpuinfo")?;
        for line in cpuinfo.lines() {
            if let Some((key, value)) = line.split_once(':') {
                if key.trim() == "model name" {
                    return Ok(value.trim().to_string());
                }
            }
        }
        bail!("/proc/cpuinfo names no CPU model")
    }

    /// The instruction set CPUID reports, which a C compiler targets as the preset `preset` of its
    /// microarchitecture.
    fn isa(preset: &str) -> Result<Isa> {
        let mut features = Vec::new();
        for (feature, detected) in detected_features() {
            if detected {
                features.push(feature);
            }
        }
        let has = |feature| features.contains(&feature);
        let mut vector_bits = vec![128];
        if has(Feature::Avx) {
            vector_bits.push(256);
        }
        let avx512 = has(Feature::Avx512f);
        if avx512 {
            vector_bits.push(512);
        }
        Ok(Isa {
            arch: Arch::X86_64,
            features,
            vector_bits,
            vector_registers: if avx512 { 32 } else { 16 },
            write_masks: if avx512 { 7 } else { 0 },
            cc_march: preset.to_string(),
            page: Some(page_bytes()?),
        })
    }

    /// The bytes of a page of this process's memory.
    fn page_bytes() -> Result<u32> {
        // SAFETY: the call reads a constant of the system.
        let bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        u32::try_from(bytes)
            .ok()
            .filter(|&page| page > 0)
            .ok_or_else(|| anyhow!("sysconf gives no size of a page: {bytes}"))
    }

    /// Each feature a target can name, and whether this CPU has it.
    fn detected_features() -> Vec<(Feature, bool)> {
        use std::arch::is_x86_feature_detected as has;
        vec![
            (Feature::Sse42, has!("sse4.2")),
            (Feature::Popcnt, has!("popcnt")),
            (Feature::Avx, has!("avx")),
            (Feature::Avx2, has!("avx2")),
            (Feature::Fma, has!("fma")),
            (Feature::F16c, has!("f16c")),
            (Feature::Bmi2, has!("bmi2")),
            (Feature::Avx512f, has!("avx512f")),
            (Feature::Avx512bw, has!("avx512bw")),
            (Feature::Avx512cd, has!("avx512cd")),
            (Feature::Avx512dq, has!("avx512dq")),
            (Feature::Avx512vl, has!("avx512vl")),
            (Feature::Avx512bf16, has!("avx512bf16")),
            (Feature::Avx512vnni, has!("avx512vnni")),
            (Feature::Avx512vbmi, has!("avx512vbmi")),
            (Feature::Avx512vbmi2, has!("avx512vbmi2")),
            (Feature::Avx512ifma, has!("avx512ifma")),
            (Feature::Avx512bitalg, has!("avx512bitalg")),
            (Feature::Avx512vpopcntdq, has!("avx512vpopcntdq")),
            (Feature::Avx512fp16, has!("avx512fp16")),
            (Feature::Avxvnni, has!("avxvnni")),
            (Feature::Gfni, has!("gfni")),
            (Feature::Vaes, has!("vaes")),
            (Feature::Vpclmulqdq, has!("vpclmulqdq")),
        ]
    }

    /// The preset of this CPU's microarchitecture, which states the costs of its instructions, from
    /// its CPUID vendor, family and model: `znver5` for the Zen 5 desktop parts it describes,
    /// Granite Ridge, AMD family 1Ah models 40h to 4Fh. Fails for any other CPU, as the probe
    /// knows no other microarchitecture.
    fn microarchitecture() -> Result<&'static str> {
        let (vendor, signature) = (__cpuid(0), __cpuid(1));
        let mut name = Vec::with_capacity(12);
        for word in [vendor.ebx, vendor.edx, vendor.ecx] {
            name.extend_from_slice(&word.to_le_bytes());
        }
        let base_family = (signature.eax >> 8) & 0xf;
        let family = match base_family {
            0xf => base_family + ((signature.eax >> 20) & 0xff),
            _ => base_family,
        };
        let model = ((signature.eax >> 4) & 0xf) | (((signature.eax >> 16) & 0xf) << 4);
        match (name.as_slice(), family, model) {
            (b"AuthenticAMD", 0x1a, 0x40..=0x4f) => Ok("znver5"),
            _ => bail!(
                "the probe has no preset for the {} ({}, family {family:X}h, model {model:X}h) \
                 and cannot measure instruction costs: name a preset or a target file",
                model_name()?,
                String::from_utf8_lossy(&name)
            ),
        }
    }

    /// The execution levels and memories of the host, and the first CPU of each core of each L3
    /// domain, which the measurements run on.
    struct Topology {
        levels: Vec<ExecutionLevel>,
        memories: Vec<Memory>,
        domain_cpus: Vec<Vec<usize>>,
    }

    fn online_cpus() -> Result<Vec<usize>> {
        let text = fs::read_to_string("/sys/devices/system/cpu/online")
            .context("reading the online CPUs")?;
        cpu_list(&text)
    }

    fn cpu_file(cpu: usize, file: &str) -> Result<String> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/{file}");
        let text = fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
        Ok(text.trim().to_string())
    }

    /// A cache size in sysfs form: `48K`, `32M`.
    fn size(text: &str) -> Result<u64> {
        let (digits, unit) = match text
            .char_indices()
            .find(|(_, character)| !character.is_ascii_digit())
        {
            Some((at, _)) => text.split_at(at),
            None => (text, ""),
        };
        let value: u64 = digits.parse()?;
        let scale = match unit {
            "" => 1,
            "K" => 1 << 10,
            "M" => 1 << 20,
            "G" => 1 << 30,
            other => bail!("a cache size in unknown units: {other}"),
        };
        Ok(value * scale)
    }

    /// One cache of a CPU as sysfs gives it: level, size, line, ways and the CPUs that share it.
    struct SysCache {
        level: u32,
        bytes: u64,
        line: u32,
        ways: u32,
        shared: String,
    }

    /// The data and unified caches of `cpu`.
    fn caches_of(cpu: usize) -> Result<Vec<SysCache>> {
        let mut caches = Vec::new();
        for index in 0.. {
            let dir = format!("cache/index{index}");
            let Ok(kind) = cpu_file(cpu, &format!("{dir}/type")) else {
                break;
            };
            if kind == "Instruction" {
                continue;
            }
            caches.push(SysCache {
                level: cpu_file(cpu, &format!("{dir}/level"))?.parse()?,
                bytes: size(&cpu_file(cpu, &format!("{dir}/size"))?)?,
                line: cpu_file(cpu, &format!("{dir}/coherency_line_size"))?.parse()?,
                ways: cpu_file(cpu, &format!("{dir}/ways_of_associativity"))?.parse()?,
                shared: cpu_file(cpu, &format!("{dir}/shared_cpu_list"))?,
            });
        }
        Ok(caches)
    }

    /// The levels of the host: its L3 domains, each the cores that share an L3, and its cores, each
    /// with its CPUs; and its memories: its registers, the L1 and L2 of a core, the L3 of a domain,
    /// the smallest when they differ, and DRAM, whose buffers start at a multiple of
    /// `buffer_alignment` bytes.
    fn topology(buffer_alignment: u64) -> Result<Topology> {
        // Each core by the first CPU of its threads, with the list of its threads.
        let mut core_threads: BTreeMap<usize, String> = BTreeMap::new();
        for cpu in online_cpus()? {
            let siblings_text = cpu_file(cpu, "topology/thread_siblings_list")?;
            let siblings = cpu_list(&siblings_text)?;
            let first = *siblings
                .first()
                .ok_or_else(|| anyhow!("CPU {cpu} has no threads"))?;
            core_threads.insert(first, siblings_text);
        }
        let first_core = *core_threads
            .keys()
            .next()
            .ok_or_else(|| anyhow!("no core is online"))?;
        let mut memories = core_memories(first_core)?;
        let domains = l3_domains(core_threads.keys().copied())?;
        memories.push(cache_memory(&domains.smallest, "l3"));
        memories.push(Memory {
            alignment: Some(buffer_alignment),
            ..plain_memory("dram", MemoryKind::Main, None)
        });
        let levels = vec![
            ExecutionLevel {
                name: "l3".to_string(),
                units: domains.lists.len() as u32,
                cpus: domains.lists,
            },
            ExecutionLevel {
                name: "core".to_string(),
                units: core_threads.len() as u32,
                cpus: core_threads.into_values().collect(),
            },
        ];
        Ok(Topology {
            levels,
            memories,
            domain_cpus: domains.first_cpus,
        })
    }

    /// The memories of one core, from the caches of the CPU `first_core`: its registers, then its
    /// L1 and L2. Fails when the core has no L3, which groups the cores into domains.
    fn core_memories(first_core: usize) -> Result<Vec<Memory>> {
        let mut core_caches = Vec::new();
        let mut last_level = 0;
        for cache in caches_of(first_core)? {
            last_level = last_level.max(cache.level);
            if cache.level < 3 {
                core_caches.push(cache);
            }
        }
        ensure!(
            last_level == 3,
            "the host has no L3 cache to group its cores by"
        );
        core_caches.sort_by_key(|cache| cache.level);
        let mut memories = vec![plain_memory(
            "registers",
            MemoryKind::Registers,
            Some("core"),
        )];
        for cache in &core_caches {
            memories.push(cache_memory(cache, "core"));
        }
        Ok(memories)
    }

    /// The L3 domains of the host: the CPUs of each, as sysfs lists them, the first CPU of each of
    /// its cores, and the smallest of their L3 caches.
    struct L3Domains {
        lists: Vec<String>,
        first_cpus: Vec<Vec<usize>>,
        smallest: SysCache,
    }

    /// The L3 domains of the cores whose first CPUs are `cores`, by the CPUs that share each L3,
    /// in the order of their first cores.
    fn l3_domains(cores: impl Iterator<Item = usize>) -> Result<L3Domains> {
        let mut domains: BTreeMap<String, (SysCache, Vec<usize>)> = BTreeMap::new();
        for core in cores {
            let l3 = caches_of(core)?
                .into_iter()
                .find(|cache| cache.level == 3)
                .ok_or_else(|| anyhow!("CPU {core} has no L3 cache"))?;
            domains
                .entry(l3.shared.clone())
                .or_insert((l3, Vec::new()))
                .1
                .push(core);
        }
        let mut ordered: Vec<(String, SysCache, Vec<usize>)> = Vec::new();
        for (cpus, (cache, first_cpus)) in domains {
            ordered.push((cpus, cache, first_cpus));
        }
        ordered.sort_by_key(|(_, _, first_cpus)| first_cpus[0]);
        let mut lists = Vec::new();
        let mut first_cpus = Vec::new();
        let mut smallest_l3: Option<SysCache> = None;
        for (cpus, cache, domain_cores) in ordered {
            lists.push(cpus);
            first_cpus.push(domain_cores);
            if smallest_l3
                .as_ref()
                .is_none_or(|known| cache.bytes < known.bytes)
            {
                smallest_l3 = Some(cache);
            }
        }
        let smallest = smallest_l3.ok_or_else(|| anyhow!("the host has no L3 domain"))?;
        Ok(L3Domains {
            lists,
            first_cpus,
            smallest,
        })
    }

    /// A memory named `name` of `kind` with no facts of a cache, which each unit of `level` has,
    /// or the whole host without one.
    fn plain_memory(name: &str, kind: MemoryKind, level: Option<&str>) -> Memory {
        Memory {
            name: name.to_string(),
            kind,
            bytes: None,
            level: level.map(str::to_string),
            line: None,
            ways: None,
            space: None,
            prefetch: None,
            fill: Fill::Loads,
            alignment: None,
        }
    }

    /// The memory of the cache `cache`, which each unit of the level `level` has: copies held for
    /// it live in private scratch, and a prefetch asks it for lines with the hint of its level, T0
    /// for the L1, T1 for the L2 and T2 for the L3.
    fn cache_memory(cache: &SysCache, level: &str) -> Memory {
        let hint = match cache.level {
            1 => Prefetch::T0,
            2 => Prefetch::T1,
            _ => Prefetch::T2,
        };
        Memory {
            name: format!("l{}", cache.level),
            kind: MemoryKind::Cache,
            bytes: Some(cache.bytes),
            level: Some(level.to_string()),
            line: Some(cache.line),
            ways: Some(cache.ways),
            space: Some(Space::Private),
            prefetch: Some(hint),
            fill: Fill::Loads,
            alignment: None,
        }
    }

    /// Pins the calling thread to `cpu`.
    fn pin(cpu: usize) -> Result<()> {
        // SAFETY: the set is a plain bit set that `CPU_SET` fills, and the call reads it.
        let status = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(cpu, &mut set);
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set)
        };
        ensure!(status == 0, "pinning a thread to CPU {cpu} failed");
        Ok(())
    }

    /// The core counts a curve measures in a domain of `cores` cores: powers of two, then all.
    fn core_counts(cores: usize) -> Vec<usize> {
        let mut counts = Vec::new();
        let mut count = 1;
        while count < cores {
            counts.push(count);
            count *= 2;
        }
        counts.push(cores);
        counts
    }

    fn measure_bandwidth(topology: &Topology) -> Result<Vec<Bandwidth>> {
        let domains = &topology.domain_cpus;
        let smallest = domains.iter().map(Vec::len).min().unwrap_or(1);
        let cache_bytes = |name: &str| {
            let memory = topology.memories.iter().find(|memory| memory.name == name);
            memory.and_then(|cache| cache.bytes)
        };
        let l3_of_domain = cache_bytes("l3").ok_or_else(|| anyhow!("the host has no L3"))?;
        let l3_total = l3_of_domain * domains.len() as u64;
        // DRAM: eight times the cache the threads could hold, so that no pass reads cache.
        let dram_bytes = (8 * l3_total).max(512 << 20) as usize;
        let mut curves = Vec::new();
        let buffer = Buffer::new(dram_bytes);
        let mut domain_counts = vec![1];
        if domains.len() > 1 {
            domain_counts.push(domains.len());
        }
        for &domain_count in &domain_counts {
            let mut curve = Bandwidth {
                memory: "dram".to_string(),
                domains: domain_count as u32,
                cores_per_domain: Vec::new(),
                read_bytes_per_s: Vec::new(),
            };
            for count in core_counts(smallest) {
                let cpus = first_cores(domains, domain_count, count);
                curve.cores_per_domain.push(count as u32);
                curve
                    .read_bytes_per_s
                    .push(read_rate(&buffer, &cpus, dram_bytes)?);
            }
            curves.push(curve);
        }
        // L3 of the first domain: half of it, shared among the threads that read it.
        let l3_bytes = (l3_of_domain / 2) as usize;
        let mut l3 = Bandwidth {
            memory: "l3".to_string(),
            domains: 1,
            cores_per_domain: Vec::new(),
            read_bytes_per_s: Vec::new(),
        };
        for count in core_counts(domains[0].len()) {
            let cpus = first_cores(domains, 1, count);
            l3.cores_per_domain.push(count as u32);
            l3.read_bytes_per_s
                .push(read_rate(&buffer, &cpus, l3_bytes)?);
        }
        curves.push(l3);
        // L2 of one core: half of it.
        if let Some(l2_bytes) = cache_bytes("l2") {
            let cpus = first_cores(domains, 1, 1);
            let rate = read_rate(&buffer, &cpus, (l2_bytes / 2) as usize)?;
            curves.push(Bandwidth {
                memory: "l2".to_string(),
                domains: 1,
                cores_per_domain: vec![1],
                read_bytes_per_s: vec![rate],
            });
        }
        Ok(curves)
    }

    /// The first CPU of each of the first `count` cores of each of the first `domain_count` domains.
    fn first_cores(domains: &[Vec<usize>], domain_count: usize, count: usize) -> Vec<usize> {
        let mut cpus = Vec::new();
        for domain in &domains[..domain_count] {
            cpus.extend_from_slice(&domain[..count]);
        }
        cpus
    }

    /// Memory the probe reads, from a cache-line boundary, written once so that every page is
    /// backed.
    struct Buffer {
        storage: Vec<u64>,
        start: usize,
    }

    impl Buffer {
        fn new(bytes: usize) -> Self {
            let mut storage = vec![0u64; bytes / 8 + 8];
            for (index, word) in storage.iter_mut().enumerate() {
                *word = index as u64;
            }
            let start = storage.as_ptr().align_offset(64);
            Self { storage, start }
        }

        /// The first `bytes` of the buffer.
        fn words(&self, bytes: usize) -> &[u64] {
            &self.storage[self.start..self.start + bytes / 8]
        }
    }

    /// The read bandwidth, bytes per second, of threads pinned to `cpus` that each read their own
    /// part of the first `bytes` of the buffer: the best of several trials.
    fn read_rate(buffer: &Buffer, cpus: &[usize], bytes: usize) -> Result<f64> {
        let words = buffer.words(bytes);
        let part = words.len() / cpus.len() / 8 * 8;
        // Enough passes that a trial reads for about 20 ms at 100 GB/s.
        let passes = (2_000_000_000 / bytes).clamp(1, 1000);
        let mut best = f64::MIN;
        for _ in 0..5 {
            let slowest = slowest_read(words, cpus, part, passes)?;
            let read_bytes = (part * 8 * cpus.len() * passes) as f64;
            best = best.max(read_bytes / slowest.as_secs_f64());
        }
        Ok(best)
    }

    /// The time the slowest of the threads pinned to `cpus` takes to read its own `part` words of
    /// `words` `passes` times, the threads starting together once each has read its words once.
    fn slowest_read(words: &[u64], cpus: &[usize], part: usize, passes: usize) -> Result<Duration> {
        let barrier = Barrier::new(cpus.len());
        let times: Vec<Result<Duration>> = std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (thread, &cpu) in cpus.iter().enumerate() {
                let own_words = &words[thread * part..(thread + 1) * part];
                let start_together = &barrier;
                handles
                    .push(scope.spawn(move || timed_reads(cpu, own_words, start_together, passes)));
            }
            handles
                .into_iter()
                .map(|handle| handle.join().expect("a probe thread finishes"))
                .collect()
        });
        let mut slowest = Duration::ZERO;
        for time in times {
            slowest = slowest.max(time?);
        }
        Ok(slowest)
    }

    /// The time `passes` reads of `words` take on `cpu`, after one read and a wait at `barrier`
    /// for the other threads.
    fn timed_reads(
        cpu: usize,
        words: &[u64],
        barrier: &Barrier,
        passes: usize,
    ) -> Result<Duration> {
        pin(cpu)?;
        read(words);
        barrier.wait();
        let start = Instant::now();
        for _ in 0..passes {
            read(words);
        }
        Ok(start.elapsed())
    }

    /// The cycles per second of the first core of the first domain: the rate of a chain of
    /// additions that each wait for the one before, one cycle apiece.
    fn measure_clock(topology: &Topology) -> Result<f64> {
        let cpu = *first_cores(&topology.domain_cpus, 1, 1)
            .first()
            .ok_or_else(|| anyhow!("the machine has no core to measure the clock on"))?;
        std::thread::scope(|scope| {
            scope
                .spawn(move || -> Result<f64> {
                    pin(cpu)?;
                    const ADDS: u64 = 400_000_000;
                    let mut best = f64::MIN;
                    for _ in 0..5 {
                        let start = Instant::now();
                        std::hint::black_box(chain(ADDS));
                        best = best.max(ADDS as f64 / start.elapsed().as_secs_f64());
                    }
                    Ok(best)
                })
                .join()
                .expect("the clock probe thread finishes")
        })
    }

    /// `adds` dependent additions, eight to a loop iteration.
    fn chain(adds: u64) -> u64 {
        let mut value = 0u64;
        for _ in 0..adds / 8 {
            // SAFETY: the instructions add to a register and touch no memory.
            unsafe {
                std::arch::asm!(
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    "add {value}, 1",
                    value = inout(reg) value,
                    options(nomem, nostack),
                );
            }
        }
        value
    }

    /// Reads every word: in 512-bit loads where the CPU has AVX-512F, else word by word into four
    /// sums.
    fn read(words: &[u64]) {
        let sum = if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU has AVX-512F.
            unsafe { read_avx512(words) }
        } else {
            let mut sums = [0u64; 4];
            for chunk in words.as_chunks::<4>().0 {
                for lane in 0..4 {
                    sums[lane] = sums[lane].wrapping_add(chunk[lane]);
                }
            }
            sums.iter().fold(0u64, |sum, &part| sum.wrapping_add(part))
        };
        std::hint::black_box(sum);
    }

    /// Sums `words` in four 512-bit accumulators.
    #[target_feature(enable = "avx512f")]
    fn read_avx512(words: &[u64]) -> u64 {
        use std::arch::x86_64::*;
        let mut sums = [_mm512_setzero_si512(); 4];
        for chunk in words.as_chunks::<32>().0 {
            for (lane, sum) in sums.iter_mut().enumerate() {
                // SAFETY: the chunk holds 32 words, so each of its four 64-byte parts is in it.
                let vector = unsafe { _mm512_loadu_si512(chunk[lane * 8..].as_ptr().cast()) };
                *sum = _mm512_add_epi64(*sum, vector);
            }
        }
        let total = _mm512_add_epi64(
            _mm512_add_epi64(sums[0], sums[1]),
            _mm512_add_epi64(sums[2], sums[3]),
        );
        _mm512_reduce_add_epi64(total) as u64
    }
}
