#[cfg(feature = "profiler")]
use std::cell::Cell;
#[cfg(feature = "profiler")]
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "profiler")]
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
#[cfg(feature = "profiler")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "profiler")]
use std::sync::RwLock;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
#[cfg(feature = "profiler")]
use std::time::Instant;

#[cfg(feature = "profiler")]
use serde::{Deserialize, Serialize};

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ProfilerKey {
    Layer {
        section: u32,
        name: &'static str,
    },
    Functional {
        section: u32,
        op: &'static str,
        implementation: &'static str,
        signature: Option<u32>,
    },
    Backend {
        section: u32,
        name: &'static str,
        signature: Option<u32>,
    },
    Compile {
        section: u32,
        name: &'static str,
    },
    CompilePass {
        section: u32,
        name: &'static str,
    },
    Cache {
        section: u32,
        name: &'static str,
    },
}

#[cfg(feature = "profiler")]
impl ProfilerKey {
    #[inline]
    fn section(&self) -> u32 {
        match *self {
            ProfilerKey::Layer { section, .. } => section,
            ProfilerKey::Functional { section, .. } => section,
            ProfilerKey::Backend { section, .. } => section,
            ProfilerKey::Compile { section, .. } => section,
            ProfilerKey::CompilePass { section, .. } => section,
            ProfilerKey::Cache { section, .. } => section,
        }
    }

    #[inline]
    fn with_section(self, section: u32) -> Self {
        match self {
            ProfilerKey::Layer { name, .. } => ProfilerKey::Layer { section, name },
            ProfilerKey::Functional {
                op,
                implementation,
                signature,
                ..
            } => ProfilerKey::Functional {
                section,
                op,
                implementation,
                signature,
            },
            ProfilerKey::Backend {
                name, signature, ..
            } => ProfilerKey::Backend {
                section,
                name,
                signature,
            },
            ProfilerKey::Compile { name, .. } => ProfilerKey::Compile { section, name },
            ProfilerKey::CompilePass { name, .. } => ProfilerKey::CompilePass { section, name },
            ProfilerKey::Cache { name, .. } => ProfilerKey::Cache { section, name },
        }
    }
}

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
#[derive(Default, Clone, Copy)]
pub struct WorkStats {
    pub elements: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub flops: u64,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
}

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
#[derive(Default, Clone)]
struct Stat {
    calls: u64,
    exclusive_ns: u128,
    inclusive_ns: u128,
    work: WorkStats,
}

impl Stat {
    #[cfg(feature = "profiler")]
    fn merge_from(&mut self, other: Stat) {
        self.calls = self.calls.saturating_add(other.calls);
        self.exclusive_ns = self.exclusive_ns.saturating_add(other.exclusive_ns);
        self.inclusive_ns = self.inclusive_ns.saturating_add(other.inclusive_ns);
        self.work.elements = self.work.elements.saturating_add(other.work.elements);
        self.work.bytes_read = self.work.bytes_read.saturating_add(other.work.bytes_read);
        self.work.bytes_written = self
            .work
            .bytes_written
            .saturating_add(other.work.bytes_written);
        self.work.flops = self.work.flops.saturating_add(other.work.flops);
        self.work.alloc_bytes = self.work.alloc_bytes.saturating_add(other.work.alloc_bytes);
        self.work.alloc_count = self.work.alloc_count.saturating_add(other.work.alloc_count);
    }
}

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
struct Profiler {
    stats: Mutex<HashMap<ProfilerKey, Stat>>,
}

impl Profiler {
    #[inline]
    #[cfg_attr(not(feature = "profiler"), allow(dead_code))]
    fn instance() -> &'static Self {
        static INSTANCE: OnceLock<Profiler> = OnceLock::new();
        INSTANCE.get_or_init(|| Profiler {
            stats: Mutex::new(HashMap::new()),
        })
    }

    #[inline]
    #[cfg_attr(not(feature = "profiler"), allow(dead_code))]
    fn record(&self, key: ProfilerKey, exclusive: Duration, inclusive: Duration, work: WorkStats) {
        #[cfg(feature = "profiler")]
        {
            if profiler_suspended() {
                return;
            }
            let mut stats = self.stats.lock().expect("profiler mutex poisoned");
            let entry = stats.entry(key).or_default();
            entry.calls = entry.calls.saturating_add(1);
            entry.exclusive_ns = entry.exclusive_ns.saturating_add(exclusive.as_nanos());
            entry.inclusive_ns = entry.inclusive_ns.saturating_add(inclusive.as_nanos());
            entry.work.elements = entry.work.elements.saturating_add(work.elements);
            entry.work.bytes_read = entry.work.bytes_read.saturating_add(work.bytes_read);
            entry.work.bytes_written = entry.work.bytes_written.saturating_add(work.bytes_written);
            entry.work.flops = entry.work.flops.saturating_add(work.flops);
            entry.work.alloc_bytes = entry.work.alloc_bytes.saturating_add(work.alloc_bytes);
            entry.work.alloc_count = entry.work.alloc_count.saturating_add(work.alloc_count);
        }
        #[cfg(not(feature = "profiler"))]
        {
            let _ = key;
            let _ = exclusive;
            let _ = inclusive;
            let _ = work;
        }
    }

    #[cfg(feature = "profiler")]
    fn reset(&self) {
        self.stats.lock().expect("profiler mutex poisoned").clear();
    }

    #[cfg(feature = "profiler")]
    fn take_stats(&self) -> HashMap<ProfilerKey, Stat> {
        let mut stats = self.stats.lock().expect("profiler mutex poisoned");
        std::mem::take(&mut *stats)
    }
}

#[cfg(feature = "profiler")]
struct GuardFrame {
    key: ProfilerKey,
    start: Instant,
    child_time: Duration,
    work: WorkStats,
}

#[cfg(feature = "profiler")]
thread_local! {
    static ACTIVE_GUARDS: RefCell<Vec<GuardFrame>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "profiler")]
thread_local! {
    static PROFILER_SUSPEND_COUNT: Cell<u32> = const { Cell::new(0) };
}

#[cfg(feature = "profiler")]
fn profiler_suspended() -> bool {
    PROFILER_SUSPEND_COUNT.with(|count| count.get() > 0)
}

#[cfg(feature = "profiler")]
static NEXT_TRACE_THREAD_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "profiler")]
thread_local! {
    static TRACE_THREAD_ID: u64 = NEXT_TRACE_THREAD_ID.fetch_add(1, Ordering::Relaxed);
}

#[cfg(feature = "profiler")]
fn current_trace_thread_id() -> u64 {
    TRACE_THREAD_ID.with(|tid| *tid)
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone, Serialize)]
struct ChromeTraceArgs {
    section: u32,
    signature: Option<u32>,
    excl_us: u64,
    elements: u64,
    bytes_read: u64,
    bytes_written: u64,
    flops: u64,
    alloc_bytes: u64,
    alloc_count: u64,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone, Serialize)]
struct ChromeTraceEvent {
    name: String,
    cat: &'static str,
    ph: &'static str,
    ts: u64,
    dur: u64,
    pid: u32,
    tid: u64,
    args: ChromeTraceArgs,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone, Serialize)]
struct ChromeTraceOtherData {
    sections: Vec<String>,
    signatures: Vec<String>,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone, Serialize)]
struct ChromeTrace {
    #[serde(rename = "traceEvents")]
    trace_events: Vec<ChromeTraceEvent>,
    #[serde(rename = "displayTimeUnit")]
    display_time_unit: &'static str,
    #[serde(rename = "otherData")]
    other_data: ChromeTraceOtherData,
}

#[cfg(feature = "profiler")]
struct TraceRecorder {
    enabled: AtomicBool,
    epoch: RwLock<Instant>,
    events: Mutex<Vec<ChromeTraceEvent>>,
}

#[cfg(feature = "profiler")]
impl TraceRecorder {
    fn instance() -> &'static Self {
        static INSTANCE: OnceLock<TraceRecorder> = OnceLock::new();
        INSTANCE.get_or_init(|| TraceRecorder {
            enabled: AtomicBool::new(false),
            epoch: RwLock::new(Instant::now()),
            events: Mutex::new(Vec::new()),
        })
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        *self
            .epoch
            .write()
            .expect("profiler trace epoch lock poisoned") = Instant::now();
        self.events
            .lock()
            .expect("profiler trace mutex poisoned")
            .clear();
    }

    fn take_events(&self) -> Vec<ChromeTraceEvent> {
        let mut events = self.events.lock().expect("profiler trace mutex poisoned");
        std::mem::take(&mut *events)
    }

    fn record(
        &self,
        key: ProfilerKey,
        start: Instant,
        elapsed: Duration,
        exclusive: Duration,
        work: WorkStats,
    ) {
        if !self.enabled() {
            return;
        }

        let epoch = *self
            .epoch
            .read()
            .expect("profiler trace epoch lock poisoned");
        let ts = start.saturating_duration_since(epoch).as_micros() as u64;
        let dur = elapsed.as_micros() as u64;
        let excl_us = exclusive.as_micros() as u64;

        let (cat, name, section, signature) = match key {
            ProfilerKey::Layer { section, name } => ("layer", name.to_string(), section, None),
            ProfilerKey::Functional {
                section,
                op,
                implementation,
                signature,
            } => (
                "functional",
                format!("{op} ({implementation})"),
                section,
                signature,
            ),
            ProfilerKey::Backend {
                section,
                name,
                signature,
            } => ("backend", name.to_string(), section, signature),
            ProfilerKey::Compile { section, name } => ("compile", name.to_string(), section, None),
            ProfilerKey::CompilePass { section, name } => {
                ("compile_pass", name.to_string(), section, None)
            }
            ProfilerKey::Cache { .. } => {
                return;
            }
        };

        let event = ChromeTraceEvent {
            name,
            cat,
            ph: "X",
            ts,
            dur,
            pid: 1,
            tid: current_trace_thread_id(),
            args: ChromeTraceArgs {
                section,
                signature,
                excl_us,
                elements: work.elements,
                bytes_read: work.bytes_read,
                bytes_written: work.bytes_written,
                flops: work.flops,
                alloc_bytes: work.alloc_bytes,
                alloc_count: work.alloc_count,
            },
        };

        self.events
            .lock()
            .expect("profiler trace mutex poisoned")
            .push(event);
    }
}

#[inline(always)]
pub fn trace_enable() {
    #[cfg(feature = "profiler")]
    {
        TraceRecorder::instance()
            .enabled
            .store(true, Ordering::Relaxed);
    }
}

#[inline(always)]
pub fn trace_disable() {
    #[cfg(feature = "profiler")]
    {
        TraceRecorder::instance()
            .enabled
            .store(false, Ordering::Relaxed);
    }
}

#[inline(always)]
pub fn trace_reset() {
    #[cfg(feature = "profiler")]
    {
        TraceRecorder::instance().reset();
    }
}

#[cfg(feature = "profiler")]
pub fn take_trace_json() -> Option<String> {
    let events = TraceRecorder::instance().take_events();
    if events.is_empty() {
        return None;
    }

    let sections = section_table()
        .lock()
        .expect("profiler section mutex poisoned")
        .snapshot();
    let signatures = signature_table()
        .lock()
        .expect("profiler signature mutex poisoned")
        .snapshot();

    let trace = ChromeTrace {
        trace_events: events,
        display_time_unit: "ms",
        other_data: ChromeTraceOtherData {
            sections,
            signatures,
        },
    };
    serde_json::to_string(&trace).ok()
}

#[cfg(not(feature = "profiler"))]
pub fn take_trace_json() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_trace_json_pretty() -> Option<String> {
    let events = TraceRecorder::instance().take_events();
    if events.is_empty() {
        return None;
    }

    let sections = section_table()
        .lock()
        .expect("profiler section mutex poisoned")
        .snapshot();
    let signatures = signature_table()
        .lock()
        .expect("profiler signature mutex poisoned")
        .snapshot();

    let trace = ChromeTrace {
        trace_events: events,
        display_time_unit: "ms",
        other_data: ChromeTraceOtherData {
            sections,
            signatures,
        },
    };
    serde_json::to_string_pretty(&trace).ok()
}

#[cfg(not(feature = "profiler"))]
pub fn take_trace_json_pretty() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
#[derive(Debug, Deserialize)]
struct TraceJson {
    #[serde(rename = "traceEvents")]
    trace_events: Vec<TraceEventJson>,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Deserialize)]
struct TraceEventJson {
    name: String,
    cat: String,
    ph: String,
    ts: u64,
    dur: u64,
    tid: u64,
    #[serde(default)]
    args: TraceArgsJson,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Default, Deserialize)]
struct TraceArgsJson {
    #[serde(default)]
    excl_us: u64,
}

#[cfg(feature = "profiler")]
pub fn chrome_trace_to_folded(trace_json: &str) -> Option<String> {
    let parsed: TraceJson = serde_json::from_str(trace_json).ok()?;
    let mut by_tid: BTreeMap<u64, Vec<TraceEventJson>> = BTreeMap::new();
    for event in parsed.trace_events {
        if event.ph != "X" || event.dur == 0 {
            continue;
        }
        by_tid.entry(event.tid).or_default().push(event);
    }

    let mut folded: BTreeMap<String, u64> = BTreeMap::new();
    for (_tid, mut events) in by_tid {
        events.sort_by(|a, b| {
            a.ts.cmp(&b.ts)
                .then_with(|| b.dur.cmp(&a.dur))
                .then_with(|| a.cat.cmp(&b.cat))
                .then_with(|| a.name.cmp(&b.name))
        });
        let mut stack: Vec<(u64, String)> = Vec::new();
        for event in events {
            while let Some((end_ts, _)) = stack.last() {
                if event.ts >= *end_ts {
                    stack.pop();
                } else {
                    break;
                }
            }
            let mut frame_names = stack
                .iter()
                .map(|(_, name)| name.clone())
                .collect::<Vec<_>>();
            let frame_name = format!("{}::{}", event.cat, event.name);
            frame_names.push(frame_name.clone());
            let stack_key = frame_names.join(";");
            let exclusive = if event.args.excl_us == 0 {
                event.dur
            } else {
                event.args.excl_us
            };
            if exclusive > 0 {
                let entry = folded.entry(stack_key).or_insert(0);
                *entry = entry.saturating_add(exclusive);
            }
            let end_ts = event.ts.saturating_add(event.dur);
            stack.push((end_ts, frame_name));
        }
    }

    let mut out = String::new();
    for (stack, micros) in folded {
        out.push_str(&format!("{stack} {micros}\n"));
    }
    Some(out)
}

#[cfg(not(feature = "profiler"))]
pub fn chrome_trace_to_folded(_trace_json: &str) -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
#[derive(Serialize)]
struct SpeedscopeFile {
    #[serde(rename = "$schema")]
    schema: &'static str,
    name: &'static str,
    exporter: &'static str,
    #[serde(rename = "activeProfileIndex")]
    active_profile_index: usize,
    shared: SpeedscopeShared,
    profiles: Vec<SpeedscopeProfile>,
}

#[cfg(feature = "profiler")]
#[derive(Serialize)]
struct SpeedscopeShared {
    frames: Vec<SpeedscopeFrame>,
}

#[cfg(feature = "profiler")]
#[derive(Serialize)]
struct SpeedscopeFrame {
    name: String,
}

#[cfg(feature = "profiler")]
#[derive(Serialize)]
struct SpeedscopeProfile {
    #[serde(rename = "type")]
    profile_type: &'static str,
    name: &'static str,
    unit: &'static str,
    #[serde(rename = "startValue")]
    start_value: u64,
    #[serde(rename = "endValue")]
    end_value: u64,
    samples: Vec<Vec<usize>>,
    weights: Vec<u64>,
}

#[cfg(feature = "profiler")]
pub fn chrome_trace_to_speedscope(trace_json: &str) -> Option<String> {
    let folded = chrome_trace_to_folded(trace_json)?;
    let mut stacks = Vec::<(Vec<String>, u64)>::new();
    for line in folded.lines().filter(|line| !line.trim().is_empty()) {
        let (stack, micros) = line.rsplit_once(' ')?;
        let weight = micros.parse::<u64>().ok()?;
        if weight == 0 {
            continue;
        }
        let frames = stack
            .split(';')
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        if frames.is_empty() {
            continue;
        }
        stacks.push((frames, weight));
    }
    stacks.sort_by(|a, b| {
        let a_key = a.0.join(";");
        let b_key = b.0.join(";");
        a_key.cmp(&b_key).then_with(|| a.1.cmp(&b.1))
    });

    let mut frame_names = BTreeSet::new();
    for (frames, _) in &stacks {
        for frame in frames {
            frame_names.insert(frame.clone());
        }
    }
    let frames = frame_names
        .into_iter()
        .map(|name| SpeedscopeFrame { name })
        .collect::<Vec<_>>();
    let frame_map = frames
        .iter()
        .enumerate()
        .map(|(idx, frame)| (frame.name.clone(), idx))
        .collect::<BTreeMap<_, _>>();

    let mut samples = Vec::<Vec<usize>>::new();
    let mut weights = Vec::<u64>::new();
    for (stack, weight) in stacks {
        let sample = stack
            .iter()
            .filter_map(|frame| frame_map.get(frame).copied())
            .collect::<Vec<_>>();
        if sample.is_empty() {
            continue;
        }
        samples.push(sample);
        weights.push(weight);
    }
    let end_value = weights
        .iter()
        .copied()
        .fold(0u64, |acc, weight| acc.saturating_add(weight));
    let file = SpeedscopeFile {
        schema: "https://www.speedscope.app/file-format-schema.json",
        name: "gpt-rs profile",
        exporter: "gpt-rs",
        active_profile_index: 0,
        shared: SpeedscopeShared { frames },
        profiles: vec![SpeedscopeProfile {
            profile_type: "sampled",
            name: "profile",
            unit: "microseconds",
            start_value: 0,
            end_value,
            samples,
            weights,
        }],
    };
    serde_json::to_string_pretty(&file).ok()
}

#[cfg(not(feature = "profiler"))]
pub fn chrome_trace_to_speedscope(_trace_json: &str) -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
#[derive(Default)]
struct InternTable {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

#[cfg(feature = "profiler")]
impl InternTable {
    fn with_default(default: &str) -> Self {
        let mut table = InternTable::default();
        table.intern(default);
        table
    }

    fn intern(&mut self, value: &str) -> u32 {
        if let Some(id) = self.ids.get(value).copied() {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(value.to_string());
        self.ids.insert(value.to_string(), id);
        id
    }

    fn resolve(&self, id: u32) -> &str {
        self.names
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }

    fn snapshot(&self) -> Vec<String> {
        self.names.clone()
    }
}

#[cfg(feature = "profiler")]
fn section_table() -> &'static Mutex<InternTable> {
    static TABLE: OnceLock<Mutex<InternTable>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(InternTable::with_default("default")))
}

#[cfg(feature = "profiler")]
fn signature_table() -> &'static Mutex<InternTable> {
    static TABLE: OnceLock<Mutex<InternTable>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(InternTable::default()))
}

#[cfg(feature = "profiler")]
fn intern_section(name: &str) -> u32 {
    let mut table = section_table()
        .lock()
        .expect("profiler section mutex poisoned");
    table.intern(name)
}

#[cfg(feature = "profiler")]
fn resolve_section(section: u32) -> String {
    let table = section_table()
        .lock()
        .expect("profiler section mutex poisoned");
    table.resolve(section).to_string()
}

#[cfg(feature = "profiler")]
fn intern_signature(signature: &str) -> u32 {
    let mut table = signature_table()
        .lock()
        .expect("profiler signature mutex poisoned");
    table.intern(signature)
}

#[cfg(feature = "profiler")]
fn resolve_signature(signature: u32) -> String {
    let table = signature_table()
        .lock()
        .expect("profiler signature mutex poisoned");
    table.resolve(signature).to_string()
}

#[cfg(feature = "profiler")]
thread_local! {
    static SECTION_STACK: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

#[inline(always)]
fn current_section_id() -> u32 {
    #[cfg(feature = "profiler")]
    {
        SECTION_STACK.with(|stack| stack.borrow().last().copied().unwrap_or(0))
    }
    #[cfg(not(feature = "profiler"))]
    {
        0
    }
}

pub struct SuspendGuard {
    _private: (),
}

#[inline(always)]
pub fn suspend() -> SuspendGuard {
    #[cfg(feature = "profiler")]
    {
        PROFILER_SUSPEND_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    }
    SuspendGuard { _private: () }
}

#[cfg(feature = "profiler")]
impl Drop for SuspendGuard {
    fn drop(&mut self) {
        PROFILER_SUSPEND_COUNT.with(|count| {
            let current = count.get();
            count.set(current.saturating_sub(1));
        });
    }
}

pub struct SectionGuard {
    #[cfg(feature = "profiler")]
    active: bool,
}

impl SectionGuard {
    #[inline(always)]
    fn new(active: bool) -> Self {
        #[cfg(feature = "profiler")]
        {
            Self { active }
        }
        #[cfg(not(feature = "profiler"))]
        {
            let _ = active;
            Self {}
        }
    }
}

#[cfg(feature = "profiler")]
impl Drop for SectionGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        SECTION_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let _ = stack.pop();
        });
    }
}

#[inline(always)]
pub fn push_section(name: &str) {
    #[cfg(feature = "profiler")]
    {
        let id = intern_section(name);
        SECTION_STACK.with(|stack| stack.borrow_mut().push(id));
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = name;
    }
}

#[inline(always)]
pub fn pop_section() -> bool {
    #[cfg(feature = "profiler")]
    {
        SECTION_STACK.with(|stack| stack.borrow_mut().pop().is_some())
    }
    #[cfg(not(feature = "profiler"))]
    {
        false
    }
}

#[inline(always)]
pub fn section_scope(name: &str) -> SectionGuard {
    #[cfg(feature = "profiler")]
    {
        if profiler_suspended() {
            return SectionGuard::new(false);
        }
        push_section(name);
        SectionGuard::new(true)
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = name;
        SectionGuard::new(false)
    }
}

pub struct ScopeGuard {
    #[cfg(feature = "profiler")]
    key: Option<ProfilerKey>,
}

impl ScopeGuard {
    #[inline(always)]
    fn new(key: ProfilerKey) -> Self {
        Self::new_with_work(key, WorkStats::default())
    }

    #[inline(always)]
    fn new_with_work(key: ProfilerKey, work: WorkStats) -> Self {
        #[cfg(feature = "profiler")]
        {
            if profiler_suspended() {
                return ScopeGuard { key: None };
            }
            ACTIVE_GUARDS.with(|stack| {
                stack.borrow_mut().push(GuardFrame {
                    key,
                    start: Instant::now(),
                    child_time: Duration::ZERO,
                    work,
                });
            });
            ScopeGuard { key: Some(key) }
        }
        #[cfg(not(feature = "profiler"))]
        {
            let _ = key;
            let _ = work;
            ScopeGuard {}
        }
    }
}

#[cfg(feature = "profiler")]
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let Some(expected) = self.key else {
            return;
        };
        ACTIVE_GUARDS.with(|stack| {
            let mut stack = stack.borrow_mut();
            let frame = stack.pop().expect("scope guard stack underflow");
            debug_assert!(frame.key == expected, "scope guard stack corrupted");

            let GuardFrame {
                key,
                start,
                child_time,
                work,
            } = frame;

            let elapsed = start.elapsed();
            let exclusive = elapsed.saturating_sub(child_time);
            Profiler::instance().record(key, exclusive, elapsed, work);
            TraceRecorder::instance().record(key, start, elapsed, exclusive, work);

            if let Some(parent) = stack.last_mut() {
                parent.child_time = parent.child_time.saturating_add(elapsed);
            }
        });
    }
}

#[inline(always)]
pub fn layer_scope(name: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::Layer {
        section: current_section_id(),
        name,
    })
}

#[inline(always)]
pub fn functional_scope(op: &'static str, implementation: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::Functional {
        section: current_section_id(),
        op,
        implementation,
        signature: None,
    })
}

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
#[derive(Default, Clone, Copy)]
pub struct ScopeMeta {
    signature: Option<u32>,
    work: WorkStats,
}

impl ScopeMeta {
    pub fn signature(signature: u32) -> Self {
        Self {
            signature: Some(signature),
            work: WorkStats::default(),
        }
    }

    pub fn with_work(mut self, work: WorkStats) -> Self {
        self.work = work;
        self
    }
}

#[inline(always)]
pub fn signature_id(signature: &str) -> Option<u32> {
    #[cfg(feature = "profiler")]
    {
        Some(intern_signature(signature))
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = signature;
        None
    }
}

pub fn tensor_spec_signature(spec: &crate::backend::spec::TensorSpec) -> String {
    let mut out = String::new();
    out.push_str(&format!("{:?}", spec.dtype).to_ascii_lowercase());
    out.push('[');
    for (idx, dim) in spec.shape.dims().iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        match dim {
            crate::backend::spec::Dimension::Static(v) => out.push_str(&v.to_string()),
            crate::backend::spec::Dimension::Dynamic(sym) => {
                out.push('?');
                out.push_str(sym.as_str());
            }
        }
    }
    out.push(']');
    out
}

#[inline(always)]
pub fn backend_scope_with_meta<F>(name: &'static str, meta: F) -> ScopeGuard
where
    F: FnOnce() -> ScopeMeta,
{
    #[cfg(feature = "profiler")]
    {
        if profiler_suspended() {
            return ScopeGuard { key: None };
        }
        let meta = meta();
        ScopeGuard::new_with_work(
            ProfilerKey::Backend {
                section: current_section_id(),
                name,
                signature: meta.signature,
            },
            meta.work,
        )
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = meta;
        backend_scope(name)
    }
}

#[inline(always)]
pub fn functional_scope_with_meta<F>(
    op: &'static str,
    implementation: &'static str,
    meta: F,
) -> ScopeGuard
where
    F: FnOnce() -> ScopeMeta,
{
    #[cfg(feature = "profiler")]
    {
        if profiler_suspended() {
            return ScopeGuard { key: None };
        }
        let meta = meta();
        ScopeGuard::new_with_work(
            ProfilerKey::Functional {
                section: current_section_id(),
                op,
                implementation,
                signature: meta.signature,
            },
            meta.work,
        )
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = meta;
        functional_scope(op, implementation)
    }
}

#[inline(always)]
pub fn backend_scope(name: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::Backend {
        section: current_section_id(),
        name,
        signature: None,
    })
}

#[inline(always)]
pub fn record_backend_aggregate(
    name: &'static str,
    calls: u64,
    duration: Duration,
    work: WorkStats,
) {
    record_backend_aggregate_with_signature(name, None, calls, duration, work);
}

#[inline(always)]
pub fn record_backend_aggregate_with_signature(
    name: &'static str,
    signature: Option<u32>,
    calls: u64,
    duration: Duration,
    work: WorkStats,
) {
    #[cfg(feature = "profiler")]
    {
        if calls == 0 || profiler_suspended() {
            return;
        }
        let mut stats = Profiler::instance()
            .stats
            .lock()
            .expect("profiler mutex poisoned");
        let entry = stats
            .entry(ProfilerKey::Backend {
                section: current_section_id(),
                name,
                signature,
            })
            .or_default();
        entry.calls = entry.calls.saturating_add(calls);
        entry.exclusive_ns = entry.exclusive_ns.saturating_add(duration.as_nanos());
        entry.inclusive_ns = entry.inclusive_ns.saturating_add(duration.as_nanos());
        entry.work.elements = entry.work.elements.saturating_add(work.elements);
        entry.work.bytes_read = entry.work.bytes_read.saturating_add(work.bytes_read);
        entry.work.bytes_written = entry.work.bytes_written.saturating_add(work.bytes_written);
        entry.work.flops = entry.work.flops.saturating_add(work.flops);
        entry.work.alloc_bytes = entry.work.alloc_bytes.saturating_add(work.alloc_bytes);
        entry.work.alloc_count = entry.work.alloc_count.saturating_add(work.alloc_count);
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = name;
        let _ = signature;
        let _ = calls;
        let _ = duration;
        let _ = work;
    }
}

#[inline(always)]
pub fn compile_scope(name: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::Compile {
        section: current_section_id(),
        name,
    })
}

#[inline(always)]
pub fn compile_pass_scope(name: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::CompilePass {
        section: current_section_id(),
        name,
    })
}

#[inline(always)]
pub fn cache_event(name: &'static str) {
    #[cfg(feature = "profiler")]
    {
        Profiler::instance().record(
            ProfilerKey::Cache {
                section: current_section_id(),
                name,
            },
            Duration::ZERO,
            Duration::ZERO,
            WorkStats::default(),
        );
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = name;
    }
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone, Serialize)]
pub struct TableRow {
    pub name: String,
    pub signature: Option<String>,
    pub calls: u64,
    pub per_ms: f64,
    pub excl_ms: f64,
    pub incl_ms: f64,
    pub percent: f64,
    pub percent_global: f64,
    pub elements: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub flops: u64,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
}

#[cfg(not(feature = "profiler"))]
#[derive(Debug, Clone, Default)]
pub struct TableRow {
    _private: (),
}

#[derive(Debug, Clone)]
pub struct ProfileFormatOptions {
    pub measured_units: Option<f64>,
    pub unit_label: Option<String>,
    pub profile_full: bool,
    pub backend_top_n: usize,
    pub backend_min_percent: f64,
}

impl Default for ProfileFormatOptions {
    fn default() -> Self {
        Self {
            measured_units: None,
            unit_label: None,
            profile_full: false,
            backend_top_n: 24,
            backend_min_percent: 1.0,
        }
    }
}

#[cfg(feature = "profiler")]
#[derive(Debug, Clone)]
pub struct ProfileSnapshot {
    pub formatted: String,
    pub report_json: String,
    pub profile_jsonl: String,
}

#[cfg(not(feature = "profiler"))]
#[derive(Debug, Clone, Default)]
pub struct ProfileSnapshot {
    _private: (),
}

#[cfg(feature = "profiler")]
#[derive(Debug, Default, Clone, Serialize)]
pub struct ProfilerTables {
    pub layers: Vec<TableRow>,
    pub functionals: Vec<TableRow>,
    pub backend: Vec<TableRow>,
    pub compilation: Vec<TableRow>,
    pub compile_passes: Vec<TableRow>,
    pub caches: Vec<TableRow>,
}

#[cfg(not(feature = "profiler"))]
#[derive(Debug, Default, Clone)]
pub struct ProfilerTables {
    _private: (),
}

#[cfg(feature = "profiler")]
#[derive(Debug, Default, Clone, Serialize)]
pub struct ProfilerSection {
    pub section: String,
    pub tables: ProfilerTables,
}

#[cfg(feature = "profiler")]
#[derive(Debug, Default, Clone, Serialize)]
pub struct ProfilerReport {
    pub sections: Vec<ProfilerSection>,
}

#[cfg(not(feature = "profiler"))]
#[derive(Debug, Default, Clone)]
pub struct ProfilerReport {
    _private: (),
}

#[cfg(feature = "profiler")]
fn summarise_tables(stats: HashMap<ProfilerKey, Stat>) -> Option<ProfilerTables> {
    if stats.is_empty() {
        return None;
    }

    let mut layers: Vec<(String, Option<String>, Stat)> = Vec::new();
    let mut functionals: Vec<(String, Option<String>, Stat)> = Vec::new();
    let mut backend: Vec<(String, Option<String>, Stat)> = Vec::new();
    let mut compilation: Vec<(String, Option<String>, Stat)> = Vec::new();
    let mut compile_passes: Vec<(String, Option<String>, Stat)> = Vec::new();
    let mut caches: Vec<(String, Stat)> = Vec::new();

    for (key, stat) in stats {
        match key {
            ProfilerKey::Layer { name, .. } => layers.push((name.to_string(), None, stat)),
            ProfilerKey::Functional {
                op,
                implementation,
                signature,
                ..
            } => {
                let signature = signature.map(resolve_signature);
                let name = format!("{op} ({implementation})");
                functionals.push((name, signature, stat));
            }
            ProfilerKey::Backend {
                name, signature, ..
            } => backend.push((name.to_string(), signature.map(resolve_signature), stat)),
            ProfilerKey::Compile { name, .. } => compilation.push((name.to_string(), None, stat)),
            ProfilerKey::CompilePass { name, .. } => {
                compile_passes.push((name.to_string(), None, stat));
            }
            ProfilerKey::Cache { name, .. } => caches.push((name.to_string(), stat)),
        }
    }

    fn finalize_by_percent(rows: &mut [TableRow]) {
        rows.sort_by(|a, b| {
            b.percent
                .partial_cmp(&a.percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn finalize_by_calls(rows: &mut [TableRow]) {
        rows.sort_by(|a, b| b.calls.cmp(&a.calls).then_with(|| a.name.cmp(&b.name)));
    }

    fn to_rows(
        items: Vec<(String, Option<String>, Stat)>,
        category_total_ns: f64,
        global_total_ns: f64,
    ) -> Vec<TableRow> {
        items
            .into_iter()
            .map(|(name, signature, stat)| {
                let excl_ms = stat.exclusive_ns as f64 / 1_000_000.0;
                let incl_ms = stat.inclusive_ns as f64 / 1_000_000.0;
                let per_ms = if stat.calls > 0 {
                    excl_ms / stat.calls as f64
                } else {
                    0.0
                };
                let percent = if category_total_ns > 0.0 {
                    (stat.exclusive_ns as f64 / category_total_ns) * 100.0
                } else {
                    0.0
                };
                let percent_global = if global_total_ns > 0.0 {
                    (stat.exclusive_ns as f64 / global_total_ns) * 100.0
                } else {
                    0.0
                };
                TableRow {
                    name,
                    signature,
                    calls: stat.calls,
                    per_ms,
                    excl_ms,
                    incl_ms,
                    percent,
                    percent_global,
                    elements: stat.work.elements,
                    bytes_read: stat.work.bytes_read,
                    bytes_written: stat.work.bytes_written,
                    flops: stat.work.flops,
                    alloc_bytes: stat.work.alloc_bytes,
                    alloc_count: stat.work.alloc_count,
                }
            })
            .collect()
    }

    fn to_cache_rows(items: Vec<(String, Stat)>) -> Vec<TableRow> {
        let total_calls: f64 = items.iter().map(|(_, stat)| stat.calls as f64).sum();
        items
            .into_iter()
            .map(|(name, stat)| {
                let percent = if total_calls > 0.0 {
                    (stat.calls as f64 / total_calls) * 100.0
                } else {
                    0.0
                };
                TableRow {
                    name,
                    signature: None,
                    calls: stat.calls,
                    per_ms: 0.0,
                    excl_ms: 0.0,
                    incl_ms: 0.0,
                    percent,
                    percent_global: percent,
                    elements: stat.work.elements,
                    bytes_read: stat.work.bytes_read,
                    bytes_written: stat.work.bytes_written,
                    flops: stat.work.flops,
                    alloc_bytes: stat.work.alloc_bytes,
                    alloc_count: stat.work.alloc_count,
                }
            })
            .collect()
    }

    let total_layers_ns: f64 = layers
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum();
    let total_functionals_ns: f64 = functionals
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum();
    let total_backend_ns: f64 = backend
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum();
    let total_compilation_ns: f64 = compilation
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum();
    let total_compile_passes_ns: f64 = compile_passes
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum();
    let global_total_ns = total_layers_ns
        + total_functionals_ns
        + total_backend_ns
        + total_compilation_ns
        + total_compile_passes_ns;

    let mut table = ProfilerTables {
        layers: to_rows(layers, total_layers_ns, global_total_ns),
        functionals: to_rows(functionals, total_functionals_ns, global_total_ns),
        backend: to_rows(backend, total_backend_ns, global_total_ns),
        compilation: to_rows(compilation, total_compilation_ns, global_total_ns),
        compile_passes: to_rows(compile_passes, total_compile_passes_ns, global_total_ns),
        caches: to_cache_rows(caches),
    };

    finalize_by_percent(&mut table.layers);
    finalize_by_percent(&mut table.functionals);
    finalize_by_percent(&mut table.backend);
    finalize_by_percent(&mut table.compilation);
    finalize_by_percent(&mut table.compile_passes);
    finalize_by_calls(&mut table.caches);

    Some(table)
}

#[cfg(feature = "profiler")]
pub fn take_tables() -> Option<ProfilerTables> {
    let stats = Profiler::instance().take_stats();
    let mut merged: HashMap<ProfilerKey, Stat> = HashMap::new();
    for (key, stat) in stats {
        merged
            .entry(key.with_section(0))
            .or_default()
            .merge_from(stat);
    }
    summarise_tables(merged)
}

#[cfg(not(feature = "profiler"))]
pub fn take_tables() -> Option<ProfilerTables> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_report() -> Option<ProfilerReport> {
    let stats = Profiler::instance().take_stats();
    if stats.is_empty() {
        return None;
    }

    let mut by_section: HashMap<u32, HashMap<ProfilerKey, Stat>> = HashMap::new();
    for (key, stat) in stats {
        by_section
            .entry(key.section())
            .or_default()
            .insert(key, stat);
    }

    let mut sections: Vec<(u32, HashMap<ProfilerKey, Stat>)> = by_section.into_iter().collect();
    sections.sort_by_key(|(section, _)| *section);

    let mut report = ProfilerReport::default();
    for (section_id, section_stats) in sections {
        if let Some(tables) = summarise_tables(section_stats) {
            report.sections.push(ProfilerSection {
                section: resolve_section(section_id),
                tables,
            });
        }
    }

    if report.sections.is_empty() {
        None
    } else {
        Some(report)
    }
}

#[cfg(not(feature = "profiler"))]
pub fn take_report() -> Option<ProfilerReport> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_report_json() -> Option<String> {
    let report = take_report()?;
    serde_json::to_string(&report).ok()
}

#[cfg(not(feature = "profiler"))]
pub fn take_report_json() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_report_json_pretty() -> Option<String> {
    let report = take_report()?;
    serde_json::to_string_pretty(&report).ok()
}

#[cfg(not(feature = "profiler"))]
pub fn take_report_json_pretty() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
fn positive_measured_units(options: &ProfileFormatOptions) -> Option<f64> {
    options.measured_units.filter(|value| *value > 0.0)
}

#[cfg(feature = "profiler")]
fn format_execution_kpis(tables: &ProfilerTables, options: &ProfileFormatOptions) -> String {
    let total_layers_ms: f64 = tables.layers.iter().map(|row| row.excl_ms).sum();
    let total_functionals_ms: f64 = tables.functionals.iter().map(|row| row.excl_ms).sum();
    let total_backend_ms: f64 = tables.backend.iter().map(|row| row.excl_ms).sum();
    let total_compilation_ms: f64 = tables.compilation.iter().map(|row| row.excl_ms).sum();
    let total_compile_passes_ms: f64 = tables.compile_passes.iter().map(|row| row.excl_ms).sum();
    let total_cache_events: u64 = tables.caches.iter().map(|row| row.calls).sum();
    let total_backend_calls: u64 = tables.backend.iter().map(|row| row.calls).sum();
    let total_measured_ms = total_layers_ms
        + total_functionals_ms
        + total_backend_ms
        + total_compilation_ms
        + total_compile_passes_ms;
    let total_compile_ms = total_compilation_ms + total_compile_passes_ms;
    let unit_label = options.unit_label.as_deref().unwrap_or("unit");
    let units = positive_measured_units(options);

    let mut rows: Vec<(String, String)> = Vec::new();
    rows.push((
        "measured_ms.total".to_string(),
        format!("{total_measured_ms:.3}"),
    ));
    rows.push((
        "layers_ms.total".to_string(),
        format!("{total_layers_ms:.3}"),
    ));
    rows.push((
        "functionals_ms.total".to_string(),
        format!("{total_functionals_ms:.3}"),
    ));
    rows.push((
        "backend_ms.total".to_string(),
        format!("{total_backend_ms:.3}"),
    ));
    rows.push((
        "compile_ms.total".to_string(),
        format!("{total_compile_ms:.3}"),
    ));
    rows.push((
        "backend_calls.total".to_string(),
        total_backend_calls.to_string(),
    ));
    rows.push((
        "cache_events.total".to_string(),
        total_cache_events.to_string(),
    ));

    if let Some(units) = units {
        rows.push((
            format!("measured_ms/{unit_label}"),
            format!("{:.3}", total_measured_ms / units),
        ));
        rows.push((
            format!("backend_ms/{unit_label}"),
            format!("{:.3}", total_backend_ms / units),
        ));
        rows.push((
            format!("compile_ms/{unit_label}"),
            format!("{:.3}", total_compile_ms / units),
        ));
        rows.push((
            format!("backend_calls/{unit_label}"),
            format!("{:.3}", (total_backend_calls as f64) / units),
        ));
    }

    let metric_width = rows
        .iter()
        .map(|(metric, _)| metric.len())
        .max()
        .unwrap_or("metric".len())
        .max("metric".len());
    let value_width = rows
        .iter()
        .map(|(_, value)| value.len())
        .max()
        .unwrap_or("value".len())
        .max("value".len());

    let mut output = String::new();
    output.push_str("Execution KPIs\n");
    output.push_str(&format!(
        "| {metric} | {value} |\n",
        metric = format_args!("{:<width$}", "metric", width = metric_width),
        value = format_args!("{:<width$}", "value", width = value_width),
    ));
    output.push_str(&format!(
        "|-{metric}-|-{value}-|\n",
        metric = "-".repeat(metric_width),
        value = "-".repeat(value_width),
    ));
    for (metric, value) in rows {
        output.push_str(&format!(
            "| {metric} | {value} |\n",
            metric = format_args!("{:<width$}", metric, width = metric_width),
            value = format_args!("{:>width$}", value, width = value_width),
        ));
    }
    output
}

#[cfg(feature = "profiler")]
fn maybe_compact_backend_rows(
    rows: &[TableRow],
    options: &ProfileFormatOptions,
) -> (Vec<TableRow>, Option<String>) {
    if options.profile_full || rows.len() <= options.backend_top_n {
        return (rows.to_vec(), None);
    }

    let mut selected = Vec::with_capacity(rows.len());
    let mut omitted_rows = 0usize;
    let mut omitted_ms = 0.0f64;
    for (idx, row) in rows.iter().enumerate() {
        if idx < options.backend_top_n || row.percent >= options.backend_min_percent {
            selected.push(row.clone());
        } else {
            omitted_rows += 1;
            omitted_ms += row.excl_ms;
        }
    }

    if omitted_rows == 0 {
        (selected, None)
    } else {
        (
            selected,
            Some(format!(
                "(compact: showing {} of {} rows, omitted {:.3} ms; use --profile-full for full backend table)",
                rows.len().saturating_sub(omitted_rows),
                rows.len(),
                omitted_ms
            )),
        )
    }
}

#[cfg(feature = "profiler")]
pub fn format_tables(tables: &ProfilerTables) -> String {
    format_tables_with_options(tables, &ProfileFormatOptions::default())
}

#[cfg(feature = "profiler")]
pub fn format_tables_with_options(
    tables: &ProfilerTables,
    options: &ProfileFormatOptions,
) -> String {
    fn display_name_len(row: &TableRow) -> usize {
        match row.signature.as_ref() {
            None => row.name.len(),
            Some(sig) => row.name.len().saturating_add(sig.len()).saturating_add(3),
        }
    }

    fn display_name(row: &TableRow) -> String {
        match row.signature.as_ref() {
            None => row.name.clone(),
            Some(sig) => format!("{} [{sig}]", row.name),
        }
    }

    fn format_table(
        label: &str,
        column: &str,
        rows: &[TableRow],
        options: &ProfileFormatOptions,
    ) -> String {
        let mut output = String::new();
        output.push_str(label);
        output.push('\n');

        let units = positive_measured_units(options);
        let include_unit_columns = units.is_some();

        let name_width = rows
            .iter()
            .map(display_name_len)
            .max()
            .unwrap_or(column.len())
            .max(column.len());

        let mut calls_width = "#".len();
        let mut per_width = "ms/call".len();
        let mut excl_width = "self_ms".len();
        let mut incl_width = "total_ms".len();
        let mut percent_width = "%tbl".len();
        let mut global_percent_width = "%all".len();
        let mut calls_per_unit_width = "calls/unit".len();
        let mut ms_per_unit_width = "ms/unit".len();

        for row in rows {
            calls_width = calls_width.max(format!("{}", row.calls).len());
            per_width = per_width.max(format!("{:.3}", row.per_ms).len());
            excl_width = excl_width.max(format!("{:.3}", row.excl_ms).len());
            incl_width = incl_width.max(format!("{:.3}", row.incl_ms).len());
            percent_width = percent_width.max(format!("{:.2}", row.percent).len());
            global_percent_width =
                global_percent_width.max(format!("{:.2}", row.percent_global).len());
            if let Some(units) = units {
                calls_per_unit_width =
                    calls_per_unit_width.max(format!("{:.3}", (row.calls as f64) / units).len());
                ms_per_unit_width =
                    ms_per_unit_width.max(format!("{:.3}", row.excl_ms / units).len());
            }
        }

        if include_unit_columns {
            output.push_str(&format!(
                "| {column} | {calls} | {per} | {excl} | {incl} | {calls_unit} | {ms_unit} | {percent} | {global} |\n",
                column = format_args!("{:<width$}", column, width = name_width),
                calls = format_args!("{:^width$}", "#", width = calls_width),
                per = format_args!("{:^width$}", "ms/call", width = per_width),
                excl = format_args!("{:^width$}", "self_ms", width = excl_width),
                incl = format_args!("{:^width$}", "total_ms", width = incl_width),
                calls_unit = format_args!("{:^width$}", "calls/unit", width = calls_per_unit_width),
                ms_unit = format_args!("{:^width$}", "ms/unit", width = ms_per_unit_width),
                percent = format_args!("{:^width$}", "%tbl", width = percent_width),
                global = format_args!("{:^width$}", "%all", width = global_percent_width),
            ));
            output.push_str(&format!(
                "|-{name}-|-{calls}-|-{per}-|-{excl}-|-{incl}-|-{calls_unit}-|-{ms_unit}-|-{percent}-|-{global}-|\n",
                name = "-".repeat(name_width),
                calls = "-".repeat(calls_width),
                per = "-".repeat(per_width),
                excl = "-".repeat(excl_width),
                incl = "-".repeat(incl_width),
                calls_unit = "-".repeat(calls_per_unit_width),
                ms_unit = "-".repeat(ms_per_unit_width),
                percent = "-".repeat(percent_width),
                global = "-".repeat(global_percent_width),
            ));
        } else {
            output.push_str(&format!(
                "| {column} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
                column = format_args!("{:<width$}", column, width = name_width),
                calls = format_args!("{:^width$}", "#", width = calls_width),
                per = format_args!("{:^width$}", "ms/call", width = per_width),
                excl = format_args!("{:^width$}", "self_ms", width = excl_width),
                incl = format_args!("{:^width$}", "total_ms", width = incl_width),
                percent = format_args!("{:^width$}", "%tbl", width = percent_width),
                global = format_args!("{:^width$}", "%all", width = global_percent_width),
            ));
            output.push_str(&format!(
                "|-{name}-|-{calls}-|-{per}-|-{excl}-|-{incl}-|-{percent}-|-{global}-|\n",
                name = "-".repeat(name_width),
                calls = "-".repeat(calls_width),
                per = "-".repeat(per_width),
                excl = "-".repeat(excl_width),
                incl = "-".repeat(incl_width),
                percent = "-".repeat(percent_width),
                global = "-".repeat(global_percent_width),
            ));
        }

        if rows.is_empty() {
            if include_unit_columns {
                output.push_str(&format!(
                    "| {value} | {calls} | {per} | {excl} | {incl} | {calls_unit} | {ms_unit} | {percent} | {global} |\n",
                    value = format_args!("{:<width$}", "(no data)", width = name_width),
                    calls = format_args!("{:>width$}", "-", width = calls_width),
                    per = format_args!("{:>width$}", "-", width = per_width),
                    excl = format_args!("{:>width$}", "-", width = excl_width),
                    incl = format_args!("{:>width$}", "-", width = incl_width),
                    calls_unit = format_args!("{:>width$}", "-", width = calls_per_unit_width),
                    ms_unit = format_args!("{:>width$}", "-", width = ms_per_unit_width),
                    percent = format_args!("{:>width$}", "-", width = percent_width),
                    global = format_args!("{:>width$}", "-", width = global_percent_width),
                ));
            } else {
                output.push_str(&format!(
                    "| {value} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
                    value = format_args!("{:<width$}", "(no data)", width = name_width),
                    calls = format_args!("{:>width$}", "-", width = calls_width),
                    per = format_args!("{:>width$}", "-", width = per_width),
                    excl = format_args!("{:>width$}", "-", width = excl_width),
                    incl = format_args!("{:>width$}", "-", width = incl_width),
                    percent = format_args!("{:>width$}", "-", width = percent_width),
                    global = format_args!("{:>width$}", "-", width = global_percent_width),
                ));
            }
            return output;
        }

        for row in rows {
            let display_name = display_name(row);
            let per = format!("{:.3}", row.per_ms);
            let excl = format!("{:.3}", row.excl_ms);
            let incl = format!("{:.3}", row.incl_ms);
            let percent = format!("{:.2}", row.percent);
            let global = format!("{:.2}", row.percent_global);
            if let Some(units) = units {
                let calls_unit = format!("{:.3}", (row.calls as f64) / units);
                let ms_unit = format!("{:.3}", row.excl_ms / units);
                output.push_str(&format!(
                    "| {name} | {calls} | {per} | {excl} | {incl} | {calls_unit} | {ms_unit} | {percent} | {global} |\n",
                    name = format_args!("{:<width$}", display_name, width = name_width),
                    calls = format_args!("{:>width$}", row.calls, width = calls_width),
                    per = format_args!("{:>width$}", per, width = per_width),
                    excl = format_args!("{:>width$}", excl, width = excl_width),
                    incl = format_args!("{:>width$}", incl, width = incl_width),
                    calls_unit = format_args!("{:>width$}", calls_unit, width = calls_per_unit_width),
                    ms_unit = format_args!("{:>width$}", ms_unit, width = ms_per_unit_width),
                    percent = format_args!("{:>width$}", percent, width = percent_width),
                    global = format_args!("{:>width$}", global, width = global_percent_width),
                ));
            } else {
                output.push_str(&format!(
                    "| {name} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
                    name = format_args!("{:<width$}", display_name, width = name_width),
                    calls = format_args!("{:>width$}", row.calls, width = calls_width),
                    per = format_args!("{:>width$}", per, width = per_width),
                    excl = format_args!("{:>width$}", excl, width = excl_width),
                    incl = format_args!("{:>width$}", incl, width = incl_width),
                    percent = format_args!("{:>width$}", percent, width = percent_width),
                    global = format_args!("{:>width$}", global, width = global_percent_width),
                ));
            }
        }
        output
    }

    fn format_cache_table(label: &str, column: &str, rows: &[TableRow]) -> String {
        let mut output = String::new();
        output.push_str(label);
        output.push('\n');

        let name_width = rows
            .iter()
            .map(|row| row.name.len())
            .max()
            .unwrap_or(column.len())
            .max(column.len());

        let mut calls_width = "#".len();
        let mut percent_width = "%".len();
        for row in rows {
            calls_width = calls_width.max(format!("{}", row.calls).len());
            percent_width = percent_width.max(format!("{:.2}", row.percent).len());
        }

        let header = format!(
            "| {column} | {calls} | {percent} |\n",
            column = format_args!("{:<width$}", column, width = name_width),
            calls = format_args!("{:^width$}", "#", width = calls_width),
            percent = format_args!("{:^width$}", "%", width = percent_width),
        );
        output.push_str(&header);
        let separator = format!(
            "|-{name}-|-{calls}-|-{percent}-|\n",
            name = "-".repeat(name_width),
            calls = "-".repeat(calls_width),
            percent = "-".repeat(percent_width),
        );
        output.push_str(&separator);

        if rows.is_empty() {
            let empty_row = format!(
                "| {value} | {calls} | {percent} |\n",
                value = format_args!("{:<width$}", "(no data)", width = name_width),
                calls = format_args!("{:>width$}", "-", width = calls_width),
                percent = format_args!("{:>width$}", "-", width = percent_width),
            );
            output.push_str(&empty_row);
            return output;
        }

        for row in rows {
            let percent = format!("{:.2}", row.percent);
            let line = format!(
                "| {name} | {calls} | {percent} |\n",
                name = format_args!("{:<width$}", row.name, width = name_width),
                calls = format_args!("{:>width$}", row.calls, width = calls_width),
                percent = format_args!("{:>width$}", percent, width = percent_width),
            );
            output.push_str(&line);
        }

        output
    }

    let (backend_rows, backend_compact_suffix) =
        maybe_compact_backend_rows(&tables.backend, options);
    let backend_label = match backend_compact_suffix {
        Some(suffix) => format!("Backend Ops {suffix}"),
        None => "Backend Ops".to_string(),
    };

    let mut out = String::new();
    out.push_str(&format_execution_kpis(tables, options));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_table("Layers", "layer", &tables.layers, options));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_table(
        "Functionals",
        "functional",
        &tables.functionals,
        options,
    ));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_table(
        backend_label.as_str(),
        "backend",
        backend_rows.as_slice(),
        options,
    ));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_table(
        "Compilation",
        "compile",
        &tables.compilation,
        options,
    ));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_table(
        "Optimizer Passes",
        "pass",
        &tables.compile_passes,
        options,
    ));
    out.push('\n');
    out.push('\n');
    out.push_str(&format_cache_table("Cache Stats", "cache", &tables.caches));
    out
}

#[cfg(feature = "profiler")]
fn format_report_with_options(report: &ProfilerReport, options: &ProfileFormatOptions) -> String {
    if report.sections.len() == 1 {
        return format_tables_with_options(&report.sections[0].tables, options);
    }

    let mut out = String::new();
    for (idx, section) in report.sections.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!("Section: {}\n\n", section.section));
        out.push_str(&format_tables_with_options(&section.tables, options));
    }
    out
}

#[cfg(feature = "profiler")]
fn kpi_rows_for_report(
    report: &ProfilerReport,
    options: &ProfileFormatOptions,
) -> Vec<(String, f64)> {
    let mut layers_ms = 0.0f64;
    let mut functionals_ms = 0.0f64;
    let mut backend_ms = 0.0f64;
    let mut compilation_ms = 0.0f64;
    let mut compile_passes_ms = 0.0f64;
    let mut backend_calls = 0u64;
    let mut cache_events = 0u64;

    for section in &report.sections {
        layers_ms += section
            .tables
            .layers
            .iter()
            .map(|row| row.excl_ms)
            .sum::<f64>();
        functionals_ms += section
            .tables
            .functionals
            .iter()
            .map(|row| row.excl_ms)
            .sum::<f64>();
        backend_ms += section
            .tables
            .backend
            .iter()
            .map(|row| row.excl_ms)
            .sum::<f64>();
        compilation_ms += section
            .tables
            .compilation
            .iter()
            .map(|row| row.excl_ms)
            .sum::<f64>();
        compile_passes_ms += section
            .tables
            .compile_passes
            .iter()
            .map(|row| row.excl_ms)
            .sum::<f64>();
        backend_calls += section
            .tables
            .backend
            .iter()
            .map(|row| row.calls)
            .sum::<u64>();
        cache_events += section
            .tables
            .caches
            .iter()
            .map(|row| row.calls)
            .sum::<u64>();
    }

    let compile_ms = compilation_ms + compile_passes_ms;
    let measured_ms = layers_ms + functionals_ms + backend_ms + compile_ms;
    let mut rows = vec![
        ("measured_ms.total".to_string(), measured_ms),
        ("layers_ms.total".to_string(), layers_ms),
        ("functionals_ms.total".to_string(), functionals_ms),
        ("backend_ms.total".to_string(), backend_ms),
        ("compile_ms.total".to_string(), compile_ms),
        ("backend_calls.total".to_string(), backend_calls as f64),
        ("cache_events.total".to_string(), cache_events as f64),
    ];
    if let Some(units) = positive_measured_units(options) {
        let unit_label = options.unit_label.as_deref().unwrap_or("unit");
        rows.push((format!("measured_ms/{unit_label}"), measured_ms / units));
        rows.push((format!("backend_ms/{unit_label}"), backend_ms / units));
        rows.push((format!("compile_ms/{unit_label}"), compile_ms / units));
        rows.push((
            format!("backend_calls/{unit_label}"),
            (backend_calls as f64) / units,
        ));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

#[cfg(feature = "profiler")]
fn section_rows_sorted(rows: &[TableRow]) -> Vec<TableRow> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| a.signature.cmp(&b.signature))
            .then_with(|| a.calls.cmp(&b.calls))
    });
    sorted
}

#[cfg(feature = "profiler")]
fn cache_miss_reason_from_row_name(name: &str) -> Option<(&'static str, &str)> {
    if let Some(reason) = name.strip_prefix("plan_cache_miss_reason.") {
        return Some(("plan_cache", reason));
    }
    if let Some(reason) = name.strip_prefix("program_cache_miss_reason.") {
        return Some(("program_cache", reason));
    }
    None
}

#[cfg(feature = "profiler")]
pub fn report_to_jsonl(report: &ProfilerReport, options: &ProfileFormatOptions) -> String {
    let mut lines = Vec::new();
    lines.push(serde_json::json!({
        "type": "meta",
        "schema": "gpt-rs.profile.v1",
        "unit_label": options.unit_label.as_deref(),
        "measured_units": options.measured_units,
        "profile_full": options.profile_full
    }));

    for (name, value) in kpi_rows_for_report(report, options) {
        lines.push(serde_json::json!({
            "type": "kpi",
            "name": name,
            "value": format!("{value:.6}")
        }));
    }

    for section in &report.sections {
        let tables = [
            (
                "layers",
                section_rows_sorted(section.tables.layers.as_slice()),
            ),
            (
                "functionals",
                section_rows_sorted(section.tables.functionals.as_slice()),
            ),
            (
                "backend",
                section_rows_sorted(section.tables.backend.as_slice()),
            ),
            (
                "compilation",
                section_rows_sorted(section.tables.compilation.as_slice()),
            ),
            (
                "compile_passes",
                section_rows_sorted(section.tables.compile_passes.as_slice()),
            ),
            (
                "caches",
                section_rows_sorted(section.tables.caches.as_slice()),
            ),
        ];
        for (table_name, rows) in tables {
            for row in rows {
                let units = positive_measured_units(options);
                let calls_per_unit = units.map(|unit| (row.calls as f64) / unit);
                let ms_per_unit = units.map(|unit| row.excl_ms / unit);
                lines.push(serde_json::json!({
                    "type": "row",
                    "section": section.section,
                    "table": table_name,
                    "name": row.name,
                    "signature": row.signature,
                    "calls": row.calls,
                    "ms_per_call": format!("{:.6}", row.per_ms),
                    "self_ms": format!("{:.6}", row.excl_ms),
                    "total_ms": format!("{:.6}", row.incl_ms),
                    "pct_section": format!("{:.6}", row.percent),
                    "pct_global": format!("{:.6}", row.percent_global),
                    "calls_per_unit": calls_per_unit.map(|v| format!("{v:.6}")),
                    "ms_per_unit": ms_per_unit.map(|v| format!("{v:.6}")),
                }));
                if table_name == "caches" {
                    if let Some((cache, reason)) = cache_miss_reason_from_row_name(&row.name) {
                        lines.push(serde_json::json!({
                            "type": "cache_miss_reason",
                            "section": section.section,
                            "cache": cache,
                            "reason": reason,
                            "count": row.calls
                        }));
                    }
                }
            }
        }
    }

    let mut out = String::new();
    for line in lines {
        if let Ok(serialized) = serde_json::to_string(&line) {
            out.push_str(serialized.as_str());
            out.push('\n');
        }
    }
    out
}

#[cfg(feature = "profiler")]
pub fn take_profile_snapshot_with_options(
    pretty_report_json: bool,
    options: &ProfileFormatOptions,
) -> Option<ProfileSnapshot> {
    let report = take_report()?;
    let formatted = format_report_with_options(&report, options);
    let report_json = if pretty_report_json {
        serde_json::to_string_pretty(&report).ok()?
    } else {
        serde_json::to_string(&report).ok()?
    };
    let profile_jsonl = report_to_jsonl(&report, options);
    Some(ProfileSnapshot {
        formatted,
        report_json,
        profile_jsonl,
    })
}

#[cfg(not(feature = "profiler"))]
pub fn take_profile_snapshot_with_options(
    _pretty_report_json: bool,
    _options: &ProfileFormatOptions,
) -> Option<ProfileSnapshot> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables() -> Option<String> {
    take_formatted_tables_with_options(&ProfileFormatOptions::default())
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables_with_options(options: &ProfileFormatOptions) -> Option<String> {
    take_report().map(|report| format_report_with_options(&report, options))
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables_and_report_json(pretty: bool) -> Option<(String, String)> {
    take_formatted_tables_and_report_json_with_options(pretty, &ProfileFormatOptions::default())
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables_and_report_json_with_options(
    pretty: bool,
    options: &ProfileFormatOptions,
) -> Option<(String, String)> {
    let snapshot = take_profile_snapshot_with_options(pretty, options)?;
    Some((snapshot.formatted, snapshot.report_json))
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables_and_report_json(_pretty: bool) -> Option<(String, String)> {
    None
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables_with_options(_options: &ProfileFormatOptions) -> Option<String> {
    None
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables_and_report_json_with_options(
    _pretty: bool,
    _options: &ProfileFormatOptions,
) -> Option<(String, String)> {
    None
}

#[cfg(feature = "profiler")]
pub fn reset() {
    Profiler::instance().reset();
}

#[cfg(not(feature = "profiler"))]
pub fn reset() {}

#[cfg(feature = "profiler")]
pub fn write_tables<W: fmt::Write>(writer: &mut W) -> fmt::Result {
    if let Some(formatted) = take_formatted_tables() {
        writer.write_str(&formatted)
    } else {
        Ok(())
    }
}

#[cfg(not(feature = "profiler"))]
pub fn write_tables<W: fmt::Write>(_writer: &mut W) -> fmt::Result {
    Ok(())
}
