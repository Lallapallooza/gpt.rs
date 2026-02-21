#[cfg(feature = "profiler")]
use std::cell::Cell;
#[cfg(feature = "profiler")]
use std::cell::RefCell;
use std::collections::HashMap;
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
use serde::Serialize;

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
pub fn format_tables(tables: &ProfilerTables) -> String {
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

    fn format_table(label: &str, column: &str, rows: &[TableRow]) -> String {
        let mut output = String::new();
        output.push_str(label);
        output.push('\n');

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

        for row in rows {
            calls_width = calls_width.max(format!("{}", row.calls).len());
            per_width = per_width.max(format!("{:.3}", row.per_ms).len());
            excl_width = excl_width.max(format!("{:.3}", row.excl_ms).len());
            incl_width = incl_width.max(format!("{:.3}", row.incl_ms).len());
            percent_width = percent_width.max(format!("{:.2}", row.percent).len());
            global_percent_width =
                global_percent_width.max(format!("{:.2}", row.percent_global).len());
        }

        let header = format!(
            "| {column} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
            column = format_args!("{:<width$}", column, width = name_width),
            calls = format_args!("{:^width$}", "#", width = calls_width),
            per = format_args!("{:^width$}", "ms/call", width = per_width),
            excl = format_args!("{:^width$}", "self_ms", width = excl_width),
            incl = format_args!("{:^width$}", "total_ms", width = incl_width),
            percent = format_args!("{:^width$}", "%tbl", width = percent_width),
            global = format_args!("{:^width$}", "%all", width = global_percent_width),
        );
        output.push_str(&header);
        let separator = format!(
            "|-{name}-|-{calls}-|-{per}-|-{excl}-|-{incl}-|-{percent}-|-{global}-|\n",
            name = "-".repeat(name_width),
            calls = "-".repeat(calls_width),
            per = "-".repeat(per_width),
            excl = "-".repeat(excl_width),
            incl = "-".repeat(incl_width),
            percent = "-".repeat(percent_width),
            global = "-".repeat(global_percent_width),
        );
        output.push_str(&separator);

        if rows.is_empty() {
            let empty_row = format!(
                "| {value} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
                value = format_args!("{:<width$}", "(no data)", width = name_width),
                calls = format_args!("{:>width$}", "-", width = calls_width),
                per = format_args!("{:>width$}", "-", width = per_width),
                excl = format_args!("{:>width$}", "-", width = excl_width),
                incl = format_args!("{:>width$}", "-", width = incl_width),
                percent = format_args!("{:>width$}", "-", width = percent_width),
                global = format_args!("{:>width$}", "-", width = global_percent_width),
            );
            output.push_str(&empty_row);
            return output;
        }

        for row in rows {
            let per_call = format!("{:.3}", row.per_ms);
            let excl = format!("{:.3}", row.excl_ms);
            let incl = format!("{:.3}", row.incl_ms);
            let percent = format!("{:.2}", row.percent);
            let global_percent = format!("{:.2}", row.percent_global);
            let display_name = display_name(row);
            let line = format!(
                "| {name} | {calls} | {per} | {excl} | {incl} | {percent} | {global} |\n",
                name = format_args!("{:<width$}", display_name, width = name_width),
                calls = format_args!("{:>width$}", row.calls, width = calls_width),
                per = format_args!("{:>width$}", per_call, width = per_width),
                excl = format_args!("{:>width$}", excl, width = excl_width),
                incl = format_args!("{:>width$}", incl, width = incl_width),
                percent = format_args!("{:>width$}", percent, width = percent_width),
                global = format_args!("{:>width$}", global_percent, width = global_percent_width),
            );
            output.push_str(&line);
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

    let mut out = String::new();
    out.push_str(&format_table("Layers", "layer", &tables.layers));
    out.push('\n');
    out.push_str(&format_table(
        "Functionals",
        "functional",
        &tables.functionals,
    ));
    out.push('\n');
    out.push_str(&format_table("Backend Ops", "backend", &tables.backend));
    out.push('\n');
    out.push_str(&format_table("Compilation", "compile", &tables.compilation));
    out.push('\n');
    out.push_str(&format_table(
        "Optimizer Passes",
        "pass",
        &tables.compile_passes,
    ));
    out.push('\n');
    out.push_str(&format_cache_table("Cache Stats", "cache", &tables.caches));
    out
}

#[cfg(feature = "profiler")]
fn format_report(report: &ProfilerReport) -> String {
    if report.sections.len() == 1 {
        return format_tables(&report.sections[0].tables);
    }

    let mut out = String::new();
    for (idx, section) in report.sections.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!("Section: {}\n\n", section.section));
        out.push_str(&format_tables(&section.tables));
    }
    out
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables() -> Option<String> {
    take_report().map(|report| format_report(&report))
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables() -> Option<String> {
    None
}

#[cfg(feature = "profiler")]
pub fn take_formatted_tables_and_report_json(pretty: bool) -> Option<(String, String)> {
    let report = take_report()?;
    let formatted = format_report(&report);
    let json = if pretty {
        serde_json::to_string_pretty(&report).ok()?
    } else {
        serde_json::to_string(&report).ok()?
    };
    Some((formatted, json))
}

#[cfg(not(feature = "profiler"))]
pub fn take_formatted_tables_and_report_json(_pretty: bool) -> Option<(String, String)> {
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
