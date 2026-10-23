//! The profiler: scopes that time layers, functionals, backend kernels and compilation, with the
//! work they do, grouped by section. Without the `profiler` feature every scope is a no-op.
//!
//! [`take_snapshot`] reads the profile as tables, JSON and JSON lines. [`trace_enable`] also
//! records each scope as a Chrome trace event, which [`take_trace_json`] reads.

mod report;
mod trace;

#[cfg(feature = "profiler")]
use std::cell::Cell;
#[cfg(feature = "profiler")]
use std::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "profiler")]
use std::collections::{BTreeMap, BTreeSet};
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

pub use self::report::*;
pub use self::trace::*;

#[cfg_attr(not(feature = "profiler"), allow(dead_code))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ProfilerKey {
    Layer {
        section: u32,
        name: &'static str,
    },
    Functional {
        section: u32,
        name: &'static str,
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
            .map(String::as_str)
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

/// Whether the profiler collects scopes on this thread now: built with it, and not suspended.
#[inline(always)]
pub fn collecting() -> bool {
    #[cfg(feature = "profiler")]
    {
        !profiler_suspended()
    }
    #[cfg(not(feature = "profiler"))]
    {
        false
    }
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
            let _ = stack.borrow_mut().pop();
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
        ACTIVE_GUARDS.with(|guards| {
            let mut stack = guards.borrow_mut();
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

/// The scope of the functional `name`, which `#[functional]` opens around its body.
#[inline(always)]
pub fn functional_scope(name: &'static str) -> ScopeGuard {
    ScopeGuard::new(ProfilerKey::Functional {
        section: current_section_id(),
        name,
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

/// The profiler label `backend.<name>` of the op named `name`.
pub fn op_label(name: &str) -> &'static str {
    intern_label(&format!("backend.{name}"))
}

/// `label` as a profiler label, which lives as long as the process. Each distinct label is
/// allocated once.
pub fn intern_label(label: &str) -> &'static str {
    static LABELS: std::sync::LazyLock<Mutex<HashMap<String, &'static str>>> =
        std::sync::LazyLock::new(Default::default);
    let mut labels = LABELS.lock().expect("profiler labels poisoned");
    if let Some(interned) = labels.get(label) {
        return interned;
    }
    let interned: &'static str = Box::leak(label.to_string().into_boxed_str());
    labels.insert(label.to_string(), interned);
    interned
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
        let scope_meta = meta();
        ScopeGuard::new_with_work(
            ProfilerKey::Backend {
                section: current_section_id(),
                name,
                signature: scope_meta.signature,
            },
            scope_meta.work,
        )
    }
    #[cfg(not(feature = "profiler"))]
    {
        let _ = meta;
        backend_scope(name)
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
pub fn reset() {
    Profiler::instance().reset();
}

#[cfg(not(feature = "profiler"))]
pub fn reset() {}
