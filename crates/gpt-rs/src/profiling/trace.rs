//! The Chrome trace of the profiler scopes, and its conversion to folded stacks and speedscope.

#[allow(unused_imports)]
use super::*;

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
pub(super) struct TraceRecorder {
    enabled: AtomicBool,
    epoch: RwLock<Instant>,
    events: Mutex<Vec<ChromeTraceEvent>>,
}

#[cfg(feature = "profiler")]
impl TraceRecorder {
    pub(super) fn instance() -> &'static Self {
        static INSTANCE: OnceLock<TraceRecorder> = OnceLock::new();
        INSTANCE.get_or_init(|| TraceRecorder {
            enabled: AtomicBool::new(false),
            epoch: RwLock::new(Instant::now()),
            events: Mutex::new(Vec::new()),
        })
    }

    pub(super) fn enabled(&self) -> bool {
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

    pub(super) fn record(
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
            ProfilerKey::Functional { section, name } => {
                ("functional", name.to_string(), section, None)
            }
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

/// Takes the recorded Chrome trace, which resets it, as JSON indented when `pretty` is true.
/// `None` without recorded events or without the `profiler` feature.
#[cfg(feature = "profiler")]
pub fn take_trace_json(pretty: bool) -> Option<String> {
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
    if pretty {
        serde_json::to_string_pretty(&trace).ok()
    } else {
        serde_json::to_string(&trace).ok()
    }
}

#[cfg(not(feature = "profiler"))]
pub fn take_trace_json(_pretty: bool) -> Option<String> {
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
