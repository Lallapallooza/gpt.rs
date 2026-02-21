use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process;
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;

use gpt_rs::backend::optimizer::{OptimizeContext, Optimizer, PassResult};
use gpt_rs::backend::spec::{Function, PortableBackend};
use gpt_rs::ops::graph::{context::with_default_arena, CachePolicy, GraphArena};

use super::common;

const BASELINE_CACHE_CAPACITY: usize = 64;

#[derive(Default, Clone, Copy)]
struct RunDurations {
    torch_ms: f64,
    gpt_ms: f64,
}

thread_local! {
    static RUN_CONTEXT: RefCell<RunDurations> = RefCell::new(RunDurations::default());
}

#[derive(Clone)]
struct TableRow {
    backend: String,
    test_name: String,
    baseline_ms: f64,
    optimized_ms: f64,
    torch_ms: f64,
    status: String,
}

static TABLE_ROWS: OnceLock<Mutex<Vec<TableRow>>> = OnceLock::new();
static TABLE_PRINTER: Once = Once::new();
static TABLE_AGGREGATOR: OnceLock<TableAggregator> = OnceLock::new();
static C_BACKEND_CACHE_CONFIG: Once = Once::new();

fn table_rows() -> &'static Mutex<Vec<TableRow>> {
    TABLE_ROWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn table_aggregator() -> &'static TableAggregator {
    TABLE_AGGREGATOR.get_or_init(TableAggregator::new)
}

fn configure_c_backend_cache_for_parity(backend_name: &str) {
    if backend_name != "c" {
        return;
    }
    if env::var_os("GPTRS_C_CACHE_DIR").is_some() {
        return;
    }
    C_BACKEND_CACHE_CONFIG.call_once(|| {
        let pid = process::id();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let cache_dir = env::temp_dir()
            .join("gpt_rs_c_backend")
            .join("torch_parity")
            .join(format!("run-{pid}-{nonce}"));
        let _ = fs::create_dir_all(&cache_dir);
        env::set_var("GPTRS_C_CACHE_DIR", cache_dir);
    });
}

fn register_table_printer() {
    TABLE_PRINTER.call_once(|| unsafe {
        let _ = table_aggregator();
        libc::atexit(dump_table as extern "C" fn());
    });
}

extern "C" fn dump_table() {
    let _ = std::panic::catch_unwind(|| {
        let rows_lock = match TABLE_ROWS.get() {
            Some(rows) => rows,
            None => return,
        };
        let mut rows = match rows_lock.lock() {
            Ok(rows) => rows,
            Err(poisoned) => poisoned.into_inner(),
        };
        if rows.is_empty() {
            return;
        }
        let rows_snapshot = std::mem::take(&mut *rows);
        let aggregator = table_aggregator();
        let _ = aggregator.write_rows(&rows_snapshot);
        if let Some(all_rows) = aggregator.try_collect_for_print() {
            print_table(&all_rows);
            let _ = aggregator.cleanup();
        }
    });
}

fn print_table(rows: &[TableRow]) {
    struct PivotRow {
        torch_ms: f64,
        status: String,
        cells: HashMap<String, (f64, f64, String)>,
    }

    let mut backends_set: HashSet<String> = HashSet::new();
    for row in rows {
        backends_set.insert(row.backend.clone());
    }
    for backend in expected_backends() {
        backends_set.insert(backend);
    }
    let mut backends: Vec<String> = backends_set.into_iter().collect();
    backends.sort();

    let mut by_test: BTreeMap<String, PivotRow> = BTreeMap::new();
    for row in rows {
        let entry = by_test
            .entry(row.test_name.clone())
            .or_insert_with(|| PivotRow {
                torch_ms: f64::NAN,
                status: "ok".to_string(),
                cells: HashMap::new(),
            });
        if row.torch_ms.is_finite() {
            entry.torch_ms = row.torch_ms;
        }
        entry.cells.insert(
            row.backend.clone(),
            (row.baseline_ms, row.optimized_ms, row.status.clone()),
        );
    }

    for entry in by_test.values_mut() {
        let mut failed = false;
        for backend in &backends {
            match entry.cells.get(backend) {
                Some((baseline, optimized, status)) => {
                    if !baseline.is_finite() || !optimized.is_finite() || status != "ok" {
                        failed = true;
                    }
                }
                None => {
                    failed = true;
                }
            }
        }
        entry.status = if failed {
            "fail".to_string()
        } else {
            "ok".to_string()
        };
    }

    #[derive(Clone, Copy)]
    enum Align {
        Left,
        Right,
        Center,
    }

    struct Col {
        header: String,
        width: usize,
        align: Align,
    }

    fn format_cell(value: &str, width: usize, align: Align) -> String {
        match align {
            Align::Left => format!("{value:<width$}"),
            Align::Right => format!("{value:>width$}"),
            Align::Center => {
                let pad = width.saturating_sub(value.len());
                let left = pad / 2;
                let right = pad - left;
                format!("{}{}{}", " ".repeat(left), value, " ".repeat(right))
            }
        }
    }

    fn border(widths: &[usize]) -> String {
        let mut line = String::new();
        line.push('+');
        for width in widths {
            line.push_str(&"-".repeat(width + 2));
            line.push('+');
        }
        line
    }

    let test_width = by_test
        .keys()
        .map(|name| name.len())
        .max()
        .unwrap_or(4)
        .max("test".len());
    let torch_width = rows
        .iter()
        .map(|row| format_ms(row.torch_ms).len())
        .max()
        .unwrap_or(5)
        .max("torch".len());
    let status_width = rows
        .iter()
        .map(|row| row.status.len())
        .max()
        .unwrap_or(2)
        .max("status".len());

    let mut cols = Vec::new();
    cols.push(Col {
        header: "test".to_string(),
        width: test_width,
        align: Align::Left,
    });
    cols.push(Col {
        header: "torch".to_string(),
        width: torch_width,
        align: Align::Right,
    });
    cols.push(Col {
        header: "status".to_string(),
        width: status_width,
        align: Align::Center,
    });

    let mut backend_cols: Vec<(String, usize, String, usize)> = Vec::new();
    for backend in &backends {
        let opt_name = format!("{backend}-opt");
        let mut base_w = backend.len().max(3);
        let mut opt_w = opt_name.len().max(3);
        for entry in by_test.values() {
            let (baseline, optimized) = entry
                .cells
                .get(backend)
                .map(|(b, o, _)| (*b, *o))
                .unwrap_or((f64::NAN, f64::NAN));
            base_w = base_w.max(format_ms(baseline).len());
            opt_w = opt_w.max(format_ms(optimized).len());
        }
        backend_cols.push((backend.clone(), base_w, opt_name, opt_w));
    }

    for (backend, base_w, opt_name, opt_w) in &backend_cols {
        cols.push(Col {
            header: backend.clone(),
            width: *base_w,
            align: Align::Right,
        });
        cols.push(Col {
            header: opt_name.clone(),
            width: *opt_w,
            align: Align::Right,
        });
    }

    let widths: Vec<usize> = cols.iter().map(|col| col.width).collect();
    let header = cols
        .iter()
        .map(|col| format_cell(&col.header, col.width, col.align))
        .collect::<Vec<_>>()
        .join(" | ");

    println!("{}", border(&widths));
    println!("| {} |", header);
    println!("{}", border(&widths));

    for (test_name, entry) in by_test {
        let mut cells = Vec::new();
        cells.push(format_cell(&test_name, test_width, Align::Left));
        cells.push(format_cell(
            &format_ms(entry.torch_ms),
            torch_width,
            Align::Right,
        ));
        cells.push(format_cell(&entry.status, status_width, Align::Center));
        for (backend, base_w, _opt_name, opt_w) in &backend_cols {
            let (baseline, optimized) = entry
                .cells
                .get(backend)
                .map(|(b, o, _)| (*b, *o))
                .unwrap_or((f64::NAN, f64::NAN));
            cells.push(format_cell(&format_ms(baseline), *base_w, Align::Right));
            cells.push(format_cell(&format_ms(optimized), *opt_w, Align::Right));
        }
        println!("| {} |", cells.join(" | "));
    }
    println!("{}", border(&widths));
}

struct TableAggregator {
    root: PathBuf,
    rows_dir: PathBuf,
    print_lock_path: PathBuf,
    printed_path: PathBuf,
    pid: u32,
}

impl TableAggregator {
    fn new() -> Self {
        let target_dir = env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("..")
                    .join("..")
                    .join("target")
            });
        let root = target_dir.join("torch_parity");
        let rows_dir = root.join("rows");
        let print_lock_path = root.join("print.lock");
        let printed_path = root.join("printed");
        let _ = fs::create_dir_all(&rows_dir);
        Self {
            root,
            rows_dir,
            print_lock_path,
            printed_path,
            pid: process::id(),
        }
    }

    fn write_rows(&self, rows: &[TableRow]) -> std::io::Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let path = self.rows_dir.join(format!("rows-{}.tsv", self.pid));
        let mut file = File::create(path)?;
        for row in rows {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}",
                row.backend,
                row.test_name,
                row.baseline_ms,
                row.optimized_ms,
                row.torch_ms,
                row.status
            )?;
        }
        Ok(())
    }

    fn read_all_rows(&self) -> Vec<TableRow> {
        let mut rows = Vec::new();
        let entries = match fs::read_dir(&self.rows_dir) {
            Ok(entries) => entries,
            Err(_) => return rows,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let contents = match fs::read_to_string(&path) {
                Ok(contents) => contents,
                Err(_) => continue,
            };
            for line in contents.lines() {
                if let Some(row) = parse_row_line(line) {
                    rows.push(row);
                }
            }
        }
        rows
    }

    fn try_collect_for_print(&self) -> Option<Vec<TableRow>> {
        let _lock = self.lock_print()?;
        if self.printed_path.exists() {
            return None;
        }
        let rows = self.read_all_rows();
        if rows.is_empty() {
            return None;
        }
        if !expected_backends_present(&rows) {
            return None;
        }
        let _ = File::create(&self.printed_path);
        Some(rows)
    }

    fn cleanup(&self) -> std::io::Result<()> {
        let _ = fs::remove_file(&self.printed_path);
        let _ = fs::remove_file(&self.print_lock_path);
        if let Ok(entries) = fs::read_dir(&self.rows_dir) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let _ = fs::remove_dir(&self.rows_dir);
        let _ = fs::remove_dir(&self.root);
        Ok(())
    }

    fn lock_print(&self) -> Option<File> {
        #[cfg(unix)]
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&self.print_lock_path)
                .ok()?;
            let fd = file.as_raw_fd();
            unsafe {
                libc::flock(fd, libc::LOCK_EX);
            }
            Some(file)
        }
        #[cfg(not(unix))]
        {
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&self.print_lock_path)
                .ok()
        }
    }
}

fn parse_row_line(line: &str) -> Option<TableRow> {
    let mut parts = line.splitn(6, '\t');
    let backend = parts.next()?.to_string();
    let test_name = parts.next()?.to_string();
    let baseline_ms = parts.next()?.parse::<f64>().ok()?;
    let optimized_ms = parts.next()?.parse::<f64>().ok()?;
    let torch_ms = parts.next()?.parse::<f64>().ok()?;
    let status = parts.next()?.to_string();
    Some(TableRow {
        backend,
        test_name,
        baseline_ms,
        optimized_ms,
        torch_ms,
        status,
    })
}

fn expected_backends() -> Vec<String> {
    if let Ok(value) = env::var("GPTRS_TORCH_PARITY_BACKENDS") {
        let backends: Vec<String> = value
            .split(',')
            .map(|entry| entry.trim())
            .filter(|entry| !entry.is_empty())
            .map(|entry| entry.to_string())
            .collect();
        if !backends.is_empty() {
            return backends;
        }
    }
    vec![
        "c".to_string(),
        "cpu-portable".to_string(),
        "faer".to_string(),
    ]
}

fn expected_backends_present(rows: &[TableRow]) -> bool {
    let expected = expected_backends();
    let mut seen = HashSet::new();
    for row in rows {
        seen.insert(row.backend.as_str());
    }
    expected
        .iter()
        .all(|backend| seen.contains(backend.as_str()))
}
#[derive(Clone, Copy)]
enum BenchMode {
    Baseline,
    Optimized,
}

struct NoopOptimizer;

impl<B: PortableBackend + 'static> Optimizer<B> for NoopOptimizer {
    fn optimize(&self, _function: &mut Function, _cx: &mut OptimizeContext<B>) -> PassResult {
        PassResult::default()
    }
}

fn arena_for_mode<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    mode: BenchMode,
) -> Arc<GraphArena<B>> {
    match mode {
        BenchMode::Baseline => {
            let optimizer: Arc<dyn Optimizer<B>> = Arc::new(NoopOptimizer);
            GraphArena::with_policy(
                Arc::clone(backend),
                CachePolicy::LazyPrograms {
                    capacity: BASELINE_CACHE_CAPACITY,
                    optimizer: Some(optimizer),
                },
            )
        }
        BenchMode::Optimized => GraphArena::new(Arc::clone(backend)),
    }
}

fn reset_durations() {
    RUN_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = RunDurations::default();
    });
}

fn take_durations() -> RunDurations {
    RUN_CONTEXT.with(|ctx| *ctx.borrow())
}

pub fn timed_torch<T, F: FnOnce() -> T>(f: F) -> T {
    let start = Instant::now();
    let output = f();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    RUN_CONTEXT.with(|ctx| {
        ctx.borrow_mut().torch_ms += elapsed;
    });
    output
}

pub fn timed_gpt<T, F: FnOnce() -> T>(f: F) -> T {
    gpt_rs::ops::graph::timing::reset_compile_time();
    let start = Instant::now();
    let output = f();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let compile = gpt_rs::ops::graph::timing::take_compile_time();
    let compile_ms = compile.as_secs_f64() * 1000.0;
    let exec_ms = (elapsed - compile_ms).max(0.0);
    RUN_CONTEXT.with(|ctx| {
        ctx.borrow_mut().gpt_ms += exec_ms;
    });
    output
}

fn run_mode<B: PortableBackend + 'static, F: Fn(&Arc<B>)>(
    backend: &Arc<B>,
    mode: BenchMode,
    f: &F,
) -> Result<(f64, RunDurations), Box<dyn std::any::Any + Send + 'static>> {
    let arena = arena_for_mode(backend, mode);
    reset_durations();
    let start = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(|| {
        with_default_arena(arena, || f(backend))
    }));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let durations = take_durations();
    match result {
        Ok(()) => Ok((elapsed_ms, durations)),
        Err(err) => Err(err),
    }
}

pub fn run_parity_test_with_modes<B: PortableBackend + 'static, F: Fn(&Arc<B>)>(
    backend: Arc<B>,
    test_name: &str,
    f: F,
) {
    let backend_name = backend.backend_name().to_string();
    if let Some(reason) = temporary_xfail_reason(&backend_name, test_name) {
        print_row(
            &backend_name,
            test_name,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            "xfail",
        );
        eprintln!("torch_parity xfail: backend={backend_name} test={test_name} reason={reason}");
        return;
    }
    configure_c_backend_cache_for_parity(&backend_name);
    let _context_guard = common::set_parity_context(&backend_name, test_name);
    let baseline = run_mode(&backend, BenchMode::Baseline, &f);
    match baseline {
        Ok((baseline_total_ms, baseline_durations)) => {
            let optimized = run_mode(&backend, BenchMode::Optimized, &f);
            match optimized {
                Ok((optimized_total_ms, optimized_durations)) => {
                    let baseline_ms = resolve_gpt_ms(baseline_total_ms, baseline_durations);
                    let optimized_ms = resolve_gpt_ms(optimized_total_ms, optimized_durations);
                    let torch_ms = resolve_torch_ms(baseline_durations, optimized_durations);
                    print_row(
                        &backend_name,
                        test_name,
                        baseline_ms,
                        optimized_ms,
                        torch_ms,
                        "ok",
                    );
                }
                Err(err) => {
                    print_row(
                        &backend_name,
                        test_name,
                        resolve_gpt_ms(baseline_total_ms, baseline_durations),
                        f64::NAN,
                        resolve_torch_ms(baseline_durations, RunDurations::default()),
                        "fail",
                    );
                    std::panic::resume_unwind(err);
                }
            }
        }
        Err(err) => {
            print_row(
                &backend_name,
                test_name,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                "fail",
            );
            std::panic::resume_unwind(err);
        }
    }
}

fn print_row(
    backend: &str,
    test_name: &str,
    baseline_ms: f64,
    optimized_ms: f64,
    torch_ms: f64,
    status: &str,
) {
    register_table_printer();
    let row = TableRow {
        backend: backend.to_string(),
        test_name: test_name.to_string(),
        baseline_ms,
        optimized_ms,
        torch_ms,
        status: status.to_string(),
    };
    let mut rows = table_rows()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    rows.push(row);
}

fn resolve_gpt_ms(total_ms: f64, durations: RunDurations) -> f64 {
    if durations.gpt_ms > 0.0 {
        durations.gpt_ms
    } else {
        total_ms
    }
}

fn resolve_torch_ms(baseline: RunDurations, optimized: RunDurations) -> f64 {
    if baseline.torch_ms > 0.0 {
        baseline.torch_ms
    } else if optimized.torch_ms > 0.0 {
        optimized.torch_ms
    } else {
        f64::NAN
    }
}

fn temporary_xfail_reason(backend_name: &str, test_name: &str) -> Option<&'static str> {
    if backend_name == "triton"
        && (test_name.starts_with("torch_vision_conv2d_")
            || test_name.starts_with("torch_vision_depthwise_conv2d_")
            || test_name.starts_with("torch_vision_group_conv2d_"))
    {
        return Some("temporary Triton xfail until generalized conv dot_general lowering lands");
    }
    None
}

fn format_ms(value: f64) -> String {
    if value.is_finite() {
        format!("{:.3}", value)
    } else {
        "n/a".to_string()
    }
}
