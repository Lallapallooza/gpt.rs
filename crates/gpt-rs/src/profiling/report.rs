//! The profile as tables, report JSON and JSON lines.

#[allow(unused_imports)]
use super::*;

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

/// A profile in its three forms: the tables as text, the report as JSON, and the rows as JSON
/// lines.
#[derive(Debug, Clone)]
pub struct ProfileSnapshot {
    pub formatted: String,
    pub report_json: String,
    pub profile_jsonl: String,
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
            ProfilerKey::Functional { name, .. } => {
                functionals.push((name.to_string(), None, stat))
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

    let total_layers_ns = total_exclusive_ns(&layers);
    let total_functionals_ns = total_exclusive_ns(&functionals);
    let total_backend_ns = total_exclusive_ns(&backend);
    let total_compilation_ns = total_exclusive_ns(&compilation);
    let total_compile_passes_ns = total_exclusive_ns(&compile_passes);
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

/// The exclusive time of the scopes of one table, in nanoseconds.
#[cfg(feature = "profiler")]
fn total_exclusive_ns(items: &[(String, Option<String>, Stat)]) -> f64 {
    items
        .iter()
        .map(|(_, _, stat)| stat.exclusive_ns as f64)
        .sum()
}

#[cfg(feature = "profiler")]
fn finalize_by_percent(rows: &mut [TableRow]) {
    rows.sort_by(|left, right| {
        right
            .percent
            .partial_cmp(&left.percent)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(feature = "profiler")]
fn finalize_by_calls(rows: &mut [TableRow]) {
    rows.sort_by(|left, right| {
        right
            .calls
            .cmp(&left.calls)
            .then_with(|| left.name.cmp(&right.name))
    });
}

#[cfg(feature = "profiler")]
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

#[cfg(feature = "profiler")]
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
fn positive_measured_units(options: &ProfileFormatOptions) -> Option<f64> {
    options.measured_units.filter(|value| *value > 0.0)
}

#[cfg(feature = "profiler")]
fn format_execution_kpis(tables: &ProfilerTables, options: &ProfileFormatOptions) -> String {
    let mut rows = Vec::new();
    for (metric, value) in execution_kpi_rows(tables, options) {
        rows.push((metric, vec![value]));
    }
    let widths = column_widths("metric", &["value"], &rows);

    let mut output = String::new();
    output.push_str("Execution KPIs\n");
    output.push_str(&table_line(&[
        format!("{:<width$}", "metric", width = widths[0]),
        format!("{:<width$}", "value", width = widths[1]),
    ]));
    output.push_str(&separator_line(&widths));
    for (metric, values) in &rows {
        output.push_str(&row_line(metric, values, &widths));
    }
    output
}

/// The metrics of the execution KPI table and their values: the totals of `tables`, then the
/// totals per measured unit when the options count units.
#[cfg(feature = "profiler")]
fn execution_kpi_rows(
    tables: &ProfilerTables,
    options: &ProfileFormatOptions,
) -> Vec<(String, String)> {
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

    let mut rows = vec![
        (
            "measured_ms.total".to_string(),
            format!("{total_measured_ms:.3}"),
        ),
        (
            "layers_ms.total".to_string(),
            format!("{total_layers_ms:.3}"),
        ),
        (
            "functionals_ms.total".to_string(),
            format!("{total_functionals_ms:.3}"),
        ),
        (
            "backend_ms.total".to_string(),
            format!("{total_backend_ms:.3}"),
        ),
        (
            "compile_ms.total".to_string(),
            format!("{total_compile_ms:.3}"),
        ),
        (
            "backend_calls.total".to_string(),
            total_backend_calls.to_string(),
        ),
        (
            "cache_events.total".to_string(),
            total_cache_events.to_string(),
        ),
    ];
    if let Some(units) = positive_measured_units(options) {
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
    rows
}

/// A table line of padded cells: `| a | b | c |`.
#[cfg(feature = "profiler")]
fn table_line(cells: &[String]) -> String {
    format!("| {} |\n", cells.join(" | "))
}

/// The line under a table header, with columns of `widths`: `|---|---|`.
#[cfg(feature = "profiler")]
fn separator_line(widths: &[usize]) -> String {
    let mut dashes = Vec::with_capacity(widths.len());
    for &width in widths {
        dashes.push("-".repeat(width));
    }
    format!("|-{}-|\n", dashes.join("-|-"))
}

/// The header line of a table: the name column left-aligned and the other headers centered, at
/// `widths`.
#[cfg(feature = "profiler")]
fn header_line(column: &str, headers: &[&str], widths: &[usize]) -> String {
    let mut padded = vec![format!("{column:<width$}", width = widths[0])];
    for (header, &width) in headers.iter().zip(&widths[1..]) {
        padded.push(format!("{header:^width$}"));
    }
    table_line(&padded)
}

/// The line of one row: its name left-aligned and its cells right-aligned, at `widths`.
#[cfg(feature = "profiler")]
fn row_line(name: &str, cells: &[String], widths: &[usize]) -> String {
    let mut padded = vec![format!("{name:<width$}", width = widths[0])];
    for (cell, &width) in cells.iter().zip(&widths[1..]) {
        padded.push(format!("{cell:>width$}"));
    }
    table_line(&padded)
}

/// The widths of the columns of a table: the name column fits `column` and every name, and each
/// other column its header and every cell under it.
#[cfg(feature = "profiler")]
fn column_widths(column: &str, headers: &[&str], rows: &[(String, Vec<String>)]) -> Vec<usize> {
    let mut widths = vec![column.len()];
    for header in headers {
        widths.push(header.len());
    }
    for (name, cells) in rows {
        widths[0] = widths[0].max(name.len());
        for (width, cell) in widths[1..].iter_mut().zip(cells) {
            *width = (*width).max(cell.len());
        }
    }
    widths
}

/// A table under `label`: the header, its separator line, and one line per row of a name and its
/// cells, or a `(no data)` line when there are no rows.
#[cfg(feature = "profiler")]
fn format_rows(
    label: &str,
    column: &str,
    headers: &[&str],
    rows: &[(String, Vec<String>)],
) -> String {
    let widths = column_widths(column, headers, rows);
    let mut output = String::new();
    output.push_str(label);
    output.push('\n');
    output.push_str(&header_line(column, headers, &widths));
    output.push_str(&separator_line(&widths));
    if rows.is_empty() {
        let dashes = vec!["-".to_string(); headers.len()];
        output.push_str(&row_line("(no data)", &dashes, &widths));
    }
    for (name, cells) in rows {
        output.push_str(&row_line(name, cells, &widths));
    }
    output
}

/// The headers of the timing columns after the name: calls, ms per call, self and total ms, calls
/// and ms per unit when `units` is given, and the two percentages.
#[cfg(feature = "profiler")]
fn timing_headers(units: Option<f64>) -> Vec<&'static str> {
    let per_unit: &[&'static str] = if units.is_some() {
        &["calls/unit", "ms/unit"]
    } else {
        &[]
    };
    [
        &["#", "ms/call", "self_ms", "total_ms"][..],
        per_unit,
        &["%tbl", "%all"],
    ]
    .concat()
}

/// The timing cells of `row`, under [`timing_headers`].
#[cfg(feature = "profiler")]
fn timing_cells(row: &TableRow, units: Option<f64>) -> Vec<String> {
    let mut cells = vec![
        row.calls.to_string(),
        format!("{:.3}", row.per_ms),
        format!("{:.3}", row.excl_ms),
        format!("{:.3}", row.incl_ms),
    ];
    if let Some(unit_count) = units {
        cells.push(format!("{:.3}", (row.calls as f64) / unit_count));
        cells.push(format!("{:.3}", row.excl_ms / unit_count));
    }
    cells.push(format!("{:.2}", row.percent));
    cells.push(format!("{:.2}", row.percent_global));
    cells
}

/// The name of `row`, with its signature in brackets when it has one.
#[cfg(feature = "profiler")]
fn display_name(row: &TableRow) -> String {
    match row.signature.as_ref() {
        None => row.name.clone(),
        Some(sig) => format!("{} [{sig}]", row.name),
    }
}

/// A timing table: per row its calls, times and shares.
#[cfg(feature = "profiler")]
fn format_table(
    label: &str,
    column: &str,
    rows: &[TableRow],
    options: &ProfileFormatOptions,
) -> String {
    let units = positive_measured_units(options);
    let mut cells = Vec::with_capacity(rows.len());
    for row in rows {
        cells.push((display_name(row), timing_cells(row, units)));
    }
    format_rows(label, column, &timing_headers(units), &cells)
}

/// The cache table: per cache event its count and its share of all events.
#[cfg(feature = "profiler")]
fn format_cache_table(label: &str, column: &str, rows: &[TableRow]) -> String {
    let mut cells = Vec::with_capacity(rows.len());
    for row in rows {
        let counts = vec![row.calls.to_string(), format!("{:.2}", row.percent)];
        cells.push((row.name.clone(), counts));
    }
    format_rows(label, column, &["#", "%"], &cells)
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
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    rows
}

#[cfg(feature = "profiler")]
fn section_rows_sorted(rows: &[TableRow]) -> Vec<TableRow> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.signature.cmp(&right.signature))
            .then_with(|| left.calls.cmp(&right.calls))
    });
    sorted
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
                    "calls_per_unit": calls_per_unit.map(|value| format!("{value:.6}")),
                    "ms_per_unit": ms_per_unit.map(|value| format!("{value:.6}")),
                }));
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

/// Takes the recorded profile, which resets it, in its three forms. The report JSON is indented
/// when `pretty_json` is true. `None` without a recorded profile or without the `profiler` feature.
#[cfg(feature = "profiler")]
pub fn take_snapshot(pretty_json: bool, options: &ProfileFormatOptions) -> Option<ProfileSnapshot> {
    let report = take_report()?;
    let report_json = if pretty_json {
        serde_json::to_string_pretty(&report).ok()?
    } else {
        serde_json::to_string(&report).ok()?
    };
    Some(ProfileSnapshot {
        formatted: format_report_with_options(&report, options),
        report_json,
        profile_jsonl: report_to_jsonl(&report, options),
    })
}

#[cfg(not(feature = "profiler"))]
pub fn take_snapshot(
    _pretty_json: bool,
    _options: &ProfileFormatOptions,
) -> Option<ProfileSnapshot> {
    None
}
