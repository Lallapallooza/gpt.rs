use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use pyo3::prelude::*;

use gpt_rs::ops::trace::{self, FileTraceOptions, FileTraceSink, ProgramDumpFilter};

static TRACE_GUARD: OnceLock<Mutex<Option<trace::TraceGuard>>> = OnceLock::new();

fn trace_guard_slot() -> &'static Mutex<Option<trace::TraceGuard>> {
    TRACE_GUARD.get_or_init(|| Mutex::new(None))
}

fn install_dump_sink(path: &Path, dump_mode: &str) -> Result<trace::TraceGuard> {
    let dump_filter = match dump_mode {
        "all" => ProgramDumpFilter::AllExecutions,
        "compile" => ProgramDumpFilter::CompileOnly,
        other => anyhow::bail!("unknown dump mode {other:?} (expected 'all' or 'compile')"),
    };

    let options = FileTraceOptions {
        dump_filter,
        ..Default::default()
    };

    let sink = FileTraceSink::with_options(path.to_path_buf(), options)
        .with_context(|| format!("failed to prepare PTIR dump directory {}", path.display()))?;
    Ok(trace::install_global_sink(std::sync::Arc::new(sink)))
}

#[pyfunction]
#[pyo3(signature = (path, dump_mode = "all"))]
pub fn set_dump_dir(path: String, dump_mode: &str) -> PyResult<()> {
    let path = PathBuf::from(path);
    let guard = install_dump_sink(&path, dump_mode).map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to enable dump dir: {err}"))
    })?;

    let mut slot = trace_guard_slot()
        .lock()
        .expect("trace guard mutex poisoned");
    *slot = Some(guard);
    Ok(())
}

#[pyfunction]
pub fn clear_dump_dir() -> PyResult<()> {
    let mut slot = trace_guard_slot()
        .lock()
        .expect("trace guard mutex poisoned");
    *slot = None;
    Ok(())
}

#[pyfunction]
pub fn profiling_reset() -> PyResult<()> {
    gpt_rs::profiling::reset();
    Ok(())
}

#[pyfunction]
pub fn profiling_take_report() -> PyResult<Option<String>> {
    Ok(gpt_rs::profiling::take_formatted_tables())
}

#[pyfunction]
pub fn profiling_push_section(name: String) -> PyResult<()> {
    gpt_rs::profiling::push_section(&name);
    Ok(())
}

#[pyfunction]
pub fn profiling_pop_section() -> PyResult<bool> {
    Ok(gpt_rs::profiling::pop_section())
}

#[pyfunction]
#[pyo3(signature = (pretty = true))]
pub fn profiling_take_report_bundle(pretty: bool) -> PyResult<Option<(String, String)>> {
    Ok(gpt_rs::profiling::take_formatted_tables_and_report_json(
        pretty,
    ))
}

#[pyfunction]
#[pyo3(signature = (pretty = false))]
pub fn profiling_take_report_json(pretty: bool) -> PyResult<Option<String>> {
    if pretty {
        Ok(gpt_rs::profiling::take_report_json_pretty())
    } else {
        Ok(gpt_rs::profiling::take_report_json())
    }
}

#[pyfunction]
pub fn profiling_trace_enable() -> PyResult<()> {
    gpt_rs::profiling::trace_enable();
    Ok(())
}

#[pyfunction]
pub fn profiling_trace_disable() -> PyResult<()> {
    gpt_rs::profiling::trace_disable();
    Ok(())
}

#[pyfunction]
pub fn profiling_trace_reset() -> PyResult<()> {
    gpt_rs::profiling::trace_reset();
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (pretty = false))]
pub fn profiling_take_trace_json(pretty: bool) -> PyResult<Option<String>> {
    if pretty {
        Ok(gpt_rs::profiling::take_trace_json_pretty())
    } else {
        Ok(gpt_rs::profiling::take_trace_json())
    }
}
