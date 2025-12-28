#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod backend;
mod debug;
mod runtime;
mod tokenizer;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<runtime::PyLoadedModel>()?;

    // Backend management
    m.add_function(wrap_pyfunction!(backend::set_backend, m)?)?;
    m.add_function(wrap_pyfunction!(backend::get_backend, m)?)?;
    m.add_function(wrap_pyfunction!(backend::list_backends, m)?)?;

    // Runtime (checkpoint-backed) models
    m.add_function(wrap_pyfunction!(runtime::load_model, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::supported_model_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::supported_backends, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::backend_features, m)?)?;
    m.add_function(wrap_pyfunction!(runtime::version_info, m)?)?;

    // Debugging/profiling helpers
    m.add_function(wrap_pyfunction!(debug::set_dump_dir, m)?)?;
    m.add_function(wrap_pyfunction!(debug::clear_dump_dir, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_reset, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_take_report, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_push_section, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_pop_section, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_take_report_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_take_report_json, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_trace_enable, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_trace_disable, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_trace_reset, m)?)?;
    m.add_function(wrap_pyfunction!(debug::profiling_take_trace_json, m)?)?;

    Ok(())
}
