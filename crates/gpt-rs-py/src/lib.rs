#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod backend;
mod debug;
mod device_ops;
mod functional;
mod gpt;
mod nn;
mod tensor;
mod tokenizer;
mod vision;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<tensor::PyTensor>()?;
    m.add_class::<tokenizer::PyTokenizer>()?;

    // Backend management
    m.add_function(wrap_pyfunction!(backend::set_backend, m)?)?;
    m.add_function(wrap_pyfunction!(backend::get_backend, m)?)?;
    m.add_function(wrap_pyfunction!(backend::list_backends, m)?)?;

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

    // Functional ops submodule
    let functional_module = PyModule::new_bound(m.py(), "functional")?;
    functional_module.add_function(wrap_pyfunction!(
        functional::softmax_last_dim,
        &functional_module
    )?)?;
    functional_module.add_function(wrap_pyfunction!(functional::gelu, &functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(
        functional::layer_norm,
        &functional_module
    )?)?;
    functional_module.add_function(wrap_pyfunction!(functional::matmul, &functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(functional::add_bias, &functional_module)?)?;
    functional_module.add_function(wrap_pyfunction!(
        functional::embedding_lookup,
        &functional_module
    )?)?;
    m.add_submodule(&functional_module)?;

    // NN layers submodule
    let nn_module = PyModule::new_bound(m.py(), "nn")?;
    nn_module.add_class::<nn::PyEmbedding>()?;
    nn_module.add_class::<nn::PyLinear>()?;
    nn_module.add_class::<nn::PyLayerNorm>()?;
    nn_module.add_class::<nn::PyFeedForward>()?;
    m.add_submodule(&nn_module)?;

    // Vision submodule
    let vision_module = PyModule::new_bound(m.py(), "vision")?;
    vision_module.add_class::<vision::PyResNet34>()?;
    vision_module.add_class::<vision::PyMobileNetV2>()?;
    vision_module.add_class::<vision::PyConv2d>()?;
    m.add_submodule(&vision_module)?;

    // GPT submodule
    let gpt_module = PyModule::new_bound(m.py(), "gpt")?;
    gpt_module.add_class::<gpt::PyGpt>()?;
    m.add_submodule(&gpt_module)?;

    Ok(())
}
