//! The backends of a build, chosen by name at run time: `ref-cpu` always, `faer` and `c` with
//! the features of those names. A model loads onto one of them, and its [`ModelHandle`] runs it
//! there without naming the backend type.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use gpt_rs::runtime::{load_model_with_options, LoadOptions, ModelHandle};

/// The backend that a caller gets without naming one: `faer` when the build has it.
pub const DEFAULT: &str = if cfg!(feature = "faer") {
    "faer"
} else {
    "ref-cpu"
};

/// The names of the backends of this build.
pub fn names() -> Vec<&'static str> {
    [
        Some("ref-cpu"),
        cfg!(feature = "faer").then_some("faer"),
        cfg!(feature = "c").then_some("c"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Loads the Hugging Face model directory `path` onto a new instance of the backend `backend`, as
/// `options` say.
pub fn load_model(
    backend: &str,
    path: impl AsRef<Path>,
    options: LoadOptions,
) -> Result<ModelHandle> {
    match backend {
        "ref-cpu" => load_model_with_options(
            Arc::new(gpt_rs_backend_ref_cpu::CpuPortableBackend::new()),
            path,
            options,
        ),
        #[cfg(feature = "faer")]
        "faer" => load_model_with_options(
            Arc::new(gpt_rs_backend_faer::FaerPortableBackend::new()),
            path,
            options,
        ),
        #[cfg(feature = "c")]
        "c" => load_model_with_options(Arc::new(gpt_rs_backend_c::CBackend::new()), path, options),
        other => bail!(
            "unknown backend '{other}'; this build has {}",
            names().join(", ")
        ),
    }
}
