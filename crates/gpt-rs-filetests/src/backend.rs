//! The runner of a backend's filetests.

use gpt_rs::backend::PortableBackend;

/// Runs the filetests under `filetests/backend/<name>/`, where `name` is the backend's name.
/// `default` is the backend's pipeline, and the backend's own passes are registered once its crate
/// is linked.
pub fn run<B: PortableBackend>(backend: &B) {
    crate::run(
        &format!("backend/{}", backend.backend_name()),
        Some(&|| backend.pipeline()),
    );
}
