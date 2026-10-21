//! The backends of a build, chosen by name at run time: `ref-cpu` always, `faer`, `c` and `x86`
//! with the features of those names. A model loads onto one of them, and its [`ModelHandle`] runs
//! it there without naming the backend type. [`BackendOptions`] hold the target and the Winograd
//! choice of a backend that compiles for a target: the program states them where it starts, and
//! they reach the backend here.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use gpt_rs::backend::PortableBackend;
use gpt_rs::runtime::{load_model_with_options, LoadOptions, ModelHandle};

/// What the `x86` backend writes beside the C it generates: the map from each line of that C to
/// the op it comes from, which a profile of the C reads.
#[cfg(feature = "x86")]
pub mod x86 {
    pub use gpt_rs_backend_x86::{LineMap, LineOrigin};
}

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
        cfg!(feature = "x86").then_some("x86"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// What a backend compiles for, as the program states it where it starts: the CLI's flags, Python's
/// arguments, or the environment.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// The target of a backend that compiles for one: a preset, `host` (the machine this runs on,
    /// probed once), or the path of a target file.
    pub target: String,
    /// Whether 3x3 convolutions of stride 1 run as Winograd's F(2x2, 3x3), which rounds differently
    /// from the convolution. Off by default.
    pub winograd: bool,
}

impl Default for BackendOptions {
    fn default() -> Self {
        BackendOptions {
            target: "host".to_string(),
            winograd: false,
        }
    }
}

impl BackendOptions {
    /// The options the environment states: `GPTRS_TARGET` (`host` when unset) and
    /// `GPTRS_WINOGRAD=1`.
    pub fn from_env() -> Self {
        let mut options = BackendOptions::default();
        if let Ok(target) = std::env::var("GPTRS_TARGET") {
            if !target.trim().is_empty() {
                options.target = target.trim().to_string();
            }
        }
        options.winograd = std::env::var("GPTRS_WINOGRAD").is_ok_and(|value| value.trim() == "1");
        options
    }
}

/// Code generic over the backend type that runs on a backend chosen by name, with
/// [`with_backend`].
pub trait BackendVisitor {
    type Output;

    fn visit<B: PortableBackend + 'static>(self, backend: Arc<B>) -> Self::Output;
}

/// Runs `visitor` on a new instance of the backend `backend`, made as `options` say.
pub fn with_backend<V: BackendVisitor>(
    backend: &str,
    #[cfg_attr(not(feature = "x86"), allow(unused_variables))] options: &BackendOptions,
    visitor: V,
) -> Result<V::Output> {
    Ok(match backend {
        "ref-cpu" => visitor.visit(Arc::new(gpt_rs_backend_ref_cpu::CpuPortableBackend::new())),
        #[cfg(feature = "faer")]
        "faer" => visitor.visit(Arc::new(gpt_rs_backend_faer::FaerPortableBackend::new())),
        #[cfg(feature = "c")]
        "c" => visitor.visit(Arc::new(gpt_rs_backend_c::CBackend::new())),
        #[cfg(feature = "x86")]
        "x86" => {
            let target = gpt_rs::gpt_rs_ir::targets::Target::resolve(&options.target)?;
            let mut x86 = gpt_rs_backend_x86::X86Options::default();
            x86.numerics.winograd = options.winograd;
            visitor.visit(Arc::new(gpt_rs_backend_x86::X86Backend::new(target, x86)?))
        }
        other => bail!(
            "unknown backend '{other}'; this build has {}",
            names().join(", ")
        ),
    })
}

/// Loads the Hugging Face model directory `path`, as `options` say, onto a new instance of the
/// backend `backend`, made as `backend_options` say.
pub fn load_model(
    backend: &str,
    backend_options: &BackendOptions,
    path: impl AsRef<Path>,
    options: LoadOptions,
) -> Result<ModelHandle> {
    struct Load<'p> {
        path: &'p Path,
        options: LoadOptions,
    }
    impl BackendVisitor for Load<'_> {
        type Output = Result<ModelHandle>;

        fn visit<B: PortableBackend + 'static>(self, backend: Arc<B>) -> Result<ModelHandle> {
            load_model_with_options(backend, self.path, self.options)
        }
    }
    with_backend(
        backend,
        backend_options,
        Load {
            path: path.as_ref(),
            options,
        },
    )?
}
