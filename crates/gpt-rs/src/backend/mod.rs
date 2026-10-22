pub mod hashing;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use foldhash::fast::FixedState;
use gpt_rs_ir::dialects::tensor::MergeContractedAxes;
use gpt_rs_ir::dialects::tile::Numerics;
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::Module;
use gpt_rs_ir::passes::{self, Pipeline};

/// A backend compiles functions of the IR and runs them on its own tensors.
pub trait PortableBackend: Send + Sync {
    /// A tensor the backend holds: host memory, a view of a mapped checkpoint, or device memory.
    type TensorHandle: Clone + Send + Sync + 'static;
    /// A compiled function, which runs many times.
    type Executable: Send + Sync + 'static;

    /// The backend's name, such as `"faer"` or `"c"`.
    fn backend_name(&self) -> &str;

    /// What else the programs it compiles depend on, such as the target it compiles for: a program
    /// compiled by one backend serves another of the same name and identity.
    fn compile_identity(&self) -> &str {
        ""
    }

    /// The passes the arena runs on a module before it compiles the module's functions, with the
    /// one rounding change of `scale_fold`. A backend that rewrites the layout of parameters adds
    /// its passes before `populate_load_time_passes`, whose `outline-static` then moves that work
    /// to load time.
    fn pipeline(&self) -> Pipeline {
        let mut pm = Pipeline::new();
        passes::populate_simplification_passes(&mut pm);
        let numerics = Numerics {
            scale_fold: true,
            ..Numerics::default()
        };
        passes::populate_tensor_passes(&mut pm, &numerics);
        pm.add(MergeContractedAxes);
        passes::populate_load_time_passes(&mut pm);
        pm
    }

    /// The parameters the backend holds, including the results of load-time functions. They live
    /// as long as the backend, so arenas share them.
    fn params(&self) -> &ParamStore<Self::TensorHandle>;

    /// Compiles the function `func` of `module`, after the pass pipeline ran on it.
    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Self::Executable>;

    /// Runs `exe` on its arguments, in argument order, and returns its results. It takes the
    /// arguments, so it may update the memory of one in place when no other handle holds it.
    fn run(
        &self,
        exe: &Self::Executable,
        args: Vec<Self::TensorHandle>,
    ) -> Result<Vec<Self::TensorHandle>>;

    /// A tensor holding the elements of `host`. A backend on host memory keeps `host`'s storage,
    /// so a view of a mapped checkpoint is never copied.
    fn upload(&self, host: HostTensor) -> Result<Self::TensorHandle>;

    /// The elements of `tensor` on the host.
    fn download(&self, tensor: &Self::TensorHandle) -> Result<HostTensor>;
}

/// Tensors by parameter id. The ids are hashes of names and contents, never chosen by an attacker,
/// so a fast fixed-seed hasher keys them.
pub struct ParamStore<H> {
    tensors: Mutex<HashMap<u128, H, FixedState>>,
}

impl<H: Clone> ParamStore<H> {
    pub fn get(&self, id: u128) -> Option<H> {
        self.lock().get(&id).cloned()
    }

    /// The tensors of `ids`, read under one lock. `None` when one of them is missing.
    pub fn get_all(&self, ids: impl IntoIterator<Item = u128>) -> Option<Vec<H>> {
        let tensors = self.lock();
        let requested = ids.into_iter();
        let mut out = Vec::with_capacity(requested.size_hint().1.unwrap_or(0));
        for id in requested {
            out.push(tensors.get(&id)?.clone());
        }
        Some(out)
    }

    pub fn set(&self, id: u128, tensor: H) {
        self.lock().insert(id, tensor);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u128, H, FixedState>> {
        self.tensors.lock().expect("parameter store poisoned")
    }
}

impl<H> Default for ParamStore<H> {
    fn default() -> Self {
        Self {
            tensors: Mutex::default(),
        }
    }
}
