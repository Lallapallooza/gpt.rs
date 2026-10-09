pub mod hashing;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{self, Pipeline};

/// A backend compiles functions of the IR and runs them on its own tensors.
pub trait PortableBackend: Send + Sync {
    /// A tensor the backend holds: host memory, a view of a mapped checkpoint, or device memory.
    type TensorHandle: Clone + Send + Sync + 'static;
    /// A compiled function, which runs many times.
    type Executable: Send + Sync + 'static;

    /// The backend's name, such as `"faer"` or `"c"`.
    fn backend_name(&self) -> &str;

    /// The passes the arena runs on a module before it compiles the module's functions. A backend
    /// that rewrites the layout of parameters adds its passes before `populate_load_time_passes`,
    /// whose `outline-static` then moves that work to load time.
    fn pipeline(&self) -> Pipeline {
        let mut pm = Pipeline::new();
        passes::populate_simplification_passes(&mut pm);
        passes::populate_tensor_passes(&mut pm);
        passes::populate_load_time_passes(&mut pm);
        pm
    }

    /// The parameters the backend holds, including the results of load-time functions. They live
    /// as long as the backend, so arenas share them.
    fn params(&self) -> &ParamStore<Self::TensorHandle>;

    /// Compiles the function `func` of `module`, after the pass pipeline ran on it.
    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Self::Executable>;

    /// Runs `exe` on its arguments, in argument order, and returns its results.
    fn run(
        &self,
        exe: &Self::Executable,
        args: &[Self::TensorHandle],
    ) -> Result<Vec<Self::TensorHandle>>;

    /// A tensor holding the elements of `host`. A backend on host memory keeps `host`'s storage,
    /// so a view of a mapped checkpoint is never copied.
    fn upload(&self, host: HostTensor) -> Result<Self::TensorHandle>;

    /// The elements of `tensor` on the host.
    fn download(&self, tensor: &Self::TensorHandle) -> Result<HostTensor>;
}

/// Tensors by parameter id.
pub struct ParamStore<H> {
    tensors: Mutex<HashMap<u128, H>>,
}

impl<H: Clone> ParamStore<H> {
    pub fn get(&self, id: u128) -> Option<H> {
        self.lock().get(&id).cloned()
    }

    /// The tensors of `ids`, read under one lock. `None` when one of them is missing.
    pub fn get_all(&self, ids: impl IntoIterator<Item = u128>) -> Option<Vec<H>> {
        let tensors = self.lock();
        let ids = ids.into_iter();
        let mut out = Vec::with_capacity(ids.size_hint().1.unwrap_or(0));
        for id in ids {
            out.push(tensors.get(&id)?.clone());
        }
        Some(out)
    }

    pub fn set(&self, id: u128, tensor: H) {
        self.lock().insert(id, tensor);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u128, H>> {
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
