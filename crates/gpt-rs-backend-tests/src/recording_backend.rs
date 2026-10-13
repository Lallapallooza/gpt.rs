//! A backend that runs nothing and records the last function the graph arena compiled.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::gpt_rs_ir::dialects::func::Func;
use gpt_rs::gpt_rs_ir::host::HostTensor;
use gpt_rs::gpt_rs_ir::ir::Module;

#[derive(Default)]
pub struct RecordingBackend {
    params: ParamStore<()>,
    last: Mutex<Option<(Arc<Module>, String)>>,
}

impl RecordingBackend {
    /// The module and the name of the function that the arena compiled last.
    pub fn recorded_or_panic(&self) -> (Arc<Module>, String) {
        self.last
            .lock()
            .expect("backend mutex poisoned")
            .clone()
            .expect("the arena compiled a function")
    }
}

impl PortableBackend for RecordingBackend {
    type TensorHandle = ();
    /// The number of results.
    type Executable = usize;

    fn backend_name(&self) -> &str {
        "recording"
    }

    fn params(&self) -> &ParamStore<()> {
        &self.params
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<usize> {
        let inst = module
            .func(func)
            .ok_or_else(|| anyhow!("the module has no function @{func}"))?;
        let results = module
            .op(inst)
            .get::<Func>()
            .expect("a function")
            .results
            .len();
        *self.last.lock().expect("backend mutex poisoned") =
            Some((Arc::clone(module), func.to_string()));
        Ok(results)
    }

    fn run(&self, results: &usize, _args: &[()]) -> Result<Vec<()>> {
        Ok(vec![(); *results])
    }

    fn upload(&self, _host: HostTensor) -> Result<()> {
        Ok(())
    }

    fn download(&self, _tensor: &()) -> Result<HostTensor> {
        Err(anyhow!("the recording backend holds no tensor data"))
    }
}
