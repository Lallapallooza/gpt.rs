//! A backend that runs another and records what the graph arena asks of it: each function it
//! compiles, and for each run whether each argument was the only handle of its elements.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::gpt_rs_ir::host::HostTensor;
use gpt_rs::gpt_rs_ir::ir::Module;
use gpt_rs::gpt_rs_ir::passes::Pipeline;

pub struct RecordingBackend<B> {
    inner: B,
    name: String,
    compiled: Mutex<Vec<(Arc<Module>, String)>>,
    runs: Mutex<Vec<Vec<bool>>>,
}

impl<B: PortableBackend<TensorHandle = HostTensor>> RecordingBackend<B> {
    /// Runs `inner`, under its name.
    pub fn new(inner: B) -> Self {
        let name = inner.backend_name().to_string();
        Self::named(&name, inner)
    }

    /// Runs `inner` under the name `name`, so the programs it compiles are cached apart from
    /// those of every other backend.
    pub fn named(name: &str, inner: B) -> Self {
        Self {
            inner,
            name: name.to_string(),
            compiled: Mutex::new(Vec::new()),
            runs: Mutex::new(Vec::new()),
        }
    }

    /// The module and the name of the function that the arena compiled last.
    pub fn last_compiled(&self) -> (Arc<Module>, String) {
        self.compiled
            .lock()
            .expect("backend mutex poisoned")
            .last()
            .cloned()
            .expect("the arena compiled a function")
    }

    /// For each run, whether each argument was the only handle of its elements.
    pub fn runs(&self) -> Vec<Vec<bool>> {
        self.runs.lock().expect("backend mutex poisoned").clone()
    }
}

impl<B: PortableBackend<TensorHandle = HostTensor>> PortableBackend for RecordingBackend<B> {
    type TensorHandle = HostTensor;
    type Executable = B::Executable;

    fn backend_name(&self) -> &str {
        &self.name
    }

    fn compile_identity(&self) -> &str {
        self.inner.compile_identity()
    }

    fn pipeline(&self) -> Pipeline {
        self.inner.pipeline()
    }

    fn params(&self) -> &ParamStore<HostTensor> {
        self.inner.params()
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Self::Executable> {
        let compiled = (Arc::clone(module), func.to_string());
        self.compiled
            .lock()
            .expect("backend mutex poisoned")
            .push(compiled);
        self.inner.compile(module, func)
    }

    fn run(&self, exe: &Self::Executable, args: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let mut unique = Vec::with_capacity(args.len());
        for arg in &args {
            unique.push(arg.is_unique());
        }
        self.runs
            .lock()
            .expect("backend mutex poisoned")
            .push(unique);
        self.inner.run(exe, args)
    }

    fn upload(&self, host: HostTensor) -> Result<HostTensor> {
        self.inner.upload(host)
    }

    fn download(&self, tensor: &HostTensor) -> Result<HostTensor> {
        self.inner.download(tensor)
    }
}
