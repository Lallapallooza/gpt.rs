//! The reference backend: it runs every function on the IR's interpreter, the semantics that
//! passes are checked against.

use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::interp;
use gpt_rs_ir::interp::HostValue;
use gpt_rs_ir::ir::Module;

#[derive(Default)]
pub struct CpuPortableBackend {
    params: ParamStore<HostTensor>,
}

impl CpuPortableBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A function that the interpreter runs.
pub struct Function {
    module: Arc<Module>,
    name: String,
}

impl PortableBackend for CpuPortableBackend {
    type TensorHandle = HostTensor;
    type Executable = Function;

    fn backend_name(&self) -> &str {
        "ref-cpu"
    }

    fn params(&self) -> &ParamStore<HostTensor> {
        &self.params
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Function> {
        ensure!(
            module.func(func).is_some(),
            "the module has no function @{func}"
        );
        Ok(Function {
            module: Arc::clone(module),
            name: func.to_string(),
        })
    }

    fn run(&self, exe: &Function, args: Vec<HostTensor>) -> Result<Vec<HostTensor>> {
        let values = args.into_iter().map(HostValue::Tensor).collect();
        interp::run(&exe.module, &exe.name, values)?
            .into_iter()
            .map(|value| match value {
                HostValue::Tensor(tensor) => Ok(tensor),
                other => Err(anyhow!(
                    "@{} returned a {}, not a tensor",
                    exe.name,
                    other.ty()
                )),
            })
            .collect()
    }

    fn upload(&self, host: HostTensor) -> Result<HostTensor> {
        Ok(host)
    }

    fn download(&self, tensor: &HostTensor) -> Result<HostTensor> {
        Ok(tensor.clone())
    }
}
