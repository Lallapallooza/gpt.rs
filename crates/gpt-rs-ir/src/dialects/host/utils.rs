//! Helpers of the `host` ops: the buffer types they give, and the buffer of each parameter of a
//! launch.

use anyhow::{anyhow, ensure, Result};

use crate::dialects::host::ir::*;
use crate::dialects::loops::{self, Access};
use crate::ir::{BufferType, DType, Dim, Ranked, Space, ValueRef, View};

/// The `global` buffer of `dtype` with the static extents `shape`.
fn static_buffer(shape: &Ranked<usize>, dtype: DType) -> BufferType {
    BufferType {
        dtype,
        dims: shape.iter().map(|&extent| Dim::Static(extent)).collect(),
        space: Space::Global,
    }
}

impl Arena {
    /// Its buffer: its bytes, in the space of an arena of its alignment.
    pub fn buffer_type(&self) -> Result<BufferType> {
        let bytes = usize::try_from(self.bytes)?;
        Ok(BufferType {
            space: Space::Arena {
                alignment: self.alignment,
            },
            ..static_buffer(&Ranked::from_iter([bytes]), DType::Ui8)
        })
    }
}

impl Slice {
    /// Its buffer, a `global` one.
    pub fn buffer_type(&self) -> BufferType {
        static_buffer(&self.shape, self.dtype)
    }
}

impl Alloc {
    /// Its buffer, a `global` one.
    pub fn buffer_type(&self) -> BufferType {
        static_buffer(&self.shape, self.dtype)
    }
}

/// The operand groups of `host.launch`, in order: the buffers of the `in`, the `out` and the
/// `inout` parameters of its callee.
const GROUPS: [&str; 3] = ["in", "out", "inout"];

/// The position in [`GROUPS`] of the group that holds the buffer of a parameter of `access`.
fn group(access: Access) -> usize {
    match access {
        Access::In => 0,
        Access::Out { .. } => 1,
        Access::Inout => 2,
    }
}

impl<V: View<Buffer: Clone>> LaunchOperands<V> {
    /// The buffer of each parameter of `func`, in order: the next of `ins` for an `in` parameter,
    /// of `outs` for an `out` one and of `inouts` for an `inout` one. Fails when a group holds
    /// more or fewer buffers than `func` has parameters of its access.
    pub fn params(&self, func: &loops::Func) -> Result<Vec<V::Buffer>> {
        let groups = [&self.ins, &self.outs, &self.inouts];
        let mut counts = [0; 3];
        for &access in &func.access {
            counts[group(access)] += 1;
        }
        for (index, name) in GROUPS.iter().enumerate() {
            ensure!(
                counts[index] == groups[index].len(),
                "@{} has {} `{name}` parameters, got {} buffers",
                func.name,
                counts[index],
                groups[index].len()
            );
        }
        let mut taken = [0; 3];
        let mut params = Vec::with_capacity(func.access.len());
        for &access in &func.access {
            let index = group(access);
            params.push(groups[index][taken[index]].clone());
            taken[index] += 1;
        }
        Ok(params)
    }

    /// The operands of a launch on `params`, the buffer of each parameter in order, of a callee
    /// whose parameters `access` describes.
    pub fn of_params(access: &[Access], params: Vec<V::Buffer>) -> Self {
        let mut groups: [Vec<V::Buffer>; 3] = Default::default();
        for (&param_access, buffer) in access.iter().zip(params) {
            groups[group(param_access)].push(buffer);
        }
        let [ins, outs, inouts] = groups;
        Self { ins, outs, inouts }
    }
}

impl<'m> LaunchOp<'m> {
    /// The buffer of each parameter of the callee, in order.
    pub fn params(&self) -> Result<Vec<ValueRef<'m>>> {
        let module = self.module();
        let name = &self.callee.0;
        let func = module
            .symbol(name)
            .and_then(|symbol| module.op(symbol).get::<loops::Func>())
            .ok_or_else(|| anyhow!("@{name} is not a loop.func"))?;
        self.operands.params(func)
    }
}
