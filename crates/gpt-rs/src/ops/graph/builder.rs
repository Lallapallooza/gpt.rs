//! Mutable builder that records operations in a [`GraphArena`](super::arena::GraphArena).

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Result};
use gpt_rs_ir::dense::Dense;
use gpt_rs_ir::dialects::tensor::{BroadcastTo, Constant};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::op::{InOp, OpDyn};
use gpt_rs_ir::types::{DType, Dim, TensorType, Type};
use smallvec::SmallVec;

use super::arena::GraphArena;
use super::state::{GraphInner, NodeState, OpRecord, ScalarKey, ValueDef, ValueRecord};
use crate::backend::PortableBackend;
use crate::ops::graph::ValueId;
use crate::tensor::{DeviceTensor, LazyHandle, Materialized};

/// An empty module. The type rules of ops without regions take a module but read none of it.
fn empty_module() -> &'static Module {
    static EMPTY: OnceLock<Module> = OnceLock::new();
    EMPTY.get_or_init(Module::new)
}

/// Records ops into a [`GraphArena`]: imports tensors and emits nodes.
pub(crate) struct GraphBuilder<'a, B: PortableBackend + 'static> {
    pub(super) arena: &'a Arc<GraphArena<B>>,
    pub(super) inner: &'a mut GraphInner<B>,
}

impl<'a, B: PortableBackend + 'static> GraphBuilder<'a, B> {
    /// Imports a tensor into the graph, returning the associated value identifier.
    /// Existing handles are reused so repeated captures share parameters.
    pub fn import(&mut self, tensor: &DeviceTensor<B>) -> Result<ValueId> {
        match tensor.lazy_handle() {
            LazyHandle::Input(input) => {
                let key = (input.role, input.id);
                if let Some(&existing) = self.inner.inputs.get(&key) {
                    return Ok(existing);
                }
                let value = self
                    .inner
                    .push_value(tensor.tensor_type(), ValueDef::Input(Arc::clone(input)));
                self.inner.inputs.insert(key, value);
                self.inner.bump_version();
                Ok(value)
            }
            LazyHandle::Node { graph, value } if Arc::ptr_eq(graph, self.arena) => Ok(*value),
            LazyHandle::Node { .. } => {
                let input = Materialized::arg(tensor.materialize()?);
                let value = self
                    .inner
                    .push_value(tensor.tensor_type(), ValueDef::Input(input));
                self.inner.bump_version();
                Ok(value)
            }
        }
    }

    /// The tensor of extents `dims` whose every element has the little-endian bytes `element` of
    /// `dtype`: a rank-0 constant, broadcast to `dims`. A graph records each scalar, and each
    /// broadcast of it, once and reuses them while they are pending.
    pub fn scalar(&mut self, dtype: DType, element: &[u8], dims: &[usize]) -> Result<ValueId> {
        let mut bits = [0u8; 8];
        bits[..element.len()].copy_from_slice(element);
        let bits = u64::from_le_bytes(bits);
        let constant = match self.pending_scalar(&(dtype, bits, SmallVec::new())) {
            Some(constant) => constant,
            None => {
                let value = Dense::splat(dtype, [], element);
                let constant = ValueId(self.emit(Constant { value }, &[])?.start);
                self.inner
                    .scalars
                    .insert((dtype, bits, SmallVec::new()), constant);
                constant
            }
        };
        if dims.is_empty() {
            return Ok(constant);
        }
        let key = (dtype, bits, SmallVec::from_slice(dims));
        if let Some(value) = self.pending_scalar(&key) {
            return Ok(value);
        }
        let shape = dims.iter().map(|&dim| Dim::Static(dim)).collect();
        let value = ValueId(self.emit(BroadcastTo { shape }, &[constant])?.start);
        self.inner.scalars.insert(key, value);
        Ok(value)
    }

    /// The scalar of `key` that this graph recorded, while it is pending.
    fn pending_scalar(&self, key: &ScalarKey) -> Option<ValueId> {
        let &value = self.inner.scalars.get(key)?;
        let record = self.inner.values.get(value.0 as usize)?;
        matches!(
            record.def,
            ValueDef::Node {
                state: NodeState::Pending,
                ..
            }
        )
        .then_some(value)
    }

    /// The type of the captured value `value`.
    pub fn type_of(&self, value: ValueId) -> Result<&TensorType> {
        Ok(self.inner.value(value)?.tensor_type())
    }

    /// Records `op` applied to `operands` and returns the ids of its results, which are
    /// consecutive. Their types come from the op's type rule, which rejects operands the op does
    /// not accept.
    pub fn emit<T: InOp + OpDyn>(&mut self, op: T, operands: &[ValueId]) -> Result<Range<u32>> {
        let results = {
            let mut types = SmallVec::<[&Type; 4]>::new();
            for &operand in operands {
                types.push(&self.inner.value(operand)?.ty);
            }
            let (results, _) = op
                .dyn_result_types(&types, None, &[], empty_module())
                .map_err(|err| anyhow!("{}: {err:#}", op.name()))?;
            results
        };
        if results.iter().any(|ty| ty.as_tensor().is_none()) {
            bail!("{} gives a result that is no tensor", op.name());
        }
        let index = self.inner.ops.len() as u32;
        let first = self.inner.values.len() as u32;
        let start = self.inner.operands.len() as u32;
        self.inner.operands.extend_from_slice(operands);
        for (result, ty) in results.into_iter().enumerate() {
            let def = ValueDef::Node {
                op: index,
                result: result as u32,
                state: NodeState::Pending,
            };
            self.inner.values.push(ValueRecord { ty, def });
        }
        self.inner.ops.push(OpRecord {
            op: op.into(),
            operands: (start, self.inner.operands.len() as u32),
            first: ValueId(first),
        });
        self.inner.bump_version();
        Ok(first..self.inner.values.len() as u32)
    }
}
