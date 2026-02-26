//! Mutable builder used to stage operations inside a [`GraphArena`](super::arena::GraphArena).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::backend::spec::{Operand, Operation, PortableBackend, TensorSpec, ValueId, ValueType};
use crate::backend::text_ir::{PtirSnippet, SnippetBindings, SnippetResult};
use crate::tensor::{DeviceTensor, InputRole, LazyHandle};

use super::arena::GraphArena;
use super::state::{GraphInner, NodeRecord, NodeState, ParamSourceRecord, ParameterRecord};

/// Context passed to graph capture closures for importing tensors and emitting nodes.
pub struct GraphBuilder<'a, B: PortableBackend + 'static> {
    pub(super) arena: Arc<GraphArena<B>>,
    pub(super) inner: &'a mut GraphInner<B>,
}

impl<'a, B: PortableBackend + 'static> GraphBuilder<'a, B> {
    /// Imports a tensor into the graph, returning the associated value identifier.
    /// Existing handles are reused so repeated captures share parameters.
    pub fn import(&mut self, tensor: &DeviceTensor<B>) -> Result<ValueId> {
        match &**tensor.lazy_handle() {
            LazyHandle::Input { .. } => {
                let role = tensor
                    .lazy_handle()
                    .role()
                    .ok_or_else(|| anyhow!("input handle missing role"))?;
                let stable_id = tensor
                    .lazy_handle()
                    .id()
                    .ok_or_else(|| anyhow!("input handle missing stable id"))?;
                let key = (role, stable_id);

                if let Some(existing) = self.inner.parameter_lookup.get(&key) {
                    return Ok(*existing);
                }

                let handle = tensor.materialize()?;
                let value = self.allocate_value();
                let spec = tensor.tensor_spec();
                if role == InputRole::Param {
                    self.arena.param_resolver.set(stable_id, handle.clone());
                }
                self.inner.push_parameter(ParameterRecord {
                    value,
                    spec,
                    handle: Some(handle),
                    role,
                    stable_id: Some(stable_id),
                });
                self.inner.parameter_lookup.insert(key, value);
                self.inner.bump_version();
                Ok(value)
            }
            LazyHandle::Param {
                base_id, source, ..
            } => {
                let stable_id = tensor
                    .lazy_handle()
                    .id()
                    .ok_or_else(|| anyhow!("param handle missing stable id"))?;
                let key = (InputRole::Param, stable_id);

                if let Some(existing) = self.inner.parameter_lookup.get(&key) {
                    return Ok(*existing);
                }

                let value = self.allocate_value();
                let spec = tensor.tensor_spec();
                self.inner.param_sources.insert(
                    stable_id,
                    ParamSourceRecord {
                        base_id: *base_id,
                        source: Arc::clone(source),
                    },
                );
                self.inner.push_parameter(ParameterRecord {
                    value,
                    spec,
                    handle: None,
                    role: InputRole::Param,
                    stable_id: Some(stable_id),
                });
                self.inner.parameter_lookup.insert(key, value);
                self.inner.bump_version();
                Ok(value)
            }
            LazyHandle::Node { graph, value } => {
                if Arc::ptr_eq(graph, &self.arena) {
                    Ok(*value)
                } else {
                    let handle = tensor.materialize()?;
                    let value_id = self.allocate_value();
                    let spec = tensor.tensor_spec();
                    self.inner.push_parameter(ParameterRecord {
                        value: value_id,
                        spec,
                        handle: Some(handle),
                        role: InputRole::Arg,
                        stable_id: None,
                    });
                    self.inner.bump_version();
                    Ok(value_id)
                }
            }
        }
    }

    /// Emits a new operation node and returns its output value identifier.
    /// The builder records insertion order to preserve dependencies during flushing.
    pub fn emit(&mut self, op: Operation, operands: Vec<Operand>, spec: TensorSpec) -> ValueId {
        let value = self.allocate_value();
        crate::backend::pattern::record_node(value, &op, &operands);
        self.inner.nodes.insert(
            value,
            NodeRecord {
                op,
                operands,
                spec,
                state: NodeState::Pending,
            },
        );
        self.inner.order.push(value);
        self.inner.bump_version();
        value
    }

    /// Returns the arena backing the builder.
    /// This lets callers spawn additional captures or force eager materialisation.
    pub fn arena(&self) -> Arc<GraphArena<B>> {
        Arc::clone(&self.arena)
    }

    /// Marks a value identifier as an exported graph output so future materialisations
    /// surface the handle without additional compilation.
    pub fn export(&mut self, value: ValueId) {
        self.inner.exports.insert(value);
    }

    /// Removes a value identifier from the exported set if it no longer needs to be surfaced.
    pub fn unexport(&mut self, value: ValueId) {
        self.inner.exports.remove(&value);
    }

    /// Emits a reusable PTIR snippet using the supplied bindings, returning the produced values.
    pub fn emit_snippet(
        &mut self,
        snippet: PtirSnippet,
        bindings: &SnippetBindings,
    ) -> Result<SnippetResult> {
        let parsed = snippet
            .instantiate(bindings)
            .map_err(|err| anyhow!(err.to_string()))?;

        let function = parsed
            .program
            .functions
            .first()
            .ok_or_else(|| anyhow!("snippet must define at least one function"))?;

        let mut id_map: HashMap<ValueId, ValueId> = HashMap::new();
        for (name, actual) in bindings.bound_values() {
            let snippet_id = parsed
                .value_names
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("snippet does not declare value `{name}`"))?;
            id_map.insert(snippet_id, actual);
        }

        for param_id in &function.parameter_ids {
            if !id_map.contains_key(param_id) {
                let name = parsed
                    .value_names
                    .iter()
                    .find_map(|(key, value)| if value == param_id { Some(key) } else { None })
                    .map(|s| s.as_str())
                    .unwrap_or("<unnamed>");
                return Err(anyhow!("missing snippet binding for parameter `%{name}`"));
            }
        }

        for instruction in &function.body {
            let mapped_operands = instruction
                .operands
                .iter()
                .map(|operand| match operand {
                    Operand::Value(value) => id_map
                        .get(value)
                        .copied()
                        .map(Operand::Value)
                        .ok_or_else(|| anyhow!("snippet references unknown value id {:?}", value)),
                    Operand::TupleElement { tuple, index } => id_map
                        .get(tuple)
                        .copied()
                        .map(|mapped| Operand::TupleElement {
                            tuple: mapped,
                            index: *index,
                        })
                        .ok_or_else(|| anyhow!("snippet references unknown tuple id {:?}", tuple)),
                    Operand::Literal(lit) => Ok(Operand::Literal(lit.clone())),
                })
                .collect::<Result<Vec<_>>>()?;

            let spec = match &instruction.output {
                ValueType::Tensor(spec) => spec.clone(),
                _ => {
                    return Err(anyhow!(
                        "non-tensor snippet outputs are not supported in graph capture"
                    ))
                }
            };

            let new_id = self.emit(instruction.op.clone(), mapped_operands, spec);
            id_map.insert(instruction.id, new_id);
        }

        let mut results = Vec::with_capacity(function.result_ids.len());
        let mut result_specs = Vec::with_capacity(function.result_ids.len());
        for (result_id, value_type) in function.result_ids.iter().zip(function.results.iter()) {
            let mapped = id_map
                .get(result_id)
                .copied()
                .ok_or_else(|| anyhow!("snippet result id {:?} missing mapping", result_id))?;
            let spec = match value_type {
                ValueType::Tensor(spec) => spec.clone(),
                other => {
                    return Err(anyhow!(
                        "snippet result type {:?} is not supported in graph capture",
                        other
                    ))
                }
            };
            results.push(mapped);
            result_specs.push(spec);
        }

        Ok(SnippetResult::new(results, result_specs))
    }

    fn allocate_value(&mut self) -> ValueId {
        let value = ValueId(self.inner.next_value);
        self.inner.next_value += 1;
        value
    }
}
