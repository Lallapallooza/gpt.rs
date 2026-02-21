//! Device-side tensor wrapper that tracks backend handles and shape metadata.

use super::lazy_tensor::InputRole;
use super::{lazy_tensor::LazyHandle, shape::Shape, spec_utils, DType, Tensor};
use anyhow::{anyhow, ensure, Result};
use std::collections::HashMap;
use std::fmt;
use std::sync::{
    atomic::{AtomicU64, Ordering as AtomicOrdering},
    Arc,
};

use crate::backend::spec::{PortableBackend, TensorInit, TensorSpec, ValueId};
use crate::ops::graph::{CompiledGraph, GraphArena};
use crate::params::{BaseParamId, ParamSource};

type ArenaGroup<B> = (Arc<GraphArena<B>>, Vec<(usize, ValueId)>);

static ARG_HANDLE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);
static PARAM_HANDLE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_arg_handle_id() -> u128 {
    u128::from(ARG_HANDLE_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed))
}

fn next_param_handle_id() -> u128 {
    u128::from(PARAM_HANDLE_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed))
}

/// Device-side tensor that wraps a backend handle and retains shape metadata.
pub struct DeviceTensor<B: PortableBackend + 'static> {
    backend: Arc<B>,
    shape: Shape,
    dtype: DType,
    handle: Arc<LazyHandle<B>>,
}

impl<B: PortableBackend + 'static> Clone for DeviceTensor<B> {
    fn clone(&self) -> Self {
        DeviceTensor {
            backend: Arc::clone(&self.backend),
            shape: self.shape.clone(),
            dtype: self.dtype,
            handle: self.handle.clone(),
        }
    }
}

impl<B: PortableBackend + 'static> AsRef<DeviceTensor<B>> for DeviceTensor<B> {
    fn as_ref(&self) -> &DeviceTensor<B> {
        self
    }
}

impl<B: PortableBackend + 'static> DeviceTensor<B> {
    /// Transfers a host tensor into backend memory, producing a fresh device tensor.
    pub fn from_host(backend: Arc<B>, tensor: Tensor) -> Result<Self> {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();
        let literal = tensor.to_literal();
        let handle = backend.materialize(TensorInit::Literal(literal))?;
        Ok(DeviceTensor {
            backend,
            shape,
            dtype,
            handle: Arc::new(LazyHandle::Input {
                id: next_arg_handle_id(),
                role: InputRole::Arg,
                tensor: handle,
            }),
        })
    }

    /// Wraps an existing backend handle with explicit metadata.
    pub fn from_handle(
        backend: Arc<B>,
        shape: Shape,
        dtype: DType,
        handle: B::TensorHandle,
    ) -> Self {
        DeviceTensor {
            backend,
            shape,
            dtype,
            handle: Arc::new(LazyHandle::Input {
                id: next_arg_handle_id(),
                role: InputRole::Arg,
                tensor: handle,
            }),
        }
    }

    /// Builds a device tensor from a lazy graph node, optionally flushing when eager mode is set.
    pub fn from_lazy(
        graph: Arc<GraphArena<B>>,
        shape: Shape,
        dtype: DType,
        value: ValueId,
    ) -> Result<Self> {
        let backend = graph.backend();

        if crate::env::eager_enabled() {
            let handle = graph.flush_until(value)?;
            Ok(DeviceTensor {
                backend,
                shape,
                dtype,
                handle: Arc::new(LazyHandle::Input {
                    id: next_arg_handle_id(),
                    role: InputRole::Arg,
                    tensor: handle,
                }),
            })
        } else {
            Ok(DeviceTensor {
                backend,
                shape,
                dtype,
                handle: Arc::new(LazyHandle::Node { graph, value }),
            })
        }
    }

    /// Materializes a zero-filled tensor on the backend.
    pub fn zeros(backend: Arc<B>, shape: Shape) -> Result<Self> {
        let spec = TensorSpec::new(
            crate::backend::spec::DType::F32,
            spec_utils::backend_shape_from_shape(&shape),
        );
        let handle = backend.materialize(TensorInit::Zeroed(spec))?;
        Ok(DeviceTensor {
            backend,
            shape,
            dtype: DType::F32,
            handle: Arc::new(LazyHandle::Input {
                id: next_arg_handle_id(),
                role: InputRole::Arg,
                tensor: handle,
            }),
        })
    }

    /// Copies the device tensor back to the host as a [`Tensor`].
    pub fn to_host(&self) -> Result<Tensor> {
        let handle = self.materialize()?;
        let literal = self.backend.to_literal(&handle)?;
        Tensor::from_literal(&literal)
    }

    /// Returns the backend instance that owns the tensor.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    /// Compiles the pending lazy graph that produces this tensor.
    pub fn compile(&self) -> Result<CompiledGraph<B>> {
        let (graph, value) = self
            .graph_value()
            .ok_or_else(|| anyhow!("tensor is already materialised; no graph to compile"))?;
        graph.compile(&[value])
    }

    /// Produces a materialized backend handle, cloning if required.
    pub fn clone_handle(&self) -> B::TensorHandle {
        self.materialize()
            .expect("device tensor materialization failed")
    }

    /// Exposes the logical shape metadata.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the scalar dtype of the device tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the raw lazy handle reference for graph wiring.
    pub(crate) fn lazy_handle(&self) -> &Arc<LazyHandle<B>> {
        &self.handle
    }

    /// Returns the owning graph arena when the tensor is still represented lazily.
    pub(crate) fn graph(&self) -> Option<Arc<GraphArena<B>>> {
        self.handle.graph()
    }

    /// Returns the graph arena and value identifier when the tensor is still lazy.
    pub(crate) fn graph_value(&self) -> Option<(Arc<GraphArena<B>>, ValueId)> {
        match &*self.handle {
            LazyHandle::Input { .. } => None,
            LazyHandle::Param { .. } => None,
            LazyHandle::Node { graph, value } => Some((Arc::clone(graph), *value)),
        }
    }

    /// Ensures the tensor is materialized on the backend and returns the handle.
    pub fn materialize(&self) -> Result<B::TensorHandle> {
        match &*self.handle {
            LazyHandle::Input { tensor, .. } => Ok(tensor.clone()),
            LazyHandle::Param {
                base_id,
                source,
                cache_enabled,
                cached,
                ..
            } => {
                if *cache_enabled {
                    if let Some(handle) = cached.get() {
                        return Ok(handle.clone());
                    }
                    let handle = source.load(*base_id)?;
                    let _ = cached.set(handle.clone());
                    return Ok(handle);
                }
                Ok(source.load(*base_id)?)
            }
            LazyHandle::Node { graph, value } => {
                if let Some(handle) = graph.try_ready_handle(*value) {
                    return Ok(handle);
                }
                let mut handles = graph.materialize_values(&[*value])?;
                handles
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("failed to materialize value {:?}", value))
            }
        }
    }

    /// Materializes the tensor and returns a detached device tensor backed by a direct input handle.
    ///
    /// This is useful for weight preparation: callers can build a small packing/fusion graph once,
    /// execute it immediately, and store the resulting handle for reuse in subsequent programs.
    pub fn freeze(&self) -> Result<Self> {
        match &*self.handle {
            LazyHandle::Input { .. } => Ok(self.clone()),
            LazyHandle::Param { id, .. } => {
                let handle = self.materialize()?;
                Ok(DeviceTensor {
                    backend: Arc::clone(&self.backend),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    handle: Arc::new(LazyHandle::Input {
                        id: *id,
                        role: InputRole::Param,
                        tensor: handle,
                    }),
                })
            }
            LazyHandle::Node { .. } => {
                let handle = self.materialize()?;
                Ok(DeviceTensor::from_handle(
                    Arc::clone(&self.backend),
                    self.shape.clone(),
                    self.dtype,
                    handle,
                ))
            }
        }
    }

    /// Returns a device tensor that is marked as a model parameter (stable identity across captures).
    ///
    /// Parameters behave differently from runtime arguments: optimizer passes may treat them as
    /// foldable inputs and cache derived representations (e.g. packed weights).
    pub fn as_param(&self) -> Result<Self> {
        match &*self.handle {
            LazyHandle::Input {
                role: InputRole::Param,
                ..
            } => Ok(self.clone()),
            LazyHandle::Param { .. } => Ok(self.clone()),
            _ => {
                let handle = self.materialize()?;
                Ok(DeviceTensor {
                    backend: Arc::clone(&self.backend),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    handle: Arc::new(LazyHandle::Input {
                        id: next_param_handle_id(),
                        role: InputRole::Param,
                        tensor: handle,
                    }),
                })
            }
        }
    }

    /// Marks the tensor as a model parameter with an explicit stable identity.
    ///
    /// This is the preferred entrypoint for binding checkpoint parameter names to deterministic
    /// ids, and for namespacing multiple co-resident model instances.
    pub fn as_param_with_id(&self, stable_id: u128) -> Result<Self> {
        match &*self.handle {
            LazyHandle::Input {
                role: InputRole::Param,
                id,
                ..
            } if *id == stable_id => Ok(self.clone()),
            LazyHandle::Param {
                id,
                base_id,
                source,
                cache_enabled,
                cached,
            } => {
                if *id == stable_id {
                    return Ok(self.clone());
                }
                let new_cached = once_cell::sync::OnceCell::new();
                if let Some(handle) = cached.get() {
                    let _ = new_cached.set(handle.clone());
                }
                Ok(DeviceTensor {
                    backend: Arc::clone(&self.backend),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    handle: Arc::new(LazyHandle::Param {
                        id: stable_id,
                        base_id: *base_id,
                        source: Arc::clone(source),
                        cache_enabled: *cache_enabled,
                        cached: new_cached,
                    }),
                })
            }
            _ => {
                let handle = self.materialize()?;
                Ok(DeviceTensor {
                    backend: Arc::clone(&self.backend),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    handle: Arc::new(LazyHandle::Input {
                        id: stable_id,
                        role: InputRole::Param,
                        tensor: handle,
                    }),
                })
            }
        }
    }

    /// Materializes a collection of tensors, grouping them by their backing graph arenas so
    /// common subgraphs execute only once.
    pub fn materialize_many(tensors: &[&DeviceTensor<B>]) -> Result<Vec<B::TensorHandle>> {
        if tensors.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Option<B::TensorHandle>> = vec![None; tensors.len()];
        let mut groups: HashMap<usize, ArenaGroup<B>> = HashMap::new();

        for (idx, tensor) in tensors.iter().enumerate() {
            match &*tensor.handle {
                LazyHandle::Input { tensor: handle, .. } => {
                    results[idx] = Some(handle.clone());
                }
                LazyHandle::Param { .. } => {
                    results[idx] = Some(tensor.materialize()?);
                }
                LazyHandle::Node { graph, value } => {
                    let key = Arc::as_ptr(graph) as usize;
                    let entry = groups
                        .entry(key)
                        .or_insert_with(|| (Arc::clone(graph), Vec::new()));
                    entry.1.push((idx, *value));
                }
            }
        }

        for (_, (graph, entries)) in groups.into_iter() {
            let value_ids = entries.iter().map(|(_, value)| *value).collect::<Vec<_>>();
            let handles = graph.materialize_values(&value_ids)?;
            for ((index, _), handle) in entries.into_iter().zip(handles.into_iter()) {
                results[index] = Some(handle);
            }
        }

        Ok(results
            .into_iter()
            .map(|entry| entry.expect("failed to materialize tensor"))
            .collect())
    }

    /// Builds a backend `TensorSpec` matching this tensor.
    pub(crate) fn tensor_spec(&self) -> TensorSpec {
        TensorSpec::new(
            spec_utils::backend_dtype(self.dtype),
            spec_utils::backend_shape_from_shape(&self.shape),
        )
    }

    pub fn lazy_param(
        backend: Arc<B>,
        shape: Shape,
        dtype: DType,
        stable_id: u128,
        base_id: BaseParamId,
        source: Arc<dyn ParamSource<B>>,
        cache_enabled: bool,
    ) -> Self {
        DeviceTensor {
            backend,
            shape,
            dtype,
            handle: Arc::new(LazyHandle::Param {
                id: stable_id,
                base_id,
                source,
                cache_enabled,
                cached: once_cell::sync::OnceCell::new(),
            }),
        }
    }
}

impl<B: PortableBackend> fmt::Debug for DeviceTensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceTensor")
            .field("backend", &self.backend.backend_name())
            .field("shape", &self.shape.dims())
            .field("dtype", &self.dtype)
            .finish()
    }
}

/// Helper trait for converting host values or existing device tensors into device tensors.
pub trait IntoDeviceTensor<B: PortableBackend + 'static> {
    /// Converts the value into a device tensor bound to the provided backend.
    fn into_device_tensor(self, backend: &Arc<B>) -> Result<DeviceTensor<B>>;
}

/// Variant of [`IntoDeviceTensor`] that lifts optional values.
pub trait IntoDeviceTensorOption<B: PortableBackend + 'static> {
    /// Converts the option into a device tensor when present, otherwise returns `None`.
    fn into_device_tensor_option(self, backend: &Arc<B>) -> Result<Option<DeviceTensor<B>>>;
}

impl<B, T> IntoDeviceTensorOption<B> for Option<T>
where
    B: PortableBackend + 'static,
    T: IntoDeviceTensor<B>,
{
    fn into_device_tensor_option(self, backend: &Arc<B>) -> Result<Option<DeviceTensor<B>>> {
        match self {
            Some(value) => value.into_device_tensor(backend).map(Some),
            None => Ok(None),
        }
    }
}

impl<B: PortableBackend + 'static> IntoDeviceTensor<B> for Tensor {
    fn into_device_tensor(self, backend: &Arc<B>) -> Result<DeviceTensor<B>> {
        DeviceTensor::from_host(Arc::clone(backend), self)
    }
}

impl<B, T> IntoDeviceTensor<B> for T
where
    B: PortableBackend + 'static,
    T: AsRef<DeviceTensor<B>>,
{
    fn into_device_tensor(self, backend: &Arc<B>) -> Result<DeviceTensor<B>> {
        let dt = self.as_ref();
        let tensor_backend = dt.backend();
        ensure!(
            Arc::ptr_eq(&tensor_backend, backend),
            "device tensor backend mismatch",
        );
        Ok(dt.clone())
    }
}
