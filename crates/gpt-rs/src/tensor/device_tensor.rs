//! Device-side tensor wrapper that tracks backend handles and shape metadata.

use super::lazy_tensor::{InputRole, Materialized};
use super::{lazy_tensor::LazyHandle, shape::Shape, DType, HostTensor};
use anyhow::Result;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::backend::PortableBackend;
use crate::ops::graph::GraphArena;
use crate::ops::graph::ValueId;
use gpt_rs_ir::types::TensorType;

type ArenaGroup<B> = (Arc<GraphArena<B>>, Vec<(usize, ValueId)>);

/// Device-side tensor that wraps a backend handle and retains shape metadata.
pub struct DeviceTensor<B: PortableBackend + 'static> {
    pub(super) backend: Arc<B>,
    shape: Shape,
    dtype: DType,
    handle: LazyHandle<B>,
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
    pub fn from_host(backend: Arc<B>, tensor: HostTensor) -> Result<Self> {
        let shape = Shape::new(&tensor.dims);
        let dtype = tensor.dtype();
        let handle = backend.upload(tensor)?;
        Ok(DeviceTensor {
            backend,
            shape,
            dtype,
            handle: LazyHandle::Input(Materialized::arg(handle)),
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
            handle: LazyHandle::Input(Materialized::arg(handle)),
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
                handle: LazyHandle::Input(Materialized::arg(handle)),
            })
        } else {
            Ok(DeviceTensor {
                backend,
                shape,
                dtype,
                handle: LazyHandle::Node { graph, value },
            })
        }
    }

    /// Materializes a zero-filled tensor on the backend.
    pub fn zeros(backend: Arc<B>, shape: Shape) -> Result<Self> {
        let ty = TensorType::new(DType::F32, shape.dims().iter().copied());
        let handle = backend.upload(HostTensor::zeros(&ty)?)?;
        Ok(DeviceTensor {
            backend,
            shape,
            dtype: DType::F32,
            handle: LazyHandle::Input(Materialized::arg(handle)),
        })
    }

    /// Copies the device tensor back to the host.
    pub fn to_host(&self) -> Result<HostTensor> {
        let handle = self.materialize()?;
        self.backend.download(&handle)
    }

    /// Returns the backend instance that owns the tensor.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
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

    /// The extent of `axis`, which counts from the end when negative: `x.size(-1)` is the last
    /// extent, as in Torch.
    #[track_caller]
    pub fn size(&self, axis: i64) -> usize {
        let dims = self.shape.dims();
        let rank = dims.len() as i64;
        let index = if axis < 0 { axis + rank } else { axis };
        assert!(
            (0..rank).contains(&index),
            "axis {axis} is out of range for a tensor of rank {rank}"
        );
        dims[index as usize]
    }

    /// Returns the scalar dtype of the device tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the raw lazy handle reference for graph wiring.
    pub(crate) fn lazy_handle(&self) -> &LazyHandle<B> {
        &self.handle
    }

    /// Returns the owning graph arena when the tensor is still represented lazily.
    pub(crate) fn graph(&self) -> Option<Arc<GraphArena<B>>> {
        self.handle.graph()
    }

    /// Stops the owning graph from exporting this value. Use it for state that is being replaced,
    /// so later plans do not compute a handle that nobody reads.
    pub(crate) fn release_export(&self) {
        if let Some((graph, value)) = self.graph_value() {
            graph.unexport(value);
        }
    }

    /// Returns the graph arena and value identifier when the tensor is still lazy.
    pub(crate) fn graph_value(&self) -> Option<(Arc<GraphArena<B>>, ValueId)> {
        match &self.handle {
            LazyHandle::Input(_) => None,
            LazyHandle::Node { graph, value } => Some((Arc::clone(graph), *value)),
        }
    }

    /// Ensures the tensor is materialized on the backend and returns the handle.
    pub fn materialize(&self) -> Result<B::TensorHandle> {
        match &self.handle {
            LazyHandle::Input(input) => Ok(input.tensor.clone()),
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
        match &self.handle {
            LazyHandle::Input(_) => Ok(self.clone()),
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

    /// This tensor as the model parameter `id` (a [`crate::params::ParamKey`]), which keeps its
    /// identity across graphs. It stores the tensor under `id` in the backend's parameter store,
    /// where programs read it, so passes can move computations over parameters to load time.
    pub fn as_param(&self, id: u128) -> Result<Self> {
        match &self.handle {
            LazyHandle::Input(input) if input.role == InputRole::Param && input.id == id => {
                Ok(self.clone())
            }
            _ => {
                let handle = self.materialize()?;
                self.backend.params().set(id, handle.clone());
                Ok(DeviceTensor {
                    backend: Arc::clone(&self.backend),
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                    handle: LazyHandle::Input(Arc::new(Materialized {
                        id,
                        role: InputRole::Param,
                        tensor: handle,
                    })),
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
            match &tensor.handle {
                LazyHandle::Input(input) => {
                    results[idx] = Some(input.tensor.clone());
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
            for ((index, _), handle) in entries.into_iter().zip(handles) {
                results[index] = Some(handle);
            }
        }

        Ok(results
            .into_iter()
            .map(|entry| entry.expect("failed to materialize tensor"))
            .collect())
    }

    /// The IR type of this tensor.
    pub(crate) fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype, self.shape.dims().iter().copied())
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
