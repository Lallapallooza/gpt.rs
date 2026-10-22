//! Builds layers from the named tensors of a checkpoint.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use gpt_rs_ir::ir::{Ranked, TensorType};

use crate::backend::PortableBackend;
use crate::tensor::{DType, DeviceTensor};

/// How a constructor reads a tensor of the checkpoint: as stored, or as a view of its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorView {
    /// As stored.
    Stored,
    /// Without its axes of one element: the `[out, in, 1, 1]` weight of a 1x1 convolution reads as
    /// `[out, in]` where neither extent is 1.
    WithoutUnitAxes,
    /// Its bits read as elements of this dtype, as `tensor.bitcast` reads them, its last axis
    /// scaled by the ratio of the widths: the int32 words of a 4-bit checkpoint read as its codes.
    Bitcast(DType),
}

impl TensorView {
    /// The type of a tensor stored as `stored`, read as this view says. It fails on dynamic
    /// extents. A bitcast also fails on a tensor with no axes, and on rows that hold no whole
    /// number of the elements it reads.
    pub fn type_of(self, stored: &TensorType) -> Result<TensorType> {
        let dims = stored
            .static_dims()
            .ok_or_else(|| anyhow!("a checkpoint tensor of type {stored} has dynamic extents"))?;
        match self {
            TensorView::Stored => Ok(stored.clone()),
            TensorView::WithoutUnitAxes => {
                let mut kept = Ranked::<usize>::new();
                for &extent in &dims {
                    if extent != 1 {
                        kept.push(extent);
                    }
                }
                Ok(TensorType::new(stored.dtype, kept))
            }
            TensorView::Bitcast(to) => {
                let (&last, outer) = dims.split_last().ok_or_else(|| {
                    anyhow!("a tensor of type {stored} has no axis to read as {to}")
                })?;
                let bits = last * stored.dtype.bits();
                ensure!(
                    bits.is_multiple_of(to.bits()),
                    "rows of {bits} bits hold no whole number of {to} elements"
                );
                let extents = [outer, &[bits / to.bits()]].concat();
                Ok(TensorType::new(to, extents))
            }
        }
    }
}

/// Reads the checkpoint tensor of a name as a [`TensorView`] says: a parameter
/// ([`DeviceTensor::as_param`]), or `None` for a name the checkpoint does not have.
pub type TensorGetter<'a, B> = dyn FnMut(&str, TensorView) -> Result<Option<DeviceTensor<B>>> + 'a;

/// The checkpoint tensors that a model's constructors read, by module path.
///
/// Constructors in a [`crate::nn::module`] impl read through it, inside the path of each field
/// (see the macro's Construction section). So `weight: p.param()?` in a `Linear` under
/// `model.layers.0.mlp.up_proj` reads `model.layers.0.mlp.up_proj.weight`. Shapes come from the
/// checkpoint.
pub struct Params<'a, B: PortableBackend + 'static> {
    backend: Arc<B>,
    get: &'a mut TensorGetter<'a, B>,
    path: Vec<String>,
    /// Tensors read so far, which tells [`Params::optional`] whether a build read any.
    reads: usize,
    linear_input_dtype: Option<DType>,
    base_model_prefix: Option<&'static str>,
}

impl<'a, B: PortableBackend + 'static> Params<'a, B> {
    /// Reads tensors by name from `get`.
    pub fn new(backend: Arc<B>, get: &'a mut TensorGetter<'a, B>) -> Self {
        Self {
            backend,
            get,
            path: Vec::new(),
            reads: 0,
            linear_input_dtype: None,
            base_model_prefix: None,
        }
    }

    /// Sets the input dtype of every projection built from these parameters.
    pub fn with_linear_input_dtype(mut self, dtype: Option<DType>) -> Self {
        self.linear_input_dtype = dtype;
        self
    }

    /// Sets the module name of the base model, like `base_model_prefix` in `transformers`. A
    /// checkpoint saved from the base model alone names `{prefix}.h.0.ln_1.weight` as
    /// `h.0.ln_1.weight`, so a lookup falls back to the name without the prefix.
    pub fn with_base_model_prefix(mut self, prefix: &'static str) -> Self {
        self.base_model_prefix = Some(prefix);
        self
    }

    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn linear_input_dtype(&self) -> Option<DType> {
        self.linear_input_dtype
    }

    /// The current path, which names the module that the constructor builds. A constructor can
    /// start its error messages with it.
    pub fn path(&self) -> String {
        self.path.join(".")
    }

    /// The tensor at the current path.
    pub fn param(&mut self) -> Result<DeviceTensor<B>> {
        match self.optional_param()? {
            Some(tensor) => Ok(tensor),
            None => Err(MissingTensor(self.path.join(".")).into()),
        }
    }

    /// The tensor at the current path, or `None` when the checkpoint has none.
    pub fn optional_param(&mut self) -> Result<Option<DeviceTensor<B>>> {
        self.optional_view(TensorView::Stored)
    }

    /// The tensor at the current path without its axes of one element
    /// ([`TensorView::WithoutUnitAxes`]).
    pub fn param_without_unit_axes(&mut self) -> Result<DeviceTensor<B>> {
        match self.optional_view(TensorView::WithoutUnitAxes)? {
            Some(tensor) => Ok(tensor),
            None => Err(MissingTensor(self.path.join(".")).into()),
        }
    }

    /// The tensor at the current path, its bits read as elements of `dtype`
    /// ([`TensorView::Bitcast`]), or `None` when the checkpoint has none.
    pub fn optional_param_as(&mut self, dtype: DType) -> Result<Option<DeviceTensor<B>>> {
        self.optional_view(TensorView::Bitcast(dtype))
    }

    /// The tensor at the current path, read as `view` says, or `None` when the checkpoint has
    /// none.
    fn optional_view(&mut self, view: TensorView) -> Result<Option<DeviceTensor<B>>> {
        let name = self.path.join(".");
        let tensor = match (self.get)(&name, view)? {
            Some(tensor) => Some(tensor),
            None => match self
                .base_model_prefix
                .and_then(|prefix| name.strip_prefix(prefix)?.strip_prefix('.'))
            {
                Some(unprefixed) => (self.get)(unprefixed, view)?,
                None => None,
            },
        };
        self.reads += usize::from(tensor.is_some());
        Ok(tensor)
    }

    /// Builds a sublayer that the checkpoint may leave out, like a `shortcut` or a
    /// `classifier`. It is `None` when its first tensor is missing. A missing tensor after the
    /// first one is an error, since the sublayer is then only partly in the checkpoint.
    pub fn optional<T>(&mut self, build: impl FnOnce(&mut Self) -> Result<T>) -> Result<Option<T>> {
        let reads = self.reads;
        match build(self) {
            Ok(built) => Ok(Some(built)),
            Err(err) if self.reads == reads && err.is::<MissingTensor>() => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Runs `build` inside the path segment `segment`.
    pub fn scoped<T>(
        &mut self,
        segment: &str,
        build: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.path.push(segment.to_string());
        let built = build(self);
        self.path.pop();
        built
    }

    /// Builds one module per item, the `i`-th inside the path segment `i`, like the entries of a
    /// `ModuleList`.
    pub fn each<I: IntoIterator, T>(
        &mut self,
        items: I,
        mut build: impl FnMut(&mut Self, I::Item) -> Result<T>,
    ) -> Result<Vec<T>> {
        items
            .into_iter()
            .enumerate()
            .map(|(index, item)| self.scoped(&index.to_string(), |p| build(p, item)))
            .collect()
    }

    /// Enters the path of a struct field or enum variant. `segment` is `None` for a field or
    /// variant that adds no segment.
    #[doc(hidden)]
    pub fn __enter(&mut self, segment: Option<&'static str>) -> Scope<'_, 'a, B> {
        if let Some(name) = segment {
            self.path.push(name.to_string());
        }
        Scope {
            pushed: segment.is_some(),
            params: self,
        }
    }
}

/// A tensor that a layer reads and the checkpoint does not have.
#[derive(Debug, thiserror::Error)]
#[error("missing tensor '{0}' in checkpoint")]
pub struct MissingTensor(String);

/// [`Params`] inside one field's or variant's path until it drops.
#[doc(hidden)]
pub struct Scope<'s, 'a, B: PortableBackend + 'static> {
    params: &'s mut Params<'a, B>,
    pushed: bool,
}

impl<'a, B: PortableBackend + 'static> Deref for Scope<'_, 'a, B> {
    type Target = Params<'a, B>;

    fn deref(&self) -> &Self::Target {
        self.params
    }
}

impl<B: PortableBackend + 'static> DerefMut for Scope<'_, '_, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.params
    }
}

impl<B: PortableBackend + 'static> Drop for Scope<'_, '_, B> {
    fn drop(&mut self) {
        if self.pushed {
            self.params.path.pop();
        }
    }
}
