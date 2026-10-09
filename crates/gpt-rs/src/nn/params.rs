//! Builds layers from the named tensors of a checkpoint.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use anyhow::Result;

use crate::backend::PortableBackend;
use crate::tensor::{DType, DeviceTensor};

/// The checkpoint tensors that a model's constructors read, by module path.
///
/// Constructors in a [`crate::nn::module`] impl read through it, inside the path of each field
/// (see the macro's Construction section). So `weight: p.param()?` in a `Linear` under
/// `model.layers.0.mlp.up_proj` reads `model.layers.0.mlp.up_proj.weight`. Shapes come from the
/// checkpoint.
pub struct Params<'a, B: PortableBackend + 'static> {
    backend: Arc<B>,
    get: &'a mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>,
    path: Vec<String>,
    /// Tensors read so far, which tells [`Params::optional`] whether a build read any.
    reads: usize,
    linear_input_dtype: Option<DType>,
    base_model_prefix: Option<&'static str>,
}

impl<'a, B: PortableBackend + 'static> Params<'a, B> {
    /// Reads tensors by name from `get`, which returns parameters ([`DeviceTensor::as_param`]), or
    /// `None` for a name the checkpoint does not have.
    pub fn new(
        backend: Arc<B>,
        get: &'a mut dyn FnMut(&str) -> Result<Option<DeviceTensor<B>>>,
    ) -> Self {
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

    /// The tensor at the current path.
    pub fn param(&mut self) -> Result<DeviceTensor<B>> {
        match self.optional_param()? {
            Some(tensor) => Ok(tensor),
            None => Err(MissingTensor(self.path.join(".")).into()),
        }
    }

    /// The tensor at the current path, or `None` when the checkpoint has none.
    pub fn optional_param(&mut self) -> Result<Option<DeviceTensor<B>>> {
        let name = self.path.join(".");
        let tensor = match (self.get)(&name)? {
            Some(tensor) => Some(tensor),
            None => match self
                .base_model_prefix
                .and_then(|prefix| name.strip_prefix(prefix)?.strip_prefix('.'))
            {
                Some(unprefixed) => (self.get)(unprefixed)?,
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
        if let Some(segment) = segment {
            self.path.push(segment.to_string());
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
