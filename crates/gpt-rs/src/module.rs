use anyhow::{ensure, Result};

use crate::backend::spec::PortableBackend;
use crate::tensor::DeviceTensor;

pub type VisitParamsFn<'a, B> = dyn FnMut(&str, TensorRole, &DeviceTensor<B>) -> Result<()> + 'a;
pub type VisitParamsMutFn<'a, B> =
    dyn FnMut(&str, TensorRole, &mut DeviceTensor<B>) -> Result<()> + 'a;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorRole {
    Parameter,
    Buffer,
}

#[derive(Default)]
struct ParamPath {
    segments: Vec<String>,
}

impl ParamPath {
    fn push(&mut self, segment: &str) -> Result<()> {
        ensure!(
            !segment.is_empty(),
            "parameter path segments must be non-empty"
        );
        ensure!(
            !segment.contains('.'),
            "parameter path segments must not contain '.', got '{segment}'"
        );
        ensure!(
            segment.is_ascii(),
            "parameter path segments must be ASCII, got '{segment}'"
        );
        self.segments.push(segment.to_string());
        Ok(())
    }

    fn pop(&mut self) {
        let _ = self.segments.pop();
    }
}

pub struct ParamVisitor<'a, B: PortableBackend + 'static> {
    path: ParamPath,
    scratch: String,
    f: &'a mut VisitParamsFn<'a, B>,
}

impl<'a, B: PortableBackend + 'static> ParamVisitor<'a, B> {
    pub fn new(f: &'a mut VisitParamsFn<'a, B>) -> Self {
        Self {
            path: ParamPath::default(),
            scratch: String::new(),
            f,
        }
    }

    pub fn scoped(
        &mut self,
        segment: &str,
        inner: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        self.path.push(segment)?;
        let out = inner(self);
        self.path.pop();
        out
    }

    pub fn param(&mut self, leaf: &str, role: TensorRole, tensor: &DeviceTensor<B>) -> Result<()> {
        ensure!(!leaf.is_empty(), "parameter leaf names must be non-empty");
        ensure!(
            !leaf.contains('.'),
            "parameter leaf names must not contain '.', got '{leaf}'"
        );
        ensure!(
            leaf.is_ascii(),
            "parameter leaf names must be ASCII, got '{leaf}'"
        );

        self.scratch.clear();
        for (i, seg) in self.path.segments.iter().enumerate() {
            if i > 0 {
                self.scratch.push('.');
            }
            self.scratch.push_str(seg);
        }
        if !self.path.segments.is_empty() {
            self.scratch.push('.');
        }
        self.scratch.push_str(leaf);

        (self.f)(self.scratch.as_str(), role, tensor)
    }
}

pub struct ParamVisitorMut<'a, B: PortableBackend + 'static> {
    path: ParamPath,
    scratch: String,
    f: &'a mut VisitParamsMutFn<'a, B>,
}

impl<'a, B: PortableBackend + 'static> ParamVisitorMut<'a, B> {
    pub fn new(f: &'a mut VisitParamsMutFn<'a, B>) -> Self {
        Self {
            path: ParamPath::default(),
            scratch: String::new(),
            f,
        }
    }

    pub fn scoped(
        &mut self,
        segment: &str,
        inner: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        self.path.push(segment)?;
        let out = inner(self);
        self.path.pop();
        out
    }

    pub fn param(
        &mut self,
        leaf: &str,
        role: TensorRole,
        tensor: &mut DeviceTensor<B>,
    ) -> Result<()> {
        ensure!(!leaf.is_empty(), "parameter leaf names must be non-empty");
        ensure!(
            !leaf.contains('.'),
            "parameter leaf names must not contain '.', got '{leaf}'"
        );
        ensure!(
            leaf.is_ascii(),
            "parameter leaf names must be ASCII, got '{leaf}'"
        );

        self.scratch.clear();
        for (i, seg) in self.path.segments.iter().enumerate() {
            if i > 0 {
                self.scratch.push('.');
            }
            self.scratch.push_str(seg);
        }
        if !self.path.segments.is_empty() {
            self.scratch.push('.');
        }
        self.scratch.push_str(leaf);

        (self.f)(self.scratch.as_str(), role, tensor)
    }
}

pub trait Module<B: PortableBackend + 'static> {
    fn visit_params(&self, v: &mut ParamVisitor<'_, B>) -> Result<()>;
    fn visit_params_mut(&mut self, v: &mut ParamVisitorMut<'_, B>) -> Result<()>;
}
