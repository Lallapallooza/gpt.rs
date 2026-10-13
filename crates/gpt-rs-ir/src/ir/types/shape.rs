//! Extents and axes: dimension symbols, the extents of an axis, lists indexed by axis and
//! permutations of axes.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{ensure, Result};
use tinyvec::ArrayVec;

use crate::ir::TensorType;

/// Name of a dimension whose extent is known only at run time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Sym(Arc<str>);

impl Sym {
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Arc::from(name.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One tensor axis: a static extent, `4`, or the extent a dimension symbol binds, `{T}`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    #[attr(format = "$0")]
    Static(usize),
    #[attr(format = "`{` $0 `}`")]
    Dynamic(Sym),
}

/// An extent of 0, which fills the unused entries of a [`Ranked`] list.
impl Default for Dim {
    fn default() -> Self {
        Dim::Static(0)
    }
}

/// The largest rank of a tensor type.
pub const MAX_RANK: usize = 8;

/// A list indexed by axis, such as the extents of a tensor type or the axes of an attribute: at
/// most [`MAX_RANK`] entries, inline.
pub type Ranked<T> = ArrayVec<[T; MAX_RANK]>;

/// `items` as a [`Ranked`] list. Fails when there are more than [`MAX_RANK`].
pub fn ranked<T: Default>(items: impl IntoIterator<Item = T>) -> Result<Ranked<T>> {
    let mut list = Ranked::new();
    let mut rank = 0;
    for item in items {
        rank += 1;
        if rank <= MAX_RANK {
            list.push(item);
        }
    }
    ensure!(
        rank <= MAX_RANK,
        "rank {rank} is above the largest rank, {MAX_RANK}"
    );
    Ok(list)
}

impl Dim {
    pub fn as_static(&self) -> Option<usize> {
        match self {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic(_) => None,
        }
    }

    /// Whether the axis has extent 1, which broadcasting repeats and reshapes add or drop freely.
    pub fn is_one(&self) -> bool {
        *self == Dim::Static(1)
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Static(n)
    }
}

/// A permutation of axes: result axis `i` is input axis `self[i]`, as `tensor.transpose` reads
/// its input. It may name axes out of range or twice until [`Perm::is_permutation_of`] checks it.
#[derive(crate::Attr, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[attr(format = "$0")]
pub struct Perm(Ranked<usize>);

impl Perm {
    /// The permutation that keeps each of `rank` axes in place.
    pub fn identity(rank: usize) -> Self {
        Self((0..rank).collect())
    }

    /// Whether every result axis reads the input axis of the same index.
    pub fn is_identity(&self) -> bool {
        self.0.iter().copied().eq(0..self.0.len())
    }

    /// Whether it names every axis below `rank` once.
    pub fn is_permutation_of(&self, rank: usize) -> bool {
        let mut seen = [false; MAX_RANK];
        self.0.len() == rank
            && self
                .0
                .iter()
                .all(|&axis| axis < rank && !std::mem::replace(&mut seen[axis], true))
    }

    /// The permutation that undoes this one: transposing by `self` and then by the inverse gives
    /// the input back.
    pub fn inverse(&self) -> Self {
        let mut inverse = self.0;
        for (result_axis, &input_axis) in self.0.iter().enumerate() {
            inverse[input_axis] = result_axis;
        }
        Self(inverse)
    }

    /// The single permutation that transposing by `first` and then by `self` amounts to.
    pub fn after(&self, first: &Perm) -> Self {
        Self(self.apply(&first.0))
    }

    /// `items` in the order this permutation reads them: `items[self[i]]` for each result axis
    /// `i`, such as the extents of the transposed type.
    pub fn apply<T: Clone + Default>(&self, items: &[T]) -> Ranked<T> {
        let mut applied = Ranked::new();
        for &axis in &self.0 {
            applied.push(items[axis].clone());
        }
        applied
    }

    /// Whether the result axes `axes`, listed in increasing order, read input axes in increasing
    /// order: the transpose keeps their order.
    pub fn keeps_order(&self, axes: &[usize]) -> bool {
        let read = self.read_axes(axes);
        read.windows(2).all(|pair| pair[0] < pair[1])
    }

    /// The input axes that the result axes `axes` read.
    pub fn read_axes(&self, axes: &[usize]) -> Ranked<usize> {
        let mut read = Ranked::new();
        for &axis in axes {
            read.push(self.0[axis]);
        }
        read
    }
}

impl std::ops::Deref for Perm {
    type Target = [usize];

    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl<'a> IntoIterator for &'a Perm {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Panics when there are more than [`MAX_RANK`] axes.
impl FromIterator<usize> for Perm {
    #[track_caller]
    fn from_iter<I: IntoIterator<Item = usize>>(axes: I) -> Self {
        match ranked(axes) {
            Ok(axes) => Self(axes),
            Err(err) => panic!("permutation: {err}"),
        }
    }
}

/// Dimension symbols bound to extents: what a call binds when its arguments meet the argument
/// types of its callee.
#[derive(Debug, Clone, Default)]
pub struct SymBindings(HashMap<Sym, Dim>);

impl SymBindings {
    /// Binds the symbols of `param` to the extents of `arg`. Fails when the types differ in
    /// dtype, rank or a static extent, or when a symbol is already bound to another extent.
    pub fn bind(&mut self, param: &TensorType, arg: &TensorType) -> Result<()> {
        ensure!(
            param.dtype == arg.dtype && param.rank() == arg.rank(),
            "{arg} does not match {param}"
        );
        for (p, a) in param.dims.iter().zip(&arg.dims) {
            match p {
                Dim::Static(_) => ensure!(p == a, "{arg} does not match {param}"),
                Dim::Dynamic(sym) => {
                    let previous = self.0.entry(sym.clone()).or_insert_with(|| a.clone());
                    ensure!(
                        previous == a,
                        "{{{}}} is bound to both {previous} and {a}",
                        sym.as_str()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, sym: &Sym) -> Option<&Dim> {
        self.0.get(sym)
    }

    /// `ty` with each bound symbol replaced by its extent.
    pub fn resolve(&self, ty: &TensorType) -> TensorType {
        let dims = ty
            .dims
            .iter()
            .map(|dim| match dim {
                Dim::Dynamic(sym) => self.0.get(sym).cloned().unwrap_or_else(|| dim.clone()),
                Dim::Static(_) => dim.clone(),
            })
            .collect();
        ty.with_dims(dims)
    }
}
