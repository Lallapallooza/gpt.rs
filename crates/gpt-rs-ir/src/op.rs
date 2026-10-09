//! What an op is: the traits `#[derive(Op)]` implements and the ones an op author implements.
//!
//! An op is a struct whose fields are its attributes. `#[op(...)]` declares its named operand,
//! result and region groups, its effects and its traits. The derive generates `<Op>Operands<V>`
//! and `<Op>Results<V>` structs with one field per group. A [`View`] `V` says what each element is.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use anyhow::Result;
use smallvec::SmallVec;

use crate::dense::Dense;
use crate::host::{HostScalar, HostTensor, HostValue};
use crate::module::{Module, Region, Value};
use crate::text::Cursor;
use crate::traits::Traits;
use crate::types::{DType, Element, TensorType, Type};
use crate::Op;

/// What one element of an operand or result group is, under some view.
pub trait View {
    type Any;
    type Tensor;
    type Index;
    type Scalar;
    fn tensor(x: Self::Any) -> Option<Self::Tensor>;
    fn index(x: Self::Any) -> Option<Self::Index>;
    fn scalar(x: Self::Any) -> Option<Self::Scalar>;
}

/// A view whose elements can be turned back into `Any`, needed to flatten a group struct.
pub trait Embed: View {
    fn from_tensor(x: Self::Tensor) -> Self::Any;
    fn from_index(x: Self::Index) -> Self::Any;
    fn from_scalar(x: Self::Scalar) -> Self::Any;
}

/// Operand types, as `result_types` and `region_signatures` see them.
pub struct Types<'a>(PhantomData<&'a ()>);

impl<'a> View for Types<'a> {
    type Any = &'a Type;
    type Tensor = &'a TensorType;
    type Index = ();
    type Scalar = DType;
    fn tensor(x: &'a Type) -> Option<&'a TensorType> {
        x.as_tensor()
    }
    fn index(x: &'a Type) -> Option<()> {
        x.is_index().then_some(())
    }
    fn scalar(x: &'a Type) -> Option<DType> {
        x.as_scalar()
    }
}

/// Result types, as `result_types` returns them.
pub struct OwnedTypes;

impl View for OwnedTypes {
    type Any = Type;
    type Tensor = TensorType;
    type Index = ();
    type Scalar = DType;
    fn tensor(x: Type) -> Option<TensorType> {
        match x {
            Type::Tensor(t) => Some(t),
            _ => None,
        }
    }
    fn index(x: Type) -> Option<()> {
        x.is_index().then_some(())
    }
    fn scalar(x: Type) -> Option<DType> {
        x.as_scalar()
    }
}

impl Embed for OwnedTypes {
    fn from_tensor(x: TensorType) -> Type {
        Type::Tensor(x)
    }
    fn from_index(_: ()) -> Type {
        Type::Index
    }
    fn from_scalar(x: DType) -> Type {
        Type::Scalar(x)
    }
}

/// Operand values, as `eval` sees them.
pub struct Host<'a>(PhantomData<&'a ()>);

impl<'a> View for Host<'a> {
    type Any = &'a HostValue;
    type Tensor = &'a HostTensor;
    type Index = i64;
    type Scalar = HostScalar;
    fn tensor(x: &'a HostValue) -> Option<&'a HostTensor> {
        x.as_tensor()
    }
    fn index(x: &'a HostValue) -> Option<i64> {
        match x {
            HostValue::Index(i) => Some(*i),
            _ => None,
        }
    }
    fn scalar(x: &'a HostValue) -> Option<HostScalar> {
        match x {
            HostValue::Scalar(s) => Some(*s),
            _ => None,
        }
    }
}

/// Result values, as `eval` returns them.
pub struct OwnedHost;

impl View for OwnedHost {
    type Any = HostValue;
    type Tensor = HostTensor;
    type Index = i64;
    type Scalar = HostScalar;
    fn tensor(x: HostValue) -> Option<HostTensor> {
        match x {
            HostValue::Tensor(t) => Some(t),
            _ => None,
        }
    }
    fn index(x: HostValue) -> Option<i64> {
        match x {
            HostValue::Index(i) => Some(i),
            _ => None,
        }
    }
    fn scalar(x: HostValue) -> Option<HostScalar> {
        match x {
            HostValue::Scalar(s) => Some(s),
            _ => None,
        }
    }
}

impl Embed for OwnedHost {
    fn from_tensor(x: HostTensor) -> HostValue {
        HostValue::Tensor(x)
    }
    fn from_index(x: i64) -> HostValue {
        HostValue::Index(x)
    }
    fn from_scalar(x: HostScalar) -> HostValue {
        HostValue::Scalar(x)
    }
}

/// SSA values, as the builder and patterns see them. The values carry no kind. `dyn_result_types`
/// checks the kinds from the value types, in the builder and in the verifier.
pub struct Values;

impl View for Values {
    type Any = Value;
    type Tensor = Value;
    type Index = Value;
    type Scalar = Value;
    fn tensor(x: Value) -> Option<Value> {
        Some(x)
    }
    fn index(x: Value) -> Option<Value> {
        Some(x)
    }
    fn scalar(x: Value) -> Option<Value> {
        Some(x)
    }
}

impl Embed for Values {
    fn from_tensor(x: Value) -> Value {
        x
    }
    fn from_index(x: Value) -> Value {
        x
    }
    fn from_scalar(x: Value) -> Value {
        x
    }
}

/// The kind an operand or result group accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Any,
    Tensor,
    Index,
    Scalar,
}

/// How many values a group holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    One,
    /// None or one.
    Optional,
    /// Any number.
    Variadic,
}

/// One named operand or result group of an op.
#[derive(Debug, Clone, Copy)]
pub struct Group {
    pub name: &'static str,
    pub kind: Kind,
    pub arity: Arity,
}

impl Group {
    /// Whether the group holds a number of values other than one.
    pub fn is_variable(&self) -> bool {
        self.arity != Arity::One
    }
}

/// What an op does besides computing its results. The default, with no effect set, is pure.
/// Passes never merge, precompute or erase as dead an op with any effect.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Effects {
    /// Effects that the IR does not describe.
    pub opaque: bool,
    /// Draws random numbers.
    pub rng: bool,
    /// Allocates memory.
    pub alloc: bool,
    /// Operand groups (by index) whose memory the op reads.
    pub reads: u64,
    /// Operand groups (by index) whose memory the op writes.
    pub writes: u64,
}

impl Effects {
    pub fn is_pure(&self) -> bool {
        *self == Self::default()
    }
}

/// The block arguments a region must have and the types its terminator must yield.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSig {
    pub args: Vec<Type>,
    pub yields: Vec<Type>,
}

impl RegionSig {
    pub fn new(
        args: impl IntoIterator<Item = Type>,
        yields: impl IntoIterator<Item = Type>,
    ) -> Self {
        Self {
            args: args.into_iter().collect(),
            yields: yields.into_iter().collect(),
        }
    }
}

/// The regions of an op that declares none.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRegions;

/// Implemented by `#[derive(Op)]`.
pub trait OpKind: Clone + PartialEq + Eq + Hash + fmt::Debug + 'static {
    type Operands<V: View>;
    type Results<V: View>;
    type Regions<R>;

    const OPERANDS: &'static [Group];
    const RESULTS: &'static [Group];
    const REGIONS: &'static [&'static str];
    /// What the reference of the dialect shows about the op.
    const INFO: OpInfo;

    fn name(&self) -> Cow<'static, str>;
    fn effects(&self) -> Effects;
    /// The op's traits: those `#[op(..., traits(...))]` lists, then those of its kind.
    fn traits(&self) -> Traits;

    /// Pushes `(name, text)` for every attribute, except an `#[attr(default)]` attribute at its
    /// default.
    fn print_attrs(&self, out: &mut Vec<(&'static str, String)>);
    /// Returns `None` when `name` is not this op. Otherwise parses the optional attribute
    /// dictionary that follows.
    fn parse(name: &str, c: &mut Cursor<'_>) -> Option<Result<Self>>;

    fn split_operands<V: View>(
        flat: impl ExactSizeIterator<Item = V::Any>,
        segments: Option<&[usize]>,
    ) -> Result<Self::Operands<V>>;
    fn flatten_operands<V: Embed>(groups: Self::Operands<V>) -> (Flat<V::Any>, Option<Vec<usize>>);
    fn split_results<V: View>(
        flat: impl ExactSizeIterator<Item = V::Any>,
        segments: Option<&[usize]>,
    ) -> Result<Self::Results<V>>;
    fn flatten_results<V: Embed>(groups: Self::Results<V>) -> (Flat<V::Any>, Option<Vec<usize>>);
    fn split_regions<R>(flat: Vec<R>) -> Result<Self::Regions<R>>;
    fn flatten_regions<R>(regions: Self::Regions<R>) -> Vec<R>;

    /// Whether `op` is the terminator that region `index` must end with.
    fn is_region_terminator(index: usize, op: &Op) -> bool;
}

/// What the reference of a dialect shows about an op, which `#[derive(Op)]` writes from its
/// declaration and its doc comment.
#[derive(Clone, Copy)]
pub struct OpInfo {
    /// The name, or for a family the pattern with `{kind}`: `tensor.{kind}`.
    pub name: &'static str,
    /// The struct of the op, and the file that defines it, relative to the workspace.
    pub rust: &'static str,
    pub file: &'static str,
    pub doc: &'static str,
    pub operands: &'static [Group],
    pub results: &'static [Group],
    pub regions: &'static [&'static str],
    pub attrs: &'static [AttrInfo],
    /// The traits of every op of the family. Each kind may add its own.
    pub traits: &'static [&'static dyn crate::traits::OpTrait],
    pub effects: Effects,
    /// The kinds of a family, or none.
    pub kinds: fn() -> Vec<KindInfo>,
}

/// An attribute of an op: a field of its struct.
#[derive(Clone, Copy)]
pub struct AttrInfo {
    pub name: &'static str,
    /// The field's Rust type.
    pub ty: &'static str,
    /// Whether the text leaves it out at its default.
    pub default: bool,
}

/// One kind of an op family.
pub struct KindInfo {
    pub mnemonic: &'static str,
    pub doc: &'static str,
    pub traits: &'static [&'static dyn crate::traits::OpTrait],
}

/// The op's type rule, implemented next to the op in its dialect's `ir/`.
pub trait OpDef: OpKind {
    /// The op's shape and dtype rule: the result types for these operand types, or why the op is
    /// invalid.
    fn result_types(
        &self,
        ins: Self::Operands<Types<'_>>,
        cx: &TypeCx<'_, Self::Regions<Region>>,
    ) -> Result<Self::Results<OwnedTypes>>;
}

/// The op's semantics, implemented in its dialect's `interp/`. The interpreter runs it, and the
/// filetests check every pass against it. `#[derive(Op)]` calls it, so an op without it does not
/// compile.
pub trait Eval: OpKind {
    fn eval(
        &self,
        ins: Self::Operands<Host<'_>>,
        cx: &mut EvalCx<'_, Self::Regions<Region>>,
    ) -> Result<Self::Results<OwnedHost>>;
}

/// What an op author implements in addition for an op that declares regions.
pub trait RegionOpDef: OpKind + DeclaresRegions {
    fn region_signatures(&self, ins: Self::Operands<Types<'_>>)
        -> Result<Self::Regions<RegionSig>>;
}

pub struct TypeCx<'a, R> {
    pub module: &'a Module,
    pub regions: R,
}

/// Runs a region of the op being evaluated.
pub trait RegionRunner {
    fn run_region(&mut self, region: Region, args: Vec<HostValue>) -> Result<Vec<HostValue>>;
}

pub struct EvalCx<'a, R> {
    pub regions: R,
    pub(crate) results: &'a [Type],
    pub(crate) runner: &'a mut dyn RegionRunner,
}

impl<R> EvalCx<'_, R> {
    /// Runs `region` with `args` as its block arguments and returns what its terminator yields.
    pub fn run(&mut self, region: Region, args: Vec<HostValue>) -> Result<Vec<HostValue>> {
        self.runner.run_region(region, args)
    }

    /// The extents of tensor result `index`. Fails when one is a dimension this run has not
    /// bound.
    pub fn result_dims(&self, index: usize) -> Result<Vec<usize>> {
        match &self.results[index] {
            Type::Tensor(tensor) => tensor.static_dims().map(SmallVec::into_vec).ok_or_else(|| {
                anyhow::anyhow!("result {index} is {tensor}, whose extents this run has not bound")
            }),
            other => anyhow::bail!("result {index} is {other}, not a tensor"),
        }
    }
}

/// An operand as a folder sees it.
#[derive(Debug, Clone, Copy)]
pub struct FoldIn<'a> {
    pub value: Value,
    pub ty: &'a Type,
    pub constant: Option<&'a Dense>,
}

impl FoldIn<'_> {
    /// Whether the operand is a constant whose every element is `value`, compared by bits.
    pub fn is_splat<T: Element>(&self, value: T) -> bool {
        self.constant.is_some_and(|c| c.is_splat(value))
    }
}

/// Operands as a folder sees them.
pub struct Folding<'a>(PhantomData<&'a ()>);

impl<'a> View for Folding<'a> {
    type Any = FoldIn<'a>;
    type Tensor = FoldIn<'a>;
    type Index = FoldIn<'a>;
    type Scalar = FoldIn<'a>;
    fn tensor(x: FoldIn<'a>) -> Option<FoldIn<'a>> {
        Some(x)
    }
    fn index(x: FoldIn<'a>) -> Option<FoldIn<'a>> {
        Some(x)
    }
    fn scalar(x: FoldIn<'a>) -> Option<FoldIn<'a>> {
        Some(x)
    }
}

/// What one result of a folded op becomes.
#[derive(Debug, Clone, PartialEq)]
pub enum Folded {
    /// A value the module already has.
    Value(Value),
    /// A new constant.
    Constant(Dense),
}

/// Results as a folder returns them.
pub struct FoldOut;

impl View for FoldOut {
    type Any = Folded;
    type Tensor = Folded;
    type Index = Folded;
    type Scalar = Folded;
    fn tensor(x: Folded) -> Option<Folded> {
        Some(x)
    }
    fn index(x: Folded) -> Option<Folded> {
        Some(x)
    }
    fn scalar(x: Folded) -> Option<Folded> {
        Some(x)
    }
}

impl Embed for FoldOut {
    fn from_tensor(x: Folded) -> Folded {
        x
    }
    fn from_index(x: Folded) -> Folded {
        x
    }
    fn from_scalar(x: Folded) -> Folded {
        x
    }
}

pub struct FoldCx<'a> {
    pub(crate) results: &'a [Type],
}

impl FoldCx<'_> {
    pub fn result_type(&self, index: usize) -> &Type {
        &self.results[index]
    }

    /// The extents of tensor result `index`, when they are all static.
    pub fn result_dims(&self, index: usize) -> Option<Vec<usize>> {
        self.results[index]
            .as_tensor()?
            .static_dims()
            .map(SmallVec::into_vec)
    }
}

/// Implemented by `#[derive(Op)]` for an op that declares `regions(..)`.
pub trait DeclaresRegions {}

/// An op's own folder, for folds no trait implies. `#[fold]` on the impl registers it, with its
/// doc comment, which says what the op folds to. It never computes on elements: each result
/// becomes an existing value, or a constant made by repeating or moving constant bytes.
pub trait Fold: OpKind {
    /// The folded results, or `None` when the op does not fold.
    fn fold(
        &self,
        ins: Self::Operands<Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<Self::Results<FoldOut>>;
}

/// A folder behind a signature without the op's types: the op, its operands, their segments and
/// the result types.
pub type FoldFn = fn(&dyn Any, &[FoldIn<'_>], Option<&[usize]>, &[Type]) -> Option<Vec<Folded>>;

/// A folder as `#[fold]` registers it.
pub struct FoldEntry {
    /// The `TypeId` of the op struct.
    pub op: fn() -> TypeId,
    /// The name of the op struct, and the file of the folder, relative to the workspace.
    pub rust: &'static str,
    pub file: &'static str,
    pub doc: &'static str,
    pub fold: FoldFn,
}

/// Every folder that `#[fold]` registers.
#[linkme::distributed_slice]
pub static FOLDS: [FoldEntry];

/// The registered folder of the op struct whose `TypeId` is `op`.
pub fn folder_of(op: TypeId) -> Option<FoldFn> {
    FOLDS
        .iter()
        .find(|entry| (entry.op)() == op)
        .map(|entry| entry.fold)
}

/// `T`'s folder as a [`FoldFn`], which `#[fold]` registers.
pub fn fold_erased<T: Fold + 'static>(
    op: &dyn Any,
    ins: &[FoldIn<'_>],
    segments: Option<&[usize]>,
    results: &[Type],
) -> Option<Vec<Folded>> {
    let op = op.downcast_ref::<T>()?;
    let typed = T::split_operands::<Folding<'_>>(ins.iter().cloned(), segments).ok()?;
    let out = op.fold(typed, &FoldCx { results })?;
    crate::doc::fired(crate::doc::Fired::Fold(crate::doc::short_name::<T>()));
    Some(T::flatten_results::<FoldOut>(out).0.into_vec())
}

pub use gpt_rs_ir_macros::fold;

/// The non-generic entry points, called through [`Op`](enum@crate::Op). `#[derive(Op)]` implements
/// it on top of [`OpDef`] and, for ops with regions, [`RegionOpDef`].
pub trait OpDyn {
    fn dyn_result_types(
        &self,
        ins: &[&Type],
        segments: Option<&[usize]>,
        regions: &[Region],
        module: &Module,
    ) -> Result<(Flat<Type>, Option<Vec<usize>>)>;

    fn dyn_region_signatures(
        &self,
        ins: &[&Type],
        segments: Option<&[usize]>,
    ) -> Result<Vec<RegionSig>>;

    /// The op's `#[fold]` folder applied to these operands, or `None` when it has no folder or
    /// does not fold.
    fn dyn_fold(
        &self,
        ins: &[FoldIn<'_>],
        segments: Option<&[usize]>,
        results: &[Type],
    ) -> Option<Vec<Folded>>;

    /// `results` are the result types with the run's dynamic dimensions bound.
    fn dyn_eval(
        &self,
        ins: &[&HostValue],
        segments: Option<&[usize]>,
        regions: &[Region],
        results: &[Type],
        runner: &mut dyn RegionRunner,
    ) -> Result<Vec<HostValue>>;
}

/// An op that has a place in the [`Op`](enum@crate::Op) union. `#[derive(Dialect)]` implements it.
pub trait InOp: OpKind + Into<Op> {
    fn cast(op: &Op) -> Option<&Self>;
}

/// An op found in a module, with its attributes and named operand and result groups. It owns its
/// data, so a pattern can hold it while it edits the module.
pub struct Matched<T: OpKind> {
    pub inst: crate::module::Inst,
    pub attrs: T,
    pub operands: T::Operands<Values>,
    pub results: T::Results<Values>,
}

/// The values of an op's operand or result groups in one list, inline up to four.
pub type Flat<T> = SmallVec<[T; 4]>;

/// The size of each group in a flat list of `count` values. With at most one optional or variadic
/// group, or when every optional or variadic group is empty, `count` gives the sizes. Otherwise
/// `segments` must give them.
pub fn group_sizes(
    groups: &[Group],
    count: usize,
    segments: Option<&[usize]>,
    what: &str,
) -> Result<Flat<usize>> {
    if let Some(segments) = segments {
        anyhow::ensure!(
            segments.len() == groups.len(),
            "expected {} {what} segments, got {}",
            groups.len(),
            segments.len()
        );
        anyhow::ensure!(
            segments.iter().sum::<usize>() == count,
            "{what} segments {segments:?} do not add up to {count}"
        );
        for (group, &size) in groups.iter().zip(segments) {
            ensure_size(group, size, what)?;
        }
        return Ok(SmallVec::from_slice(segments));
    }
    let fixed = groups.iter().filter(|g| !g.is_variable()).count();
    let variable = groups.len() - fixed;
    match variable {
        0 => {
            anyhow::ensure!(count == fixed, "expected {fixed} {what}s, got {count}");
            Ok(SmallVec::from_elem(1, groups.len()))
        }
        1 => {
            anyhow::ensure!(
                count >= fixed,
                "expected at least {fixed} {what}s, got {count}"
            );
            let sizes: Flat<usize> = groups
                .iter()
                .map(|g| if g.is_variable() { count - fixed } else { 1 })
                .collect();
            for (group, &size) in groups.iter().zip(&sizes) {
                ensure_size(group, size, what)?;
            }
            Ok(sizes)
        }
        _ if count == fixed => Ok(groups
            .iter()
            .map(|g| usize::from(!g.is_variable()))
            .collect()),
        _ => anyhow::bail!(
            "an op with {variable} optional or variadic {what} groups needs explicit segments"
        ),
    }
}

/// Checks that `group` can hold `size` values.
fn ensure_size(group: &Group, size: usize, what: &str) -> Result<()> {
    match group.arity {
        Arity::One => anyhow::ensure!(
            size == 1,
            "{what} group `{}` takes exactly one value, got {size}",
            group.name
        ),
        Arity::Optional => anyhow::ensure!(
            size <= 1,
            "{what} group `{}` takes at most one value, got {size}",
            group.name
        ),
        Arity::Variadic => {}
    }
    Ok(())
}

/// Takes the next element of a group and projects it to the group's kind.
pub fn take_one<T, A>(
    items: &mut impl Iterator<Item = A>,
    project: impl FnOnce(A) -> Option<T>,
    group: &Group,
    what: &str,
) -> Result<T> {
    let item = items.next().expect("group sizes match the element count");
    project(item)
        .ok_or_else(|| anyhow::anyhow!("{what} `{}` must be {}", group.name, kind_name(group.kind)))
}

/// Takes the element of an optional group that holds `n` (0 or 1) and projects it to the group's
/// kind.
pub fn take_optional<T, A>(
    items: &mut impl Iterator<Item = A>,
    n: usize,
    project: impl FnOnce(A) -> Option<T>,
    group: &Group,
    what: &str,
) -> Result<Option<T>> {
    (n == 1)
        .then(|| take_one(items, project, group, what))
        .transpose()
}

/// Takes the next `n` elements of a variadic group and projects each to the group's kind.
pub fn take_many<T, A>(
    items: &mut impl Iterator<Item = A>,
    n: usize,
    project: impl Fn(A) -> Option<T>,
    group: &Group,
    what: &str,
) -> Result<Vec<T>> {
    (0..n)
        .map(|_| take_one(items, &project, group, what))
        .collect()
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Any => "a value",
        Kind::Tensor => "a tensor",
        Kind::Index => "an index",
        Kind::Scalar => "a scalar",
    }
}

/// Segments to store for a flattened group struct: only needed with more than one optional or
/// variadic group, one of which holds a value.
pub fn segments_if_needed(groups: &[Group], sizes: &[usize]) -> Option<Vec<usize>> {
    let variable = groups.iter().zip(sizes).filter(|(g, _)| g.is_variable());
    let (count, filled) = variable.fold((0, false), |(count, filled), (_, &size)| {
        (count + 1, filled || size > 0)
    });
    (count > 1 && filled).then(|| sizes.to_vec())
}
