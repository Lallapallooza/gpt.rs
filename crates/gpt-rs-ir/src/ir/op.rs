//! What an op is: the traits `#[op]` implements and the ones an op author implements.
//!
//! An op is a struct whose fields are its attributes. `#[op(...)]` declares its named operand,
//! result and region groups, its effects and its traits. It generates `<Op>Operands<V>` and
//! `<Op>Results<V>` structs with one field per group. A [`View`] `V` says what each element is.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use anyhow::Result;
use smallvec::SmallVec;

use crate::host::{HostScalar, HostTensor};
use crate::interp::{HostBuffer, HostValue, Unit};
use crate::ir::{
    BufferType, DType, Dense, Element, IndexEnv, Inst, Module, Region, RegionRef, TensorType, Type,
    Value, ValueRef,
};
use crate::text::Cursor;
use crate::traits::Traits;
use crate::Op;

/// What one element of an operand or result group is, under some view.
pub trait View {
    type Any;
    type Tensor;
    type Buffer;
    type Index;
    type Scalar;
    fn tensor(x: Self::Any) -> Option<Self::Tensor>;
    fn buffer(x: Self::Any) -> Option<Self::Buffer>;
    fn index(x: Self::Any) -> Option<Self::Index>;
    fn scalar(x: Self::Any) -> Option<Self::Scalar>;
}

/// A view whose elements can be turned back into `Any`, needed to flatten a group struct.
pub trait Embed: View {
    fn from_tensor(x: Self::Tensor) -> Self::Any;
    fn from_buffer(x: Self::Buffer) -> Self::Any;
    fn from_index(x: Self::Index) -> Self::Any;
    fn from_scalar(x: Self::Scalar) -> Self::Any;
}

/// Operand types, as `result_types` and `region_signatures` see them.
pub struct Types<'a>(PhantomData<&'a ()>);

impl<'a> View for Types<'a> {
    type Any = &'a Type;
    type Tensor = &'a TensorType;
    type Buffer = &'a BufferType;
    type Index = ();
    type Scalar = DType;
    fn tensor(x: &'a Type) -> Option<&'a TensorType> {
        x.as_tensor()
    }
    fn buffer(x: &'a Type) -> Option<&'a BufferType> {
        x.as_buffer()
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
    type Buffer = BufferType;
    type Index = ();
    type Scalar = DType;
    fn tensor(x: Type) -> Option<TensorType> {
        match x {
            Type::Tensor(t) => Some(t),
            _ => None,
        }
    }
    fn buffer(x: Type) -> Option<BufferType> {
        match x {
            Type::Buffer(b) => Some(b),
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
    fn from_buffer(x: BufferType) -> Type {
        Type::Buffer(x)
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
    type Buffer = &'a HostBuffer;
    type Index = i64;
    type Scalar = HostScalar;
    fn tensor(x: &'a HostValue) -> Option<&'a HostTensor> {
        x.as_tensor()
    }
    fn buffer(x: &'a HostValue) -> Option<&'a HostBuffer> {
        match x {
            HostValue::Buffer(b) => Some(b),
            _ => None,
        }
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
    type Buffer = HostBuffer;
    type Index = i64;
    type Scalar = HostScalar;
    fn tensor(x: HostValue) -> Option<HostTensor> {
        match x {
            HostValue::Tensor(t) => Some(t),
            _ => None,
        }
    }
    fn buffer(x: HostValue) -> Option<HostBuffer> {
        match x {
            HostValue::Buffer(b) => Some(b),
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
    fn from_buffer(x: HostBuffer) -> HostValue {
        HostValue::Buffer(x)
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
    type Buffer = Value;
    type Index = Value;
    type Scalar = Value;
    fn tensor(x: Value) -> Option<Value> {
        Some(x)
    }
    fn buffer(x: Value) -> Option<Value> {
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
    fn from_buffer(x: Value) -> Value {
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
    Buffer,
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

/// The block arguments a region must have and the types its terminator must yield. A region whose
/// yields are `None` may yield any types, and its op's `result_types` checks them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionSig {
    pub args: Vec<Type>,
    pub yields: Option<Vec<Type>>,
}

impl RegionSig {
    pub fn new(
        args: impl IntoIterator<Item = Type>,
        yields: impl IntoIterator<Item = Type>,
    ) -> Self {
        Self {
            args: args.into_iter().collect(),
            yields: Some(yields.into_iter().collect()),
        }
    }

    /// A region with these block arguments that may yield any types.
    pub fn args_only(args: impl IntoIterator<Item = Type>) -> Self {
        Self {
            args: args.into_iter().collect(),
            yields: None,
        }
    }
}

/// The regions of an op that declares none.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRegions;

/// The operand, result and region groups of an op, in one struct each. The ops of a family share
/// their family's; `#[family]` implements it for the family, and `#[op]` for an op of its own.
pub trait Groups: 'static {
    type Operands<V: View>;
    type Results<V: View>;
    type Regions<R>;
    /// What creating the op gives: its result for an op of one result, the list for an op of
    /// one variadic result group, nothing for an op without results, and its named result groups
    /// otherwise.
    type Created;

    const OPERANDS: &'static [Group];
    const RESULTS: &'static [Group];
    const REGIONS: &'static [&'static str];

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
    /// The results of a created op as `create` gives them.
    fn created(results: Self::Results<Values>) -> Self::Created;
    fn flatten_regions<R>(regions: Self::Regions<R>) -> Vec<R>;

    /// Whether `op` is the terminator that region `index` must end with.
    fn is_region_terminator(index: usize, op: &Op) -> bool;
}

/// The operand groups of the op `T` under the view `V`.
pub type Operands<T, V> = <<T as OpKind>::Groups as Groups>::Operands<V>;
/// The result groups of the op `T` under the view `V`.
pub type Results<T, V> = <<T as OpKind>::Groups as Groups>::Results<V>;
/// The region groups of the op `T`, each an `R`.
pub type Regions<T, R> = <<T as OpKind>::Groups as Groups>::Regions<R>;
/// What creating the op `T` gives.
pub type Created<T> = <<T as OpKind>::Groups as Groups>::Created;

/// Implemented by `#[op]`.
pub trait OpKind: Clone + PartialEq + Eq + Hash + fmt::Debug + 'static {
    /// The op's groups: its own, or its family's.
    type Groups: Groups;

    /// What the reference of the dialect shows about the op.
    const INFO: OpInfo;

    fn name(&self) -> Cow<'static, str>;
    fn effects(&self) -> Effects;
    /// The op's traits: those of its family, then those `#[op(..., traits(...))]` lists.
    fn traits(&self) -> Traits;

    /// Pushes `(name, text)` for every attribute, except an `#[attr(default)]` attribute at its
    /// default.
    fn print_attrs(&self, out: &mut Vec<(&'static str, String)>);
    /// Returns `None` when `name` is not this op. Otherwise parses the optional attribute
    /// dictionary that follows, handing a key the op does not have to `other_attr`, with its
    /// position.
    fn parse(
        name: &str,
        c: &mut Cursor<'_>,
        other_attr: &mut dyn FnMut(&str, &mut Cursor<'_>, usize) -> Result<()>,
    ) -> Option<Result<Self>>;
    /// Returns `None` unless this op is a [`SymbolOp`] named `name`. Otherwise parses its
    /// signature.
    fn parse_symbol(_name: &str, _c: &mut Cursor<'_>) -> Option<Result<ParsedSymbol<Self>>> {
        None
    }
}

/// A symbol op read from its signature, and the arguments of its body.
pub type ParsedSymbol<T> = (T, Vec<NamedArg>);

/// An argument of a region as the text writes it: its name, where the name is, and its type.
pub struct NamedArg {
    pub name: String,
    /// The byte offset of the name, for errors about it.
    pub at: usize,
    pub ty: Type,
}

/// An op that holds a constant value, which constant queries and folds read. `#[op(..., constant)]`
/// declares an op as one.
pub trait ConstantOp {
    fn value(&self) -> &Dense;
}

/// An op that defines a top-level symbol: its text form is its signature, then its body,
/// `func.func @main(%0: tensor<4xf32>, ...) -> tensor<4xf32> attributes {...} { ... }`.
/// `#[op(..., symbol)]` declares an op as one and implements this trait from its fields, through
/// which the printer and the parser write and read its signature.
pub trait SymbolOp {
    /// The symbol's name, without its `@`.
    fn symbol(&self) -> &str;
    /// Writes the signature after the op's name, up to its body: `@name(%a: T, ...) -> ...`,
    /// with `arg_names` the names of the body's arguments.
    fn print_signature(&self, arg_names: &[String], out: &mut String);
    /// The signature after the op's name, up to its body: the op, and the arguments of its body.
    fn parse_signature(c: &mut Cursor<'_>) -> Result<ParsedSymbol<Self>>
    where
        Self: Sized;
}

/// What the reference of a dialect shows about an op, which `#[op]` writes from its declaration and
/// its doc comment.
#[derive(Clone, Copy)]
pub struct OpInfo {
    pub name: &'static str,
    /// The struct of the op, and the file that defines it, relative to the workspace.
    pub rust: &'static str,
    pub file: &'static str,
    pub doc: &'static str,
    pub operands: &'static [Group],
    pub results: &'static [Group],
    pub regions: &'static [&'static str],
    pub attrs: &'static [AttrInfo],
    /// The traits of its family, then its own.
    pub traits: &'static [&'static dyn crate::traits::OpTrait],
    pub effects: Effects,
    /// The family of the op, if it is in one.
    pub family: Option<FamilyInfo>,
}

/// An attribute of an op: a field of its struct.
#[derive(Clone, Copy)]
pub struct AttrInfo {
    pub name: &'static str,
    /// The type as the reference pages name it, [`Attr::type_name`](crate::ir::Attr::type_name).
    pub type_name: fn() -> String,
    /// The type whose values the text writes, [`Attr::ITEM`](crate::ir::Attr::ITEM).
    pub item: &'static str,
    /// Whether the text leaves it out at its default.
    pub default: bool,
}

/// An op family: ops that share their operands, results, type rule and semantics, and differ in
/// what the family trait gives each, such as the `arith` op a `tensor` elementwise op applies.
#[derive(Clone, Copy)]
pub struct FamilyInfo {
    /// The family trait.
    pub name: &'static str,
    /// Its doc comment: the rules every op of the family follows.
    pub doc: &'static str,
}
/// The op's type rule, implemented in its dialect's `verify/`.
pub trait OpDef: OpKind {
    /// The op's shape and dtype rule: the result types for these operand types, or why the op is
    /// invalid.
    fn result_types(
        &self,
        ins: Operands<Self, Types<'_>>,
        cx: &TypeCx<'_, Regions<Self, Region>>,
    ) -> Result<Results<Self, OwnedTypes>>;
}

/// The op's semantics, implemented in its dialect's `interp/`. The interpreter runs it, and the
/// filetests check every pass against it. `#[op]` calls it, so an op without it does not compile.
pub trait Eval: OpKind {
    fn eval(
        &self,
        ins: Operands<Self, Host<'_>>,
        cx: &mut EvalCx<'_, Regions<Self, Region>>,
    ) -> Result<Results<Self, OwnedHost>>;
}

/// What an op author implements in addition for an op that declares regions.
pub trait RegionOpDef: OpKind + DeclaresRegions {
    fn region_signatures(&self, ins: Operands<Self, Types<'_>>)
        -> Result<Regions<Self, RegionSig>>;
}

pub struct TypeCx<'a, R> {
    pub module: &'a Module,
    pub regions: R,
}

/// Runs a region of the op being evaluated.
pub trait RegionRunner {
    fn run_region(&mut self, region: Region, args: Vec<HostValue>) -> Result<Vec<HostValue>>;
    /// The module being run.
    fn module(&self) -> &Module;
    /// The extent this run bound to the dimension symbol `sym`.
    fn sym(&self, sym: &crate::ir::Sym) -> Option<usize>;
    /// Runs the top-level function-like op named `callee` on `args` and returns its results.
    fn call(&mut self, callee: &str, args: Vec<HostValue>) -> Result<Vec<HostValue>>;
    /// The point of the enclosing `tile.block` that a body op runs at, which `tile.index` reads.
    fn index_env(&self) -> Option<&dyn IndexEnv> {
        None
    }

    /// The unit of work an op runs in: the iteration of each `loop.forall` around it.
    fn unit(&self) -> &Unit {
        static OUTSIDE: Unit = Unit::NONE;
        &OUTSIDE
    }

    /// Runs `region` once for each index from `start` up to `end`, each as a unit of one run of a
    /// forall, which may run at the same time as the others.
    fn run_units(&mut self, _region: Region, _start: i64, _end: i64) -> Result<()> {
        anyhow::bail!("only a run of a function runs units")
    }
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

    /// The module being run.
    pub fn module(&self) -> &Module {
        self.runner.module()
    }

    /// The extent this run bound to the dimension symbol `sym`.
    pub fn sym(&self, sym: &crate::ir::Sym) -> Option<usize> {
        self.runner.sym(sym)
    }

    /// Runs the top-level function-like op named `callee` on `args`.
    pub fn call(&mut self, callee: &str, args: Vec<HostValue>) -> Result<Vec<HostValue>> {
        self.runner.call(callee, args)
    }

    /// The point of the enclosing `tile.block`, for an op of its body.
    pub fn index_env(&self) -> Option<&dyn IndexEnv> {
        self.runner.index_env()
    }

    /// The unit of work the op runs in.
    pub fn unit(&self) -> &Unit {
        self.runner.unit()
    }

    /// Runs `region` once for each index from `start` up to `end`, as units of one run of a forall.
    pub fn run_units(&mut self, region: Region, start: i64, end: i64) -> Result<()> {
        self.runner.run_units(region, start, end)
    }

    /// The result types, with this run's dimensions bound.
    pub fn result_types(&self) -> &[Type] {
        self.results
    }

    /// The extents of tensor result `index`. Fails when one is a dimension this run has not
    /// bound.
    pub fn result_dims(&self, index: usize) -> Result<Vec<usize>> {
        match &self.results[index] {
            Type::Tensor(tensor) => {
                tensor
                    .static_dims()
                    .map(|dims| dims.to_vec())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "result {index} is {tensor}, whose extents this run has not bound"
                        )
                    })
            }
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
    type Buffer = FoldIn<'a>;
    type Index = FoldIn<'a>;
    type Scalar = FoldIn<'a>;
    fn tensor(x: FoldIn<'a>) -> Option<FoldIn<'a>> {
        Some(x)
    }
    fn buffer(x: FoldIn<'a>) -> Option<FoldIn<'a>> {
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

impl From<FoldIn<'_>> for Folded {
    fn from(input: FoldIn<'_>) -> Self {
        Folded::Value(input.value)
    }
}

/// Results as a folder returns them.
pub struct FoldOut;

impl View for FoldOut {
    type Any = Folded;
    type Tensor = Folded;
    type Buffer = Folded;
    type Index = Folded;
    type Scalar = Folded;
    fn tensor(x: Folded) -> Option<Folded> {
        Some(x)
    }
    fn buffer(x: Folded) -> Option<Folded> {
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
    fn from_buffer(x: Folded) -> Folded {
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

    /// `input` itself, when it has the type of the op's one result: the fold of an op that leaves
    /// its input as it is.
    pub fn identity(&self, input: FoldIn<'_>) -> Option<Folded> {
        (input.ty == self.result_type(0)).then(|| input.into())
    }

    /// The extents of tensor result `index`, when they are all static.
    pub fn result_dims(&self, index: usize) -> Option<Vec<usize>> {
        self.results[index]
            .as_tensor()?
            .static_dims()
            .map(|dims| dims.to_vec())
    }
}

/// Implemented by `#[op]` for an op that declares `regions(..)`.
pub trait DeclaresRegions {}

/// An op's own folder, for folds no trait implies. `#[fold]` on the impl registers it, with its
/// doc comment, which says what the op folds to. It never computes on elements: each result
/// becomes an existing value, or a constant made by repeating or moving constant bytes.
pub trait Fold: OpKind {
    /// The folded results, or `None` when the op does not fold.
    fn fold(
        &self,
        ins: Operands<Self, Folding<'_>>,
        cx: &FoldCx<'_>,
    ) -> Option<Results<Self, FoldOut>>;
}

/// A folder behind a signature without the op's types: the op, its operands, their segments and
/// the result types.
pub type FoldFn = fn(&dyn Any, &[FoldIn<'_>], Option<&[usize]>, &[Type]) -> Option<Vec<Folded>>;

/// A folder as `#[fold]` registers it.
pub struct FoldEntry {
    /// The `TypeId` of the op struct.
    pub op: fn() -> TypeId,
    /// The name of the op, `tensor.add`.
    pub name: &'static str,
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
    let typed = T::Groups::split_operands::<Folding<'_>>(ins.iter().cloned(), segments).ok()?;
    let out = op.fold(typed, &FoldCx { results })?;
    crate::rewrite::fired(crate::rewrite::Fired::Fold(T::INFO.name));
    Some(T::Groups::flatten_results::<FoldOut>(out).0.into_vec())
}

pub use gpt_rs_ir_macros::fold;

/// The non-generic entry points, called through [`Op`](enum@crate::Op). `#[op]` implements it on
/// top of [`OpDef`] and, for ops with regions, [`RegionOpDef`].
pub trait OpDyn {
    /// The op struct itself, for the facts that registries keep by its `TypeId`, such as its
    /// `#[cost]`.
    fn dyn_any(&self) -> &dyn Any;

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

    /// The value of a [`ConstantOp`].
    fn dyn_constant(&self) -> Option<&Dense> {
        None
    }

    /// The op as a [`SymbolOp`], when it is one.
    fn dyn_symbol(&self) -> Option<&dyn SymbolOp> {
        None
    }
}

/// An op that has a place in the [`Op`](enum@crate::Op) union. `#[derive(Dialect)]` implements it.
pub trait InOp: OpKind + Into<Op> {
    fn cast(op: &Op) -> Option<&Self>;
}

/// An op of a module as a `T`, read together with the module. Its attributes are the fields of
/// `T`, through `Deref`. It has one accessor per operand, result and region group, named as the
/// op or its family declares them. The accessors return [`ValueRef`]s and [`RegionRef`]s.
/// `#[op]` and `#[family]` implement them on the groups `G`, so code generic over the ops of a
/// family can call them too. It holds references only, so it is free to make and pass around.
pub struct Matched<'m, T: OpKind, G: Groups = <T as OpKind>::Groups> {
    inst: Inst,
    op: &'m T,
    module: &'m Module,
    pub(crate) operands: G::Operands<Refs<'m>>,
    pub(crate) results: G::Results<Refs<'m>>,
    pub(crate) regions: G::Regions<RegionRef<'m>>,
}

impl<'m, T: InOp> Matched<'m, T> {
    /// The op of `inst` as a `T`, when it is one.
    pub(crate) fn new(module: &'m Module, inst: Inst) -> Option<Self> {
        let op = T::cast(module.op(inst))?;
        let refs = |values: &[Value]| {
            values
                .iter()
                .map(|&value| module.value_ref(value))
                .collect::<Vec<_>>()
        };
        let operands = T::Groups::split_operands::<Refs<'m>>(
            refs(module.operands(inst)).into_iter(),
            module.operand_segments(inst),
        )
        .ok()?;
        let results = T::Groups::split_results::<Refs<'m>>(
            refs(module.results(inst)).into_iter(),
            module.result_segments(inst),
        )
        .ok()?;
        let regions = module
            .regions(inst)
            .iter()
            .map(|&region| module.region_ref(region))
            .collect();
        let regions = T::Groups::split_regions(regions).ok()?;
        Some(Self {
            inst,
            op,
            module,
            operands,
            results,
            regions,
        })
    }
}

impl<'m, T: OpKind, G: Groups> Matched<'m, T, G> {
    pub fn inst(&self) -> Inst {
        self.inst
    }

    /// The op's attributes.
    pub fn attrs(&self) -> &'m T {
        self.op
    }

    pub fn module(&self) -> &'m Module {
        self.module
    }

    /// Operand `index` of all the op's operands, in order: for code generic over the op.
    pub fn operand(&self, index: usize) -> ValueRef<'m> {
        self.module
            .value_ref(self.module.operands(self.inst)[index])
    }

    /// Result `index` of all the op's results, in order: for code generic over the op.
    pub fn result(&self, index: usize) -> ValueRef<'m> {
        self.module.value_ref(self.module.results(self.inst)[index])
    }
}

impl<T: OpKind, G: Groups> std::ops::Deref for Matched<'_, T, G> {
    type Target = T;

    fn deref(&self) -> &T {
        self.op
    }
}

/// Operand and result values read together with their module, as [`Matched`] holds them.
pub struct Refs<'m>(PhantomData<&'m ()>);

impl<'m> View for Refs<'m> {
    type Any = ValueRef<'m>;
    type Tensor = ValueRef<'m>;
    type Buffer = ValueRef<'m>;
    type Index = ValueRef<'m>;
    type Scalar = ValueRef<'m>;
    fn tensor(x: ValueRef<'m>) -> Option<ValueRef<'m>> {
        Some(x)
    }
    fn buffer(x: ValueRef<'m>) -> Option<ValueRef<'m>> {
        Some(x)
    }
    fn index(x: ValueRef<'m>) -> Option<ValueRef<'m>> {
        Some(x)
    }
    fn scalar(x: ValueRef<'m>) -> Option<ValueRef<'m>> {
        Some(x)
    }
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
        Kind::Buffer => "a buffer",
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
