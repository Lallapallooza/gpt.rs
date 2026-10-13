//! Building and rewriting IR. [`Builder`] builds ops into a module directly. A [`Pattern`] reads
//! its root op as a [`Matched`] or a [`WithTrait`], and changes the module only through its
//! [`Rewriter`], which stages the changes; the driver applies them when the pattern returns `Ok`.
//! A pattern that does not apply returns [`NoMatch`], with the reason, and changes nothing.

mod build;
mod builder;
mod doc;
mod editor;
mod fold;
mod greedy;
mod rewriter;

use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use anyhow::{bail, Result};
pub use gpt_rs_ir_macros::{matched, pattern};

pub use self::build::{Build, Chain};
pub use self::builder::Builder;
pub use self::doc::*;
pub(crate) use self::editor::Editor;
pub use self::fold::*;
pub use self::greedy::{Greedy, GreedyReport, Rewrites};
pub use self::rewriter::{Rewriter, Staged};
use crate::ir::{InOp, Inst, Matched, Module, Value, ValueRef};
use crate::traits::OpTrait;
use crate::Op;

/// Why a pattern does not apply to its root. A literal reason costs nothing; a formatted one is
/// formatted only when the pattern fails.
#[derive(Debug, Clone)]
pub struct NoMatch(Cow<'static, str>);

impl NoMatch {
    /// The reason when the op is not the one a pattern is rooted at.
    const NOT_ROOT: &'static str = "the op is not the pattern's root";

    fn is_not_root(&self) -> bool {
        self.0 == Self::NOT_ROOT
    }

    /// The reason `args` gives, formatted only when it has arguments.
    pub fn because(args: fmt::Arguments<'_>) -> Self {
        match args.as_str() {
            Some(reason) => Self(Cow::Borrowed(reason)),
            None => Self(Cow::Owned(args.to_string())),
        }
    }
}

impl From<&'static str> for NoMatch {
    fn from(reason: &'static str) -> Self {
        Self(Cow::Borrowed(reason))
    }
}

impl fmt::Display for NoMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Returns `Err(NoMatch)` with the reason from the pattern unless `cond` holds:
/// `require!(weight.known_before_run(), "the weight changes at run time")`, or unless a pattern
/// matches, binding its names: `require!(let Some(dims) = value.static_dims(), "...")`. The
/// reason may take format arguments, which it formats only on failure.
#[macro_export]
macro_rules! require {
    (let $pattern:pat = $value:expr, $($reason:tt)+) => {
        let $pattern = $value else {
            return ::std::result::Result::Err($crate::rewrite::NoMatch::because(
                ::std::format_args!($($reason)+),
            ));
        };
    };
    ($cond:expr, $($reason:tt)+) => {
        if !$cond {
            return ::std::result::Result::Err($crate::rewrite::NoMatch::because(
                ::std::format_args!($($reason)+),
            ));
        }
    };
}

pub use crate::require;

/// An instruction, or what holds one: an op a pattern matched.
pub trait AsInst {
    fn inst(&self) -> Inst;
}

impl AsInst for Inst {
    fn inst(&self) -> Inst {
        *self
    }
}

impl<T: crate::ir::OpKind, G: crate::ir::Groups> AsInst for Matched<'_, T, G> {
    fn inst(&self) -> Inst {
        Matched::inst(self)
    }
}

impl<T: OpTrait> AsInst for WithTrait<'_, T> {
    fn inst(&self) -> Inst {
        self.inst
    }
}

impl<A: AsInst> AsInst for &A {
    fn inst(&self) -> Inst {
        (*self).inst()
    }
}

/// Values in order, as edits take them: one value, or a list of them.
pub trait IntoValues {
    fn into_values(self) -> Vec<Value>;
}

impl IntoValues for Value {
    fn into_values(self) -> Vec<Value> {
        vec![self]
    }
}

impl IntoValues for ValueRef<'_> {
    fn into_values(self) -> Vec<Value> {
        vec![self.id()]
    }
}

impl<V: Into<Value>> IntoValues for Vec<V> {
    fn into_values(self) -> Vec<Value> {
        self.into_iter().map(Into::into).collect()
    }
}

impl<V: Into<Value> + Copy> IntoValues for &[V] {
    fn into_values(self) -> Vec<Value> {
        self.iter().map(|&value| value.into()).collect()
    }
}

impl<V: Into<Value>, const N: usize> IntoValues for [V; N] {
    fn into_values(self) -> Vec<Value> {
        self.into_iter().map(Into::into).collect()
    }
}

/// What a pattern is rooted at: ops of one type, or every op that has a trait ([`Has`]).
pub trait Root {
    /// The root as the pattern reads it.
    type Matched<'m>;
    fn find(module: &Module, inst: Inst) -> Option<Self::Matched<'_>>;
    fn info() -> RootInfo;
}

/// What a pattern is rooted at, as the reference pages show it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootInfo {
    /// The ops of this name: `tile.dispatch`.
    Op(&'static str),
    /// Every op with the op trait of this name.
    Trait(&'static str),
}

impl<T: InOp> Root for T {
    type Matched<'m> = Matched<'m, T>;
    fn find(module: &Module, inst: Inst) -> Option<Matched<'_, T>> {
        module.op_as(inst)
    }
    fn info() -> RootInfo {
        RootInfo::Op(T::INFO.name)
    }
}

/// Roots a pattern at every op that has the trait `T`.
pub struct Has<T>(PhantomData<T>);

/// An op found by its trait `T`, read together with its module.
pub struct WithTrait<'m, T: 'static> {
    pub inst: Inst,
    pub op: &'m Op,
    /// The op's `T`, with its parameters.
    pub fact: &'static T,
    pub operands: Vec<ValueRef<'m>>,
    pub results: Vec<ValueRef<'m>>,
}

impl<T: OpTrait> Root for Has<T> {
    type Matched<'m> = WithTrait<'m, T>;
    fn find(module: &Module, inst: Inst) -> Option<WithTrait<'_, T>> {
        let op = module.op(inst);
        let fact = op.get_trait::<T>()?;
        let refs = |values: &[Value]| values.iter().map(|&v| module.value_ref(v)).collect();
        Some(WithTrait {
            inst,
            op,
            fact,
            operands: refs(module.operands(inst)),
            results: refs(module.results(inst)),
        })
    }
    fn info() -> RootInfo {
        RootInfo::Trait(short_name::<T>())
    }
}

/// A local rewrite rooted at `Root`. `#[pattern]` on the struct implements [`Doc`] from its doc
/// comment, and `#[pattern(canonical)]` also registers it with `canonicalize`.
pub trait Pattern: Doc {
    type Root: Root;
    /// Rewrites `root` through `rewriter` and returns `Ok`, or returns why it does not apply. It
    /// reads the module through `root` and the values it reaches; the rewriter stages what it
    /// changes.
    fn rewrite<'m>(
        &self,
        root: <Self::Root as Root>::Matched<'m>,
        rewriter: &mut Rewriter<'m>,
    ) -> Result<(), NoMatch>;
}

/// A [`Pattern`] whose root type is erased, so different patterns fit one list.
pub trait AnyPattern: Doc {
    /// The changes the pattern stages on `inst`, or why it does not apply.
    fn try_rewrite(&self, module: &Module, inst: Inst) -> Result<rewriter::Staged, NoMatch>;
    fn root(&self) -> RootInfo;
}

impl<P: Pattern> AnyPattern for P {
    fn try_rewrite(&self, module: &Module, inst: Inst) -> Result<rewriter::Staged, NoMatch> {
        let root = P::Root::find(module, inst).ok_or(NoMatch::from(NoMatch::NOT_ROOT))?;
        let mut rewriter = Rewriter::new(module, inst);
        self.rewrite(root, &mut rewriter)?;
        Ok(rewriter.finish())
    }
    fn root(&self) -> RootInfo {
        P::Root::info()
    }
}

/// What applying a pattern set to an op did.
pub enum Applied {
    /// The pattern of this name rewrote the op, and these instructions changed.
    Rewritten {
        pattern: &'static str,
        touched: Vec<Inst>,
    },
    /// No pattern applied, for these reasons, each after its pattern's name.
    NoMatch(Vec<String>),
}

/// The patterns a greedy rewrite tries on each op, in order.
#[derive(Default)]
pub struct PatternSet {
    patterns: Vec<Box<dyn AnyPattern>>,
}

impl PatternSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `pattern` at the end.
    pub fn add(&mut self, pattern: impl AnyPattern + 'static) -> &mut Self {
        self.patterns.push(Box::new(pattern));
        self
    }

    /// Adds the boxed `pattern` at the end.
    pub fn add_boxed(&mut self, pattern: Box<dyn AnyPattern>) -> &mut Self {
        self.patterns.push(pattern);
        self
    }

    /// Tries the patterns on `inst` in order, and applies what the first that applies staged.
    /// With `explain`, it keeps the reasons the patterns rooted at the op gave for not applying.
    pub fn apply(&self, module: &mut Module, inst: Inst, explain: bool) -> Result<Applied> {
        let mut reasons = Vec::new();
        for pattern in self.iter() {
            let staged = match pattern.try_rewrite(module, inst) {
                Ok(staged) => staged,
                Err(reason) => {
                    if explain && !reason.is_not_root() {
                        reasons.push(format!("{}: {reason}", pattern.name()));
                    }
                    continue;
                }
            };
            if staged.is_empty() {
                bail!(
                    "pattern {} returned Ok but changed nothing; a pattern that does not apply \
                     returns NoMatch",
                    pattern.name()
                );
            }
            let mut editor = Editor::new(module);
            editor.apply(staged)?;
            fired(Fired::Pattern(pattern.name()));
            return Ok(Applied::Rewritten {
                pattern: pattern.name(),
                touched: editor.touched,
            });
        }
        Ok(Applied::NoMatch(reasons))
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn AnyPattern> {
        self.patterns.iter().map(Box::as_ref)
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// The canonical patterns, each added by the function that `#[pattern(canonical)]` registers.
#[linkme::distributed_slice]
pub static CANONICAL_PATTERNS: [fn(&mut PatternSet)];

/// Every canonical pattern, in the order of their names: the patterns of `canonicalize`.
pub fn canonical_patterns() -> PatternSet {
    let mut set = PatternSet::new();
    for add in CANONICAL_PATTERNS {
        add(&mut set);
    }
    set.patterns.sort_by_key(|pattern| pattern.name());
    set
}
