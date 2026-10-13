//! What the reference pages show of a rewrite: its name, doc comment and file, and where it fires.

use std::any::type_name;
use std::cell::RefCell;

/// The name, doc comment and file of a pattern or an op trait, which `#[pattern]` and
/// `#[derive(Doc)]` write from the struct.
pub trait Doc {
    /// The name of the struct.
    fn name(&self) -> &'static str;
    fn doc(&self) -> &'static str;
    /// The file that defines the struct, relative to the workspace.
    fn file(&self) -> &'static str;
}

/// A rewrite that changed an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fired {
    /// The `#[fold]` folder of the op of this name.
    Fold(&'static str),
    /// The fold of the op trait of this name.
    TraitFold(&'static str),
    /// The pattern of this name.
    Pattern(&'static str),
}

thread_local! {
    static FIRED: RefCell<Option<Vec<Fired>>> = const { RefCell::new(None) };
}

/// Runs `f`, and returns its result with the rewrites that fired on this thread meanwhile, in
/// order.
pub fn record_fired<R>(f: impl FnOnce() -> R) -> (R, Vec<Fired>) {
    let outer = FIRED.with(|fired| fired.replace(Some(Vec::new())));
    let result = f();
    let fired = FIRED.with(|fired| fired.replace(outer)).unwrap_or_default();
    (result, fired)
}

/// Notes that `rewrite` fired, when [`record_fired`] is recording on this thread.
pub(crate) fn fired(rewrite: Fired) {
    FIRED.with(|fired| {
        if let Some(list) = fired.borrow_mut().as_mut() {
            list.push(rewrite);
        }
    });
}

/// The name of the type `T` without its module path.
pub(crate) fn short_name<T: ?Sized>() -> &'static str {
    let name = type_name::<T>();
    name.rsplit("::").next().unwrap_or(name)
}
