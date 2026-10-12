//! The passes of the `tile` dialect.

mod canonicalize;
mod seal_dispatches;

pub use self::seal_dispatches::SealDispatches;
