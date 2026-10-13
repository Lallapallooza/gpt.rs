//! The core of the IR, which every dialect is made of:
//!
//! - the module and its values;
//! - what an op is;
//! - the value types;
//! - attributes and their text forms;
//! - the index language that `tile` and `loop` share;
//! - locations;
//! - the verifier.
//!
//! Every public item of these modules is exported here, as `gpt_rs_ir::ir::Module`.

mod attr;
mod index;
mod location;
mod module;
mod op;
mod scope;
mod types;
mod verify;

pub use self::attr::*;
pub use self::index::*;
pub use self::location::*;
pub use self::module::*;
pub use self::op::*;
pub(crate) use self::scope::ScopedMap;
pub use self::types::*;
pub use self::verify::*;
