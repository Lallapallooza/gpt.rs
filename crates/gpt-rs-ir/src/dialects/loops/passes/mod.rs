//! The passes of the `loop` dialect, one file per pass, and its canonical patterns.

mod canonicalize;
mod contract;

pub use self::contract::Contract;
