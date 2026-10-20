//! The passes of the `arith` dialect, one file per pass.

mod expand_math;
mod expand_random;

pub use self::expand_math::ExpandMath;
pub use self::expand_random::ExpandRandom;
