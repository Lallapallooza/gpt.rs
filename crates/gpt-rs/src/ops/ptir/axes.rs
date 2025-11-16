//! Axis helper utilities used by the PTIR DSL.

use smallvec::SmallVec;

/// Compact representation of axis selections.
pub type Axes = SmallVec<[usize; 4]>;

/// Macro helper that collects arguments into an [`Axes`].
#[macro_export]
macro_rules! axes {
    ($($axis:expr),* $(,)?) => {{
        let mut tmp = $crate::ops::ptir::axes::Axes::new();
        $(tmp.push($axis as usize);)*
        tmp
    }};
}

/// Builds an axes iterator from any sequence.
pub fn axes_iter<I>(iter: I) -> Axes
where
    I: IntoIterator<Item = usize>,
{
    iter.into_iter().collect()
}
