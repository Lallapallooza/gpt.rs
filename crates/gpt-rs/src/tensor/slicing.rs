//! Python-style indexing: [`s!`](crate::s) builds the index that [`DeviceTensor::slice`] takes.
//!
//! [`DeviceTensor::slice`]: super::DeviceTensor::slice

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use gpt_rs_ir::dialects::tensor::Slice;
use smallvec::SmallVec;

/// One entry of a Python-style index, as [`s!`](crate::s) writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisIndex {
    /// The elements from `start` to `end` of an axis, every `step`-th: Python's `start:end:step`.
    /// A negative bound counts from the end, a missing one is the start or the end of the axis, and
    /// bounds clamp to the axis.
    Range {
        start: Option<i64>,
        end: Option<i64>,
        step: usize,
    },
    /// One element, which drops its axis: Python's integer index. A negative position counts from
    /// the end.
    At(i64),
    /// As many whole axes as the other entries leave: Python's `...`.
    Ellipsis,
    /// A new axis of extent 1: Python's `None`.
    NewAxis,
}

impl AxisIndex {
    /// This range, taking every `step`-th element.
    #[track_caller]
    pub fn step(self, step: usize) -> Self {
        let AxisIndex::Range { start, end, .. } = self else {
            panic!("only a range takes a step, not {self:?}");
        };
        assert!(step > 0, "a step must be positive, got 0");
        AxisIndex::Range { start, end, step }
    }
}

/// A position on an axis: a `usize` computed from a shape, or a literal, which may be negative.
pub trait Position: Copy {
    fn position(self) -> i64;
}

impl Position for usize {
    fn position(self) -> i64 {
        self as i64
    }
}

impl Position for i32 {
    fn position(self) -> i64 {
        self.into()
    }
}

impl Position for i64 {
    fn position(self) -> i64 {
        self
    }
}

fn range(start: Option<i64>, end: Option<i64>) -> AxisIndex {
    AxisIndex::Range {
        start,
        end,
        step: 1,
    }
}

/// The exclusive end after the inclusive `end`. The last element, -1, ends at the axis' end.
fn after(end: i64) -> Option<i64> {
    (end != -1).then_some(end + 1)
}

impl From<RangeFull> for AxisIndex {
    fn from(_: RangeFull) -> Self {
        range(None, None)
    }
}

impl<T: Position> From<Range<T>> for AxisIndex {
    fn from(r: Range<T>) -> Self {
        range(Some(r.start.position()), Some(r.end.position()))
    }
}

impl<T: Position> From<RangeFrom<T>> for AxisIndex {
    fn from(r: RangeFrom<T>) -> Self {
        range(Some(r.start.position()), None)
    }
}

impl<T: Position> From<RangeTo<T>> for AxisIndex {
    fn from(r: RangeTo<T>) -> Self {
        range(None, Some(r.end.position()))
    }
}

impl<T: Position> From<RangeInclusive<T>> for AxisIndex {
    fn from(r: RangeInclusive<T>) -> Self {
        range(Some(r.start().position()), after(r.end().position()))
    }
}

impl<T: Position> From<RangeToInclusive<T>> for AxisIndex {
    fn from(r: RangeToInclusive<T>) -> Self {
        range(None, after(r.end.position()))
    }
}

impl<T: Position> From<T> for AxisIndex {
    fn from(position: T) -> Self {
        AxisIndex::At(position.position())
    }
}

/// A Python-style index for [`DeviceTensor::slice`](crate::tensor::DeviceTensor::slice):
/// `x.slice(s![.., 1..3])` is `x[:, 1:3]`.
///
/// - `..`, `a..b`, `a..`, `..b`, `a..=b` and `..=b` take ranges, and `a..b;2` every second element
///   of one, Python's `a:b:2`;
/// - a single position takes one element and drops its axis;
/// - `...` stands for as many whole axes as the other entries leave, and `None` adds an axis of
///   extent 1.
///
/// Negative positions count from the end, range bounds clamp to the axis, and the axes that the
/// index leaves out at the end are taken whole, all as in Python.
#[macro_export]
macro_rules! s {
    (@index [$($done:expr,)*]) => { [$($done),*] };
    (@index [$($done:expr,)*] ... $(, $($rest:tt)*)?) => {
        $crate::s!(@index [$($done,)* $crate::tensor::AxisIndex::Ellipsis,] $($($rest)*)?)
    };
    (@index [$($done:expr,)*] None $(, $($rest:tt)*)?) => {
        $crate::s!(@index [$($done,)* $crate::tensor::AxisIndex::NewAxis,] $($($rest)*)?)
    };
    (@index [$($done:expr,)*] $range:expr ; $step:expr $(, $($rest:tt)*)?) => {
        $crate::s!(
            @index [$($done,)* $crate::tensor::AxisIndex::from($range).step($step),]
            $($($rest)*)?
        )
    };
    (@index [$($done:expr,)*] $entry:expr $(, $($rest:tt)*)?) => {
        $crate::s!(@index [$($done,)* $crate::tensor::AxisIndex::from($entry),] $($($rest)*)?)
    };
    ($($entries:tt)*) => { $crate::s!(@index [] $($entries)*) };
}

/// The strided slice that `index` takes of a tensor of `dims`, and the extents of the result once
/// the axes of single positions are dropped and new axes added.
#[track_caller]
pub(super) fn resolve(dims: &[usize], index: &[AxisIndex]) -> (Slice, SmallVec<[usize; 4]>) {
    let rank = dims.len();
    let taken = index
        .iter()
        .filter(|entry| matches!(entry, AxisIndex::Range { .. } | AxisIndex::At(_)))
        .count();
    let ellipses = index
        .iter()
        .filter(|entry| **entry == AxisIndex::Ellipsis)
        .count();
    assert!(
        ellipses <= 1,
        "an index holds one `...` at most, got {ellipses}"
    );
    assert!(
        taken <= rank,
        "an index of {taken} axes does not fit a tensor of rank {rank}"
    );
    // The entries with `...` spelled out as whole axes, and the axes left out at the end added.
    let whole = range(None, None);
    let mut entries: SmallVec<[AxisIndex; 8]> = SmallVec::new();
    for &entry in index {
        match entry {
            AxisIndex::Ellipsis => entries.extend(std::iter::repeat_n(whole, rank - taken)),
            entry => entries.push(entry),
        }
    }
    if ellipses == 0 {
        entries.extend(std::iter::repeat_n(whole, rank - taken));
    }
    let (mut starts, mut sizes, mut strides, mut out) = (
        SmallVec::new(),
        SmallVec::new(),
        SmallVec::<[usize; 4]>::new(),
        SmallVec::new(),
    );
    let mut axis = 0;
    for entry in entries {
        match entry {
            AxisIndex::NewAxis => out.push(1),
            AxisIndex::At(position) => {
                let extent = dims[axis] as i64;
                let at = if position < 0 {
                    position + extent
                } else {
                    position
                };
                assert!(
                    (0..extent).contains(&at),
                    "position {position} is out of range for axis {axis} of extent {extent}"
                );
                starts.push(at as usize);
                sizes.push(1);
                strides.push(1);
                axis += 1;
            }
            AxisIndex::Range { start, end, step } => {
                let extent = dims[axis] as i64;
                let clamp = |bound: i64| {
                    if bound < 0 {
                        (bound + extent).max(0)
                    } else {
                        bound.min(extent)
                    }
                };
                let start = start.map_or(0, clamp);
                let end = end.map_or(extent, clamp);
                let size = ((end - start).max(0) as usize).div_ceil(step);
                // An empty range starts at 0, which fits any axis.
                starts.push(if size == 0 { 0 } else { start as usize });
                sizes.push(size);
                strides.push(step);
                out.push(size);
                axis += 1;
            }
            AxisIndex::Ellipsis => unreachable!("spelled out above"),
        }
    }
    if strides.iter().all(|&step| step == 1) {
        strides.clear();
    }
    let slice = Slice {
        starts,
        sizes,
        strides,
    };
    (slice, out)
}
