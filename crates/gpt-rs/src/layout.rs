//! Tensor layout helpers.
//!
//! The portable kernels and models operate on a small set of canonical layouts.
//! This module defines shared layout enums and axis permutation helpers without
//! introducing wrapper tensor types.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout4D {
    NCHW,
    NHWC,
}

impl Layout4D {
    pub const fn perm_nchw_to_nhwc(self) -> [usize; 4] {
        let _ = self;
        [0, 2, 3, 1]
    }

    pub const fn perm_nhwc_to_nchw(self) -> [usize; 4] {
        let _ = self;
        [0, 3, 1, 2]
    }
}
