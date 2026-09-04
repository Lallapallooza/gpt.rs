//! Validation macros for [`#[functional]`](crate::functional) bodies.
//!
//! Each macro checks its tensor arguments. When a check fails, the macro returns an error from the
//! functional. Each message starts with the name of the functional, which is the `FUNCTIONAL`
//! constant that the attribute defines. The message names each argument as written:
//! `gelu: x must have rank >= 1, got []`.

use anyhow::anyhow;

#[doc(hidden)]
pub fn error(functional: &str, message: std::fmt::Arguments<'_>) -> anyhow::Error {
    anyhow!("{functional}: {message}")
}

/// Returns `functional: <message>` as the error of the enclosing functional unless `cond` holds.
#[doc(hidden)]
#[macro_export]
macro_rules! __functional_ensure {
    ($cond:expr, $($message:tt)+) => {
        if !$cond {
            return ::core::result::Result::Err(::core::convert::From::from(
                $crate::ops::functional::validate::error(FUNCTIONAL, format_args!($($message)+)),
            ));
        }
    };
}

/// `ensure_rank!(x, 2)` checks that `x` has rank 2.
#[macro_export]
macro_rules! ensure_rank {
    ($tensor:expr, $rank:expr $(,)?) => {
        $crate::__functional_ensure!(
            $tensor.shape().rank() == $rank,
            "{} must have rank {}, got {:?}",
            stringify!($tensor),
            $rank,
            $tensor.shape().dims()
        )
    };
}

/// `ensure_rank_at_least!(x, 1)` checks that `x` has rank 1 or more.
#[macro_export]
macro_rules! ensure_rank_at_least {
    ($tensor:expr, $rank:expr $(,)?) => {
        $crate::__functional_ensure!(
            $tensor.shape().rank() >= $rank,
            "{} must have rank >= {}, got {:?}",
            stringify!($tensor),
            $rank,
            $tensor.shape().dims()
        )
    };
}

/// `ensure_last_dim!(x, n)` checks that the last dimension of `x` is `n`.
#[macro_export]
macro_rules! ensure_last_dim {
    ($tensor:expr, $len:expr $(,)?) => {
        $crate::__functional_ensure!(
            $tensor.shape().dims().last() == Some(&$len),
            "{} must have last dimension {}, got {:?}",
            stringify!($tensor),
            $len,
            $tensor.shape().dims()
        )
    };
}

/// `ensure_dtype!(x, F32)` or `ensure_dtype!(x, F32 | BF16)` checks that `x` has one of the
/// [`DType`](crate::tensor::DType)s.
#[macro_export]
macro_rules! ensure_dtype {
    ($tensor:expr, $($dtype:ident)|+ $(,)?) => {
        $crate::__functional_ensure!(
            matches!($tensor.dtype(), $($crate::tensor::DType::$dtype)|+),
            "{} must have dtype {}, got {:?}",
            stringify!($tensor),
            stringify!($($dtype)|+),
            $tensor.dtype()
        )
    };
}

/// `ensure_same_dtype!(a, b, ...)` checks that every tensor has the dtype of `a`.
#[macro_export]
macro_rules! ensure_same_dtype {
    ($first:expr, $($other:expr),+ $(,)?) => {
        $(
            $crate::__functional_ensure!(
                $first.dtype() == $other.dtype(),
                "{} dtype {:?} must match {} dtype {:?}",
                stringify!($other),
                $other.dtype(),
                stringify!($first),
                $first.dtype()
            );
        )+
    };
}

/// `ensure_same_shape!(a, b, ...)` checks that every tensor has the shape of `a`.
#[macro_export]
macro_rules! ensure_same_shape {
    ($first:expr, $($other:expr),+ $(,)?) => {
        $(
            $crate::__functional_ensure!(
                $first.shape().dims() == $other.shape().dims(),
                "{} shape {:?} must match {} shape {:?}",
                stringify!($other),
                $other.shape().dims(),
                stringify!($first),
                $first.shape().dims()
            );
        )+
    };
}

/// `ensure_same_backend!(a, b, ...)` checks that every tensor is on the backend instance of `a`.
#[macro_export]
macro_rules! ensure_same_backend {
    ($first:expr, $($other:expr),+ $(,)?) => {
        $(
            $crate::__functional_ensure!(
                ::std::sync::Arc::ptr_eq(&$first.backend(), &$other.backend()),
                "{} and {} must be on the same backend, got {} and {}",
                stringify!($first),
                stringify!($other),
                $first.backend().backend_name(),
                $other.backend().backend_name()
            );
        )+
    };
}
