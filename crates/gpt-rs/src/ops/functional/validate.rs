//! Validation macros for [`#[functional]`](crate::functional) bodies.
//!
//! Each macro checks its tensor arguments and panics when a check fails, as a shape that does not
//! fit is a bug in the caller. Each message starts with the name of the functional, which is the
//! `FUNCTIONAL` constant that the attribute defines. The message names each argument as written:
//! `gelu: x must have rank >= 1, got []`.

/// Panics with `functional: <message>` unless `cond` holds.
#[doc(hidden)]
#[macro_export]
macro_rules! __functional_ensure {
    ($cond:expr, $($message:tt)+) => {
        if !$cond {
            ::core::panic!("{}: {}", FUNCTIONAL, format_args!($($message)+));
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

/// `ensure_dtype!(x, F32)` or `ensure_dtype!(x, F32 | Bf16)` checks that `x` has one of the
/// [`DType`](crate::tensor::DType)s.
#[macro_export]
macro_rules! ensure_dtype {
    ($tensor:expr, $($dtype:ident)|+ $(,)?) => {
        $crate::__functional_ensure!(
            matches!($tensor.dtype(), $($crate::tensor::DType::$dtype)|+),
            "{} must have dtype {}, got {}",
            stringify!($tensor),
            [$($crate::tensor::DType::$dtype.to_string()),+].join(" | "),
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
                "{} dtype {} must match {} dtype {}",
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
