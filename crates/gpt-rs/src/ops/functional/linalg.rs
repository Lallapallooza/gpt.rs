//! Linear algebra primitives expressed through backend DotGeneral operations.
//!
//! At the moment the module focuses on matrix multiplication with support for batched inputs and
//! appropriate validation around shapes, dtypes, and backend ownership.

use anyhow::{bail, ensure, Result};

use crate::backend::spec::{DType, PortableBackend};
use crate::ops::ptir::{self, DotAttrs, DotDims};
use crate::tensor::spec_utils::backend_dtype;
use crate::tensor::DeviceTensor;
use crate::{
    capture, ensure_dtype, ensure_rank, ensure_same_backend, ensure_same_dtype, functional,
};

struct MatmulPlan {
    dot_dims: DotDims,
    output_shape: Vec<usize>,
}

/// Checks the ranks and contraction dimensions of a 2D or batched 3D matmul and derives the PTIR
/// dot-general spec.
fn matmul_plan<B: PortableBackend + 'static>(
    a: &DeviceTensor<B>,
    b: &DeviceTensor<B>,
) -> Result<MatmulPlan> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    let (dot_dims, result_dims) = match (a_shape.rank(), b_shape.rank()) {
        (2, 2) => {
            let m = a_shape.dims()[0];
            let k_a = a_shape.dims()[1];
            let k_b = b_shape.dims()[0];
            let n = b_shape.dims()[1];

            if k_a != k_b {
                bail!(
                    "matmul contract dimension mismatch: lhs {} vs rhs {}",
                    k_a,
                    k_b
                );
            }

            (
                DotDims::new(ptir::axes_iter([]), crate::axes!(1), crate::axes!(0)),
                vec![m, n],
            )
        }
        (3, 3) => {
            let batch = a_shape.dims()[0];
            if b_shape.dims()[0] != batch {
                bail!(
                    "matmul batch dimension mismatch: lhs {} vs rhs {}",
                    batch,
                    b_shape.dims()[0]
                );
            }

            let m = a_shape.dims()[1];
            let k_a = a_shape.dims()[2];
            let k_b = b_shape.dims()[1];
            let n = b_shape.dims()[2];

            if k_a != k_b {
                bail!(
                    "batched matmul contract dimension mismatch: lhs {} vs rhs {}",
                    k_a,
                    k_b
                );
            }

            (
                DotDims::new(crate::axes!(0), crate::axes!(2), crate::axes!(1)),
                vec![batch, m, n],
            )
        }
        (lhs_rank, rhs_rank) => {
            bail!(
                "matmul expects rank-2 or rank-3 tensors; got ranks {} and {}",
                lhs_rank,
                rhs_rank
            )
        }
    };

    Ok(MatmulPlan {
        dot_dims,
        output_shape: result_dims,
    })
}

/// Performs matrix multiplication (or batched matmul) between `a` and `b`.
///
/// Rank-2 operands multiply as `[m, k] x [k, n]`. Rank-3 operands multiply per batch, as
/// `[b, m, k] x [b, k, n]`.
#[functional]
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    ensure_same_dtype!(a, b);
    ensure_same_backend!(a, b);
    let plan = matmul_plan(a, b)?;
    let tensor = capture!(|a, b| a.dot_general(&b, &plan.dot_dims, &DotAttrs::default()))?;
    debug_assert_eq!(tensor.shape().dims(), plan.output_shape.as_slice());
    debug_assert_eq!(tensor.dtype(), a.dtype());
    Ok(tensor)
}

/// PyTorch-style linear projection `y = x @ weight^T` with an f32 result. `weight` is stored as
/// `[out_features, in_features]`.
///
/// The functional casts `weight` to the dtype of `x`, so a bf16 `x` rounds f32 and f16 weights to
/// bf16. The products accumulate in f32. Backends can fuse the `cast` into the
/// `dot_general` to read reduced-precision weights directly.
#[functional]
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    ensure_same_backend!(x, weight);
    ensure_rank!(x, 2);
    ensure_rank!(weight, 2);
    ensure_dtype!(x, F32 | BF16);
    ensure_dtype!(weight, F32 | BF16 | F16);
    let (in_features, weight_in) = (x.shape().dims()[1], weight.shape().dims()[1]);
    ensure!(
        in_features == weight_in,
        "{FUNCTIONAL}: weight in_features {weight_in} must match x in_features {in_features}"
    );
    let cast_to = (weight.dtype() != x.dtype()).then(|| backend_dtype(x.dtype()));
    capture!(|x, weight| {
        let weight = match cast_to {
            Some(dtype) => weight.cast(dtype),
            None => weight,
        };
        x.dot_general(
            &weight,
            &DotDims::new(ptir::axes_iter([]), crate::axes!(1), crate::axes!(1)),
            &DotAttrs {
                accum_dtype: Some(DType::F32),
                out_dtype: Some(DType::F32),
            },
        )
    })
}
