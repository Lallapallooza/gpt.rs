//! Linear algebra primitives expressed through backend DotGeneral operations.
//!
//! At the moment the module focuses on matrix multiplication with support for batched inputs and
//! appropriate validation around shapes, dtypes, and backend ownership.

use anyhow::{bail, Result};
use gpt_rs_macros::{capture_ptir, support_runtime_overload};

use crate::backend::spec::{PortableBackend, ValueId};
use crate::ops::functional::common::{
    ensure_same_backend, ensure_same_dtype, CaptureIntoDeviceTensor,
};
use crate::ops::ptir::{self, DotAttrs, DotDims};
use crate::tensor::DeviceTensor;

struct MatmulPlan {
    dot_dims: DotDims,
    output_shape: Vec<usize>,
    requires_grad: bool,
}

/// Validates shapes/dtypes for 2D and batched 3D matmul and derives the PTIR dot-general spec.
///
/// Guards dtype/back-end consistency, checks contraction axes, and bails on unsupported ranks.
/// Tests: `functional_softmax.rs::matmul_emits_dot_general_instruction` plus Torch parity
/// (workspace + backend suites).
fn validate_matmul<B: PortableBackend + 'static>(
    a: &DeviceTensor<B>,
    b: &DeviceTensor<B>,
) -> Result<MatmulPlan> {
    ensure_same_dtype("matmul lhs", a, "rhs", b)?;

    ensure_same_backend("matmul", a, b)?;

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
        requires_grad: a.requires_grad_flag() || b.requires_grad_flag(),
    })
}

/// Captures the PTIR `dot_general` given a validated matmul plan.
fn capture_matmul<B: PortableBackend + 'static>(
    plan: MatmulPlan,
    a: &DeviceTensor<B>,
    b: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let tensor = capture_ptir!({ a, b }, |_session| -> anyhow::Result<ValueId> {
        let result = a.dot_general(&b, &plan.dot_dims, &DotAttrs::default());
        Ok::<ValueId, anyhow::Error>(result.id())
    })?
    .into_device_tensor(plan.requires_grad)?;

    debug_assert_eq!(tensor.shape().dims(), plan.output_shape.as_slice());
    debug_assert_eq!(tensor.dtype(), a.dtype());

    Ok(tensor)
}

/// Performs matrix multiplication (or batched matmul) between `a` and `b`.
/// Shape and dtype checks mirror the expectations of GPT projection layers while remaining
/// portable across backends.
///
/// Execution order:
/// - verify that operand dtypes, backends, and contraction dimensions align with the requested variant;
/// - derive the backend `DotGeneral` spec, including batch and contract axes;
/// - import operands into the active graph arena (or spawn a new one) and emit the dot product node;
/// - wrap the resulting value identifier in a [`DeviceTensor`] carrying the inferred shape and dtype.
#[support_runtime_overload]
pub fn matmul<B: PortableBackend + 'static>(
    _backend: &B,
    a: &DeviceTensor<B>,
    b: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_matmul(a, b)?;
    capture_matmul(plan, a, b)
}
