//! Matrix products: [`matmul`] and the [`linear`] projection, and [`linear_weight`], which finds
//! the projections that [`linear`] records.

use gpt_rs_ir::dialects::tensor::{Binary, BinaryKind, BroadcastTo, Cast, DotGeneral};
use gpt_rs_ir::module::{Inst, Module, Value};

use crate::tensor::DType;
use crate::{ensure_dtype, ensure_rank, ensure_same_dtype, functional};

/// Performs matrix multiplication (or batched matmul) between `a` and `b`.
///
/// Rank-2 operands multiply as `[m, k] x [k, n]`. Rank-3 operands multiply per batch, as
/// `[b, m, k] x [b, k, n]`.
#[functional]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    ensure_same_dtype!(a, b);
    let (a_dims, b_dims) = (a.shape().dims(), b.shape().dims());
    let dims = match (a_dims.len(), b_dims.len()) {
        (2, 2) => DotGeneral::contract([1], [0]),
        (3, 3) => {
            assert!(
                a_dims[0] == b_dims[0],
                "{FUNCTIONAL}: batch {} of a must match batch {} of b",
                a_dims[0],
                b_dims[0]
            );
            DotGeneral::contract([2], [1]).batch([0], [0])
        }
        (a_rank, b_rank) => {
            panic!("{FUNCTIONAL}: a and b must both have rank 2 or 3, got {a_rank} and {b_rank}")
        }
    };
    let (k_a, k_b) = (a_dims[a_dims.len() - 1], b_dims[b_dims.len() - 2]);
    assert!(
        k_a == k_b,
        "{FUNCTIONAL}: the last dimension {k_a} of a must match dimension {k_b} of b"
    );
    a.dot_general(b, dims)
}

/// Layout of the weight of a [`linear`] projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// `[out_features, in_features]`, like PyTorch `nn.Linear`.
    OutIn,
    /// `[in_features, out_features]`, like `transformers`' `Conv1D`.
    InOut,
}

/// Linear projection of `x` (`[N, in_features]`) to `[N, out_features]` with an f32 result:
/// `x @ weight^T` for [`WeightLayout::OutIn`], `x @ weight` for [`WeightLayout::InOut`].
///
/// The functional casts `weight` to the dtype of `x`, so a bf16 `x` rounds f32 and f16 weights to
/// bf16. The products accumulate in f32. A backend can fuse the `cast` into the `dot_general` to
/// read reduced-precision weights in place.
#[functional]
pub fn linear(x: &Tensor, weight: &Tensor, layout: WeightLayout) -> Tensor {
    ensure_rank!(x, 2);
    ensure_rank!(weight, 2);
    ensure_dtype!(x, F32 | Bf16);
    ensure_dtype!(weight, F32 | Bf16 | F16 | F8E4M3);
    let weight_in_axis = match layout {
        WeightLayout::OutIn => 1,
        WeightLayout::InOut => 0,
    };
    let (in_features, weight_in) = (x.size(1), weight.size(weight_in_axis));
    assert!(
        in_features == weight_in,
        "{FUNCTIONAL}: weight in_features {weight_in} must match x in_features {in_features}"
    );
    let dims = DotGeneral::contract([1], [weight_in_axis]).with_dtypes(DType::F32, DType::F32);
    x.dot_general(&weight.cast(x.dtype()), dims)
}

/// A projection that [`linear`] recorded, with its weight as stored: the operands of a kernel that
/// reads the weight in place instead of a widened copy.
pub struct LinearWeight {
    pub layout: WeightLayout,
    pub x: Value,
    pub weight: Value,
    /// The one-element scale of an FP8 weight that `nn::Linear` dequantizes.
    pub scale: Option<Value>,
    /// The ops between the stored weight and the `dot_general`: the casts, and the multiply and
    /// the broadcast of the scale.
    pub chain: Vec<Inst>,
}

impl LinearWeight {
    /// The dtypes of `x`, the weight and the scale.
    pub fn dtypes(&self, module: &Module) -> (DType, DType, Option<DType>) {
        let dtype = |value| {
            module
                .ty(value)
                .as_tensor()
                .expect("a matched tensor")
                .dtype
        };
        (dtype(self.x), dtype(self.weight), self.scale.map(dtype))
    }
}

/// The projection whose `dot_general` is `inst`: `x [M, K]` times a weight operand `[N, K]` or
/// `[K, N]`, with f32 sums and an f32 result. The weight operand is a stored weight `w`,
/// `cast(w)`, or the dequantized `cast(w) * broadcast_to(s)` of `nn::Linear` with `s` one element,
/// in either operand order and cast to the dtype of `x` or not. `x` and `w` have static extents.
pub fn linear_weight(module: &Module, inst: Inst) -> Option<LinearWeight> {
    let dot = module.op_as::<DotGeneral>(inst)?;
    let attrs = &dot.attrs;
    let layout = match attrs.contract_rhs.as_slice() {
        [1] => WeightLayout::OutIn,
        [0] => WeightLayout::InOut,
        _ => return None,
    };
    if !attrs.batch_lhs.is_empty()
        || !attrs.batch_rhs.is_empty()
        || attrs.contract_lhs[..] != [1]
        || !matches!(attrs.accum, None | Some(DType::F32))
        || module.ty(dot.results.output).as_tensor()?.dtype != DType::F32
    {
        return None;
    }
    let mut chain = Vec::new();
    let mut read = dot.operands.rhs;
    if let Some(cast) = module.def_as::<Cast>(read) {
        chain.push(cast.inst);
        read = cast.operands.input;
    }
    let (weight, scale) = match dequantized(module, read, &mut chain) {
        Some((weight, scale)) => (weight, Some(scale)),
        None => (read, None),
    };
    let x = dot.operands.lhs;
    let static_rank2 = |value| {
        let dims = module.ty(value).as_tensor().and_then(|ty| ty.static_dims());
        dims.is_some_and(|dims| dims.len() == 2)
    };
    (static_rank2(x) && static_rank2(weight)).then_some(LinearWeight {
        layout,
        x,
        weight,
        scale,
        chain,
    })
}

/// `(w, s)` when `value` is `cast(w) * broadcast_to(s)` in either operand order, with `s` one
/// element. Adds the three ops to `chain`.
fn dequantized(module: &Module, value: Value, chain: &mut Vec<Inst>) -> Option<(Value, Value)> {
    let mul = module.def_as::<Binary>(value)?;
    if mul.attrs.kind != BinaryKind::Mul {
        return None;
    }
    let (lhs, rhs) = (mul.operands.lhs, mul.operands.rhs);
    let (cast, broadcast) = match module.def_as::<Cast>(lhs) {
        Some(cast) => (cast, rhs),
        None => (module.def_as::<Cast>(rhs)?, lhs),
    };
    let broadcast = module.def_as::<BroadcastTo>(broadcast)?;
    let scale = broadcast.operands.input;
    let elements = module.ty(scale).as_tensor()?.element_count()?;
    if elements != 1 {
        return None;
    }
    chain.extend([mul.inst, cast.inst, broadcast.inst]);
    Some((cast.operands.input, scale))
}
