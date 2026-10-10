//! The patterns of `c-pack-weights`: the weights of the dots that C kernels compute, put in the
//! layout each kernel reads.

use std::collections::HashSet;

use anyhow::Result;
use gpt_rs_ir::dialects::tensor::{DotGeneral, ExtractPatches, Reshape, Transpose};
use gpt_rs_ir::module::{Inst, Module, Value, ValueDef};
use gpt_rs_ir::op::Matched;
use gpt_rs_ir::rewrite::{pattern, Pattern, Rewriter};
use gpt_rs_ir::types::{DType, Dim};

use crate::kernels::{LINEAR_PACK_MIN_ROWS, PANEL_N};

/// The extents of `value` when it is an f32 tensor with static extents.
fn f32_dims(module: &Module, value: Value) -> Option<Vec<usize>> {
    let ty = module.ty(value).as_tensor()?;
    let dims = ty.static_dims().filter(|_| ty.dtype == DType::F32)?;
    Some(dims.into_vec())
}

/// Rewrites an f32 `x [.., K] . w`, which codegen sends to the GEMM (a convolution's patches
/// among them), into `reshape(x . packed)`, where `packed` is `w` in panels of `PANEL_N` columns,
/// `[N / PANEL_N, K, PANEL_N]`. `w` is `[K, N]` or `[N, K]`. The packed dot gives
/// `[.., N / PANEL_N, PANEL_N]`, the same elements as `[.., N]`.
///
/// It packs only a static `w` whose packed form takes no more bytes than the parameters it is
/// computed from, so `outline-static` moves the packing to load time. It skips a `w` whose `N` is
/// not a multiple of `PANEL_N`, since the panels would pad it. A run-time `x` with one row, or with
/// fewer than `LINEAR_PACK_MIN_ROWS` rows against an `[N, K]` weight, leaves `w` unpacked: the GEMM
/// or the dot-product kernels read `w` in place.
#[pattern]
pub(super) struct PackGemmWeight;

impl Pattern for PackGemmWeight {
    type Root = DotGeneral;

    fn rewrite(&self, dot: Matched<DotGeneral>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let attrs = &dot.attrs;
        let x = dot.operands.lhs;
        let Some(x_dims) = f32_dims(rw.module, x) else {
            return Ok(false);
        };
        let Some(out_dims) = f32_dims(rw.module, dot.results.output) else {
            return Ok(false);
        };
        let Some(w) = rw.module.ty(dot.operands.rhs).as_tensor() else {
            return Ok(false);
        };
        let axes = attrs.axes(x_dims.len(), w.rank())?;
        let nk = match axes.contract_rhs.as_slice() {
            [0] => false,
            [1] => true,
            _ => return Ok(false),
        };
        let last = x_dims.len().saturating_sub(1);
        let plain = axes.batch_lhs.is_empty()
            && axes.batch_rhs.is_empty()
            && x_dims.len() >= 2
            && axes.contract_lhs[..] == [last]
            && matches!(attrs.accum, None | Some(DType::F32));
        let m: usize = x_dims[..last].iter().product();
        // `dot_kernel` sends fewer than LINEAR_PACK_MIN_ROWS rows of x against an [N, K] w to the
        // dot-product kernels, and the GEMM multiplies a single row in place.
        let gemm = m > 1 && !(nk && m < LINEAR_PACK_MIN_ROWS);
        // A dot of two static operands moves to load time whole.
        let run_time = !rw.module.is_static(x);
        if !plain || !gemm || !run_time {
            return Ok(false);
        }
        let Some(packed) = pack(rw, dot.inst, dot.operands.rhs, nk)? else {
            return Ok(false);
        };
        let panels = DotGeneral {
            accum: attrs.accum,
            out: attrs.out,
            ..DotGeneral::contract([last as i64], [1])
        };
        let at = dot.inst;
        let product = rw.insert_before(at, panels, (x, packed))?.output;
        let shape = out_dims.into_iter().map(Dim::Static).collect();
        let out = rw.insert_before(at, Reshape { shape }, product)?.output;
        rw.replace(at, &[out])?;
        Ok(true)
    }
}

/// `w` packed as `[N / PANEL_N, K, PANEL_N]`, built before `at`, where `w` is f32 `[N, K]` when
/// `nk` and `[K, N]` otherwise. `None` when `outline-static` would leave the packing at run time:
/// `w` is not static, or the packed `w` takes more bytes than the parameters it is computed from,
/// as when N is not a multiple of PANEL_N and the panels pad it with zeros.
fn pack(rw: &mut Rewriter<'_>, at: Inst, w: Value, nk: bool) -> Result<Option<Value>> {
    let Some(&[rows, cols]) = f32_dims(rw.module, w).as_deref() else {
        return Ok(None);
    };
    let (n, k) = if nk { (rows, cols) } else { (cols, rows) };
    if n % PANEL_N != 0 || !fits_parameters(rw.module, w) {
        return Ok(None);
    }
    // [N, K] -> [N / P, P, K] -> [N / P, K, P], or [K, N] -> [K, N / P, P] -> [N / P, K, P].
    let (split, perm) = if nk {
        ([n / PANEL_N, PANEL_N, k], vec![0, 2, 1])
    } else {
        ([k, n / PANEL_N, PANEL_N], vec![1, 0, 2])
    };
    let shape = split.into_iter().map(Dim::Static).collect();
    let split = rw.insert_before(at, Reshape { shape }, w)?.output;
    let perm = perm.into();
    let packed = rw.insert_before(at, Transpose { perm }, split)?.output;
    Ok(Some(packed))
}

/// Whether `w` is static and takes no more bytes than the parameters it is computed from.
/// `outline-static` stores no other value.
fn fits_parameters(module: &Module, w: Value) -> bool {
    if !module.is_static(w) {
        return false;
    }
    let bytes = |value: Value| {
        let ty = module.ty(value).as_tensor()?;
        ty.element_count()?.checked_mul(ty.dtype.size_in_bytes()?)
    };
    let mut seen = HashSet::new();
    let mut stack = vec![w];
    let mut params = Some(0usize);
    while let Some(value) = stack.pop() {
        if !seen.insert(value) {
            continue;
        }
        match module.def(value) {
            ValueDef::Arg(..) => {
                params = params
                    .zip(bytes(value))
                    .and_then(|(sum, b)| sum.checked_add(b));
            }
            ValueDef::Result(inst, _) => stack.extend(module.operands(inst)),
        }
    }
    params
        .zip(bytes(w))
        .is_some_and(|(have, need)| need <= have)
}

/// Lays out the weight of a grouped convolution for the grouped kernel. The dot of the patches
/// `[N, OH, OW, KH, KW, G, C_in / G]` with the weight `[G, C_out / G, C_in / G, KH, KW]`, batched
/// over the groups, becomes the same dot with the weight transposed to
/// `[KH, KW, C_in / G, G, C_out / G]`, the order of `[KH, KW, C_in / G, C_out]` that the kernel
/// reads. The result keeps its axes. `outline-static` moves the transpose of a static weight to
/// load time.
#[pattern]
pub(super) struct GroupedConvWeight;

impl Pattern for GroupedConvWeight {
    type Root = DotGeneral;

    fn rewrite(&self, dot: Matched<DotGeneral>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let attrs = &dot.attrs;
        let grouped = attrs.batch_lhs[..] == [5]
            && attrs.batch_rhs[..] == [0]
            && attrs.contract_lhs[..] == [3, 4, 6]
            && attrs.contract_rhs[..] == [3, 4, 2];
        let (lhs, rhs) = (dot.operands.lhs, dot.operands.rhs);
        let from_patches = rw.def_as::<Reshape>(lhs).is_some_and(|reshape| {
            rw.def_as::<ExtractPatches>(reshape.operands.input)
                .is_some()
        });
        let weight = f32_dims(rw.module, rhs).is_some_and(|dims| dims.len() == 5);
        if !grouped || !from_patches || !weight {
            return Ok(false);
        }
        let at = dot.inst;
        let perm = [3, 4, 2, 0, 1].into_iter().collect();
        let w = rw.insert_before(at, Transpose { perm }, rhs)?.output;
        let laid_out = DotGeneral {
            batch_rhs: [3].into_iter().collect(),
            contract_rhs: [0, 1, 2].into_iter().collect(),
            ..attrs.clone()
        };
        let out = rw.insert_before(at, laid_out, (lhs, w))?.output;
        rw.replace(at, &[out])?;
        Ok(true)
    }
}
