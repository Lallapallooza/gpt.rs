//! The faer linear kernel for f32 or bf16 activations and a weight in a narrower stored dtype. It
//! reads the weight in place and widens it to f32 as it goes, so no f32 copy of the weight exists.
//! A weight with a per-tensor scale widens to its stored value times the scale, rounded as
//! [`Scaled`] says. bf16 activations widen to f32 before the products, which loses nothing.
//!
//! Up to [`MATVEC_ROWS`] activation rows against an `[N, K]` weight take a matrix-vector loop that
//! widens each weight element in registers. Other shapes take blocks of weight rows widened into a
//! per-thread scratch buffer and multiplied by faer's matmul.

use std::borrow::Cow;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::Range;

use anyhow::{anyhow, Result};
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::reborrow::ReborrowMut;
use faer::traits::pulp::{Arch, Simd, WithSimd};
use faer::{Accum, Par};
use gemm_common::cache::CACHE_INFO;
use gpt_rs::ops::functional::WeightLayout;
use gpt_rs_ir::host::{Buffer, HostTensor, TensorData};
use gpt_rs_ir::ir::{DType, TensorType, F8E4M3};
use half::bf16;

use crate::{dims_of, output_tensor};

/// The most activation rows the matrix-vector loop keeps in registers at once. Up to this many,
/// it widens every weight element once, as the matmul path does, without the scratch round trip.
/// The loop loads all rows of `x` again for every weight row. Past this many rows, the matmul
/// path loads less, because its register tiles reuse each load of `x` for several weight rows.
const MATVEC_ROWS: usize = 4;

/// How the kernel turns a stored weight element into the f32 it multiplies.
trait Widen: Copy + Send + Sync {
    type Stored: Copy + Send + Sync;
    fn widen(self, stored: Self::Stored) -> f32;
}

/// The stored value.
#[derive(Clone, Copy)]
struct Decoded<W>(PhantomData<W>);

impl Widen for Decoded<bf16> {
    type Stored = bf16;

    #[inline(always)]
    fn widen(self, stored: bf16) -> f32 {
        // A bf16 is the upper half of an f32.
        f32::from_bits(u32::from(stored.to_bits()) << 16)
    }
}

impl Widen for Decoded<F8E4M3> {
    type Stored = F8E4M3;

    #[inline(always)]
    fn widen(self, stored: F8E4M3) -> f32 {
        stored.to_f32()
    }
}

/// The stored value times the scale in f32, rounded to bf16 when `TO_BF16`. The IR computes the
/// product in the dtype of the scale and casts it to the dtype of `x`, so it rounds to bf16 when
/// either is bf16. Every stored dtype the kernel reads holds bf16 values, so the f32 product with a
/// bf16 scale is exact and its one rounding gives the bf16 product.
#[derive(Clone, Copy)]
struct Scaled<W, const TO_BF16: bool> {
    scale: f32,
    stored: PhantomData<W>,
}

impl<W: Copy + Send + Sync, const TO_BF16: bool> Widen for Scaled<W, TO_BF16>
where
    Decoded<W>: Widen<Stored = W>,
{
    type Stored = W;

    #[inline(always)]
    fn widen(self, stored: W) -> f32 {
        let product = Decoded(PhantomData).widen(stored) * self.scale;
        if TO_BF16 {
            bf16::from_f32(product).to_f32()
        } else {
            product
        }
    }
}

/// Whether the kernel takes an `x` of dtype `x` and a weight stored in `weight`, with a scale of
/// dtype `scale` when there is one.
pub fn reads(x: DType, weight: DType, scale: Option<DType>) -> bool {
    use DType::{Bf16, F32, F8E4M3};
    matches!(x, F32 | Bf16)
        && matches!(weight, Bf16 | F8E4M3)
        && matches!(scale, None | Some(Bf16 | F32))
}

/// The name of `layout` in the profiler signature.
fn layout_name(layout: WeightLayout) -> &'static str {
    match layout {
        WeightLayout::OutIn => "out_in",
        WeightLayout::InOut => "in_out",
    }
}

/// Extents of a projection of `x` `[x_rows, in_features]` to `[x_rows, out_features]`.
#[derive(Clone, Copy)]
struct Shape {
    x_rows: usize,
    in_features: usize,
    out_features: usize,
    layout: WeightLayout,
}

/// Runs the linear kernel on `x` `[M, K]` in f32 or bf16, the weight `[N, K]` or `[K, N]` in its
/// stored dtype as `layout` says, and an optional one-element bf16 or f32 scale of the weight.
pub fn run(
    x: &HostTensor,
    weight: &HostTensor,
    scale: Option<&HostTensor>,
    layout: WeightLayout,
    output: &TensorType,
) -> Result<HostTensor> {
    let x_values = match &x.data {
        TensorData::F32(values) => Cow::Borrowed(&values[..]),
        TensorData::Bf16(values) => Cow::Owned(values.iter().map(|value| value.to_f32()).collect()),
        _ => return Err(anyhow!("the linear kernel expects f32 or bf16 x",)),
    };
    let x_bf16 = x.dtype() == DType::Bf16;
    // The scale, and whether the scaled weight rounds to bf16.
    let scaling = match scale.map(|scale_tensor| &scale_tensor.data) {
        None => None,
        Some(TensorData::Bf16(values)) if values.len() == 1 => Some((values[0].to_f32(), true)),
        Some(TensorData::F32(values)) if values.len() == 1 => Some((values[0], x_bf16)),
        Some(_) => {
            return Err(anyhow!(
                "the linear kernel expects a one-element bf16 or f32 scale",
            ))
        }
    };
    let (&[x_rows, in_features], &[weight_rows, weight_cols], &[out_rows, out_features]) = (
        x.dims.clone().as_slice(),
        weight.dims.clone().as_slice(),
        dims_of(output)?.as_slice(),
    ) else {
        return Err(anyhow!(
            "the linear kernel expects rank-2 x, weight and result",
        ));
    };
    let weight_dims = match layout {
        WeightLayout::OutIn => [out_features, in_features],
        WeightLayout::InOut => [in_features, out_features],
    };
    if [weight_rows, weight_cols] != weight_dims || out_rows != x_rows || output.dtype != DType::F32
    {
        return Err(anyhow!("the linear kernel operand shapes do not match",));
    }
    let shape = Shape {
        x_rows,
        in_features,
        out_features,
        layout,
    };

    let _prof_guard = gpt_rs::profiling::backend_scope_with_meta("backend.faer.linear", || {
        let signature = format!(
            "x={} weight={} layout={}",
            x.tensor_type(),
            weight.tensor_type(),
            layout_name(layout),
        );
        let (rows, inputs, outputs) = (x_rows as u64, in_features as u64, out_features as u64);
        let bytes = |tensor: &HostTensor| tensor.dtype().size_in_bytes().unwrap_or(0) as u64;
        let weight_bytes = (outputs * inputs).saturating_mul(bytes(weight));
        let work = gpt_rs::profiling::WorkStats {
            elements: rows * outputs,
            bytes_read: (rows * inputs * bytes(x)).saturating_add(weight_bytes),
            bytes_written: rows * outputs * 4,
            flops: 2 * rows * outputs * inputs,
            alloc_bytes: rows * outputs * 4,
            alloc_count: 1,
        };
        gpt_rs::profiling::signature_id(&signature)
            .map(gpt_rs::profiling::ScopeMeta::signature)
            .unwrap_or_default()
            .with_work(work)
    });
    let out = match &weight.data {
        TensorData::Bf16(stored) => widened(&x_values, &stored[..], scaling, shape)?,
        TensorData::F8E4M3(stored) => widened(&x_values, &stored[..], scaling, shape)?,
        _ => {
            return Err(anyhow!(format!(
                "the linear kernel cannot read {:?} weights",
                weight.dtype()
            )))
        }
    };
    output_tensor(output, TensorData::F32(Buffer::from(out)))
}

/// [`linear`] with the widening that `scaling` asks for. `scaling` holds the scale and whether the
/// scaled weight rounds to bf16.
fn widened<W: Copy + Send + Sync>(
    x: &[f32],
    weight: &[W],
    scaling: Option<(f32, bool)>,
    shape: Shape,
) -> Result<Vec<f32>>
where
    Decoded<W>: Widen<Stored = W>,
{
    let stored = PhantomData;
    match scaling {
        None => linear(x, weight, Decoded(stored), shape),
        Some((scale, false)) => linear(x, weight, Scaled::<W, false> { scale, stored }, shape),
        Some((scale, true)) => linear(x, weight, Scaled::<W, true> { scale, stored }, shape),
    }
}

/// `x @ w^T` for [`WeightLayout::OutIn`], `x @ w` for [`WeightLayout::InOut`], with `w` widened
/// by `widen`.
fn linear<D: Widen>(x: &[f32], weight: &[D::Stored], widen: D, shape: Shape) -> Result<Vec<f32>> {
    let Shape {
        x_rows,
        in_features,
        out_features,
        ..
    } = shape;
    if x.len() != x_rows * in_features || weight.len() != out_features * in_features {
        return Err(anyhow!("the linear kernel operand size mismatch",));
    }
    let mut out = vec![0.0f32; x_rows * out_features];
    if out.is_empty() || in_features == 0 {
        return Ok(out);
    }
    // The result `[x_rows, out_features]` row-major is its transpose `[out_features, x_rows]`
    // column-major: a task computes a range of its rows, the outputs of a range of weight rows.
    let transposed = MatMut::from_column_major_slice_mut(&mut out, out_features, x_rows);
    let par = crate::faer_parallelism();
    let threads = par.degree();
    let (path, granule, tasks) = if shape.layout == WeightLayout::OutIn && x_rows <= MATVEC_ROWS {
        // A task streams at least an L2 of weights, which pays for the split.
        let row_bytes = in_features * size_of::<D::Stored>();
        let min_rows = (CACHE_INFO[1].cache_bytes / row_bytes).max(1);
        (Path::Matvec, 1, threads.min(out_features / min_rows))
    } else {
        let blocks = Blocks::new(shape, threads);
        let tasks = threads.min(out_features.div_ceil(blocks.rows));
        (Path::Matmul(blocks), blocks.rows, tasks)
    };
    split_rows(par, transposed, 0, granule, tasks.max(1), &|first, dst| {
        Arch::new().dispatch(RowsTask {
            x,
            weight,
            widen,
            shape,
            path,
            first,
            dst,
        })
    });
    Ok(out)
}

#[derive(Clone, Copy)]
enum Path {
    Matvec,
    Matmul(Blocks),
}

/// The rows `first..` of the transposed result. Its kernels inline into
/// [`WithSimd::with_simd`], so they compile for the vector instructions [`Arch`] picks.
struct RowsTask<'a, D: Widen> {
    x: &'a [f32],
    weight: &'a [D::Stored],
    widen: D,
    shape: Shape,
    path: Path,
    first: usize,
    dst: MatMut<'a, f32>,
}

impl<D: Widen> WithSimd for RowsTask<'_, D> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            x,
            weight,
            widen,
            shape,
            path,
            first,
            dst,
        } = self;
        match path {
            Path::Matvec => matvec(simd, x, weight, widen, shape.in_features, first, dst),
            Path::Matmul(blocks) => matmul_blocks(x, weight, widen, shape, blocks, first, dst),
        }
    }
}

/// Runs `task(first, rows)` in parallel on `tasks` consecutive row ranges of `dst`, where `first`
/// is the index of the range's first row. The ranges hold whole granules of `granule` rows, so
/// only the last range ends in a partial granule, and their granule counts differ by at most one.
/// Each half of a split takes granules in proportion to its tasks, so that an odd task count
/// still gives every task its share.
fn split_rows<F>(
    parallelism: Par,
    dst: MatMut<f32>,
    first: usize,
    granule: usize,
    tasks: usize,
    task: &F,
) where
    F: Fn(usize, MatMut<'_, f32>) + Sync,
{
    if tasks <= 1 {
        return task(first, dst);
    }
    let top_tasks = tasks / 2;
    let mid = dst.nrows().div_ceil(granule) * top_tasks / tasks * granule;
    let (top, bottom) = dst.split_at_row_mut(mid);
    faer::utils::thread::join_raw(
        |par| split_rows(par, top, first, granule, top_tasks, task),
        |par| split_rows(par, bottom, first + mid, granule, tasks - top_tasks, task),
        parallelism,
    );
}

/// The rows `first..` of the transposed result: weight rows `[N, K]` dotted with the at most
/// [`MATVEC_ROWS`] rows of `x`, all in one pass over each weight row.
#[inline(always)]
fn matvec<S: Simd, D: Widen>(
    simd: S,
    x: &[f32],
    weight: &[D::Stored],
    widen: D,
    in_features: usize,
    first: usize,
    mut dst: MatMut<f32>,
) {
    let x_row = |index: usize| &x[index * in_features..][..in_features];
    for row in 0..dst.nrows() {
        let w_row = &weight[(first + row) * in_features..][..in_features];
        // At most eight vector accumulators, enough independent sums to hide the latency of the
        // multiply-adds.
        match dst.ncols() {
            1 => store(
                &mut dst,
                row,
                dot_rows::<S, D, 1, 8>(simd, [x_row(0)], w_row, widen),
            ),
            2 => {
                let sums = dot_rows::<S, D, 2, 4>(simd, [x_row(0), x_row(1)], w_row, widen);
                store(&mut dst, row, sums)
            }
            3 => {
                let activations = [x_row(0), x_row(1), x_row(2)];
                let sums = dot_rows::<S, D, 3, 2>(simd, activations, w_row, widen);
                store(&mut dst, row, sums)
            }
            _ => {
                let activations = std::array::from_fn(x_row);
                let sums = dot_rows::<S, D, MATVEC_ROWS, 2>(simd, activations, w_row, widen);
                store(&mut dst, row, sums)
            }
        }
    }
}

#[inline(always)]
fn store<const R: usize>(dst: &mut MatMut<f32>, row: usize, sums: [f32; R]) {
    for (col, sum) in sums.into_iter().enumerate() {
        dst[(row, col)] = sum;
    }
}

/// The dot products of `R` rows of `x` with one weight row. Each chunk of `V` vectors of weight
/// elements is widened once and multiplied into `V` accumulators per row.
#[inline(always)]
fn dot_rows<S: Simd, D: Widen, const R: usize, const V: usize>(
    simd: S,
    x: [&[f32]; R],
    weight: &[D::Stored],
    widen: D,
) -> [f32; R] {
    let zero = simd.splat_f32s(0.0);
    let mut acc = [[zero; V]; R];
    let chunk = V * S::F32_LANES;
    let chunks = weight.chunks_exact(chunk);
    let tail = chunks.remainder();
    for (chunk_index, weights) in chunks.enumerate() {
        let mut widened = [zero; V];
        let lanes: &mut [f32] = bytemuck::cast_slice_mut(&mut widened);
        for (lane, stored) in lanes.iter_mut().zip(weights) {
            *lane = widen.widen(*stored);
        }
        for (row_acc, x_row) in acc.iter_mut().zip(&x) {
            let (x_chunk, _) = S::as_simd_f32s(&x_row[chunk_index * chunk..][..chunk]);
            for ((vector_acc, &weight_vector), &x_vector) in
                row_acc.iter_mut().zip(&widened).zip(x_chunk)
            {
                *vector_acc = simd.mul_add_e_f32s(weight_vector, x_vector, *vector_acc);
            }
        }
    }
    let body = weight.len() - tail.len();
    std::array::from_fn(|row| {
        let tail_sum: f32 = tail
            .iter()
            .zip(&x[row][body..])
            .map(|(&stored, x_value)| widen.widen(stored) * x_value)
            .sum();
        let vector_sum = acc[row]
            .iter()
            .fold(zero, |sum, &vector_acc| simd.add_f32s(sum, vector_acc));
        simd.reduce_sum_f32s(vector_sum) + tail_sum
    })
}

/// Rows and inner extent of the weight blocks the matmul path widens at a time.
#[derive(Clone, Copy)]
struct Blocks {
    rows: usize,
    inner: usize,
}

impl Blocks {
    fn new(shape: Shape, threads: usize) -> Self {
        // Rows come in whole cache lines of f32 outputs.
        let line = (CACHE_INFO[0].cache_line_bytes / size_of::<f32>()).max(1);
        // A widened block fills half of the L2 a thread has, the other half holding the
        // matmul's packed panels of it and of `x`.
        let floats = (CACHE_INFO[1].cache_bytes / 2 / size_of::<f32>()).max(line * line);
        // A square block reads `x` and the result about equally often per weight element: `x`
        // once per block of rows, the result once per block of the inner extent.
        let side = (floats.isqrt() / line * line).max(line);
        let inner = shape.in_features.div_ceil(shape.in_features.div_ceil(side));
        let max_rows = (floats / inner / line * line).max(line);
        // A thread's share of the rows splits into equal blocks no larger than that, so every
        // thread gets the same number of blocks.
        let per_thread = shape.out_features.div_ceil(threads.max(1));
        let rows = per_thread
            .div_ceil(per_thread.div_ceil(max_rows))
            .next_multiple_of(line);
        Self { rows, inner }
    }
}

thread_local! {
    static WIDENED_BLOCK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// The rows `first..` of the transposed result, a block of weight rows at a time: each block is
/// widened into a scratch buffer [`Blocks::inner`] elements of `K` at a time and multiplied with
/// the matching columns of `x` by faer's matmul.
#[inline(always)]
fn matmul_blocks<D: Widen>(
    x: &[f32],
    weight: &[D::Stored],
    widen: D,
    shape: Shape,
    blocks: Blocks,
    first: usize,
    mut dst: MatMut<f32>,
) {
    let x_t = MatRef::from_row_major_slice(x, shape.x_rows, shape.in_features).transpose();
    // Taken rather than borrowed in a closure, which would compile outside the caller's vector
    // instructions.
    let mut scratch = WIDENED_BLOCK.take();
    scratch.resize(blocks.rows * blocks.inner, 0.0);
    let total = dst.nrows();
    for start in (0..total).step_by(blocks.rows) {
        let rows = blocks.rows.min(total - start);
        let mut out = dst.rb_mut().subrows_mut(start, rows);
        for k0 in (0..shape.in_features).step_by(blocks.inner) {
            let inner = blocks.inner.min(shape.in_features - k0);
            let block = &mut scratch[..rows * inner];
            let row = first + start;
            let lhs = widen_block(weight, widen, shape, row..row + rows, k0..k0 + inner, block);
            let accum = if k0 == 0 { Accum::Replace } else { Accum::Add };
            let rhs = x_t.subrows(k0, inner);
            matmul(out.rb_mut(), accum, lhs, rhs, 1.0, Par::Seq);
        }
    }
    WIDENED_BLOCK.set(scratch);
}

/// Widens the elements `ks` of weight rows `rows` into `block`, and views it as that
/// `[rows.len(), ks.len()]` matrix.
#[inline(always)]
fn widen_block<'a, D: Widen>(
    weight: &[D::Stored],
    widen: D,
    shape: Shape,
    rows: Range<usize>,
    ks: Range<usize>,
    block: &'a mut [f32],
) -> MatRef<'a, f32> {
    let (row, k0) = (rows.start, ks.start);
    match shape.layout {
        WeightLayout::OutIn => {
            widen_runs(
                weight,
                widen,
                row * shape.in_features + k0,
                shape.in_features,
                ks.len(),
                block,
            );
            MatRef::from_row_major_slice(block, rows.len(), ks.len())
        }
        WeightLayout::InOut => {
            widen_runs(
                weight,
                widen,
                k0 * shape.out_features + row,
                shape.out_features,
                rows.len(),
                block,
            );
            MatRef::from_row_major_slice(block, ks.len(), rows.len()).transpose()
        }
    }
}

/// Fills `dst` with runs of `len` widened elements of `src`, the runs `stride` apart from `start`.
#[inline(always)]
fn widen_runs<D: Widen>(
    src: &[D::Stored],
    widen: D,
    start: usize,
    stride: usize,
    len: usize,
    dst: &mut [f32],
) {
    for (run_index, run) in dst.chunks_exact_mut(len).enumerate() {
        let src_run = &src[start + run_index * stride..][..len];
        for (out, &value) in run.iter_mut().zip(src_run) {
            *out = widen.widen(value);
        }
    }
}
