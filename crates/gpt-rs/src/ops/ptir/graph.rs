//! PTIR (Portable Tensor IR) graph construction DSL and snippet emission infrastructure.
//!
//! This module provides the high-level interface for building portable tensor computation graphs
//! using a fluent, expression-oriented API. The PTIR DSL allows functional operators to describe
//! their computations abstractly, which are then lowered to backend-specific programs.
//!
//! ## Architecture
//!
//! The PTIR layer sits between functional operators and the graph arena:
//!
//! ```text
//! Functional Ops (ops/functional/*)
//!         |
//!         | capture_ptir! macro
//!         v
//! PTIR DSL (this module)
//!         |
//!         | emit operations
//!         v
//! GraphBuilder (ops/graph/builder.rs)
//!         |
//!         | record nodes
//!         v
//! GraphArena (ops/graph/arena.rs)
//! ```
//!
//! ## Key Types
//!
//! - [`PtirSession`] - Entry point for PTIR graph construction, provides import and constant creation
//! - [`Tensor`] - Represents a PTIR tensor value with fluent operation methods
//! - [`PtirGraph`] - Internal graph state maintaining value mappings and emitting operations
//! - [`PtirValue`] - Metadata about a tensor value (shape, dtype, name)
//!
//! ## Usage Pattern
//!
//! Functional operators use the `capture_ptir!` macro to create PTIR sessions:
//!
//! ```rust,ignore
//! capture_ptir! {
//!     { x, y },  // Import device tensors as PTIR tensors
//!     |session| {
//!         // Build computation graph using PTIR DSL
//!         let sum = x.try_add(&y)?;
//!         let result = sum.sqrt();
//!         Ok(result.id())  // Return the ValueId
//!     }
//! }
//! ```
//!
//! The DSL provides methods like:
//! - Binary ops: `try_add`, `try_mul`, `try_div`, `try_sub`, `try_maximum`, `try_minimum`
//! - Unary ops: `sqrt`, `exp`, `log`, `tanh`, `erf`, `try_neg`, `try_abs`
//! - Shape ops: `reshape`, `transpose`, `broadcast`, `slice`, `concat`
//! - Reductions: `reduce_sum`, `reduce_max`, `reduce_mean`
//! - Matrix ops: `dot_general` (batched matmul with flexible dimension specs)
//!
//! ## Snippet Emission
//!
//! For complex operations, the DSL supports emitting custom PTIR snippets (text-based IR
//! fragments) via [`SnippetEmitter`]. This allows backend-agnostic optimization patterns
//! to be captured in a portable format before final lowering.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};
use half::{bf16, f16};

use crate::backend::ptir_utils::tensor_spec_static as tensor_spec_from;
use crate::backend::spec::{
    BroadcastToSpec, CompareSpec, ComparisonOp, ConcatSpec, CustomCallAttr, CustomCallSpec, DType,
    Dimension, DotGeneralSpec, DynamicSliceSpec, DynamicUpdateSliceSpec, ElementwiseBinaryOp,
    ElementwiseUnaryOp, ExtractPatchesSpec, GatherSpec, IotaSpec, Literal, Operand, Operation,
    PortableBackend, ReduceKind, ReduceSpec, ReduceWindowSpec, ReshapeDim, ReshapeSpec,
    RngUniformSpec, SliceSpec, TensorLiteral, TensorSpec, TransposeSpec, ValueId,
};
use crate::backend::text_ir::{PtirSnippet, SnippetBindings, SnippetResult};
use crate::ops::graph::GraphBuilder;
use crate::ops::ptir::axes::Axes;
use crate::ops::ptir::tensor::{TensorDType, TensorPlaceholder};

/// High-level wrapper around [`GraphBuilder`] providing DSL conveniences.
///
/// `PtirGraph` maintains mappings from value IDs to metadata and delegates operation
/// emission to the underlying graph builder. This type is wrapped in a `RefCell` and
/// accessed through [`PtirSession`] to enable interior mutability within capture closures.
pub struct PtirGraph<'ctx, 'gb, B: PortableBackend + 'static> {
    ctx: &'ctx mut GraphBuilder<'gb, B>,
    values: HashMap<String, PtirValue>,
    values_by_id: HashMap<ValueId, PtirValue>,
}

/// Entry point for PTIR graph construction sessions.
///
/// Created by the `capture_ptir!` macro, this session provides methods for importing
/// device tensors, creating constants, and accessing the PTIR graph for snippet emission.
/// Sessions use interior mutability (`Rc<RefCell<...>>`) to allow multiple tensor handles
/// to reference the same graph within capture closures.
#[derive(Clone)]
pub struct PtirSession<'ctx, 'gb, B: PortableBackend + 'static> {
    graph: Rc<RefCell<PtirGraph<'ctx, 'gb, B>>>,
}

/// Metadata describing a PTIR tensor value.
///
/// Stores shape information, dtype, and optional naming for debugging. The `static_dims`
/// field caches resolved dimensions when all sizes are known at graph construction time.
#[derive(Debug, Clone)]
pub struct PtirValue {
    name: Option<String>,
    value: ValueId,
    spec: TensorSpec,
    dtype: DType,
    rank: usize,
    static_dims: Option<Vec<usize>>,
}

/// Collection of PTIR values returned from snippet emission or multi-output operations.
#[derive(Debug, Clone)]
pub struct PtirResults {
    values: Vec<PtirValue>,
}

/// A PTIR tensor handle providing fluent operation composition.
///
/// `Tensor` is a lightweight, copyable handle to a value in the PTIR graph. It provides
/// methods for all supported tensor operations (arithmetic, shape manipulation, reductions,
/// etc.) that emit corresponding graph nodes. The `Copy` trait enables natural expression
/// composition without explicit cloning.
///
/// Many operations are provided in two forms:
/// - `try_*` variants that return `Result` and surface validation/emission failures
/// - convenience methods without the `try_` prefix that panic on error
///
/// Prefer the `try_*` APIs in library code and keep the panicking helpers for tests or trusted
/// internal call sites.
///
/// # Lifetime Parameters
///
/// - `'ctx`: Lifetime of the PTIR session/graph
/// - `'gb`: Lifetime of the underlying graph builder
/// - `B`: Backend type implementing `PortableBackend`
#[derive(Debug)]
pub struct Tensor<'ctx, 'gb, B: PortableBackend + 'static> {
    graph: NonNull<RefCell<PtirGraph<'ctx, 'gb, B>>>,
    value: ValueId,
    _marker: PhantomData<&'ctx mut GraphBuilder<'gb, B>>,
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Copy for Tensor<'ctx, 'gb, B> {}

impl<'ctx, 'gb, B: PortableBackend + 'static> Clone for Tensor<'ctx, 'gb, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> PtirSession<'ctx, 'gb, B> {
    pub fn new(ctx: &'ctx mut GraphBuilder<'gb, B>) -> Self {
        Self {
            graph: Rc::new(RefCell::new(PtirGraph::new(ctx))),
        }
    }

    pub fn import<T>(
        &self,
        name: impl Into<String>,
        value: ValueId,
        placeholder: &TensorPlaceholder<T>,
    ) -> Tensor<'ctx, 'gb, B>
    where
        T: TensorDType,
    {
        let ptir_value = self.graph.borrow_mut().import(name, value, placeholder);
        Tensor::from_session(&self.graph, ptir_value.id())
    }

    pub fn import_spec(
        &self,
        name: impl Into<String>,
        value: ValueId,
        spec: TensorSpec,
    ) -> Tensor<'ctx, 'gb, B> {
        let ptir_value = self.graph.borrow_mut().import_spec(name, value, spec);
        Tensor::from_session(&self.graph, ptir_value.id())
    }

    pub fn graph(&self) -> Rc<RefCell<PtirGraph<'ctx, 'gb, B>>> {
        Rc::clone(&self.graph)
    }

    pub fn scalar(&self, value: f32) -> Tensor<'ctx, 'gb, B> {
        let ptir_value = self
            .graph
            .borrow_mut()
            .scalar_literal(value, DType::F32, None)
            .expect("PTIR scalar literal creation failed");
        Tensor::from_session(&self.graph, ptir_value.id())
    }

    pub fn iota(
        &self,
        dims: impl Into<Vec<usize>>,
        axis: usize,
        dtype: DType,
    ) -> Tensor<'ctx, 'gb, B> {
        let dims_vec = dims.into();
        let ptir_value = self
            .graph
            .borrow_mut()
            .iota(&dims_vec, axis, dtype, None)
            .expect("PTIR iota creation failed");
        Tensor::from_session(&self.graph, ptir_value.id())
    }

    pub fn rng_uniform(&self, dims: impl Into<Vec<usize>>, dtype: DType) -> Tensor<'ctx, 'gb, B> {
        let dims_vec = dims.into();
        let ptir_value = self
            .graph
            .borrow_mut()
            .rng_uniform(&dims_vec, dtype, None)
            .expect("PTIR rng_uniform creation failed");
        Tensor::from_session(&self.graph, ptir_value.id())
    }

    /// Emits a backend-provided fused kernel as a `custom_call`.
    pub fn custom_call(
        &self,
        target: impl Into<String>,
        attrs: BTreeMap<String, CustomCallAttr>,
        operands: &[Tensor<'ctx, 'gb, B>],
        output_spec: TensorSpec,
    ) -> Tensor<'ctx, 'gb, B> {
        let operands = operands
            .iter()
            .map(|tensor| Operand::Value(tensor.value))
            .collect::<Vec<_>>();
        let ptir_value = self
            .graph
            .borrow_mut()
            .custom_call(target.into(), attrs, operands, output_spec, None)
            .expect("PTIR custom_call emission failed");
        Tensor::from_session(&self.graph, ptir_value.id())
    }
}

/// Optional attributes for configuring dot_general (batched matrix multiplication) operations.
///
/// Allows specifying a custom accumulation dtype for intermediate dot product results,
/// which can improve numerical stability or enable mixed-precision computation.
#[derive(Debug, Default, Clone)]
pub struct DotAttrs {
    /// Optional dtype for accumulation (e.g., F32 accumulation for F16 inputs).
    pub accum_dtype: Option<DType>,
    /// Optional output dtype (defaults to the input dtype).
    pub out_dtype: Option<DType>,
}

/// Dimension specification for batched matrix multiplication (dot_general operation).
///
/// Describes which axes participate in batching, contraction (reduction), and output.
/// Follows the XLA dot_general semantics, supporting flexible dimension arrangements
/// beyond simple matrix multiplication.
///
/// # Example
///
/// For a standard batched matmul `[B, M, K] @ [B, K, N] -> [B, M, N]`:
/// ```rust,ignore
/// DotDims::new(
///     axes![0],  // batch: axis 0 on both sides
///     axes![2],  // contracting_lhs: K dimension (axis 2 of lhs)
///     axes![1],  // contracting_rhs: K dimension (axis 1 of rhs)
/// )
/// ```
#[derive(Debug, Clone)]
pub struct DotDims {
    /// Axes that form batch dimensions (must match on lhs and rhs).
    pub batch: Axes,
    /// Axes on the left-hand side that will be contracted (summed over).
    pub contracting_lhs: Axes,
    /// Axes on the right-hand side that will be contracted (summed over).
    pub contracting_rhs: Axes,
    /// Optional override for batch axes on the right-hand side (for asymmetric batching).
    pub batch_rhs: Option<Axes>,
}

impl DotDims {
    pub fn new(batch: Axes, contracting_lhs: Axes, contracting_rhs: Axes) -> Self {
        Self {
            batch,
            contracting_lhs,
            contracting_rhs,
            batch_rhs: None,
        }
    }

    pub fn with_rhs_batch(mut self, rhs: Axes) -> Self {
        self.batch_rhs = Some(rhs);
        self
    }
}

pub struct SnippetEmitter<'a, 'ctx, 'gb, B: PortableBackend + 'static> {
    graph: &'a mut PtirGraph<'ctx, 'gb, B>,
    snippet: PtirSnippet,
    bindings: SnippetBindings,
}

impl<'ctx, 'gb, B: PortableBackend + 'static> PtirGraph<'ctx, 'gb, B> {
    pub fn new(ctx: &'ctx mut GraphBuilder<'gb, B>) -> Self {
        Self {
            ctx,
            values: HashMap::new(),
            values_by_id: HashMap::new(),
        }
    }

    pub fn import<T>(
        &mut self,
        name: impl Into<String>,
        value: ValueId,
        placeholder: &TensorPlaceholder<T>,
    ) -> PtirValue
    where
        T: TensorDType,
    {
        self.import_spec(name, value, placeholder.spec().clone())
    }

    pub fn import_spec(
        &mut self,
        name: impl Into<String>,
        value: ValueId,
        spec: TensorSpec,
    ) -> PtirValue {
        let name = name.into();
        let value_ref = self.register_value(Some(name.clone()), value, spec);
        self.values.insert(name, value_ref.clone());
        value_ref
    }

    pub fn emit_snippet(
        &'ctx mut self,
        snippet: PtirSnippet,
    ) -> SnippetEmitter<'ctx, 'ctx, 'gb, B> {
        SnippetEmitter {
            graph: self,
            snippet,
            bindings: SnippetBindings::new(),
        }
    }

    pub fn graph_builder(&mut self) -> &mut GraphBuilder<'gb, B> {
        self.ctx
    }

    fn register_value(
        &mut self,
        name: Option<String>,
        value: ValueId,
        spec: TensorSpec,
    ) -> PtirValue {
        let dtype = spec.dtype;
        let rank = spec.shape.rank();
        let static_dims = collect_static_dims(&spec);
        let registered = PtirValue {
            name,
            value,
            spec,
            dtype,
            rank,
            static_dims,
        };
        self.values_by_id.insert(value, registered.clone());
        registered
    }

    fn fetch_value(&self, value: ValueId) -> Result<PtirValue> {
        self.values_by_id
            .get(&value)
            .cloned()
            .ok_or_else(|| anyhow!("value {:?} is not registered in PTIR graph", value))
    }

    fn register_snippet_results(&mut self, result: SnippetResult) -> Vec<PtirValue> {
        let (value_ids, specs) = result.into_parts();
        value_ids
            .into_iter()
            .zip(specs)
            .map(|(value, spec)| self.register_value(None, value, spec))
            .collect()
    }

    fn elementwise_binary(
        &mut self,
        op: ElementwiseBinaryOp,
        lhs: ValueId,
        rhs: ValueId,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let lhs_value = self.fetch_value(lhs)?;
        let rhs_value = self.fetch_value(rhs)?;
        ensure!(
            lhs_value.dtype() == rhs_value.dtype(),
            "elementwise binary operands must share dtype"
        );
        ensure!(
            lhs_value.rank() == rhs_value.rank(),
            "elementwise binary operands must share rank"
        );
        let lhs_dims = lhs_value.dims()?;
        let rhs_dims = rhs_value.dims()?;
        ensure!(
            lhs_dims == rhs_dims,
            "elementwise binary operands must share shape"
        );

        let result_spec = tensor_spec_from(lhs_value.dtype(), lhs_dims);
        let operands = vec![Operand::Value(lhs), Operand::Value(rhs)];
        let output = self.ctx.emit(
            Operation::ElementwiseBinary(op),
            operands,
            result_spec.clone(),
        );
        let registered = self.register_value(name.map(|s| s.to_string()), output, result_spec);
        Ok(registered)
    }

    fn elementwise_unary(
        &mut self,
        op: ElementwiseUnaryOp,
        value: ValueId,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let result_spec = value_meta.spec().clone();
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::ElementwiseUnary(op),
            operands,
            result_spec.clone(),
        );
        let registered = self.register_value(name.map(|s| s.to_string()), output, result_spec);
        Ok(registered)
    }

    fn reduce(
        &mut self,
        kind: ReduceKind,
        value: ValueId,
        axes: &[usize],
        keepdims: bool,
        accum_dtype: Option<DType>,
        out_dtype: Option<DType>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let axes = validate_axes(axes, value_meta.rank())?;
        let base_dims = value_meta.dims()?.to_vec();
        let mut result_dims = Vec::new();
        if keepdims {
            result_dims = base_dims.clone();
            for axis in &axes {
                result_dims[*axis] = 1;
            }
        } else {
            for (index, dim) in base_dims.iter().enumerate() {
                if !axes.contains(&index) {
                    result_dims.push(*dim);
                }
            }
            ensure!(
                !result_dims.is_empty(),
                "reduce without keepdims removed all dimensions"
            );
        }

        let result_dtype = out_dtype.unwrap_or(value_meta.dtype());
        let result_spec = tensor_spec_from(result_dtype, &result_dims);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::Reduce(ReduceSpec {
                kind,
                axes,
                keepdims,
                accum_dtype,
                out_dtype,
            }),
            operands,
            result_spec.clone(),
        );
        let registered = self.register_value(None, output, result_spec);
        Ok(registered)
    }

    fn custom_call(
        &mut self,
        target: String,
        attrs: BTreeMap<String, CustomCallAttr>,
        operands: Vec<Operand>,
        output_spec: TensorSpec,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        for operand in &operands {
            match operand {
                Operand::Value(id) => {
                    let _ = self.fetch_value(*id)?;
                }
                Operand::TupleElement { tuple, .. } => {
                    let _ = self.fetch_value(*tuple)?;
                }
                Operand::Literal(_) => {}
            }
        }

        let output = self.ctx.emit(
            Operation::CustomCall(CustomCallSpec { target, attrs }),
            operands,
            output_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, output_spec))
    }

    fn extract_patches(
        &mut self,
        value: ValueId,
        spec: &ExtractPatchesSpec,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let dims = value_meta.dims()?;
        ensure!(
            dims.len() >= 3,
            "extract_patches expects [N, spatial..., C] input, got rank {}",
            dims.len()
        );

        let spatial_rank = dims.len() - 2;
        ensure!(
            spec.window.len() == spatial_rank,
            "extract_patches window rank {} must match spatial rank {}",
            spec.window.len(),
            spatial_rank
        );
        ensure!(
            spec.strides.len() == spatial_rank,
            "extract_patches strides rank {} must match spatial rank {}",
            spec.strides.len(),
            spatial_rank
        );
        ensure!(
            spec.dilation.len() == spatial_rank,
            "extract_patches dilation rank {} must match spatial rank {}",
            spec.dilation.len(),
            spatial_rank
        );
        ensure!(
            spec.padding.len() == spatial_rank,
            "extract_patches padding rank {} must match spatial rank {}",
            spec.padding.len(),
            spatial_rank
        );

        let batch = dims[0];
        let channels = dims[dims.len() - 1];
        ensure!(channels > 0, "extract_patches requires C > 0");

        let mut out_spatial = Vec::with_capacity(spatial_rank);
        for axis in 0..spatial_rank {
            let input = dims[1 + axis];
            let window = spec.window[axis];
            let stride = spec.strides[axis];
            let dilation = spec.dilation[axis];
            let (pad_before, pad_after) = spec.padding[axis];
            ensure!(input > 0, "extract_patches requires input dims > 0");
            ensure!(window > 0, "extract_patches window dim must be > 0");
            ensure!(stride > 0, "extract_patches stride must be > 0");
            ensure!(dilation > 0, "extract_patches dilation must be > 0");

            let effective = (window - 1)
                .checked_mul(dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| anyhow!("extract_patches effective window overflow"))?;
            let padded = input
                .checked_add(pad_before)
                .and_then(|v| v.checked_add(pad_after))
                .ok_or_else(|| anyhow!("extract_patches padded dimension overflow"))?;
            ensure!(
                padded >= effective,
                "extract_patches window ({}) exceeds padded input ({}) on axis {}",
                effective,
                padded,
                axis
            );
            let out = (padded - effective) / stride + 1;
            out_spatial.push(out);
        }

        let patch_elems = spec
            .window
            .iter()
            .try_fold(1usize, |acc, &v| acc.checked_mul(v))
            .ok_or_else(|| anyhow!("extract_patches window element product overflow"))?;
        let patch_dim = patch_elems
            .checked_mul(channels)
            .ok_or_else(|| anyhow!("extract_patches patch size overflow"))?;

        let mut result_dims = Vec::with_capacity(dims.len());
        result_dims.push(batch);
        result_dims.extend(out_spatial);
        result_dims.push(patch_dim);

        let result_spec = tensor_spec_from(value_meta.dtype(), &result_dims);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::ExtractPatches(spec.clone()),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn reduce_window(
        &mut self,
        value: ValueId,
        spec: &ReduceWindowSpec,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let dims = value_meta.dims()?;
        let rank = dims.len();

        ensure!(
            spec.window_dims.len() == rank,
            "reduce_window window_dims rank {} must match operand rank {}",
            spec.window_dims.len(),
            rank
        );
        ensure!(
            spec.strides.len() == rank,
            "reduce_window strides rank {} must match operand rank {}",
            spec.strides.len(),
            rank
        );
        ensure!(
            spec.padding.len() == rank,
            "reduce_window padding rank {} must match operand rank {}",
            spec.padding.len(),
            rank
        );
        ensure!(
            spec.base_dilation.len() == rank,
            "reduce_window base_dilation rank {} must match operand rank {}",
            spec.base_dilation.len(),
            rank
        );
        ensure!(
            spec.window_dilation.len() == rank,
            "reduce_window window_dilation rank {} must match operand rank {}",
            spec.window_dilation.len(),
            rank
        );

        let mut result_dims = Vec::with_capacity(rank);
        for (axis, &input) in dims.iter().enumerate() {
            let window = spec.window_dims[axis];
            let stride = spec.strides[axis];
            let base_dilation = spec.base_dilation[axis];
            let window_dilation = spec.window_dilation[axis];
            let (pad_before, pad_after) = spec.padding[axis];
            ensure!(input > 0, "reduce_window requires input dims > 0");
            ensure!(window > 0, "reduce_window window dim must be > 0");
            ensure!(stride > 0, "reduce_window stride must be > 0");
            ensure!(base_dilation > 0, "reduce_window base_dilation must be > 0");
            ensure!(
                window_dilation > 0,
                "reduce_window window_dilation must be > 0"
            );

            let dilated_input = (input - 1)
                .checked_mul(base_dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| anyhow!("reduce_window dilated input overflow"))?;
            let padded = dilated_input
                .checked_add(pad_before)
                .and_then(|v| v.checked_add(pad_after))
                .ok_or_else(|| anyhow!("reduce_window padded dimension overflow"))?;
            let effective_window = (window - 1)
                .checked_mul(window_dilation)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| anyhow!("reduce_window effective window overflow"))?;

            ensure!(
                padded >= effective_window,
                "reduce_window window ({}) exceeds padded input ({}) on axis {}",
                effective_window,
                padded,
                axis
            );

            let out = (padded - effective_window) / stride + 1;
            result_dims.push(out);
        }

        let result_spec = tensor_spec_from(value_meta.dtype(), &result_dims);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::ReduceWindow(spec.clone()),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn broadcast_to(
        &mut self,
        value: ValueId,
        result_shape: &[usize],
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let operand_rank = value_meta.rank();
        let operand_dims = value_meta.dims()?;
        let result_rank = result_shape.len();
        ensure!(
            result_rank >= operand_rank,
            "broadcast result rank must be >= operand rank"
        );
        let rank_diff = result_rank - operand_rank;
        for (axis, &dim) in operand_dims.iter().enumerate() {
            let out_dim = result_shape[rank_diff + axis];
            ensure!(
                dim == 1 || dim == out_dim,
                "broadcast_to dim mismatch at axis {}: {} vs {}",
                axis,
                dim,
                out_dim
            );
        }

        let result_spec = tensor_spec_from(value_meta.dtype(), result_shape);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::BroadcastTo(BroadcastToSpec {
                result_shape: result_spec.shape.clone(),
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn scalar_literal(
        &mut self,
        value: f32,
        dtype: DType,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let literal = scalar_literal_tensor(dtype, value)?;
        let result_spec = tensor_spec_from(dtype, &[]);
        let output = self.ctx.emit(
            Operation::BroadcastTo(BroadcastToSpec {
                result_shape: result_spec.shape.clone(),
            }),
            vec![Operand::Literal(literal)],
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn scalar_literal_broadcast(
        &mut self,
        target: ValueId,
        value: f32,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let target_meta = self.fetch_value(target)?;
        let literal = scalar_literal_tensor(target_meta.dtype(), value)?;
        let result_spec = target_meta.spec().clone();
        let output = self.ctx.emit(
            Operation::BroadcastTo(BroadcastToSpec {
                result_shape: result_spec.shape.clone(),
            }),
            vec![Operand::Literal(literal)],
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn dot_general(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        dims: &DotDims,
        attrs: &DotAttrs,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let lhs_value = self.fetch_value(lhs)?;
        let rhs_value = self.fetch_value(rhs)?;
        ensure!(
            lhs_value.dtype() == rhs_value.dtype(),
            "dot_general operands must share dtype"
        );

        let lhs_dims = lhs_value.dims()?;
        let rhs_dims = rhs_value.dims()?;

        let batch_rhs = dims
            .batch_rhs
            .as_ref()
            .cloned()
            .unwrap_or_else(|| dims.batch.clone());

        ensure!(
            dims.batch.len() == batch_rhs.len(),
            "dot_general batch axes length mismatch"
        );
        ensure!(
            dims.contracting_lhs.len() == dims.contracting_rhs.len(),
            "dot_general contracting axes length mismatch"
        );

        validate_axes(&dims.batch, lhs_value.rank())?;
        validate_axes(&batch_rhs, rhs_value.rank())?;
        validate_axes(&dims.contracting_lhs, lhs_value.rank())?;
        validate_axes(&dims.contracting_rhs, rhs_value.rank())?;

        for (&lhs_axis, &rhs_axis) in dims.batch.iter().zip(batch_rhs.iter()) {
            ensure!(
                lhs_dims[lhs_axis] == rhs_dims[rhs_axis],
                "dot_general batch axis dimension mismatch"
            );
        }

        for (&lhs_axis, &rhs_axis) in dims.contracting_lhs.iter().zip(dims.contracting_rhs.iter()) {
            ensure!(
                lhs_dims[lhs_axis] == rhs_dims[rhs_axis],
                "dot_general contracting axis dimension mismatch"
            );
        }

        let mut result_dims = Vec::new();
        for &axis in &dims.batch {
            result_dims.push(lhs_dims[axis]);
        }

        let mut lhs_excluded: HashSet<usize> = dims.batch.iter().copied().collect();
        lhs_excluded.extend(dims.contracting_lhs.iter().copied());
        for (idx, size) in lhs_dims.iter().enumerate() {
            if !lhs_excluded.contains(&idx) {
                result_dims.push(*size);
            }
        }

        let mut rhs_excluded: HashSet<usize> = batch_rhs.iter().copied().collect();
        rhs_excluded.extend(dims.contracting_rhs.iter().copied());
        for (idx, size) in rhs_dims.iter().enumerate() {
            if !rhs_excluded.contains(&idx) {
                result_dims.push(*size);
            }
        }

        ensure!(
            !result_dims.is_empty(),
            "dot_general result shape cannot be empty"
        );

        let out_dtype = attrs.out_dtype.unwrap_or(lhs_value.dtype());
        let result_spec = tensor_spec_from(out_dtype, &result_dims);
        let operands = vec![Operand::Value(lhs), Operand::Value(rhs)];
        let output = self.ctx.emit(
            Operation::DotGeneral(DotGeneralSpec {
                batch_lhs: dims.batch.iter().copied().collect(),
                batch_rhs: batch_rhs.iter().copied().collect(),
                contract_lhs: dims.contracting_lhs.iter().copied().collect(),
                contract_rhs: dims.contracting_rhs.iter().copied().collect(),
                accum_dtype: attrs.accum_dtype,
                out_dtype: attrs.out_dtype,
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn reshape(&mut self, value: ValueId, dims: &[usize], name: Option<&str>) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let source_dims = value_meta.dims()?.to_vec();
        let src_elems: usize = source_dims.iter().product();
        let dst_elems: usize = dims.iter().product();
        ensure!(
            src_elems == dst_elems,
            "reshape requires the same number of elements (got {src_elems} vs {dst_elems})"
        );

        let reshape_spec = ReshapeSpec {
            new_shape: dims
                .iter()
                .copied()
                .map(|dim| ReshapeDim::Explicit(Dimension::from_usize(dim)))
                .collect(),
        };
        let result_spec = tensor_spec_from(value_meta.dtype(), dims);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::Reshape(reshape_spec),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn dynamic_slice(
        &mut self,
        value: ValueId,
        starts: ValueId,
        sizes: &[usize],
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let start_meta = self.fetch_value(starts)?;

        ensure!(
            value_meta.rank() == sizes.len(),
            "dynamic_slice sizes length must match operand rank"
        );

        let start_dims = start_meta.dims()?;
        ensure!(
            start_meta.dtype() == DType::Si32,
            "dynamic_slice start indices must be si32"
        );
        ensure!(
            start_dims.len() == 1 && start_dims[0] == sizes.len(),
            "dynamic_slice start indices must be 1-D of length equal to rank"
        );

        let result_spec = tensor_spec_from(value_meta.dtype(), sizes);
        let operands = vec![Operand::Value(value), Operand::Value(starts)];
        let output = self.ctx.emit(
            Operation::DynamicSlice(DynamicSliceSpec {
                sizes: sizes.to_vec(),
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn dynamic_update_slice(
        &mut self,
        base: ValueId,
        update: ValueId,
        starts: ValueId,
        sizes: &[usize],
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let base_meta = self.fetch_value(base)?;
        let update_meta = self.fetch_value(update)?;
        let start_meta = self.fetch_value(starts)?;

        ensure!(
            base_meta.rank() == sizes.len(),
            "dynamic_update_slice sizes length must match base rank"
        );
        ensure!(
            base_meta.dtype() == update_meta.dtype(),
            "dynamic_update_slice dtype mismatch"
        );

        let update_dims = update_meta.dims()?;
        ensure!(
            update_dims == sizes,
            "dynamic_update_slice update shape must match sizes"
        );

        let start_dims = start_meta.dims()?;
        ensure!(
            start_meta.dtype() == DType::Si32,
            "dynamic_update_slice start indices must be si32"
        );
        ensure!(
            start_dims.len() == 1 && start_dims[0] == sizes.len(),
            "dynamic_update_slice start indices must be 1-D of length equal to rank"
        );

        let result_spec = base_meta.spec().clone();
        let operands = vec![
            Operand::Value(base),
            Operand::Value(update),
            Operand::Value(starts),
        ];
        let output = self.ctx.emit(
            Operation::DynamicUpdateSlice(DynamicUpdateSliceSpec {
                sizes: sizes.to_vec(),
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn transpose(
        &mut self,
        value: ValueId,
        perm: &[usize],
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        ensure!(
            perm.len() == value_meta.rank(),
            "transpose permutation length must equal rank"
        );
        let mut seen = HashSet::new();
        let mut result_dims = Vec::with_capacity(perm.len());
        let dims = value_meta.dims()?;
        for &axis in perm {
            ensure!(axis < value_meta.rank(), "transpose axis out of range");
            ensure!(seen.insert(axis), "transpose permutation must be unique");
            result_dims.push(dims[axis]);
        }

        let result_spec = tensor_spec_from(value_meta.dtype(), &result_dims);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::Transpose(TransposeSpec {
                perm: perm.to_vec(),
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn slice(
        &mut self,
        value: ValueId,
        starts: &[usize],
        sizes: &[usize],
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        ensure!(
            starts.len() == value_meta.rank() && sizes.len() == value_meta.rank(),
            "slice starts/sizes must match operand rank"
        );
        let dims = value_meta.dims()?;
        for axis in 0..value_meta.rank() {
            ensure!(
                starts[axis] + sizes[axis] <= dims[axis],
                "slice exceeds dimension {axis}"
            );
        }

        let result_spec = tensor_spec_from(value_meta.dtype(), sizes);
        let operands = vec![Operand::Value(value)];
        let output = self.ctx.emit(
            Operation::Slice(SliceSpec {
                starts: starts.to_vec(),
                sizes: sizes.to_vec(),
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn concat(&mut self, values: &[ValueId], axis: usize, name: Option<&str>) -> Result<PtirValue> {
        ensure!(!values.is_empty(), "concat requires at least one operand");
        let first_meta = self.fetch_value(values[0])?;
        ensure!(axis < first_meta.rank(), "concat axis out of range");
        let mut result_dims = first_meta.dims()?.to_vec();
        let mut operands = Vec::with_capacity(values.len());
        operands.push(Operand::Value(values[0]));
        for &value in &values[1..] {
            let rhs = self.fetch_value(value)?;
            ensure!(rhs.rank() == first_meta.rank(), "concat rank mismatch");
            for (idx, (&lhs_dim, &rhs_dim)) in first_meta
                .dims()?
                .iter()
                .zip(rhs.dims()?.iter())
                .enumerate()
            {
                if idx == axis {
                    continue;
                }
                ensure!(
                    lhs_dim == rhs_dim,
                    "concat dimension mismatch at axis {idx}"
                );
            }
            result_dims[axis] += rhs.dims()?[axis];
            operands.push(Operand::Value(value));
        }

        let result_spec = tensor_spec_from(first_meta.dtype(), &result_dims);
        let output = self.ctx.emit(
            Operation::Concat(ConcatSpec {
                axis: axis as isize,
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn compare(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        op: ComparisonOp,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let lhs_meta = self.fetch_value(lhs)?;
        let rhs_meta = self.fetch_value(rhs)?;
        ensure!(
            lhs_meta.dtype() == rhs_meta.dtype(),
            "compare dtype mismatch"
        );
        ensure!(lhs_meta.rank() == rhs_meta.rank(), "compare rank mismatch");
        ensure!(
            lhs_meta.dims()? == rhs_meta.dims()?,
            "compare shape mismatch"
        );

        let result_spec = tensor_spec_from(DType::I1, lhs_meta.dims()?);
        let operands = vec![Operand::Value(lhs), Operand::Value(rhs)];
        let output = self.ctx.emit(
            Operation::Compare(CompareSpec { op }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn select(
        &mut self,
        predicate: ValueId,
        on_true: ValueId,
        on_false: ValueId,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let pred_meta = self.fetch_value(predicate)?;
        ensure!(
            pred_meta.dtype() == DType::I1,
            "select predicate must be boolean"
        );
        let true_meta = self.fetch_value(on_true)?;
        let false_meta = self.fetch_value(on_false)?;
        ensure!(
            true_meta.dtype() == false_meta.dtype(),
            "select branch dtype mismatch"
        );
        ensure!(
            true_meta.dims()? == false_meta.dims()?,
            "select branch shape mismatch"
        );
        let result_spec = true_meta.spec().clone();
        let operands = vec![
            Operand::Value(predicate),
            Operand::Value(on_true),
            Operand::Value(on_false),
        ];
        let output = self
            .ctx
            .emit(Operation::Select, operands, result_spec.clone());
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn iota(
        &mut self,
        dims: &[usize],
        axis: usize,
        dtype: DType,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        ensure!(axis < dims.len(), "iota axis out of range");
        let result_spec = tensor_spec_from(dtype, dims);
        let output = self.ctx.emit(
            Operation::Iota(IotaSpec {
                shape: result_spec.shape.clone(),
                dtype,
                axis,
            }),
            Vec::new(),
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn take(&mut self, params: ValueId, indices: ValueId, name: Option<&str>) -> Result<PtirValue> {
        let params_meta = self.fetch_value(params)?;
        let indices_meta = self.fetch_value(indices)?;
        ensure!(params_meta.rank() >= 1, "take requires operand rank >= 1");
        ensure!(
            indices_meta.dtype() == DType::Si32,
            "take indices must be si32"
        );

        let params_dims = params_meta.dims()?;
        let mut result_dims = indices_meta.dims()?.to_vec();
        result_dims.extend_from_slice(&params_dims[1..]);

        let result_spec = tensor_spec_from(params_meta.dtype(), &result_dims);
        let operands = vec![Operand::Value(params), Operand::Value(indices)];
        let output = self
            .ctx
            .emit(Operation::Take, operands, result_spec.clone());
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn gather(
        &mut self,
        value: ValueId,
        indices: ValueId,
        axis: isize,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let value_meta = self.fetch_value(value)?;
        let indices_meta = self.fetch_value(indices)?;
        ensure!(
            indices_meta.dtype() == DType::Si32,
            "gather indices must be si32"
        );
        ensure!(
            indices_meta.rank() == value_meta.rank(),
            "gather requires indices to have the same rank as the operand"
        );

        let rank = value_meta.rank();
        let axis = normalize_axis(axis, rank)?;
        let value_dims = value_meta.dims()?;
        let indices_dims = indices_meta.dims()?;

        for dim in 0..rank {
            if dim == axis {
                continue;
            }
            ensure!(
                indices_dims[dim] == value_dims[dim],
                "gather shape mismatch at axis {}: {} vs {}",
                dim,
                indices_dims[dim],
                value_dims[dim]
            );
        }

        let result_spec = tensor_spec_from(value_meta.dtype(), indices_dims);
        let operands = vec![Operand::Value(value), Operand::Value(indices)];
        let output = self.ctx.emit(
            Operation::Gather(GatherSpec {
                axis: axis as isize,
            }),
            operands,
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }

    fn rng_uniform(
        &mut self,
        dims: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> Result<PtirValue> {
        let result_spec = tensor_spec_from(dtype, dims);
        let output = self.ctx.emit(
            Operation::RngUniform(RngUniformSpec {
                shape: result_spec.shape.clone(),
                dtype,
            }),
            Vec::new(),
            result_spec.clone(),
        );
        Ok(self.register_value(name.map(|s| s.to_string()), output, result_spec))
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Tensor<'ctx, 'gb, B> {
    fn from_session(session: &Rc<RefCell<PtirGraph<'ctx, 'gb, B>>>, value: ValueId) -> Self {
        let raw = Rc::as_ptr(session) as *mut RefCell<PtirGraph<'ctx, 'gb, B>>;
        let graph = unsafe { NonNull::new_unchecked(raw) };
        Self {
            graph,
            value,
            _marker: PhantomData,
        }
    }

    fn from_parts(graph: NonNull<RefCell<PtirGraph<'ctx, 'gb, B>>>, value: ValueId) -> Self {
        Self {
            graph,
            value,
            _marker: PhantomData,
        }
    }

    fn ensure_same_session(&self, other: &Self) {
        assert!(
            self.graph == other.graph,
            "PTIR tensors originate from different sessions"
        );
    }

    fn with_graph_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&PtirGraph<'ctx, 'gb, B>) -> R,
    {
        let cell = unsafe { self.graph.as_ref() };
        let borrow = cell.borrow();
        f(&borrow)
    }

    fn with_graph_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut PtirGraph<'ctx, 'gb, B>) -> R,
    {
        let cell = unsafe { self.graph.as_ref() };
        let mut borrow = cell.borrow_mut();
        f(&mut borrow)
    }

    fn elementwise_binary(&self, rhs: &Self, op: ElementwiseBinaryOp) -> Result<Self> {
        self.ensure_same_session(rhs);
        let result =
            self.with_graph_mut(|graph| graph.elementwise_binary(op, self.value, rhs.value, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    fn elementwise_unary(&self, op: ElementwiseUnaryOp) -> Result<Self> {
        let result = self.with_graph_mut(|graph| graph.elementwise_unary(op, self.value, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    fn with_literal_binary(&self, scalar: f32, op: ElementwiseBinaryOp) -> Result<Self> {
        let literal =
            self.with_graph_mut(|graph| graph.scalar_literal_broadcast(self.value, scalar, None))?;
        let result = self
            .with_graph_mut(|graph| graph.elementwise_binary(op, self.value, literal.id(), None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    fn unwrap(result: Result<Self>, op: &str) -> Self {
        result.unwrap_or_else(|err| panic!("PTIR tensor {op} failed: {err}"))
    }

    pub fn id(&self) -> ValueId {
        self.value
    }

    pub fn ptir_bind(self, name: &'static str) -> Self {
        crate::backend::pattern::record_bind(name, self.value);
        self
    }

    pub fn spec(&self) -> TensorSpec {
        self.with_graph_ref(|graph| {
            graph
                .fetch_value(self.value)
                .expect("tensor value must exist")
                .spec()
                .clone()
        })
    }

    pub fn try_transpose(&self, perm: impl Into<Vec<usize>>) -> Result<Self> {
        let perm_vec = perm.into();
        let result = self.with_graph_mut(|graph| graph.transpose(self.value, &perm_vec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn transpose(&self, perm: impl Into<Vec<usize>>) -> Self {
        Self::unwrap(self.try_transpose(perm), "transpose")
    }

    pub fn try_slice(
        &self,
        starts: impl Into<Vec<usize>>,
        sizes: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        let starts_vec = starts.into();
        let sizes_vec = sizes.into();
        let result =
            self.with_graph_mut(|graph| graph.slice(self.value, &starts_vec, &sizes_vec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn slice(&self, starts: impl Into<Vec<usize>>, sizes: impl Into<Vec<usize>>) -> Self {
        Self::unwrap(self.try_slice(starts, sizes), "slice")
    }

    pub fn try_dynamic_slice(
        &self,
        starts: &Tensor<'ctx, 'gb, B>,
        sizes: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        let sizes_vec = sizes.into();
        let result = self.with_graph_mut(|graph| {
            graph.dynamic_slice(self.value, starts.value, &sizes_vec, None)
        })?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn dynamic_slice(
        &self,
        starts: &Tensor<'ctx, 'gb, B>,
        sizes: impl Into<Vec<usize>>,
    ) -> Self {
        Self::unwrap(self.try_dynamic_slice(starts, sizes), "dynamic_slice")
    }

    pub fn try_dynamic_update_slice(
        &self,
        update: &Tensor<'ctx, 'gb, B>,
        starts: &Tensor<'ctx, 'gb, B>,
        sizes: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        let sizes_vec = sizes.into();
        let result = self.with_graph_mut(|graph| {
            graph.dynamic_update_slice(self.value, update.value, starts.value, &sizes_vec, None)
        })?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn dynamic_update_slice(
        &self,
        update: &Tensor<'ctx, 'gb, B>,
        starts: &Tensor<'ctx, 'gb, B>,
        sizes: impl Into<Vec<usize>>,
    ) -> Self {
        Self::unwrap(
            self.try_dynamic_update_slice(update, starts, sizes),
            "dynamic_update_slice",
        )
    }

    pub fn try_broadcast_to(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape_vec = shape.into();
        let result =
            self.with_graph_mut(|graph| graph.broadcast_to(self.value, &shape_vec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn broadcast_to(&self, shape: impl Into<Vec<usize>>) -> Self {
        Self::unwrap(self.try_broadcast_to(shape), "broadcast_to")
    }

    pub fn try_add(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Add)
    }

    pub fn add(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_add(rhs), "add")
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Sub)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_sub(rhs), "sub")
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Mul)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_mul(rhs), "mul")
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Div)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_div(rhs), "div")
    }

    pub fn try_add_scalar(&self, scalar: f32) -> Result<Self> {
        self.with_literal_binary(scalar, ElementwiseBinaryOp::Add)
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        Self::unwrap(self.try_add_scalar(scalar), "add_scalar")
    }

    pub fn try_sub_scalar(&self, scalar: f32) -> Result<Self> {
        self.with_literal_binary(scalar, ElementwiseBinaryOp::Sub)
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        Self::unwrap(self.try_sub_scalar(scalar), "sub_scalar")
    }

    pub fn try_mul_scalar(&self, scalar: f32) -> Result<Self> {
        self.with_literal_binary(scalar, ElementwiseBinaryOp::Mul)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Self::unwrap(self.try_mul_scalar(scalar), "mul_scalar")
    }

    pub fn try_div_scalar(&self, scalar: f32) -> Result<Self> {
        self.with_literal_binary(scalar, ElementwiseBinaryOp::Div)
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        Self::unwrap(self.try_div_scalar(scalar), "div_scalar")
    }

    pub fn try_tanh(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Tanh)
    }

    pub fn tanh(&self) -> Self {
        Self::unwrap(self.try_tanh(), "tanh")
    }

    pub fn try_exp(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Exp)
    }

    pub fn exp(&self) -> Self {
        Self::unwrap(self.try_exp(), "exp")
    }

    pub fn try_neg(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Neg)
    }

    pub fn neg(&self) -> Self {
        Self::unwrap(self.try_neg(), "neg")
    }

    pub fn try_log(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Log)
    }

    pub fn log(&self) -> Self {
        Self::unwrap(self.try_log(), "log")
    }

    pub fn try_erf(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Erf)
    }

    pub fn erf(&self) -> Self {
        Self::unwrap(self.try_erf(), "erf")
    }

    pub fn try_reciprocal(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Reciprocal)
    }

    pub fn reciprocal(&self) -> Self {
        Self::unwrap(self.try_reciprocal(), "reciprocal")
    }

    pub fn try_sqrt(&self) -> Result<Self> {
        let rsqrt = self.try_rsqrt()?;
        rsqrt.try_reciprocal()
    }

    pub fn sqrt(&self) -> Self {
        Self::unwrap(self.try_sqrt(), "sqrt")
    }

    pub fn try_powf(&self, power: f32) -> Result<Self> {
        let log = self.try_log()?;
        let scaled = log.try_mul_scalar(power)?;
        scaled.elementwise_unary(ElementwiseUnaryOp::Exp)
    }

    pub fn powf(&self, power: f32) -> Self {
        Self::unwrap(self.try_powf(power), "powf")
    }

    pub fn try_reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims_vec = dims.into();
        let result = self.with_graph_mut(|graph| graph.reshape(self.value, &dims_vec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Self {
        Self::unwrap(self.try_reshape(dims), "reshape")
    }

    pub fn try_reduce_sum(&self, axes: impl Into<Vec<usize>>, keepdims: bool) -> Result<Self> {
        let axes_vec = axes.into();
        let result = self.with_graph_mut(|graph| {
            graph.reduce(ReduceKind::Sum, self.value, &axes_vec, keepdims, None, None)
        })?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn reduce_sum(&self, axes: impl Into<Vec<usize>>, keepdims: bool) -> Self {
        Self::unwrap(self.try_reduce_sum(axes, keepdims), "reduce_sum")
    }

    pub fn try_extract_patches(
        &self,
        window: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
        dilation: impl Into<Vec<usize>>,
        padding: impl Into<Vec<(usize, usize)>>,
        pad_value: Literal,
    ) -> Result<Self> {
        let spec = ExtractPatchesSpec {
            window: window.into(),
            strides: strides.into(),
            dilation: dilation.into(),
            padding: padding.into(),
            pad_value,
        };
        let result = self.with_graph_mut(|graph| graph.extract_patches(self.value, &spec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn extract_patches(
        &self,
        window: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
        dilation: impl Into<Vec<usize>>,
        padding: impl Into<Vec<(usize, usize)>>,
        pad_value: Literal,
    ) -> Self {
        Self::unwrap(
            self.try_extract_patches(window, strides, dilation, padding, pad_value),
            "extract_patches",
        )
    }

    pub fn try_reduce_window(&self, spec: ReduceWindowSpec) -> Result<Self> {
        let result = self.with_graph_mut(|graph| graph.reduce_window(self.value, &spec, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn reduce_window(&self, spec: ReduceWindowSpec) -> Self {
        Self::unwrap(self.try_reduce_window(spec), "reduce_window")
    }

    pub fn try_reduce_max(&self, axes: impl Into<Vec<usize>>, keepdims: bool) -> Result<Self> {
        let axes_vec = axes.into();
        let result = self.with_graph_mut(|graph| {
            graph.reduce(ReduceKind::Max, self.value, &axes_vec, keepdims, None, None)
        })?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn reduce_max(&self, axes: impl Into<Vec<usize>>, keepdims: bool) -> Self {
        Self::unwrap(self.try_reduce_max(axes, keepdims), "reduce_max")
    }

    pub fn try_broadcast_like(&self, target: &Self) -> Result<Self> {
        self.ensure_same_session(target);
        let target_dims = self
            .with_graph_ref(|graph| graph.fetch_value(target.value))?
            .dims()?
            .to_vec();
        let result =
            self.with_graph_mut(|graph| graph.broadcast_to(self.value, &target_dims, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn broadcast_like(&self, target: &Self) -> Self {
        Self::unwrap(self.try_broadcast_like(target), "broadcast_like")
    }

    pub fn try_maximum(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Maximum)
    }

    pub fn maximum(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_maximum(rhs), "maximum")
    }

    pub fn try_minimum(&self, rhs: &Self) -> Result<Self> {
        self.elementwise_binary(rhs, ElementwiseBinaryOp::Minimum)
    }

    pub fn minimum(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_minimum(rhs), "minimum")
    }

    pub fn try_abs(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Abs)
    }

    pub fn abs(&self) -> Self {
        Self::unwrap(self.try_abs(), "abs")
    }

    pub fn try_rsqrt(&self) -> Result<Self> {
        self.elementwise_unary(ElementwiseUnaryOp::Rsqrt)
    }

    pub fn rsqrt(&self) -> Self {
        Self::unwrap(self.try_rsqrt(), "rsqrt")
    }

    pub fn try_dot_general(&self, rhs: &Self, dims: &DotDims, attrs: &DotAttrs) -> Result<Self> {
        self.ensure_same_session(rhs);
        let result = self
            .with_graph_mut(|graph| graph.dot_general(self.value, rhs.value, dims, attrs, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn dot_general(&self, rhs: &Self, dims: &DotDims, attrs: &DotAttrs) -> Self {
        Self::unwrap(self.try_dot_general(rhs, dims, attrs), "dot_general")
    }

    pub fn try_concat(axis: usize, tensors: &[Self]) -> Result<Self> {
        ensure!(!tensors.is_empty(), "concat requires inputs");
        let first = tensors[0];
        let mut ids = Vec::with_capacity(tensors.len());
        ids.push(first.value);
        for tensor in &tensors[1..] {
            assert!(
                first.graph == tensor.graph,
                "concat tensors must share session"
            );
            ids.push(tensor.value);
        }
        let result = first.with_graph_mut(|graph| graph.concat(&ids, axis, None))?;
        Ok(Tensor::from_parts(first.graph, result.id()))
    }

    pub fn concat(axis: usize, tensors: &[Self]) -> Self {
        Self::unwrap(Self::try_concat(axis, tensors), "concat")
    }

    pub fn try_greater_equal(&self, rhs: &Self) -> Result<Self> {
        self.ensure_same_session(rhs);
        let result = self.with_graph_mut(|graph| {
            graph.compare(self.value, rhs.value, ComparisonOp::GreaterEqual, None)
        })?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn greater_equal(&self, rhs: &Self) -> Self {
        Self::unwrap(self.try_greater_equal(rhs), "greater_equal")
    }

    pub fn try_select(predicate: &Self, when_true: &Self, when_false: &Self) -> Result<Self> {
        assert!(predicate.graph == when_true.graph && predicate.graph == when_false.graph);
        let result = when_true.with_graph_mut(|graph| {
            graph.select(predicate.value, when_true.value, when_false.value, None)
        })?;
        Ok(Tensor::from_parts(when_true.graph, result.id()))
    }

    pub fn select(predicate: &Self, when_true: &Self, when_false: &Self) -> Self {
        Self::unwrap(Self::try_select(predicate, when_true, when_false), "select")
    }

    pub fn try_take(&self, indices: &Self) -> Result<Self> {
        self.ensure_same_session(indices);
        let result = self.with_graph_mut(|graph| graph.take(self.value, indices.value, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn take(&self, indices: &Self) -> Self {
        Self::unwrap(self.try_take(indices), "take")
    }

    pub fn try_gather(&self, indices: &Self, axis: isize) -> Result<Self> {
        self.ensure_same_session(indices);
        let result =
            self.with_graph_mut(|graph| graph.gather(self.value, indices.value, axis, None))?;
        Ok(Tensor::from_parts(self.graph, result.id()))
    }

    pub fn gather(&self, indices: &Self, axis: isize) -> Self {
        Self::unwrap(self.try_gather(indices, axis), "gather")
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<&Tensor<'ctx, 'gb, B>> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        self.add(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<f32> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: f32) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<&Tensor<'ctx, 'gb, B>> for f32 {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        rhs.add_scalar(self)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Sub<&Tensor<'ctx, 'gb, B>> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn sub(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        self.sub(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Sub<f32> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn sub(self, rhs: f32) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<&Tensor<'ctx, 'gb, B>> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        self.mul(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<f32> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<&Tensor<'ctx, 'gb, B>> for f32 {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Div<&Tensor<'ctx, 'gb, B>> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn div(self, rhs: &Tensor<'ctx, 'gb, B>) -> Self::Output {
        self.div(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Div<f32> for &Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn div(self, rhs: f32) -> Self::Output {
        self.div_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<Tensor<'ctx, 'gb, B>> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        Tensor::add(&self, &rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<f32> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: f32) -> Self::Output {
        self.add_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Add<Tensor<'ctx, 'gb, B>> for f32 {
    type Output = Tensor<'ctx, 'gb, B>;

    fn add(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        rhs.add_scalar(self)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Sub<Tensor<'ctx, 'gb, B>> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn sub(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        Tensor::sub(&self, &rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Sub<f32> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn sub(self, rhs: f32) -> Self::Output {
        self.sub_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<Tensor<'ctx, 'gb, B>> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        Tensor::mul(&self, &rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<f32> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.mul_scalar(rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Mul<Tensor<'ctx, 'gb, B>> for f32 {
    type Output = Tensor<'ctx, 'gb, B>;

    fn mul(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        rhs.mul_scalar(self)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Div<Tensor<'ctx, 'gb, B>> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn div(self, rhs: Tensor<'ctx, 'gb, B>) -> Self::Output {
        Tensor::div(&self, &rhs)
    }
}

impl<'ctx, 'gb, B: PortableBackend + 'static> Div<f32> for Tensor<'ctx, 'gb, B> {
    type Output = Tensor<'ctx, 'gb, B>;

    fn div(self, rhs: f32) -> Self::Output {
        self.div_scalar(rhs)
    }
}

impl PtirValue {
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn id(&self) -> ValueId {
        self.value
    }

    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn dims(&self) -> Result<&[usize]> {
        self.static_dims
            .as_deref()
            .ok_or_else(|| dynamic_dims_error(&self.spec))
    }

    pub fn add<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        rhs: &PtirValue,
    ) -> Result<PtirValue> {
        graph.elementwise_binary(ElementwiseBinaryOp::Add, self.id(), rhs.id(), None)
    }

    pub fn sub<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        rhs: &PtirValue,
    ) -> Result<PtirValue> {
        graph.elementwise_binary(ElementwiseBinaryOp::Sub, self.id(), rhs.id(), None)
    }

    pub fn mul<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        rhs: &PtirValue,
    ) -> Result<PtirValue> {
        graph.elementwise_binary(ElementwiseBinaryOp::Mul, self.id(), rhs.id(), None)
    }

    pub fn div<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        rhs: &PtirValue,
    ) -> Result<PtirValue> {
        graph.elementwise_binary(ElementwiseBinaryOp::Div, self.id(), rhs.id(), None)
    }

    pub fn exp<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
    ) -> Result<PtirValue> {
        graph.elementwise_unary(ElementwiseUnaryOp::Exp, self.id(), None)
    }

    pub fn tanh<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
    ) -> Result<PtirValue> {
        graph.elementwise_unary(ElementwiseUnaryOp::Tanh, self.id(), None)
    }

    pub fn reduce_max<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        axes: impl IntoIterator<Item = usize>,
        keepdims: bool,
    ) -> Result<PtirValue> {
        graph.reduce(
            ReduceKind::Max,
            self.id(),
            &axes.into_iter().collect::<Vec<_>>(),
            keepdims,
            None,
            None,
        )
    }

    pub fn reduce_sum<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        axes: impl IntoIterator<Item = usize>,
        keepdims: bool,
    ) -> Result<PtirValue> {
        graph.reduce(
            ReduceKind::Sum,
            self.id(),
            &axes.into_iter().collect::<Vec<_>>(),
            keepdims,
            None,
            None,
        )
    }

    pub fn broadcast_like<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        target: &PtirValue,
    ) -> Result<PtirValue> {
        let target_dims = target.dims()?;
        graph.broadcast_to(self.id(), target_dims, None)
    }

    pub fn add_scalar<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        scalar: f32,
    ) -> Result<PtirValue> {
        let literal = self.literal_broadcast(graph, scalar)?;
        self.add(graph, &literal)
    }

    pub fn sub_scalar<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        scalar: f32,
    ) -> Result<PtirValue> {
        let literal = self.literal_broadcast(graph, scalar)?;
        self.sub(graph, &literal)
    }

    pub fn mul_scalar<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        scalar: f32,
    ) -> Result<PtirValue> {
        let literal = self.literal_broadcast(graph, scalar)?;
        self.mul(graph, &literal)
    }

    pub fn div_scalar<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        scalar: f32,
    ) -> Result<PtirValue> {
        let literal = self.literal_broadcast(graph, scalar)?;
        self.div(graph, &literal)
    }

    pub fn literal_broadcast<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        value: f32,
    ) -> Result<PtirValue> {
        graph.scalar_literal_broadcast(self.id(), value, None)
    }

    pub fn dot_general<'ctx, 'gb, B: PortableBackend + 'static>(
        &self,
        graph: &mut PtirGraph<'ctx, 'gb, B>,
        rhs: &PtirValue,
        dims: &DotDims,
        attrs: &DotAttrs,
    ) -> Result<PtirValue> {
        graph.dot_general(self.id(), rhs.id(), dims, attrs, None)
    }
}

impl PtirResults {
    pub fn new(values: Vec<PtirValue>) -> Self {
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[PtirValue] {
        &self.values
    }

    pub fn into_values(self) -> Vec<PtirValue> {
        self.values
    }

    pub fn tuple_element(&self, index: usize) -> Result<PtirValue> {
        self.values
            .get(index)
            .cloned()
            .ok_or_else(|| anyhow!("tuple element {} out of range (len={})", index, self.len()))
    }

    pub fn into_tuple_element(self, index: usize) -> Result<PtirValue> {
        if index >= self.len() {
            return Err(anyhow!(
                "tuple element {} out of range (len={})",
                index,
                self.len()
            ));
        }
        Ok(self.values.into_iter().nth(index).unwrap())
    }

    pub fn into_value(self) -> Result<PtirValue> {
        ensure!(
            self.len() == 1,
            "expected single value result, got {}",
            self.len()
        );
        Ok(self.values.into_iter().next().unwrap())
    }
}

impl<'a, 'ctx, 'gb, B: PortableBackend + 'static> SnippetEmitter<'a, 'ctx, 'gb, B> {
    pub fn value(mut self, name: &str, value: &PtirValue) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.value(name, value.id());
        self.bindings = bindings;
        self
    }

    pub fn int(mut self, name: &str, value: usize) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.int(name, value);
        self.bindings = bindings;
        self
    }

    pub fn bool(mut self, name: &str, value: bool) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.bool(name, value);
        self.bindings = bindings;
        self
    }

    pub fn dtype(mut self, name: &str, dtype: DType) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.dtype(name, dtype);
        self.bindings = bindings;
        self
    }

    pub fn shape(mut self, name: &str, dims: impl IntoIterator<Item = usize>) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.shape(name, dims);
        self.bindings = bindings;
        self
    }

    pub fn dims(mut self, name: &str, dims: impl IntoIterator<Item = usize>) -> Self {
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.dims(name, dims);
        self.bindings = bindings;
        self
    }

    pub fn shape_like(mut self, name: &str, value: &PtirValue) -> Result<Self> {
        let dims = value.dims()?.to_vec();
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.shape(name, dims);
        self.bindings = bindings;
        Ok(self)
    }

    pub fn dims_like(mut self, name: &str, value: &PtirValue) -> Result<Self> {
        let dims = value.dims()?.to_vec();
        let mut bindings = std::mem::take(&mut self.bindings);
        bindings = bindings.dims(name, dims);
        self.bindings = bindings;
        Ok(self)
    }

    pub fn finish(self) -> Result<PtirResults> {
        let snippet_result = self.graph.ctx.emit_snippet(self.snippet, &self.bindings)?;
        let values = self.graph.register_snippet_results(snippet_result);
        Ok(PtirResults::new(values))
    }

    pub fn finish_value(self) -> Result<PtirValue> {
        self.finish()?.into_value()
    }
}

fn validate_axes(axes: &[usize], rank: usize) -> Result<Vec<usize>> {
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    for &axis in axes {
        ensure!(axis < rank, "axis {} out of range for rank {}", axis, rank);
        if seen.insert(axis) {
            ordered.push(axis);
        }
    }
    ordered.sort_unstable();
    Ok(ordered)
}

fn normalize_axis(axis: isize, rank: usize) -> Result<usize> {
    let axis = if axis < 0 { axis + rank as isize } else { axis };
    ensure!(
        axis >= 0 && axis < rank as isize,
        "axis {} out of range for rank {}",
        axis,
        rank
    );
    Ok(axis as usize)
}

fn collect_static_dims(spec: &TensorSpec) -> Option<Vec<usize>> {
    let mut dims = Vec::with_capacity(spec.shape.rank());
    for dim in spec.shape.dims() {
        match dim {
            Dimension::Static(size) => dims.push(*size),
            Dimension::Dynamic(_) => return None,
        }
    }
    Some(dims)
}

fn dynamic_dims_error(spec: &TensorSpec) -> anyhow::Error {
    for dim in spec.shape.dims() {
        if let Dimension::Dynamic(symbol) = dim {
            return anyhow!(
                "dynamic dimension `{}` is not supported in PTIR snippets",
                symbol.as_str()
            );
        }
    }
    anyhow!("dynamic dimension is not supported in PTIR snippets")
}

fn scalar_literal_tensor(dtype: DType, value: f32) -> Result<TensorLiteral> {
    let bytes = match dtype {
        DType::F32 => value.to_le_bytes().to_vec(),
        DType::F16 => f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
        DType::Bf16 => bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
        DType::F64 => (value as f64).to_le_bytes().to_vec(),
        other => {
            return Err(anyhow!(
                "scalar literal broadcast does not support dtype {:?}",
                other
            ))
        }
    };
    Ok(TensorLiteral::new(
        tensor_spec_from(dtype, &[]),
        Arc::from(bytes.into_boxed_slice()),
    ))
}
