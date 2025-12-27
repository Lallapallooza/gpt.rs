# Operations Layer Architecture

This document describes the design and architecture of the `ops` module in gpt-rs, which provides the core abstraction layer between high-level neural network operations and low-level backend execution.

## Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Subsystems](#core-subsystems)
  - [Functional Operations](#functional-operations)
  - [Graph Arena](#graph-arena)
  - [PTIR (Portable Tensor IR)](#ptir-portable-tensor-ir)
  - [Trace System](#trace-system)
- [Execution Flow](#execution-flow)
- [Design Patterns](#design-patterns)
- [Key Abstractions](#key-abstractions)

## Overview

The `ops` module implements a three-layer architecture that separates:

1. **Frontend**: High-level functional operators (attention, layer norm, matmul, etc.)
2. **IR Layer**: Portable tensor intermediate representation (PTIR)
3. **Backend**: Device-specific execution engines

This separation enables:
- **Backend portability**: Same functional code runs on CPU, GPU, or custom accelerators
- **Lazy execution**: Operations are captured in graphs and compiled on-demand
- **Graph optimization**: IR rewrites before backend lowering
- **Custom kernels**: Backends can override functional implementations

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                           │
│                    (Neural Network Layers)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             |
                             | calls functional ops
                             v
┌─────────────────────────────────────────────────────────────────────┐
│                    Functional Operations Layer                      │
│                     (ops/functional/*)                              │
│                                                                     │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐      │
│  │ Activation │  │ Attention│  │ LinAlg   │  │ Normalization │      │
│  │ (softmax,  │  │ (scaled  │  │ (matmul, │  │ (layer_norm,  │      │
│  │  gelu)     │  │  dot     │  │  concat) │  │  rms_norm)    │      │
│  └────────────┘  │  product)│  └──────────┘  └───────────────┘      │
│                  └──────────┘                                       │
│                        |                                            │
│                        | validate + capture                         │
│                        v                                            │
│                 ┌─────────────┐                                     │
│                 │  Registry   │  <-- Optional custom impls          │
│                 │  (runtime   │                                     │
│                 │   dispatch) │                                     │
│                 └─────────────┘                                     │
└────────────────────────┬────────────────────────────────────────────┘
                         |
                         | capture_ptir! macro
                         v
┌─────────────────────────────────────────────────────────────────────┐
│                      PTIR DSL Layer                                 │
│                     (ops/ptir/*)                                    │
│                                                                     │
│  ┌──────────────┐       ┌──────────────┐      ┌─────────────────┐   │
│  │ PtirSession  │ ----> │  PtirGraph   │ ---> │  Tensor<'ctx>   │   │
│  │ (entry point)│       │  (builder)   │      │  (DSL handle)   │   │
│  └──────────────┘       └──────────────┘      └─────────────────┘   │
│                                                                     │
│  Operations: try_add, try_mul, reshape, transpose, dot_general,     │
│              reduce_sum, slice, broadcast, etc.                     │
└────────────────────────┬────────────────────────────────────────────┘
                         |
                         | emit graph nodes
                         v
┌─────────────────────────────────────────────────────────────────────┐
│                      Graph Arena Layer                              │
│                     (ops/graph/*)                                   │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ GraphArena   │ <------ Arc<Backend>                              │
│  │  (storage)   │                                                   │
│  └──────┬───────┘                                                   │
│         |                                                           │
│         +----> GraphInner (nodes, parameters, exports)              │
│         |                                                           │
│         +----> PlanCache (compiled program cache)                   │
│         |                                                           │
│         +----> Optimizer (graph rewrites)                           │
│                                                                     │
│  ┌──────────────┐       ┌──────────────┐                            │
│  │ GraphBuilder │       │ CachedPlan   │                            │
│  │ (node emit)  │       │ (compiled    │                            │
│  └──────────────┘       │  program)    │                            │
│                         └──────────────┘                            │
└────────────────────────┬────────────────────────────────────────────┘
                         |
                         | compile & execute
                         v
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend Layer                                  │
│                 (backend/spec.rs + backend crates)                  │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ RefCpuBackend   │  │ FaerBackend  │  │ Custom Backends... │      │
│  │ (interpreter)   │  │ (optimized)  │  │                    │      │
│  └─────────────────┘  └──────────────┘  └────────────────────┘      │
│                                                                     │
│  Common Interface: PortableBackend trait                            │
│    - execute_program(Program) -> results                            │
│    - backend_name() -> &str                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Subsystems

### Functional Operations

**Location**: `crates/gpt-rs/src/ops/functional/`

Functional operators provide the primary API for tensor computations. Each functional module follows a consistent pattern:

#### Structure

```text
ops/functional/
├── mod.rs           # Module organization & re-exports
├── common.rs        # DeviceTensorOps trait & validation helpers
├── registry.rs      # Runtime dispatch system
├── runtime.rs       # Thread-local registry management
├── activation.rs    # Softmax, GELU, etc.
├── attention.rs     # Scaled dot-product attention
├── embedding.rs     # Embedding lookup
├── linalg.rs        # Matrix multiplication
├── normalization.rs # Layer norm, RMS norm
├── stochastic.rs    # Dropout
└── tensor_ops.rs    # Elementwise ops, bias addition
```

#### Validate-Capture Pattern

Every functional operation follows a two-phase pattern:

1. **Validation Phase**: `validate_*` function
   - Checks tensor shapes, dtypes, and backends
   - Returns a `*Plan` struct containing validated metadata
   - Pure function, no side effects

2. **Capture Phase**: `capture_*` function
   - Takes the plan and input tensors
   - Emits PTIR operations into a graph
   - Returns a `DeviceTensor` wrapping the lazy result

**Example** (from `functional/linalg.rs`):

```rust
// Plan struct holds validated metadata
struct MatmulPlan {
    dot_dims: DotDims,
    output_shape: Vec<usize>,
    requires_grad: bool,
}

// Validation: check shapes, compute output shape
fn validate_matmul(a: &DeviceTensor<B>, b: &DeviceTensor<B>) -> Result<MatmulPlan> {
    ensure_rank_at_least("matmul lhs", a, 2)?;
    ensure_rank_at_least("matmul rhs", b, 2)?;
    // ... shape checks ...
    Ok(MatmulPlan { dot_dims, output_shape, requires_grad })
}

// Capture: emit PTIR operations
fn capture_matmul(plan: MatmulPlan, a: &DeviceTensor<B>, b: &DeviceTensor<B>)
    -> Result<DeviceTensor<B>>
{
    capture_ptir! {
        { lhs = a, rhs = b },
        |session| {
            let result = lhs.dot_general(&rhs, &plan.dot_dims, &DotAttrs::default())?;
            Ok(result.id())
        }
    }?.into_device_tensor(plan.requires_grad)
}

// Public API combines validation + capture
#[support_runtime_overload]
pub fn matmul<B: PortableBackend>(
    _backend: &B,  // Required by macro for registry lookup
    a: &DeviceTensor<B>,
    b: &DeviceTensor<B>,
) -> Result<DeviceTensor<B>> {
    let plan = validate_matmul(a, b)?;
    capture_matmul(plan, a, b)
}
```

#### Registry System

The registry enables runtime selection of functional implementations:

```text
┌──────────────────────────────────────────────────┐
│            FunctionalRegistry<B>                 │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │ OpRegistry<AttentionImpl>                │    │
│  │  - implementations: Vec<Arc<Impl>>       │    │
│  │  - policy: FunctionalPolicy              │    │
│  │  - benchmark_cache: LruCache             │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  Policies:                                       │
│   - Default: first supporting impl               │
│   - Force: specific impl by name                 │
│   - Benchmark: run all, cache fastest            │
└──────────────────────────────────────────────────┘
```

**Configuration** (via `FunctionalOverrides`):

```json
{
  "attention": "force=flash_attention",
  "matmul": "benchmark(cache=64)"
}
```

### Graph Arena

**Location**: `crates/gpt-rs/src/ops/graph/`

The graph arena is the central orchestrator for lazy execution. It maintains a mutable graph of pending operations and compiles them into backend programs on demand.

#### Components

```text
ops/graph/
├── arena.rs   # GraphArena (main storage + cache)
├── builder.rs # GraphBuilder (node emission API)
├── context.rs # Thread-local default arena stack
├── plan.rs    # Plan caching & deduplication
├── state.rs   # GraphInner (mutable graph state)
└── mod.rs     # Re-exports
```

#### Lazy Execution Flow

```text
1. Capture Phase
   ┌──────────────┐
   │ Functional   │
   │ Operation    │
   └──────┬───────┘
          |
          | arena.capture(|builder| { ... })
          v
   ┌──────────────┐
   │ GraphBuilder │ ---> Emits nodes into GraphInner
   └──────────────┘
          |
          | Returns ValueId
          v
   ┌──────────────┐
   │ DeviceTensor │ ---> Lazy (not materialized yet)
   │ (lazy mode)  │
   └──────────────┘

2. Materialization (when tensor data is accessed)
   ┌──────────────┐
   │ DeviceTensor │
   │ .to_host()   │
   └──────┬───────┘
          |
          | arena.compile_and_execute(exported_ids)
          v
   ┌──────────────┐
   │ PlanCache    │ ---> Check cache by signature
   │ lookup       │
   └──────┬───────┘
          |
          | Miss or stale?
          v
   ┌──────────────┐
   │ Compile Plan │ ---> Topological sort, optimizer pass
   │              │
   └──────┬───────┘
          |
          | Backend.execute_program(program)
          v
   ┌──────────────┐
   │ Backend      │ ---> Returns tensors
   │ Execution    │
   └──────┬───────┘
          |
          | Store in DeviceTensor
          v
   ┌──────────────┐
   │ DeviceTensor │
   │ (eager mode) │
   └──────────────┘
```

#### Version-Based Staleness

The arena maintains a version counter that increments when nodes are added:

```rust
struct GraphInner<B> {
    nodes: HashMap<ValueId, NodeRecord>,
    parameters: Vec<ParameterSpec>,
    exports: HashSet<ValueId>,
    version: u64,  // Incremented on each modification
}

struct CachedPlan {
    version: u64,       // Graph version at compile time
    program: Program,   // Compiled backend program
    // ...
}
```

**Staleness detection**:
- When reusing a cached plan, compare `plan.version` vs `arena.version`
- If `arena.version > plan.version`, recompile to include new nodes
- This allows multiple ops to share graphs while invalidating stale caches

### PTIR (Portable Tensor IR)

**Location**: `crates/gpt-rs/src/ops/ptir/`

PTIR is a domain-specific language for constructing backend-portable tensor computation graphs. It provides a fluent API for building operations that are lowered to backend-specific programs.

#### Structure

```text
ops/ptir/
├── mod.rs       # Public API & helper traits
├── graph.rs     # PtirSession, PtirGraph, Tensor<'ctx>
├── tensor.rs    # TensorPlaceholder, dtype traits
└── axes.rs      # Axis specification helpers
```

#### Key Types

```text
┌──────────────────┐
│ PtirSession      │ <--- Entry point (created by capture_ptir!)
│ - import()       │
│ - scalar()       │
│ - iota()         │
└────────┬─────────┘
         |
         | wraps
         v
┌──────────────────┐
│ PtirGraph        │ <--- Internal builder state
│ - values map     │
│ - GraphBuilder   │
└────────┬─────────┘
         |
         | creates
         v
┌──────────────────┐
│ Tensor<'ctx>     │ <--- Copyable DSL handle
│ - try_add()      │
│ - try_mul()      │
│ - reshape()      │
│ - dot_general()  │
│ - reduce_sum()   │
│ - ...            │
└──────────────────┘
```

#### DSL Operations

**Binary Ops**:
- `try_add`, `try_sub`, `try_mul`, `try_div`
- `try_maximum`, `try_minimum`

**Unary Ops**:
- `sqrt`, `exp`, `log`, `tanh`, `erf`
- `try_neg`, `try_abs`

**Shape Ops**:
- `reshape(new_dims)` - Change shape
- `transpose(permutation)` - Permute axes
- `broadcast(target_shape, broadcast_dims)` - Expand dimensions
- `slice(starts, limits)` - Extract subtensor
- `concat(other, axis)` - Concatenate along axis

**Reductions**:
- `reduce_sum(axes)` - Sum over axes
- `reduce_max(axes)` - Max over axes
- `reduce_mean(axes)` - Mean over axes

**Matrix Ops**:
- `dot_general(rhs, dims, attrs)` - Generalized batched matmul

#### Example Usage

```rust
capture_ptir! {
    { x, y },  // Import device tensors
    |session| {
        // Build computation graph
        let sum = x.try_add(&y)?;
        let scaled = sum.try_mul(&session.scalar(2.0))?;
        let normed = scaled.reduce_mean(axes![1])?;
        Ok(normed.id())  // Return ValueId
    }
}
```

### Trace System

**Location**: `crates/gpt-rs/src/ops/trace/`

The trace system provides instrumentation hooks for debugging, profiling, and program dumping.

#### Features

- **Program Dumping**: Write compiled PTIR programs to disk for inspection
- **Profiling**: Time backend operations and log statistics
- **Debug Output**: Print tensor shapes, dtypes, and intermediate values

#### Configuration

```rust
let trace_config = TraceConfig {
    dump_programs: Some("dumps/".into()),  // Write programs to directory
    profile_backend: true,                  // Time operations
    debug_tensors: true,                    // Print tensor metadata
};
```

## Execution Flow

### End-to-End Example: `layer_norm(x)`

```text
1. Application calls layer_norm()
   |
   v
2. Functional: validate_layer_norm(x)
   - Check shape, dtype
   - Compute normalization axes
   - Return LayerNormPlan
   |
   v
3. Functional: capture_layer_norm(plan, x)
   - capture_ptir! creates PtirSession
   - Import x as PTIR tensor
   |
   v
4. PTIR DSL: build computation graph
   - mean = x.reduce_mean(axes)
   - var = (x - mean).try_mul(&(x - mean))?.reduce_mean(axes)
   - normalized = (x - mean) / sqrt(var + eps)
   |
   v
5. PTIR emits operations to GraphBuilder
   - GraphBuilder records nodes in GraphArena
   - Returns ValueId for normalized tensor
   |
   v
6. DeviceTensor created in lazy mode
   - Wraps Arc<GraphArena> + ValueId
   - Marks ValueId as exported
   |
   v
7. Application accesses tensor.to_host()
   - Arena compiles plan (if not cached)
   - Topological sort of graph nodes
   - Optimizer rewrites (constant folding, fusion, etc.)
   - Lower to backend Program
   |
   v
8. Backend executes program
   - Returns materialized tensors
   - DeviceTensor transitions to eager mode
   |
   v
9. Data returned to application
```

## Design Patterns

### 1. Separation of Concerns

**Functional Layer**:
- High-level semantics
- Shape validation
- Backend-agnostic

**PTIR Layer**:
- Portable IR
- Graph construction
- Operation emission

**Backend Layer**:
- Device-specific kernels
- Memory management
- Execution

### 2. Lazy Evaluation

Operations are captured in graphs and deferred until materialization. Benefits:

- **Fusion**: Multiple ops can be fused into a single kernel
- **Optimization**: Graph rewrites before execution
- **Memory**: Avoid materializing intermediate tensors

### 3. Graph Reuse

Multiple operations can share the same `GraphArena`:

```rust
let arena = GraphArena::new(backend);

// Operation 1
let y = with_graph(some_func(x), Some(arena.clone()))?;

// Operation 2 (shares graph with operation 1)
let z = with_graph(other_func(y), Some(arena.clone()))?;

// Both operations are fused into a single program
let result = z.to_host()?;
```

### 4. Validation Helpers

Common validation patterns are extracted into reusable helpers in `functional/common.rs`:

```rust
ensure_same_backend(op, lhs, rhs)?;     // Check same backend instance
ensure_same_dtype(lhs_name, lhs, rhs_name, rhs)?;
ensure_rank(tensor_name, tensor, expected_rank)?;
ensure_shape_matches(lhs_name, lhs, rhs_name, rhs)?;
ensure_axis_in_bounds(tensor_name, tensor, axis)?;
```

These provide consistent error messages and reduce boilerplate.

### 5. Extension Traits

`DeviceTensorOps` trait provides ergonomic method syntax:

```rust
// Instead of:
let result = functional::add(backend, &x, &y)?;

// Write:
let result = x.add(&y)?;
```

## Key Abstractions

### DeviceTensor

**Location**: `crates/gpt-rs/src/tensor/device_tensor.rs`

Represents a tensor with two modes:

- **Eager**: Data is materialized in backend storage
- **Lazy**: Data is represented as `(GraphArena, ValueId)` pair

```rust
pub enum DeviceTensor<B: PortableBackend> {
    Eager {
        backend: Arc<B>,
        storage: B::Storage,
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
    },
    Lazy {
        arena: Arc<GraphArena<B>>,
        shape: Shape,
        dtype: DType,
        value: ValueId,
        requires_grad: bool,
    },
}
```

### PortableBackend

**Location**: `crates/gpt-rs/src/backend/spec.rs`

Trait defining the interface all backends must implement:

```rust
pub trait PortableBackend: Send + Sync {
    type Storage;

    fn backend_name(&self) -> &str;

    fn execute_program(&self, program: &Program) -> Result<ProgramResults>;

    fn allocate(&self, spec: &TensorSpec) -> Result<Self::Storage>;

    fn read_storage(&self, storage: &Self::Storage, spec: &TensorSpec) -> Result<Vec<u8>>;

    // ... more methods ...
}
```

### Program

**Location**: `crates/gpt-rs/src/backend/spec.rs`

Represents a compiled computation graph ready for backend execution:

```rust
pub struct Program {
    pub parameters: Vec<TensorSpec>,  // Input placeholders
    pub operations: Vec<Operation>,   // Computation nodes
    pub outputs: Vec<ValueId>,        // Results to return
}
```

Operations include:
- `Constant(TensorLiteral)` - Literal values
- `ElementwiseBinary(op, lhs, rhs)` - Binary ops
- `ElementwiseUnary(op, input)` - Unary ops
- `DotGeneral(lhs, rhs, spec)` - Matrix multiplication
- `Reduce(input, kind, axes)` - Reductions
- `Reshape(input, spec)` - Shape manipulation
- And more...

## Summary

The `ops` module provides a sophisticated multi-layer architecture for tensor operations:

1. **Functional operators** provide a high-level, ergonomic API
2. **PTIR DSL** enables backend-portable graph construction
3. **Graph arena** manages lazy execution and caching
4. **Backends** implement device-specific kernels

This design enables portability, performance, and extensibility while maintaining a clean separation of concerns.
