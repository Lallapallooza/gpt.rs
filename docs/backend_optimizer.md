# PTIR rewrite system: patterns, greedy driver, and passes

## Goals

* Typed, MLIR-style `match_and_rewrite` on `Program` with `Function` and `Region` bodies. ([mlir.llvm.org][1])
* Greedy fixpoint driver with folding and DCE. Optional CSE pass. ([mlir.llvm.org][2])
* Event-driven worklist. Avoid global rescans on large graphs.
* Multi-op rewrites anchored at a root.
* Debug parity: success ⇒ IR changed, failure ⇒ no change, IR verifies. ([mlir.llvm.org][3])

## Non-goals

* No PDLL interpreter yet. The API leaves space to add it later. ([mlir.llvm.org][4])

---

## Core IR assumptions

* Single basic block per `Function.body`. Control-flow via `Region` reference ops (`cond`, `while`, `scan`).
* All ops are pure (no observable mutation). CSE is safe for pure ops with identical inputs; RNG ops must be modeled explicitly. ([mlir.llvm.org][5])

---

## Key abstractions

### Stable instruction identity

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InstId(pub u32);
```

Maintain:

* `pos_of: HashMap<InstId, usize>`
* `def_of: HashMap<ValueId, InstId>`
* `users: HashMap<ValueId, SmallVec<[InstId; 4]>>`
* `version[InstId] : u32` to invalidate failure cache entries.

### Operation key

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpKey { Cast, Reshape, Transpose, BroadcastTo, ElementwiseUnary, ElementwiseBinary, /* ... */ }
```

### Program rewriter (critical API)

```rust
pub struct ProgramRewriter<'a> {
    pub func: &'a mut Function,
    pos_of: HashMap<InstId, usize>,
    def_of: HashMap<ValueId, InstId>,
    users: HashMap<ValueId, SmallVec<[InstId; 4]>>,
    version: HashMap<InstId, u32>,
    next_value: u32,
    next_inst:  u32,
}

impl<'a> ProgramRewriter<'a> {
    // inspection
    pub fn op(&self, r: InstId) -> &Operation;
    pub fn operands(&self, r: InstId) -> &[Operand];
    pub fn value_of(&self, r: InstId) -> ValueId;
    pub fn type_of(&self, v: ValueId) -> Option<&ValueType>;
    pub fn inst_of(&self, v: ValueId) -> Option<InstId>;
    pub fn users_of(&self, v: ValueId) -> &[InstId];
    pub fn op_key(op: &Operation) -> OpKey;

    // edits
    pub fn replace_all_uses(&mut self, from: ValueId, to: ValueId);
    pub fn erase_inst(&mut self, r: InstId);
    pub fn insert_before(
        &mut self, at: InstId, op: Operation, ops: Vec<Operand>, out: ValueType
    ) -> (InstId, ValueId);
    pub fn materialize_constant(
        &mut self, at: InstId, lit: TensorLiteral, ty: ValueType
    ) -> (InstId, ValueId);

    // verification and versions
    pub fn verify(&self) -> bool;
    fn bump_version(&mut self, r: InstId);
}
```

All IR mutations must go through this API, like MLIR’s `PatternRewriter`. ([mlir.llvm.org][6])

---

## Pattern system

### Untyped base

```rust
pub trait Pattern {
    fn roots(&self) -> &'static [OpKey];
    fn benefit(&self) -> u16 { 1 }
    fn match_and_rewrite(&self, root: InstId, rw: &mut ProgramRewriter) -> bool;
}
```

### Typed trait (MLIR parity)

```rust
pub struct CastView<'a> { pub root: InstId, pub spec: &'a CastSpec }

pub trait OpRewritePattern<T> {
    fn benefit(&self) -> u16 { 1 }
    fn may_match(&self, _op: &T, _rw: &ProgramRewriter) -> bool { true }
    fn match_and_rewrite(&self, op: T, rw: &mut ProgramRewriter) -> bool;
}
```

### Adapter to `Pattern`

Implement `Pattern` once per typed pattern with a small adapter:

* Check `Operation::$Variant` at `root`, build view `T`, call the typed implementation.
  This provides `OpRewritePattern<OpT>` ergonomics like MLIR `OpRewritePattern`. ([mlir.llvm.org][1])

### Optional mini prelude

Ergonomic helpers for common checks:

* `OpRef` view with `result_tensor`, `operand_value`, `as_broadcast()`, `as_unary()`.
* `guard!` and `o!` macros for early returns.

---

## PatternSet

```rust
pub type PatternId = u32;

pub struct PatternSet { store: Vec<Box<dyn Pattern>> }
impl PatternSet {
    pub fn add<P: Pattern + 'static>(&mut self, p: P);
    pub fn freeze(self) -> FrozenPatternSet;
}

pub struct FrozenPatternSet {
    pub patterns: Vec<Box<dyn Pattern>>,
    pub by_root: HashMap<OpKey, SmallVec<[PatternId; 8]>>, // pre-sorted by benefit desc
}
```

---

## Folding and constants

Add first-class constants to enable RAUW to literals.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Operation { Constant(TensorLiteral), /* ... */ }
```

Per-op fold hook interface:

```rust
pub enum FoldResult { Nothing, ReplacedBy(ValueId), Materialize(TensorLiteral) }

pub trait Folder {
    fn try_fold(root: InstId, rw: &mut ProgramRewriter) -> FoldResult;
}
```

Called after each successful rewrite and on standalone dequeues. Matches MLIR’s “apply patterns and fold greedily”. ([mlir.llvm.org][2])

---

## Greedy driver

### Config and stats

```rust
pub struct GreedyConfig {
    pub max_iterations: usize,
    pub enable_constant_fold: bool,
    pub enable_dce: bool,
    pub cost_model: Option<fn(InstId, PatternId, &ProgramRewriter) -> u16>,
    pub expensive_checks: bool,
}

pub struct GreedyStats { pub rewrites: usize, pub folds: usize, pub erased: usize }
```

### Algorithm

* Seed a worklist with all ops except constants.
* Pop bottom-up.
* Dispatch `by_root[OpKey]`. Order by static benefit or `cost_model`.
* Failure cache: skip pattern `pid` on `(inst, version)` if it failed before.
* On success: fold, requeue users and defs, bump version, count rewrite.
* If no success: try standalone fold.
* If enabled: run cheap DCE sweep this round.
* Stop on fixpoint or `max_iterations`.

This matches MLIR greedy worklist behavior with benefit ordering and fixpoint convergence. Include expensive checks as in MLIR to assert correct pattern behavior. ([mlir.llvm.org][2])

---

## PTIR DSL and snippets

The `gpt_rs::ops::ptir` module layers a domain-specific language on top of
`GraphBuilder` so backend programs can be authored as idiomatic Rust. The DSL
keeps snippet placeholders available for tests while providing fluent helpers
that infer result metadata automatically.

### Core building blocks

* `PtirGraph::new(builder)` wraps a `GraphBuilder` during capture.
* `PtirGraph::import_spec` and `PtirGraph::import` register known SSA values
  along with their `TensorSpec`s so downstream helpers know the operand dtype
  and shape.
* `PtirValue` offers high-level operations:
  * binary ops: `.add`, `.sub`, `.div`
  * unary ops: `.exp`
  * reductions: `.reduce_max`, `.reduce_sum`
  * broadcasts: `.broadcast_like`
  * dot products: `.dot_general(&rhs, &DotDims, &DotAttrs)`
  * metadata accessors: `.dtype()`, `.rank()`, `.dims()` (cached) for shape inference and validation
  * tuple helpers: snippet captures return `PtirResults`, offering `.into_value()` for single-output snippets and `.tuple_element(index)` for multi-result cases

Each helper emits the corresponding PTIR `Operation` while statically checking
axis bounds and dtype compatibility.

### Softmax rewritten with the DSL

```rust
let backend_spec = TensorSpec::new(
    backend_dtype(dtype),
    backend_shape_from_device_shape(x.shape()),
);

let value = graph.capture(|ctx| {
    let x_id = ctx.import(x)?;
    let mut ptir = PtirGraph::new(ctx);
    let x_val = ptir.import_spec("x", x_id, backend_spec.clone());

    let max = x_val.reduce_max(&mut ptir, [axis], true)?;
    let max_broadcast = max.broadcast_like(&mut ptir, &x_val)?;
    let shifted = x_val.sub(&mut ptir, &max_broadcast)?;
    let exp = shifted.exp(&mut ptir)?;
    let sum = exp.reduce_sum(&mut ptir, [axis], true)?;
    let sum_broadcast = sum.broadcast_like(&mut ptir, &x_val)?;
    let softmax = exp.div(&mut ptir, &sum_broadcast)?;
    Ok(softmax.id())
})?;
```

The example mirrors the old handwritten builder sequence but compresses it into
clear, type-safe operations. `crates/gpt-rs/tests/functional_softmax.rs` records
the generated IR to guard against regressions.

### Dot general ergonomics

`PtirValue::dot_general` consumes a `DotDims` description and optional
`DotAttrs`, handling batch and contracting axis validation before emitting a
`DotGeneral` instruction:

```rust
let dims = DotDims::new(axes![0], axes![2], axes![1]);
let attrs = DotAttrs::default();
let product = lhs.dot_general(&mut graph, &rhs, &dims, &attrs)?;
```

Associated unit tests live in `crates/gpt-rs/tests/ptir_dsl.rs`, with snippet
parsing coverage in `crates/gpt-rs/tests/ptir_snippet.rs`.

Whenever authoring new functional kernels or rewrite patterns, prefer the DSL
over raw `ctx.emit(...)` calls—doing so keeps PTIR concise, verifiable, and
easier to audit.

---

## PTIR tracing and dumps

The `gpt_rs::ops::trace` module exposes hooks for recording every PTIR program
that a lazy graph executes. Installing an `ExecutionTraceSink` (for example
`FileTraceSink`) yields callbacks before and after each `run_program`, including
metadata such as the graph id, target `ValueId`, backend, and execution
duration. The CLI wires this up via `--dump-dir`, emitting `program.ptir`,
`program.json`, and `meta/stats` files per execution. Library integrations can
call `trace::install_global_sink(...)` to plug in custom collectors or telemetry
pipelines, and future pass-manager wiring will forward `PassEvent` records
through the same interface.

---

## DCE and CSE

* **DCE:** remove ops with no users that are not function results. Run each round if enabled.
* **CSE pass (separate):** structural hash `(OpKey, attrs, operand ids, result type)` for pure ops only. Ensure dominance: defining op must dominate all uses in the region. Mirrors MLIR `-cse`. ([mlir.llvm.org][5])

---

## Regions and recursion

Apply greedy to every `Function.body` and recurse into each `Region.body` referenced by control-flow ops. Mirrors `applyPatternsGreedily(Operation*)`. ([mlir.llvm.org][2])

---

## Verification and debug

* Fast verifier per instruction.
* If `expensive_checks`: after every mutation, assert “success ⇒ changed”, “failure ⇒ unchanged”, and IR verifies. This copies MLIR’s greedy checker. ([mlir.llvm.org][3])

---

## Performance

* Event-driven. Only dirty ops are re-matched.
* Failure cache keyed by `(InstId, version)` reduces redundant attempts.
* Bottom-up pops shorten DCE cascades.
* Complexity scales with number of successful rewrites plus local neighborhoods, not `ops × patterns`.
* CSE left out of the inner loop to keep the driver lean.

---

## Pass interface and pipeline

```rust
pub trait Pass { fn name(&self) -> &'static str; fn run_on_program(&self, p: &mut Program); }

pub struct PassManager { passes: Vec<Box<dyn Pass>> }
impl PassManager {
    pub fn add<P: Pass + 'static>(&mut self, p: P);
    pub fn run(&self, p: &mut Program);
}
```

Ship:

* `CanonicalizePass(GreedyConfig)`
* `CSEPass`
* `DCEPass`
  Default pipeline: `Canonicalize → CSE → DCE`. Mirrors MLIR’s `-canonicalize` and `-cse` flows. ([mlir.llvm.org][7])

---

## Extensibility

* PDLL-like front-end can compile to `OpRewritePattern<T>` later. ([mlir.llvm.org][4])
* Add per-op `populate_canonicalization(ps: &mut PatternSet)` to co-locate canonicalizers, like MLIR. ([mlir.llvm.org][8])

---

# Five usage examples

Below, `rw.*` calls are methods from `ProgramRewriter`. All examples compile conceptually and follow the typed-pattern adapter.

## 1) Remove no-op cast

```rust
pub struct RemoveNoOpCast;

impl OpRewritePattern<CastView<'_>> for RemoveNoOpCast {
    fn benefit(&self) -> u16 { 1 }
    fn may_match(&self, op: &CastView<'_>, rw: &ProgramRewriter) -> bool {
        let [Operand::Value(src)] = rw.operands(op.root) else { return false; };
        matches!(rw.type_of(src), Some(ValueType::Tensor(t))) && t.dtype == op.spec.dtype
    }
    fn match_and_rewrite(&self, op: CastView<'_>, rw: &mut ProgramRewriter) -> bool {
        let [Operand::Value(src)] = rw.operands(op.root) else { return false; };
        rw.replace_all_uses(rw.value_of(op.root), src);
        rw.erase_inst(op.root);
        true
    }
}

impl Pattern for RemoveNoOpCast {
    fn roots(&self) -> &'static [OpKey] { &[OpKey::Cast] }
    fn benefit(&self) -> u16 { <Self as OpRewritePattern<CastView<'_>>>::benefit(self) }
    fn match_and_rewrite(&self, root: InstId, rw: &mut ProgramRewriter) -> bool {
        let Operation::Cast(spec) = rw.op(root) else { return false; };
        <Self as OpRewritePattern<CastView<'_>>>::match_and_rewrite(self, CastView{root, spec}, rw)
    }
}
```

Matches MLIR’s canonical “erase redundant cast”. ([mlir.llvm.org][8])

## 2) Fuse unary through broadcast

`un(broadcast(x)) → broadcast(un(x))` when shape-preserving and dtype matches.

```rust
pub struct FuseUnaryThroughBroadcast;

pub struct UnaryView { pub root: InstId, pub kind: ElementwiseUnaryOp }
impl OpRewritePattern<UnaryView> for FuseUnaryThroughBroadcast {
    fn benefit(&self) -> u16 { 3 }
    fn match_and_rewrite(&self, op: UnaryView, rw: &mut ProgramRewriter) -> bool {
        let [Operand::Value(v)] = rw.operands(op.root) else { return false; };
        let Some(bid) = rw.inst_of(v) else { return false; };
        let Operation::BroadcastTo(spec) = rw.op(bid).clone() else { return false; };

        // dtype check
        let ValueType::Tensor(out_t) = rw.type_of(rw.value_of(op.root))?.clone() else { return false; };
        let ValueType::Tensor(x_t)   = rw.type_of(rw.value_of(bid))?.clone() else { return false; };
        if out_t.dtype != x_t.dtype { return false; }

        // rewrite
        let x = match rw.operands(bid) { [Operand::Value(x)] => *x, _ => return false };
        let (_u_i, u_v) = rw.insert_before(bid, Operation::ElementwiseUnary(op.kind),
                                           vec![Operand::Value(x)], ValueType::Tensor(x_t));
        let (_n_i, n_v) = rw.insert_before(op.root, Operation::BroadcastTo(spec),
                                           vec![Operand::Value(u_v)], ValueType::Tensor(out_t));
        rw.replace_all_uses(rw.value_of(op.root), n_v);
        rw.erase_inst(op.root);
        true
    }
}
impl Pattern for FuseUnaryThroughBroadcast {
    fn roots(&self) -> &'static [OpKey] { &[OpKey::ElementwiseUnary] }
    fn match_and_rewrite(&self, root: InstId, rw: &mut ProgramRewriter) -> bool {
        if let Operation::ElementwiseUnary(k) = rw.op(root) {
            <Self as OpRewritePattern<UnaryView>>::match_and_rewrite(self, UnaryView{root, kind:*k}, rw)
        } else { false }
    }
}
```

Standard canonicalization move. ([mlir.llvm.org][8])

## 3) Fold transpose∘transpose to identity

```rust
pub struct FoldTransposeOfTranspose;

pub struct TransposeView<'a> { pub root: InstId, pub spec: &'a TransposeSpec }
impl OpRewritePattern<TransposeView<'_>> for FoldTransposeOfTranspose {
    fn benefit(&self) -> u16 { 2 }
    fn match_and_rewrite(&self, op: TransposeView<'_>, rw: &mut ProgramRewriter) -> bool {
        let [Operand::Value(mid)] = rw.operands(op.root) else { return false; };
        let Some(t1) = rw.inst_of(mid) else { return false; };
        let Operation::Transpose(spec1) = rw.op(t1) else { return false; };

        // check inverse permutation
        let p0 = &op.spec.perm; let p1 = &spec1.perm;
        if p0.len() != p1.len() { return false; }
        let is_inv = p0.iter().enumerate().all(|(i,&v)| p1[v] == i);
        if !is_inv { return false; }

        // rewrite to producer of t1
        let src = match rw.operands(t1) { [Operand::Value(src)] => *src, _ => return false };
        rw.replace_all_uses(rw.value_of(op.root), src);
        rw.erase_inst(op.root);
        true
    }
}
impl Pattern for FoldTransposeOfTranspose {
    fn roots(&self) -> &'static [OpKey] { &[OpKey::Transpose] }
    fn match_and_rewrite(&self, root: InstId, rw: &mut ProgramRewriter) -> bool {
        if let Operation::Transpose(spec) = rw.op(root) {
            <Self as OpRewritePattern<TransposeView<'_>>>::match_and_rewrite(self, TransposeView{root, spec}, rw)
        } else { false }
    }
}
```

A canonical, local fold. ([mlir.llvm.org][8])

## 4) Algebraic: `mul(x, 1) → x` and `add(x, 0) → x`

Assumes `Constant` op and a `literal_is_one/zero(dtype)` helper.

```rust
pub struct FoldMulByOne;

pub struct BinaryView<'a> { pub root: InstId, pub kind: ElementwiseBinaryOp, pub lhs: &'a Operand, pub rhs: &'a Operand }
impl OpRewritePattern<BinaryView<'_>> for FoldMulByOne {
    fn benefit(&self) -> u16 { 2 }
    fn may_match(&self, op: &BinaryView<'_>, _rw: &ProgramRewriter) -> bool {
        matches!(op.kind, ElementwiseBinaryOp::Mul)
    }
    fn match_and_rewrite(&self, op: BinaryView<'_>, rw: &mut ProgramRewriter) -> bool {
        let one = |o: &Operand| matches!(o, Operand::Literal(t) if literal_is_one(&t.spec.dtype, &t.bytes));
        match (one(op.lhs), one(op.rhs)) {
            (true, false) => if let Operand::Value(x) = op.rhs { rw.replace_all_uses(rw.value_of(op.root), *x); rw.erase_inst(op.root); return true; }
            (false, true) => if let Operand::Value(x) = op.lhs { rw.replace_all_uses(rw.value_of(op.root), *x); rw.erase_inst(op.root); return true; }
            _ => {}
        }
        false
    }
}
impl Pattern for FoldMulByOne {
    fn roots(&self) -> &'static [OpKey] { &[OpKey::ElementwiseBinary] }
    fn match_and_rewrite(&self, root: InstId, rw: &mut ProgramRewriter) -> bool {
        if let Operation::ElementwiseBinary(k) = rw.op(root) {
            let ops = rw.operands(root);
            <Self as OpRewritePattern<BinaryView<'_>>>::match_and_rewrite(
                self, BinaryView{ root, kind:*k, lhs:&ops[0], rhs:&ops[1] }, rw)
        } else { false }
    }
}
```

This mirrors typical MLIR algebraic canonicalizers. ([mlir.llvm.org][8])

## 5) Build and run a pipeline

```rust
// 1) Populate patterns
let mut ps = PatternSet { store: vec![] };
ps.add(RemoveNoOpCast);
ps.add(FuseUnaryThroughBroadcast);
ps.add(FoldTransposeOfTranspose);
ps.add(FoldMulByOne);
let frozen = ps.freeze();

// 2) Canonicalize pass
pub struct CanonicalizePass { cfg: GreedyConfig, pats: FrozenPatternSet }
impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str { "ptir-canonicalize" }
    fn run_on_program(&self, p: &mut Program) {
        for f in &mut p.functions {
            let _stats = apply_patterns_and_fold_greedily(f, &self.pats, &self.cfg);
        }
        // recurse into p.regions if needed
    }
}

// 3) CSE and DCE passes (not shown here) match MLIR’s `-cse` and final DCE sweep.
let mut pm = PassManager { passes: vec![] };
pm.add(CanonicalizePass {
    cfg: GreedyConfig {
        max_iterations: usize::MAX,
        enable_constant_fold: true,
        enable_dce: true,
        cost_model: None,
        expensive_checks: cfg!(debug_assertions),
    },
    pats: frozen,
});
pm.add(CSEPass {});
pm.add(DCEPass {});
pm.run(&mut program);
```

This mirrors MLIR’s `-canonicalize` then `-cse` pattern. ([mlir.llvm.org][7])

---

## Testing and quality gates

* Golden before/after tests for each pattern.
* Driver step log with “changed op”, “folded”, “erased”.
* Enable expensive checks in debug builds. Mirrors MLIR sanity checks. ([mlir.llvm.org][3])

That is the full design. If you want, I can write the `apply_patterns_and_fold_greedily` function and the DCE sweep next.

[1]: https://mlir.llvm.org/docs/PatternRewriter/ "Pattern Rewriting : Generic DAG-to-DAG Rewriting"
[2]: https://mlir.llvm.org/doxygen/GreedyPatternRewriteDriver_8h.html "GreedyPatternRewriteDriver.h File Reference"
[3]: https://mlir.llvm.org/doxygen/GreedyPatternRewriteDriver_8cpp_source.html "lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp Source File"
[4]: https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/ "Quickstart tutorial to adding MLIR graph rewrite"
[5]: https://mlir.llvm.org/doxygen/CSE_8cpp_source.html "lib/Transforms/CSE.cpp Source File - MLIR - LLVM.org"
[6]: https://mlir.llvm.org/doxygen/classmlir_1_1PatternRewriter.html "mlir::PatternRewriter Class Reference"
[7]: https://mlir.llvm.org/docs/Passes/ "Passes - MLIR - LLVM"
[8]: https://mlir.llvm.org/docs/Canonicalization/ "Operation Canonicalization - MLIR - LLVM"
