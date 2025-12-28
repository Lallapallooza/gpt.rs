# PTIR optimizer: passes, patterns, and the pipeline

This document describes the PTIR optimizer used by the graph arena before backend execution.

It is intentionally implementation-oriented; the PTIR op semantics live in `docs/backend.md`.

## Where it lives

- `crates/gpt-rs/src/backend/optimizer/`: pass traits + optimize context + entry signatures.
- `crates/gpt-rs/src/backend/pipeline.rs`: default pipeline definition (fixed-point + passes).
- `crates/gpt-rs/src/backend/passes/`: built-in canonicalization / simplification / DCE / CSE passes.
- `crates/gpt-rs/src/backend/pattern/`: typed operation views + pattern helpers.
- `crates/gpt-rs/src/backend/driver.rs`: greedy rewrite driver (worklist + failure cache) used by passes.
- `crates/gpt-rs/src/backend/rewriter.rs`: `ProgramRewriter` (stable `InstId`, SSA edits).

## High-level flow

When a `GraphArena` compiles a plan, it:

1. Builds a PTIR `Function` from captured nodes.
2. Computes a stable `EntrySignature` (parameter roles + stable ids).
3. Runs the optimizer (`Optimizer::optimize(function, cx)`).
4. Caches the optimized program by signature.

The optimizer does not know about "training vs inference" modes; it rewrites pure PTIR.

## Pass model

All passes implement:

- `FunctionPass<B>`: `run(function, cx) -> PassResult`

The default optimizer is `PipelineOptimizer<B>`, which runs:

- a bounded fixed-point loop of canonicalization/simplification passes
- backend-specific hooks (optional)
- param-only hoisting (`ParamOnlyFoldToParamPass`)
- another bounded fixed-point cleanup loop (including `CommonSubexpressionEliminationPass` and `DeadCodeEliminationPass`)

Tuning/debug knobs:

- `GPTRS_OPT_PRE_ITERS`: fixed-point iterations for the "pre" loop (default: 2).
- `GPTRS_OPT_POST_ITERS`: fixed-point iterations for the "post" loop (default: 4).
- `GPTRS_PASS_STATS=1`: print per-pass stats to stdout.

## Pattern system

Most canonicalization/simplification passes are pattern-driven:

- `OpRewritePattern<V>`: typed pattern operating on a view `V`
- `V: OperationView`: typed extractor for a specific op shape (matcher + field access)

Operation views live under `crates/gpt-rs/src/backend/pattern/` and are generated from `#[ptir_pattern]`
annotations. Use:

```bash
cargo run -p gpt-rs-cli -- patterns
```

to list all available generated views.

The greedy driver (`apply_patterns_and_fold_greedily`) maintains:

- a worklist of `InstId`s
- a per-(pattern, inst) failure cache keyed by instruction version
- optional dead-code elimination for pure ops

## Adding a new rewrite

Pick the smallest unit that fits:

- **Pattern**: if it is local and anchored at one root op (recommended).
- **Pass**: if it needs analysis, ordering, or multiple pattern sets.

Checklist:

1. Add/extend an operation view (via `#[ptir_pattern]`) if needed.
2. Implement `OpRewritePattern<View>` and return `true` only when you actually mutate IR.
3. Wire the pattern into a pass under `crates/gpt-rs/src/backend/passes/`.
4. Register the pass in `crates/gpt-rs/src/backend/pipeline.rs` (or a backend-specific pipeline hook).
5. Validate via Torch parity and/or dumped PTIR (see `docs/testing.md`).

## Debugging optimizer behavior

Use `--dump-dir` on `gpt-rs-cli` (or the Python runner) to capture PTIR programs and pass events.
The trace JSON includes whether a plan/program cache was hit and can include pass/rewrite events when enabled.
