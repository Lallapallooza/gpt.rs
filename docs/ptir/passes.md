<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# Passes

The graph arena optimizes every program it compiles with the backend's pipeline of these passes
before the backend runs it. The ops' semantics are their `Eval` impls, which the filetests run
before and after each case's pass list. The pass catalog, `docs/ptir/passes.md`, is generated from
the code. It lists the pipeline of each backend, and every pass with its options, statistics, folds
and patterns.

Each pass is one file, named after the pass: `merge-contracted-axes` is `MergeContractedAxes` in
`merge_contracted_axes.rs`. The passes over any dialect are in this directory, a dialect's own in
`dialects/<dialect>/passes/`, next to its canonical patterns in `canonicalize.rs`, a conversion
between dialects in `conversion/<from>_to_<to>/`, and a backend's in the `passes` module of its
crate.

## Writing a pass

A pass is a struct with `#[pass]` and an `impl Pass`. Most passes apply patterns: local rewrites
that the `Greedy` driver applies to each op until nothing changes. A pattern is a struct with
`#[pattern]` and an `impl Pattern`, rooted at one op type, or at every op with a trait
(`type Root = Has<Commutative>`). A pass that rewrites each of its roots once, such as one that
outlines every group of a function, returns its rewrites as once-rewrites, `Rewrites::once(..)`. The
driver applies them to each op once, after the patterns. The example on this page compiles as a doc
test. It is not a pass of the crate.

```rust
use anyhow::Result;
use gpt_rs_ir::dialects::tensor::{Div, DivOp, Mul, Reciprocal};
use gpt_rs_ir::passes::{pass, run_patterns, Pass};
use gpt_rs_ir::require;
use gpt_rs_ir::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// `x / s` is `x * reciprocal(s)` for a static float `s`: the backend computes the reciprocal
/// once at load time, and a run multiplies, which costs less than dividing.
#[pattern]
struct DivideByStatic;

impl Pattern for DivideByStatic {
    type Root = Div;

    fn rewrite(&self, div: DivOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        // Match: the op and its operands, read through the module.
        let (dividend, divisor) = (div.lhs(), div.rhs());
        require!(divisor.tensor_type().dtype.is_float(), "an integer division truncates");
        require!(divisor.known_before_run(), "the divisor changes at run time");
        // Rewrite: create the new ops before the root, then replace the root by the product.
        // The result types come from the ops' type rules.
        let reciprocal = rewriter.create(Reciprocal, divisor);
        let product = rewriter.create(Mul, (dividend, reciprocal));
        rewriter.replace_op(&div, product);
        Ok(())
    }
}

/// Multiplies by the reciprocal of a static divisor instead of dividing by it.
///
/// The reciprocal rounds once and the product once more, so a result may differ from the
/// quotient in its last bit. That is why this is a pass a pipeline chooses, not part of
/// `canonicalize`.
#[pass(
    name = "multiply-by-reciprocal",
    stats(rewritten = "divisions by a static value replaced")
)]
pub struct MultiplyByReciprocal;

fn populate_multiply_by_reciprocal_patterns(set: &mut PatternSet) {
    set.add(DivideByStatic);
}

impl Pass for MultiplyByReciprocal {
    // What the pass applies: `run` passes it to the driver, and the catalog lists it.
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_multiply_by_reciprocal_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::REWRITTEN)
    }
}
```

The pass:

- The doc comment is required. Its first paragraph is the pass's summary in the catalog.
- `run` gets the module as a `Rewriting`, which it reads through `Deref` and changes only through
  the drivers: `run_patterns`, `Greedy`, `apply_conversion` and `cse`. A pass never holds
  `&mut Module`, and the raw edits of `Module` take an `EditKey` that only the drivers hold.
- `run` returns whether the module changed; the filetest runner checks that against the text.
  The pipeline verifies the module after every pass.
- `stats(name = "doc")` declares a counter, the constant `Self::NAME` of type `Stat`. The pass adds
  to it as it runs, and with `--dump-dir`, `GPTRS_PASS_STATS=1` logs it.
- The fields of the struct are its options, each with a doc comment, their defaults from
  `Default`. A pipeline sets them in code (`Canonicalize { fuel: 10 }`); a filetest sets them in
  the text form of op attributes (`// RUN: pass canonicalize {fuel = 10}, cse`).
- `#[pass]` registers the pass in the crate that defines it, so filetests can name it once the
  crate is linked. Two passes with one name are an error.

The pattern:

- The doc comment of the pattern is required: it states the rewrite, and the catalog shows it.
- It is rooted at an op type, such as `Div`; at each op of a family through a pattern generic over
  the family trait, `struct ReduceToTile<T>` with `impl<T: Reduce> Pattern for ReduceToTile<T>`;
  or at every op with a trait, `type Root = Has<Commutative>`.
- It reads the IR through its root, a view `OpRef<'m, T>` that `#[op]` names `<Name>Op`
  (`DivOp`): the op's attributes through `Deref` and an accessor per operand, result and region
  group. It reads on through the `ValueRef`s those give: `defining_op::<T>()`, `uses()`,
  `has_one_use()`, `tensor_type()`, `constant()`, `splat()`, `known_before_run()` and
  `depends_on_parameter()`.
- It changes an op's attributes through the op's view, `op.update(rewriter, |op| ...)` or a
  generated `op.set_<field>(rewriter, value)`, and the rest of the IR through the rewriter:
  `create`, `create_op`, `create_with_regions` (whose region builder can `move_here` an op of the
  module), `replace_op`, `replace_all_uses`, `set_operand`, `set_operands`, `erase`, `remove_arg`,
  `add_arg` and `add_symbol`. New ops go before the root unless `set_insertion_point` moves them,
  and take the root's location unless `set_origin` names another op. The rewriter stages the
  changes, and the driver applies them when the pattern returns `Ok`.
- When it does not apply it returns `Err(NoMatch)` with the reason: `require!(cond, "reason")`,
  `require!(let Some(x) = .., "reason")` or `.ok_or("reason")?`. The driver drops what it staged,
  so a pattern that does not apply changes nothing. A failing conversion reports the reasons.
- Creating an op its type rule rejects is a bug in the pattern, and panics.
- A pattern that does not keep every result bit for bit, or that chooses a layout for one backend,
  belongs to the pass that adds it, through the pass's `populate_<pass>_patterns`.

## Canonical patterns and folds

A pattern is canonical when it keeps every result bit for bit, apart from a NaN's sign and payload.
It must also hold on every backend, never add work and lead to one normal form, so it needs no cost
model. `#[pattern(canonical)]` registers it, and `canonicalize` applies every registered one. A
rewrite that can add work or move it between ops, such as fusing an op into its readers, is a pass
of its own. A pattern generic over its root op names the ops it registers for:

```rust
/// `t(t(x))` is one `t` of `x` for a movement op `t`.
#[pattern(canonical(Reshape, BroadcastTo, Slice, Transpose))]
struct CollapseChain<T>(PhantomData<T>);
```

An op's own folder is `#[fold]` on `impl Fold for <op>`, in its dialect's `fold/`, with a doc
comment that says what the op folds to. The traits of an op fold it too, by the laws they state.

```rust
/// A transpose by the identity `perm` is the input.
#[fold]
impl Fold for Transpose {
    fn fold(
        &self,
        ins: TransposeOperands<Folding>,
        _cx: &FoldCx,
    ) -> Option<TransposeResults<FoldOut>> {
        let identity = self.perm.iter().copied().eq(0..self.perm.len());
        identity.then_some(TransposeResults {
            output: ins.input.into(),
        })
    }
}
```

`cargo run -p gpt-rs-docs` runs the pass filetests and records where each fold and pattern
fires. It fails on:

- a folder, or a pattern of a pass, that fires in no filetest;
- a pass that no filetest runs;
- an op or kind that no roundtrip filetest uses.

## Pipelines

A pipeline is code: `PortableBackend::pipeline` adds passes in the order they run, mostly in the
groups that the `populate_*_passes` functions add. The C backend's:

```rust
let mut pm = Pipeline::new();
passes::populate_simplification_passes(&mut pm);
let numerics = Numerics {
    scale_fold: true,
    ..Numerics::default()
};
passes::populate_tensor_passes(&mut pm, &numerics);
pm.add(MergeContractedAxes).add(PackWeights);
passes::populate_load_time_passes(&mut pm);
```

A pass that changes rounding runs only where the backend's numerics allow it. The backend's
allowance and each dispatch's record of the changes it may hold are both `Numerics` values.

Some passes of a backend put parameters in the layouts its kernels read, such as the reshape and
transpose into the C GEMM's panels. They go before `populate_load_time_passes`, whose
`outline-static` moves their work to load time. Such a pass writes plain ops, so the module keeps
its meaning and the interpreter checks the pass. The catalog lists the pipeline of each backend,
the x86 backend's for the `x86-64-v4` preset.

The modules that the ref-cpu, faer and C backends compile hold only `tensor` ops. Such a backend
chooses its kernels when it compiles: it recognizes op sequences, such as a convolution or a
projection over a stored bf16 weight, and runs one kernel in their place. The x86 backend's
pipeline lowers the module through `tile`, `loop` and `isa` to the buffer form of `host`, and the
backend writes C for every op of it.

## Where the pipeline runs

When the graph arena (`crates/gpt-rs/src/ops/graph/arena.rs`) compiles a plan, it:

1. builds a module whose function `@captured` holds the recorded ops. Its arguments are the plan
   inputs, bound by stable id when they are parameters;
2. runs the backend's pipeline (`PortableBackend::pipeline`), verifying the module after each pass;
3. drops the parameters that nothing reads (`prune-params`);
4. runs on the backend each load-time function whose results the backend's parameter store lacks,
   and stores the results there;
5. compiles `@captured` on the backend (`PortableBackend::compile`) and caches the result by plan
   key.

## Debugging

- `--dump-dir` writes the module of each program the backend runs, after the pipeline, as
  `program.ptir` in the run's directory, and the module after each pass under `passes/`.
- With `--dump-dir`, `GPTRS_PASS_STATS=1` also records, per pass, its options, whether it changed
  the module, how many ops remain and the pass's statistics (`passes.jsonl`).
- `canonicalize {fuel = N}` stops after N steps. Use it to bisect a pattern that breaks a program.

## The pipelines of the backends

`ref-cpu` and `faer`:

1. `canonicalize`
2. `cse`
3. `dot-scale-fold`
4. `canonicalize`
5. `merge-contracted-axes`
6. `outline-static`
7. `canonicalize`
8. `cse`

`c`:

1. `canonicalize`
2. `cse`
3. `dot-scale-fold`
4. `canonicalize`
5. `merge-contracted-axes`
6. `c-pack-weights`
7. `outline-static`
8. `canonicalize`
9. `cse`

`x86`:

1. `canonicalize`
2. `cse`
3. `dot-scale-fold`
4. `canonicalize`
5. `tensor-to-tile`
6. `pad-with-identity`
7. `inline-static-maps`
8. `canonicalize`
9. `widen-reused-maps`
10. `plan-layouts`
11. `canonicalize`
12. `widen-narrow-floats`
13. `expand-math {numerics = <contract = true, skip_pad = true, reassoc = true, scale_fold = true, expand_math = true>}`
14. `expand-random`
15. `form-groups`
16. `seal-dispatches`
17. `name-target {target = "x86-64-v4", numerics = <contract = true, skip_pad = true, reassoc = true, scale_fold = true, expand_math = true>}`
18. `select-atoms {target = "x86-64-v4"}`
19. `split-lane-folds {target = "x86-64-v4"}`
20. `schedule {target = "x86-64-v4"}`
21. `place-epilogues`
22. `walk-order-layouts`
23. `outline-static`
24. `canonicalize`
25. `cse`
26. `tile-to-loop`
27. `canonicalize`
28. `contract`
29. `loop-to-isa {target = "x86-64-v4"}`
30. `plan-memory {target = "x86-64-v4"}`

## Folds

The passes that fold an op try its own `#[fold]` folder, then the folds of its traits. A fold replaces the op by an existing value or by a constant that repeats or moves constant bytes; it never computes on elements.

| Op | Folds | Fires in |
|---|---|---|
| `loop.index` | An index whose expression is one of its operands is that operand. | [loop/canonicalize/index.ptir](../../filetests/loop/canonicalize/index.ptir) |
| `tensor.bitcast` | A bitcast to the input's own type is the input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.broadcast_to` | A broadcast to the input's own type is the input. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.cast` | A cast to the input's own type is the input. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `tensor.concat` | A concat of one input is that input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.dynamic_slice` | A slice as large as its input starts at 0 whatever `starts` holds, so it is the input. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
| `tensor.dynamic_update_slice` | A patch as large as the base replaces all of it, so the result is the patch. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
| `tensor.reshape` | A reshape to the input's own type is the input, and a reshape of a constant is that constant with the new extents, its elements in the same row-major order. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.select` | `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are the same value, and `on_false` when `pred` is all false. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `tensor.slice` | A slice with the input's extents covers the whole input, so it is the input. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.transpose` | A transpose by the identity `perm` is the input. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |

The traits whose folds fire in the filetests; [tensor.md](tensor.md#traits) describes every trait.

| Trait | | Fires in |
|---|---|---|
| `Absorbing` | `f(a, z) = z`, for the integers and `i1`. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `Commutative` | `f(a, b) = f(b, a)`. Folding puts a run-time operand first and a constant last (see `Module::known_before_run`), so the other folds, the patterns and CSE see a constant on the right. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir), [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |
| `Idempotent` | For a unary op `f(f(a)) = f(a)`, for a binary one `f(a, a) = a`. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `Involution` | `f(f(a)) = a`. | [tensor/canonicalize/fuel.ptir](../../filetests/tensor/canonicalize/fuel.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `Movement` | Each result element is an element of one of the data operands: the first `data` operands, or all of them when `data` is `None`. The op moves elements without computing on them, so a splat stays a splat. The fold builds that splat only when it holds no more elements than the operands: a constant stores every element, so a broadcast splat stays a broadcast. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `RightIdentity` | `f(a, e) = a`. A commutative op has `e` on the left too, since folding moves constants to the right. | [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |

## Passes

### `c-pack-weights`

Puts the weights of the dots that C kernels compute in the layout each kernel reads: GEMM
weights in the GEMM's panels, and grouped convolution weights tap by tap.

Each layout is a transpose of the weight, after a reshape for the GEMM's panels. The dot then
contracts the weight in that layout, so the IR keeps its meaning. When the weight is static,
`outline-static` moves the layout to load time, and the kernel reads the laid-out parameter.
GEMM weights that are not static, activations among them, stay as they are, and the GEMM packs
the panels it needs on each call.

Defined by `PackWeights` in [crates/gpt-rs-backend-c/src/passes/pack_weights.rs](../../crates/gpt-rs-backend-c/src/passes/pack_weights.rs).

| Statistic | |
|---|---|
| `packed` | weights laid out for a kernel |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `PackGemmWeight` | `tensor.dot_general` | Rewrites an f32 `x [.., K] . w`, which codegen sends to the GEMM (a convolution's patches among them), into `reshape(x . packed)`, where `packed` is `w` in panels of `PANEL_N` columns, `[N / PANEL_N, K, PANEL_N]`. `w` is `[K, N]` or `[N, K]`. The packed dot gives `[.., N / PANEL_N, PANEL_N]`, the same elements as `[.., N]`.<br><br>It packs only a static `w` whose packed form takes no more bytes than the parameters it is computed from, so `outline-static` moves the packing to load time. It skips a `w` whose `N` is not a multiple of `PANEL_N`, since the panels would pad it. A run-time `x` with one row, or with fewer than `LINEAR_PACK_MIN_ROWS` rows against an `[N, K]` weight, leaves `w` unpacked: the GEMM or the dot-product kernels read `w` in place. | [backend/c/pack_weights.ptir](../../filetests/backend/c/pack_weights.ptir), [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir) |
| `GroupedConvWeight` | `tensor.dot_general` | Lays out the weight of a grouped convolution for the grouped kernel. The dot of the patches `[N, OH, OW, KH, KW, G, C_in / G]` with the weight `[G, C_out / G, C_in / G, KH, KW]`, batched over the groups, becomes the same dot with the weight transposed to `[KH, KW, C_in / G, G, C_out / G]`, the order of `[KH, KW, C_in / G, C_out]` that the kernel reads. The result keeps its axes. `outline-static` moves the transpose of a static weight to load time. | [backend/c/pack_weights.ptir](../../filetests/backend/c/pack_weights.ptir) |

Filetests: [backend/c/pack_weights.ptir](../../filetests/backend/c/pack_weights.ptir).

### `canonicalize`

Simplifies the module without changing what it computes: folds ops, erases dead ones and
applies every canonicalization pattern until nothing changes.

The folds are the ops' `#[fold]` folders and the folds of their traits (`Commutative`,
`Involution`, ...). The patterns are those that `#[pattern(canonical)]` registers. Each keeps
the results bit for bit, apart from the sign and payload of a NaN, which IEEE 754 leaves
unspecified.

Defined by `Canonicalize` in [crates/gpt-rs-ir/src/passes/canonicalize.rs](../../crates/gpt-rs-ir/src/passes/canonicalize.rs).

| Option | Type | Default | |
|---|---|---|---|
| `fuel` | `u64` | `0` | Stops after this many folds, erasures and rewrites. 0 runs to a fixed point. Use it to bisect a pattern that loops or a rewrite that breaks a program. |

| Statistic | |
|---|---|
| `folded` | ops replaced by an existing value or a constant |
| `erased` | dead ops erased |
| `rewritten` | ops rewritten by a pattern |

It applies every [fold](#folds).

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `BroadcastOfLeadingUnitAxes` | `tensor.broadcast_to` | A `broadcast_to` of a reshape that only adds leading axes of extent 1 is the `broadcast_to` of the reshape's input: broadcasting aligns trailing axes already. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |
| `CanonicalIndices` | `tile.block` | Every index expression is in canonical form: sums with merged coefficients, `floordiv` and `mod` reduced where the iterators' extents allow, and `min` and `max` decided where the extents decide them. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir) |
| `CollapseChain` | `tensor.reshape`, `tensor.broadcast_to`, `tensor.slice`, `tensor.transpose` | `t(t(x))` is one `t` of `x` for a movement op `t`: a reshape or `broadcast_to` to the last shape, the slice whose starts are `s1.starts + s2.starts * s1.strides` and whose strides are `s1.strides * s2.strides`, and the transpose `q[i] = p1[p2[i]]`. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `CompactInputs` | `tile.block` | A read whose value the body does not use is removed, with its body argument. Operands that hold the same value merge into the first, and operands the block no longer names are removed. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/inline_static_maps/weights.ptir](../../filetests/tile/inline_static_maps/weights.ptir) |
| `CompareWithItself` | `tensor.less`, `tensor.less_equal`, `tensor.equal`, `tensor.greater_equal`, `tensor.greater`, `tensor.not_equal` | A comparison of an integer or `i1` value with itself holds everywhere or nowhere, as its relation does on equal operands. Equal constants are one value once the driver merges them. Floats are left alone: NaN is not equal to itself. | [tensor/canonicalize/constants.ptir](../../filetests/tensor/canonicalize/constants.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `ComposeCopyReads` | `tile.block` | A read of the result of a copy block reads the copy's operand, at the copy's coordinates of the point it reads. A read that stays inside the result takes the copy's `pad`. A padded read composes where each of its coordinates that may leave the result leaves the operand with it. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir) |
| `ComposeIndices` | `loop.index` | A `loop.index` is one expression over the values that no `loop.index` computes, such as loop variables and the coordinates of atoms, in canonical form where the loops around bound those values: the variable of a loop is its start plus an offset below the most points the loop runs, and a coordinate of a `loop.atom` is below the atom's tile. The op becomes that expression when it computes fewer `floordiv` and `mod`, or as many and fewer operations. It becomes the value its expression names when the expression is one value. The indices it took stay for their other uses. | [loop/canonicalize/index.ptir](../../filetests/loop/canonicalize/index.ptir) |
| `ConcatOfConcat` | `tensor.concat` | A concat of a concat on the same axis is one concat of the inner concat's inputs. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `DelinearizeIters` | `tile.block` | An iterator of static extent `outer * inner` that no fresh write and no value's iterators name splits into two in its place, of extents `outer` and `inner`, when that removes a `floordiv` or `mod` from the block's expressions. The points keep their order, so every fold keeps its bits. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/inline_static_maps/weights.ptir](../../filetests/tile/inline_static_maps/weights.ptir) |
| `DotOfOneProduct` | `tensor.dot_general` | A dot whose contracted axes hold one element sums one product. It is the product of the operands, laid out along the result's axes and cast to the accumulator, plus 0: the sum starts from 0, so a product of `-0.0` gives `0.0`. The result is cast to `out`. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir) |
| `DropDefaultDtypes` | `tensor.dot_general`, `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or`, `tensor.reduce_window_sum`, `tensor.reduce_window_max`, `tensor.reduce_window_min`, `tensor.reduce_window_and`, `tensor.reduce_window_or` | An `out` equal to the input dtype, or an `accum` equal to its default accumulator, is left out. | [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir) |
| `DropUnitContractions` | `tensor.dot_general` | A pair of contracted axes of one element each adds nothing to the sums: the dot contracts the operands reshaped without such pairs, summing the same products in the same order. A dot that sums one product is [`DotOfOneProduct`]'s. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir) |
| `DropUnitIters` | `tile.block` | The pattern removes an iterator of extent 1, which is always 0. It writes 0 for it in every expression and write, and moves the later iterators down one position. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/canonicalize/updates.ptir](../../filetests/tile/canonicalize/updates.ptir) |
| `EraseIdentityBlock` | `tile.block` | A block that reads its operand at its own iterators, in the order it writes them, into a result of the operand's type, and yields what it reads, is that operand. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `Factor` | every op with `DistributesOver` | `g(f(a, b), f(a, c))` is `f(a, g(b, c))` when `f` distributes over `g`, which saves an `f`. For a commutative `f` the shared operand may be on either side. The pattern is rooted at either `f` and rewrites its only use, `g`. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `FoldIndexArith` | `arith.add`, `arith.sub`, `arith.mul`, `arith.minimum`, `arith.maximum`, `arith.neg` | Integer `arith` on `tile.index` values and constants is one `tile.index` of the combined expression, so that index math is in the index language. It folds where no value leaves the range of the dtype: always for si64, and for another dtype where the block's iterator extents bound the values. | [tile/canonicalize/index.ptir](../../filetests/tile/canonicalize/index.ptir) |
| `GroupStatic` | every op with `Associative` | `f(f(x, a), b)` is `f(x, f(a, b))` when `f` is associative, `a` and `b` are static, and one of them reads a parameter. The backend then computes `f(a, b)` once at load time, so a run computes one `f` where it computed two. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `HoistBroadcast` | every op with `Elementwise` | `f(broadcast_to(a), broadcast_to(b))` is `broadcast_to(f(a, b))` when `a` and `b` have the same shape, so `f` computes on fewer elements. The pattern splats a splat constant operand to that shape. | [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `InlineSingleTrip` | `loop.for` | A `loop.for` whose end is its start plus 1, where the loops around bound them, is its body, with the index at the start, when the body allocates no buffer: a buffer lives as long as the region that allocates it, so the loop bounds its life. | [loop/canonicalize/index.ptir](../../filetests/loop/canonicalize/index.ptir) |
| `IotaOfUnitAxis` | `tensor.iota` | A `tensor.iota` along an axis of extent 1 counts no further than 0: it is 0, broadcast. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `MergeEqualReads` | `tile.block` | Two reads of the same operand at the same coordinates, with the same pad, are one: the body takes the value of the first for both. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `MergeReductions` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | `reduce(reduce(x, a), b)` of one fold is one reduction of `x` over the axes of both, reshaped to the result, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `NonNegativeAxis` | `tensor.argmax`, `tensor.top_k`, `tensor.bottom_k`, `tensor.concat`, `tensor.iota`, `tensor.gather`, `tensor.scatter_add`, `tensor.scatter_max`, `tensor.scatter_min`, `tensor.scatter_replace` | A negative `axis` is written as the axis it counts back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `NonNegativeDotAxes` | `tensor.dot_general` | Negative batch and contract axes are written as the axes they count back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `OrderItersByWrite` | `tile.block` | The parallel iterators that the block writes at come first, in the order its writes first take them, and its other iterators follow in their order. The iterators only name the block's points, so this renames them: a block and the blocks that run on its result element by element then name each element's point alike. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir) |
| `ReduceOfBroadcast` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A reduction of `broadcast_to(x)` is `x` reshaped to the result when its fold is idempotent, the reduction covers every axis the broadcast repeats `x` along, and its other axes have extent 1. Each repeated axis must have a static extent above zero, since a reduction over no elements gives its initial value. | [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `ReduceOfConcat` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A reduction of `concat(x1, x2, ...)` that reduces the concatenated axis is its fold applied to the reductions of the parts, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `SliceOfConcat` | `tensor.slice` | A slice of a concat that reads one input along the concatenated axis is a slice of that input. A slice that spans several inputs stays: as a concat of slices, it would copy what it reads twice. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `SortedReduceAxes` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | The reduced axes are written non-negative and in increasing order. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `SplatReads` | `tile.block` | A read without `pad` of a value whose elements are all equal, a constant or a block that fills one, is that element, an `arith.constant` in the body. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir) |
| `TransposeIntoDot` | `tensor.dot_general` | A transpose that feeds a dot folds into the dot's axes when it keeps the order of the operand's free axes: the dot reads the transpose's input and names its batch and contracted axes there. Each sum keeps its order, which follows the listed contracted axes. | [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `TransposeOfUnitAxes` | `tensor.transpose` | A transpose that moves only axes of extent 1 keeps the row-major order of the elements, so it is a reshape to its result shape. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `UnitStrides` | `tensor.slice` | Steps of 1 on every axis are written as no strides. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `WriteInCopyOrder` | `tile.block` | When the only reader of a block's result is a copy that reorders its axes, the block writes that result in the copy's order. The block's new result replaces the copy. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir) |
| `WriteIntoUpdate` | `tile.block` | When the only reader of a map's result is a copy into another operand, the map writes into that operand itself, so its values go straight where the copy put them. This holds when the copy reads at each point the element the map writes at that point, and the map writes each point at an element of its own. | [tile/canonicalize/updates.ptir](../../filetests/tile/canonicalize/updates.ptir) |

Filetests: [loop/canonicalize/index.ptir](../../filetests/loop/canonicalize/index.ptir), [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir), [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir), [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/canonicalize/constants.ptir](../../filetests/tensor/canonicalize/constants.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/fuel.ptir](../../filetests/tensor/canonicalize/fuel.ptir), [tensor/canonicalize/group_static.ptir](../../filetests/tensor/canonicalize/group_static.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/canonicalize/index.ptir](../../filetests/tile/canonicalize/index.ptir), [tile/canonicalize/placed.ptir](../../filetests/tile/canonicalize/placed.ptir), [tile/canonicalize/updates.ptir](../../filetests/tile/canonicalize/updates.ptir), [tile/inline_static_maps/weights.ptir](../../filetests/tile/inline_static_maps/weights.ptir).

### `contract`

In each `loop.func` that contracts, an `arith.add` of an f32 or f64 `arith.mul` that nothing
else takes is one `arith.fma`, which rounds the product and the sum once: the product the add
takes second, else the first.

Defined by `Contract` in [crates/gpt-rs-ir/src/dialects/loops/passes/contract.rs](../../crates/gpt-rs-ir/src/dialects/loops/passes/contract.rs).

| Statistic | |
|---|---|
| `contracted` | products and sums written as one arith.fma |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `FuseProduct` | `arith.add` | `arith.add(x, arith.mul(a, b))` or `arith.add(arith.mul(a, b), x)`, in a function that contracts, is `arith.fma(a, b, x)`, when nothing else takes the product and it is f32 or f64: the product the add takes second, when it takes two. | [loop/contract/products.ptir](../../filetests/loop/contract/products.ptir) |

Filetests: [loop/contract/products.ptir](../../filetests/loop/contract/products.ptir).

### `cse`

Replaces each op by an earlier op that computes the same values: the same op with the same
attributes and operands. Ops with effects or regions, and terminators, are never replaced. An
op in a region also matches an op of an enclosing region, unless a region isolated from above
lies between them.

Defined by `Cse` in [crates/gpt-rs-ir/src/passes/cse.rs](../../crates/gpt-rs-ir/src/passes/cse.rs).

| Statistic | |
|---|---|
| `replaced` | ops replaced by an earlier op |

Filetests: [tensor/cse/cse.ptir](../../filetests/tensor/cse/cse.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `dce`

Erases the ops that have no effect and whose results nothing reads, until nothing changes.

Defined by `Dce` in [crates/gpt-rs-ir/src/passes/dce.rs](../../crates/gpt-rs-ir/src/passes/dce.rs).

| Statistic | |
|---|---|
| `erased` | dead ops erased |

Filetests: [tensor/cse/cse.ptir](../../filetests/tensor/cse/cse.ptir), [tensor/dce/dce.ptir](../../filetests/tensor/dce/dce.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir), [tensor/outline_static/budget.ptir](../../filetests/tensor/outline_static/budget.ptir), [tensor/outline_static/rerun.ptir](../../filetests/tensor/outline_static/rerun.ptir), [tensor/outline_static/static.ptir](../../filetests/tensor/outline_static/static.ptir).

### `dot-scale-fold`

Moves the static scales of dot results into the weights, so a convolution absorbs the
BatchNorm after it and a run saves the multiply.

The two sides are equal in real arithmetic, not bit for bit: the scale rounds into every
weight before the sum, instead of into the sum once. So it is a pass of its own, not part of
`canonicalize`.

Defined by `DotScaleFold` in [crates/gpt-rs-ir/src/dialects/tensor/passes/dot_scale_fold.rs](../../crates/gpt-rs-ir/src/dialects/tensor/passes/dot_scale_fold.rs).

| Statistic | |
|---|---|
| `moved` | scales moved into a weight |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ScaleIntoDot` | `tensor.mul` | `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are static and `s` varies only along result axes that come from `w`: its batch axes and its free axes. `s'` is `s` laid out along those axes of `w` and broadcast to its shape. The product may reach the multiply through transposes and reshapes that only add or remove axes of extent 1, each the one user of the value before it, which then move the scaled product. The scaled weight is static and reads a parameter, so `outline-static` moves it to load time, and the rewrite applies only where that pass would store it ([`OutlineStatic::would_store`]): a weight cast to a wider dtype, which would take more bytes than its parameter, stays unscaled. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/dot_scale_fold/scales.ptir](../../filetests/tensor/dot_scale_fold/scales.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |

Filetests: [tensor/dot_scale_fold/cast.ptir](../../filetests/tensor/dot_scale_fold/cast.ptir), [tensor/dot_scale_fold/scales.ptir](../../filetests/tensor/dot_scale_fold/scales.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `expand-math`

Writes `arith.exp`, `tanh`, `log`, `sin`, `cos` and `erf` of f32 scalars as polynomials in
`arith`, for a target that has no lane-wise form of these functions, where the compilation
allows their rounding (`numerics = <expand_math = true>`). The sine and the cosine compute in
f64 and on ui64 words, so the target needs f64 and 64-bit integer lanes.

Measured on every f32 input against the interpreter's math functions, the `libm` crate's f64
functions rounded to f32: each expansion is within 1 ulp of its reference and gives its
infinities and NaNs. The polynomials round otherwise than the reference, so this is a pass of
its own, not part of `canonicalize`.

Defined by `ExpandMath` in [crates/gpt-rs-ir/src/dialects/arith/passes/expand_math.rs](../../crates/gpt-rs-ir/src/dialects/arith/passes/expand_math.rs).

| Option | Type | Default | |
|---|---|---|---|
| `numerics` | `Numerics` | `<>` | The rounding changes the compilation allows: the pass writes the polynomials only where they name `expand_math`. |

| Statistic | |
|---|---|
| `expanded` | math functions written as polynomials |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `Expand` | `arith.exp`, `arith.tanh`, `arith.log`, `arith.sin`, `arith.cos`, `arith.erf` | `arith.exp`, `tanh`, `log`, `sin`, `cos` or `erf` of an f32 becomes its polynomial in `arith`, as the function's `Expanded` impl writes it, where the compilation allows the polynomials' rounding. | [arith/expand_math/math.ptir](../../filetests/arith/expand_math/math.ptir) |

Filetests: [arith/expand_math/allowed.ptir](../../filetests/arith/expand_math/allowed.ptir), [arith/expand_math/math.ptir](../../filetests/arith/expand_math/math.ptir).

### `expand-random`

Writes each `arith.rng_uniform` as the `arith` of its draw: the integer Philox4x32-10 block that
holds the element's words of the stream, and the element they give. It keeps every bit, so a
backend that computes no draw of its own runs it, before it decides how to compute the ops.

Defined by `ExpandRandom` in [crates/gpt-rs-ir/src/dialects/arith/passes/expand_random.rs](../../crates/gpt-rs-ir/src/dialects/arith/passes/expand_random.rs).

| Statistic | |
|---|---|
| `expanded` | random draws written as arith |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ExpandRngUniform` | `arith.rng_uniform` | `arith.rng_uniform` becomes its draw in `arith`: the Philox4x32-10 block that holds the element's words of the stream, by integer ops on ui64 values of 32 bits each, the words picked from it, and the element they give: the low bits of the words for an integer, wrapped to its dtype, and for a float as many top bits of the words as its significand has, scaled into `[0, 1)`. The words of one element lie in one block, since their count divides the block's. | [arith/expand_random/uniform.ptir](../../filetests/arith/expand_random/uniform.ptir) |

Filetests: [arith/expand_random/uniform.ptir](../../filetests/arith/expand_random/uniform.ptir).

### `fold`

Replaces each op that folds by an existing value or a constant, until nothing changes. The
folds are those of `canonicalize`, without its patterns.

Defined by `Fold` in [crates/gpt-rs-ir/src/passes/fold.rs](../../crates/gpt-rs-ir/src/passes/fold.rs).

| Statistic | |
|---|---|
| `folded` | ops replaced by an existing value or a constant |

It applies every [fold](#folds).

Filetests: [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir).

### `form-groups`

Groups each block with the chain of blocks that run on its result element by element, its
epilogue, into a `tile.group`, which `seal-dispatches` outlines into one dispatch. Grouping
moves blocks and changes none: each keeps its body, reads, writes and folds, so it is exact.

- A block joins the chain when it is the one use of the result of the block before it and
  takes it as an epilogue (`Block::takes_epilogue`): it runs on each element of that result,
  where that block writes it, over the same leading parallel iterators.
- A block that runs at each call joins no chain of a block known before the run: what runs
  once, when the parameters load, stays apart from what runs at each call.
- The group yields the result of the chain's last block, and stands where that block stood.

Defined by `FormGroups` in [crates/gpt-rs-ir/src/dialects/tile/passes/form_groups.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/form_groups.rs).

| Statistic | |
|---|---|
| `grouped` | blocks grouped with the block before them |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `GroupChain` | `tile.block` | A block of a function, outside a group, and the blocks after it that each take the one before as an epilogue move into a `tile.group`, which stands where the last of them stood and yields its result. The blocks go in the order of the function, so a block that reads the results of two blocks joins the chain of the first one. | [tile/form_groups/epilogue.ptir](../../filetests/tile/form_groups/epilogue.ptir) |

Filetests: [tile/form_groups/epilogue.ptir](../../filetests/tile/form_groups/epilogue.ptir).

### `inline-static-maps`

Computes each map known before the run that `outline-static` would not store, because its result
takes more bytes than the parameters it reads ([`OutlineStatic::would_store_over`]). The blocks
that read the map compute it at each point they read. A weight cast to a wider dtype or
dequantized is then read as stored. No copy of it larger than the parameters is made at load
time or at each call. A block that reads an element at several points computes it at each, so it
is a pass rather than a canonical form.

- The map writes each point of its parallel iterators at an element of its own, at a
  coordinate that is the iterator or 0, and its body reads no `tile.index`.
- The read of its result has no pad, holding or placement.

Defined by `InlineStaticMaps` in [crates/gpt-rs-ir/src/dialects/tile/passes/inline_static_maps.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/inline_static_maps.rs).

| Statistic | |
|---|---|
| `inlined` | reads of maps computed in their readers |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ComputeMapInReader` | `tile.block` | A read of the result of a map known before the run that takes more bytes than its operands is the map's body on the map's reads, at the point of the map that the read reaches. | [tile/inline_static_maps/weights.ptir](../../filetests/tile/inline_static_maps/weights.ptir) |

Filetests: [tile/inline_static_maps/weights.ptir](../../filetests/tile/inline_static_maps/weights.ptir).

### `loop-to-isa`

Computes the points of each `loop.atom` at once, in the vectors of its lanes, and checks the
vector code against the target. The rules of the target's architecture must have code for each
`isa` op, and the target must have the extensions that code needs. A register tile's code must
hold at once no more vector and mask registers than the target has. The count goes register by
register, in the order every backend follows ([`isa::RegisterCode::order`]). The
`tile.dispatch` verifier checks the rules for the lane-wise ops and the register widths of each
block that computes in vectors, but counts no registers. The code of a verified schedule may
hold more registers than the target has, which fails here. A `loop.func` decided for another
target is an error. The lowering decides nothing:

- The atom's tile is a group of registers: one axis per tile axis, the last one counting the
  vectors of lanes along the tile's last axis. A value of the body takes extent 1 along the
  axes it does not vary on, and `isa.expand` repeats it where an op needs it whole.
- An index of the body is a base plus a multiple of each coordinate of the point. A load whose
  last index moves with the lanes by 1 is `isa.load`, one whose indices do not move with them
  is `isa.broadcast`, and one at the same element for every point stays a `loop.load`, before
  the atom. `isa.load` and `isa.broadcast` keep the load's pad where a lane may fall outside
  the buffer. A store is `isa.store`, contiguous along the lanes.
- An op on values the same at every point stays as it is, before the atom, when it has no
  effects and no regions and gives one value. The atom runs at least one point, so the op
  computes what the body would. Each such value enters the registers as one `isa.splat`. Each
  other `arith` op that has a lane-wise `isa` op is that op. Any other op is an error.
- An access of memory that moves along a tile axis whose extent may stop short of the tile
  takes an `isa.tile_mask` that varies along those axes alone. An access of a `reg` buffer,
  which holds the whole tile, takes none.

An atom whose body this cannot compute, such as one that reads with a stride along the lanes,
is an error.

Defined by `LoopToIsa` in [crates/gpt-rs-ir/src/conversion/loop_to_isa/mod.rs](../../crates/gpt-rs-ir/src/conversion/loop_to_isa/mod.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target the vector code is for. |

| Statistic | |
|---|---|
| `realized` | atoms computed in vectors |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `AtomToIsa` | `loop.atom` | `loop.atom` becomes vector code before it, which computes every point of its tile at once. | [isa/loop_to_isa/atom.ptir](../../filetests/isa/loop_to_isa/atom.ptir), [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [isa/loop_to_isa/lanewise.ptir](../../filetests/isa/loop_to_isa/lanewise.ptir), [isa/loop_to_isa/locations.ptir](../../filetests/isa/loop_to_isa/locations.ptir), [isa/loop_to_isa/masks.ptir](../../filetests/isa/loop_to_isa/masks.ptir), [isa/loop_to_isa/once.ptir](../../filetests/isa/loop_to_isa/once.ptir), [isa/loop_to_isa/pad.ptir](../../filetests/isa/loop_to_isa/pad.ptir), [isa/loop_to_isa/tile.ptir](../../filetests/isa/loop_to_isa/tile.ptir) |

Filetests: [isa/loop_to_isa/atom.ptir](../../filetests/isa/loop_to_isa/atom.ptir), [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [isa/loop_to_isa/lanewise.ptir](../../filetests/isa/loop_to_isa/lanewise.ptir), [isa/loop_to_isa/locations.ptir](../../filetests/isa/loop_to_isa/locations.ptir), [isa/loop_to_isa/masks.ptir](../../filetests/isa/loop_to_isa/masks.ptir), [isa/loop_to_isa/no_masks.ptir](../../filetests/isa/loop_to_isa/no_masks.ptir), [isa/loop_to_isa/no_rules.ptir](../../filetests/isa/loop_to_isa/no_rules.ptir), [isa/loop_to_isa/no_target.ptir](../../filetests/isa/loop_to_isa/no_target.ptir), [isa/loop_to_isa/once.ptir](../../filetests/isa/loop_to_isa/once.ptir), [isa/loop_to_isa/one_mask.ptir](../../filetests/isa/loop_to_isa/one_mask.ptir), [isa/loop_to_isa/other_target.ptir](../../filetests/isa/loop_to_isa/other_target.ptir), [isa/loop_to_isa/pad.ptir](../../filetests/isa/loop_to_isa/pad.ptir), [isa/loop_to_isa/reg_buffers.ptir](../../filetests/isa/loop_to_isa/reg_buffers.ptir), [isa/loop_to_isa/registers.ptir](../../filetests/isa/loop_to_isa/registers.ptir), [isa/loop_to_isa/tile.ptir](../../filetests/isa/loop_to_isa/tile.ptir), [isa/loop_to_isa/widths.ptir](../../filetests/isa/loop_to_isa/widths.ptir).

### `merge-contracted-axes`

Puts static dot operands in the layout the dot reads. When the lhs of a dot splits one axis into
its contracted axes, the dot reads that axis whole. The pass transposes and reshapes the static
weight to match.

Defined by `MergeContractedAxes` in [crates/gpt-rs-ir/src/dialects/tensor/passes/merge_contracted_axes.rs](../../crates/gpt-rs-ir/src/dialects/tensor/passes/merge_contracted_axes.rs).

| Statistic | |
|---|---|
| `merged` | dots whose contracted axes merged |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `MergeDotAxes` | `tensor.dot_general` | `dot_general(reshape(x), w)` becomes `dot_general(x, w')` when the reshape only splits one axis `a` of `x` into the contracted axes of the lhs, which are adjacent and in order, and `w` is static. `w'` is `w` with its batch axes first, then its contracted axes in the order the lhs contracts them, merged into one, then its free axes. The new dot contracts `a` with that axis.<br><br>The rewrite puts the weight of a convolution, `[C_out, C_in, KH, KW]`, in the layout `[KH * KW * C_in, C_out]` that a matrix product of the patches reads. `w'` is static, so `outline-static` computes it once at load time. Both dots sum the same products in the same order, row-major over the contracted axes, so they agree bit for bit. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/merge_contracted_axes/merge.ptir](../../filetests/tensor/merge_contracted_axes/merge.ptir) |

Filetests: [tensor/merge_contracted_axes/merge.ptir](../../filetests/tensor/merge_contracted_axes/merge.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `name-target`

Names on each dispatch the target it is compiled for, with the rounding changes that the pass
allows there ([`Numerics::on_target`]): `contract`, a product and the sum it enters rounded
once, only where the target computes `isa.fma`. The passes after it choose for that target,
and the dispatch's verifier checks it against that target.

Defined by `NameTarget` in [crates/gpt-rs-ir/src/dialects/tile/passes/name_target.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/name_target.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target the dispatches are compiled for. |
| `numerics` | `Numerics` | `<>` | The rounding changes the dispatches may make. |

| Statistic | |
|---|---|
| `named` | dispatches that name their target |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `NameDispatchTarget` | `tile.dispatch` | A dispatch that names no target names the pass's, with the numerics it allows there. | [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [tile/name_target/contract.ptir](../../filetests/tile/name_target/contract.ptir), [tile/select_atoms/vectors.ptir](../../filetests/tile/select_atoms/vectors.ptir), [tile/split_lane_folds/lane_folds.ptir](../../filetests/tile/split_lane_folds/lane_folds.ptir), [tile/split_lane_folds/maxima.ptir](../../filetests/tile/split_lane_folds/maxima.ptir), [tile/split_lane_folds/padded.ptir](../../filetests/tile/split_lane_folds/padded.ptir) |

Filetests: [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [tile/name_target/contract.ptir](../../filetests/tile/name_target/contract.ptir), [tile/name_target/features.ptir](../../filetests/tile/name_target/features.ptir), [tile/name_target/forms.ptir](../../filetests/tile/name_target/forms.ptir), [tile/select_atoms/vectors.ptir](../../filetests/tile/select_atoms/vectors.ptir), [tile/split_lane_folds/lane_folds.ptir](../../filetests/tile/split_lane_folds/lane_folds.ptir), [tile/split_lane_folds/maxima.ptir](../../filetests/tile/split_lane_folds/maxima.ptir), [tile/split_lane_folds/padded.ptir](../../filetests/tile/split_lane_folds/padded.ptir).

### `outline-static`

Moves the static values that the run-time ops of a function read into one load-time function,
which the runtime runs once when it loads the parameters: it takes the parameters they read
and gives each value as a parameter of its own. The function that reads a value gets a new
argument bound to that parameter.

- What moves is the value before an op that only repeats its elements, such as a
  `broadcast_to`, so the runtime keeps the smaller tensor.
- A static value computed from no parameter stays: as a parameter, it would be stored once for
  each shape a program is compiled for.
- The values stored from a parameter take at most the parameter's bytes in total. The pass
  allocates those bytes in the order the values are defined. So the values a function stores
  never take more bytes than the parameters they read, with one exception. A value of their
  dtype may take up to one slice of its outermost axis more, less one byte, as a layout in
  panels does when the parameter does not fill the last panel. A value that does not fit, such
  as a weight cast to a wider dtype, stays, and the static values it reads move instead. A value
  or parameter of unknown size never fits.
- The id of a value's parameter is a hash of the text of a function that computes it alone from
  the parameters it reads, so equal computations share one parameter across functions and
  programs, and the runtime skips a load-time function whose results it holds. Ops that
  several values share are computed once.

Defined by `OutlineStatic` in [crates/gpt-rs-ir/src/passes/outline_static.rs](../../crates/gpt-rs-ir/src/passes/outline_static.rs).

| Statistic | |
|---|---|
| `outlined` | values moved into load-time functions |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `OutlineStaticValues` | `func.func` | A run-time function's static values that run-time ops read, and that parameters go into, move into one load-time function: the function gains an argument bound to the parameter each computes, which the ops read in place of the value. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/outline_static/budget.ptir](../../filetests/tensor/outline_static/budget.ptir), [tensor/outline_static/rerun.ptir](../../filetests/tensor/outline_static/rerun.ptir), [tensor/outline_static/static.ptir](../../filetests/tensor/outline_static/static.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |

Filetests: [tensor/outline_static/budget.ptir](../../filetests/tensor/outline_static/budget.ptir), [tensor/outline_static/rerun.ptir](../../filetests/tensor/outline_static/rerun.ptir), [tensor/outline_static/static.ptir](../../filetests/tensor/outline_static/static.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `pad-with-identity`

Pads the read of a block with the identity of the fold it feeds, in place of the constraints
that keep the read inside its operand: the points those constraints skipped read the identity,
which leaves the fold as it is. `tensor-to-tile` writes a padded window as such constraints,
which hold for every dtype; a block whose read is padded can hold a register tile, which a
block with constraints cannot. The points it adds compute the body at the border of the
operand, so it is a pass rather than a canonical form.

- The block reads one operand, of static extents, in one read without a pad, and folds what
  its body yields, the read or casts of it, in one fold into new results whose combine has a
  right identity.
- The read's dtype and the dtype of each cast hold the identity exactly, as the same f64, so
  that each cast carries it bit for bit: a cast through a dtype without it, as `f32` through
  `si8` under `max`, would saturate it.
- Each constraint it drops is `0 <= c` or `c <= n - 1` for the coordinate `c` of the read on an
  axis of `n` elements; the other constraints stay.

Defined by `PadWithIdentity` in [crates/gpt-rs-ir/src/dialects/tile/passes/pad_with_identity.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/pad_with_identity.rs).

| Statistic | |
|---|---|
| `padded` | reads padded with the identity of their fold |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `BoundsToPad` | `tile.block` | `0 <= c` and `c <= n - 1` for the coordinate `c` of a block's one read on an axis of `n` elements are a pad of the identity of the block's one fold, which takes the read. | [tile/pad_with_identity/windows.ptir](../../filetests/tile/pad_with_identity/windows.ptir) |

Filetests: [tile/pad_with_identity/windows.ptir](../../filetests/tile/pad_with_identity/windows.ptir).

### `place-epilogues`

Places each block of a dispatch after the first at the register tile of the first block (`at
reg`), when the blocks so placed keep the verifier's placement rules ([`check_placements`]).
Each then runs where the block before it finishes its elements: on each element as it is
finished, such as an epilogue, or once the tile is finished, such as the block that adds the
lanes of the first block's sums. A dispatch whose blocks cannot all run there keeps them apart.

Defined by `PlaceEpilogues` in [crates/gpt-rs-ir/src/dialects/tile/passes/place_epilogues.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/place_epilogues.rs).

| Statistic | |
|---|---|
| `placed` | blocks placed in the loops of the block before them |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `PlaceAtRegisterTile` | `tile.dispatch` | The blocks after the first of a dispatch, none placed, each reading the result of the block before it: each placed at the register tile of the first, where the verifier accepts them all so placed. | [tile/place_epilogues/epilogues.ptir](../../filetests/tile/place_epilogues/epilogues.ptir) |

Filetests: [tile/place_epilogues/epilogues.ptir](../../filetests/tile/place_epilogues/epilogues.ptir).

### `plan-layouts`

Lays out each value known before the run that one block computes and another alone reads, so
that the reader takes the lanes of its vectors from consecutive elements. When the read takes
the iterator that the reader's lanes run along alone, in a coordinate other than the last, the
value is read from a copy that moves that axis last and keeps the others in their order.
`canonicalize` folds the copy into the block that computes the value, which then writes it in
the copy's order. A value of the checkpoint keeps its layout: a copy of it would hold the
weights in memory twice.

Defined by `PlanLayouts` in [crates/gpt-rs-ir/src/dialects/tile/passes/plan_layouts.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/plan_layouts.rs).

| Statistic | |
|---|---|
| `laid_out` | values written with the axis of their reader's lanes last |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `LanesLast` | `tile.block` | A block that reads a value known before the run, which another block computes for it alone, with its lanes iterator alone in a coordinate other than the last: it reads a copy of the value with that axis last. | [tile/plan_layouts/lanes.ptir](../../filetests/tile/plan_layouts/lanes.ptir) |

Filetests: [tile/plan_layouts/lanes.ptir](../../filetests/tile/plan_layouts/lanes.ptir).

### `plan-memory`

Gives every value that a function's dispatches compute a buffer, and calls each `loop.func`
with `host.launch` on buffers: the buffer form of the `host` dialect.

- Values live in one arena, `host.arena`. A greedy best fit places them, the largest first. Each
  goes at the lowest offset of the smallest gap between the placed values live at the same time,
  or after all of them when no gap fits. So values that are never live at once share bytes. The
  arena, each value in it and the memory of each allocation start at a multiple of the alignment
  that the target states for its buffers.
- An `inout` parameter updates the buffer of its argument when that argument is the result of
  an earlier launch, or an argument of the function that only launches use, and dies at this
  launch: the launch takes it once and nothing uses it later. It updates a copy otherwise.
  An argument of the function that an `inout` parameter updates last is taken (`host.take`),
  so a runtime updates the caller's memory when the caller holds the tensor no more.
- An `out<over = k>` parameter writes over the buffer of the argument of `in` parameter `k`
  when that argument is a result of an earlier launch, has the type of the parameter's result
  and dies at this launch: the launch takes it once and nothing uses it later. Of the results
  of one launch, only the first writes over a given argument.
- A value that the function returns, or that an op other than a launch reads, lives in memory
  of its own (`host.alloc`), or in the taken memory of the argument it updates.
- The function keeps its tensor arguments and results. `host.to_buffer` gives launches the
  elements of an argument or of the value of another op, and `host.to_tensor` gives other ops
  and the return the elements of a buffer.

It plans the dispatches of a function's body. A dispatch in a region of another op keeps the
value form. Every top-level `host.dispatch` must call a `loop.func`, and every result of one
must have static extents.

Defined by `PlanMemory` in [crates/gpt-rs-ir/src/dialects/host/passes/plan_memory.rs](../../crates/gpt-rs-ir/src/dialects/host/passes/plan_memory.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target that states the alignment of buffers: the arena and each allocation start at a multiple of it. |

| Statistic | |
|---|---|
| `planned` | functions given a memory plan |
| `in_place` | values written over a value that dies |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `PlanFunction` | `func.func` | A function that calls `loop.func`s takes the buffer form under its memory plan. | [host/plan_memory/arena.ptir](../../filetests/host/plan_memory/arena.ptir), [host/plan_memory/escapes.ptir](../../filetests/host/plan_memory/escapes.ptir), [host/plan_memory/gpu.ptir](../../filetests/host/plan_memory/gpu.ptir), [host/plan_memory/npu.ptir](../../filetests/host/plan_memory/npu.ptir), [host/plan_memory/update.ptir](../../filetests/host/plan_memory/update.ptir) |

Filetests: [host/plan_memory/arena.ptir](../../filetests/host/plan_memory/arena.ptir), [host/plan_memory/escapes.ptir](../../filetests/host/plan_memory/escapes.ptir), [host/plan_memory/gpu.ptir](../../filetests/host/plan_memory/gpu.ptir), [host/plan_memory/npu.ptir](../../filetests/host/plan_memory/npu.ptir), [host/plan_memory/update.ptir](../../filetests/host/plan_memory/update.ptir).

### `prune-params`

Removes from every function the arguments bound to a parameter that nothing reads. A run-time
argument stays even when nothing reads it, so the run-time arguments keep their positions.

Defined by `PruneParams` in [crates/gpt-rs-ir/src/passes/prune_params.rs](../../crates/gpt-rs-ir/src/passes/prune_params.rs).

| Statistic | |
|---|---|
| `pruned` | parameter arguments removed |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `RemoveUnreadParams` | `func.func` | A function loses the arguments bound to a parameter that nothing reads. | [tensor/prune_params/prune_params.ptir](../../filetests/tensor/prune_params/prune_params.ptir) |

Filetests: [tensor/prune_params/prune_params.ptir](../../filetests/tensor/prune_params/prune_params.ptir).

### `schedule`

Chooses the schedule of the first block of each dispatch on the target. The rules are
constraints on the target's facts, and counts of what the schedule moves, compared in order.
Issue and fill rates enter only the constraints: no time is predicted for a schedule, and none
ranks schedules. The schedule is the block's modes, outermost first:
1. the workers' shares (`d by s : workers<level = ..>`);
2. the loops over the pieces of the caches that need one (`d by t in m`);
3. the loops over the parallel iterators that the register tile does not hold;
4. the loops over the reduced iterators;
5. the register tile (`d by r : reg`) and its lanes.

The register tile:
- It is `R` rows, along one or two parallel iterators, by `V` vectors of the lanes. Each row
  iterator is one that the result's coordinates take and no `tile.index` of the body takes.
- The iterators of the rows are tried in groups. The group along which the fewest of the body's
  ops vary comes first, as the rows share the other ops. Where the rows of a group share some
  ops, the tile without rows is left out: it would do them at each point.
- Its registers fit the target's: the accumulators, the body's temporaries (one for each row
  where they vary along the rows, one for each vector where the rows share them), and the
  registers of one step's operands or of the blocks placed at the tile.
- With a fold, the accumulators cover the latency of the op that steps the state, at the rate
  the steps run. That rate is the op's issue rate, or less where a cache that is filled from
  another cache takes in the busiest worker's bytes more slowly, at the rate the target
  measured. Where no tile holds that many accumulators, the tiles with the most do.

The choice of the tile and the order of the loops:
- The candidates are the tiles whose loads and broadcasts take no more cycles, at their issue
  rates, than the fold's steps at its op's issue rate, when any tile does. The cycles of loads
  and of broadcasts add, as a broadcast from memory is a load.
- Of these, the rules take the tile and the order of the parallel loops whose busiest worker
  moves the fewest bytes into each cache, the outermost cache first. A tie goes to the one that
  loads the fewest vectors, then holds the fewest registers, then runs the fewest tiles. Among
  orders that still tie, the largest read's iterators run outermost.
- A cache holds the reads of the largest band of innermost loops whose reads fit all its ways
  but one, shared among the workers that share the cache. It takes in once a footprint of a read
  that several of those workers take. The count of a cache is the busiest worker's part of what
  its group moves in. The counts take each tile of a loop as whole, and count bytes, not lines.
- The loops over the reduced iterators run inside the parallel loops, so the fold's state stays
  in the register tile. The innermost of them moves the fewest reads, so that a read it leaves
  in place, such as the scale of a group of points, stays outside it. Of those that move as
  many, the one along which the reads move least is innermost.

Workers and caches:
- The workers split the outermost parallel loops in their order. Each loop splits into equal
  parts of whole tiles, up to as many as it has tiles and the workers left allow, so that
  consecutive workers take consecutive iterations.
- A value that a dispatch scheduled before, in module order, wrote in the run is in each cache
  where the workers that share that cache with its writers wrote it, up to the cache's capacity.
  The workers move into the cache none of the bytes of their reads that those workers wrote. The
  counts take workers that share a cache to be consecutive, as the runtime pins worker t to unit
  t.
- A cache gets the largest piece of the parallel iterators that fits it, where that piece moves
  fewer bytes into it, when the bytes it takes in cannot arrive within the time of the fold's
  steps at the rate the target measured, or the target measured none.
- Only the innermost cache may also cut the reduced iterators: where the reads of one register
  tile over every reduced point do not fit it, its piece cuts them, but only where the cut lets
  that cache keep up with the steps. The fold's state over the piece's parallel points is then
  held in scratch across the loops over the cuts (`held scratch<at = ..>`), and its bytes count
  in and out at each of their iterations.
- A read is copied into scratch, inside each worker's loops, at the outermost loop whose
  elements fit beside the other copies, where that takes fewer bytes from outside than its
  register tiles load in place.

Clips: the block clips each iterator along which a padded read's coordinate leaves its operand,
where the pad adds nothing to the fold, except for a read that a copy holds, as the copy holds
the pad. The register tile is cut at each tap to its points in range.

The pass fails where the target lacks a fact that a constraint needs, such as an op's cost or
latency, or the units of the parallel level, and names it.

Defined by `Schedule` in [crates/gpt-rs-ir/src/dialects/tile/passes/schedule/mod.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/schedule/mod.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target the schedules are chosen for. |

| Statistic | |
|---|---|
| `scheduled` | first blocks of dispatches given a schedule |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ScheduleFirstBlock` | `tile.dispatch` | The first block of a dispatch, with no modes but its lanes: its schedule. | [tile/place_epilogues/epilogues.ptir](../../filetests/tile/place_epilogues/epilogues.ptir), [tile/schedule/caches.ptir](../../filetests/tile/schedule/caches.ptir), [tile/schedule/cores.ptir](../../filetests/tile/schedule/cores.ptir), [tile/schedule/cpu.ptir](../../filetests/tile/schedule/cpu.ptir), [tile/schedule/gpu.ptir](../../filetests/tile/schedule/gpu.ptir), [tile/schedule/npu.ptir](../../filetests/tile/schedule/npu.ptir) |

Filetests: [tile/place_epilogues/epilogues.ptir](../../filetests/tile/place_epilogues/epilogues.ptir), [tile/schedule/caches.ptir](../../filetests/tile/schedule/caches.ptir), [tile/schedule/cores.ptir](../../filetests/tile/schedule/cores.ptir), [tile/schedule/cpu.ptir](../../filetests/tile/schedule/cpu.ptir), [tile/schedule/gpu.ptir](../../filetests/tile/schedule/gpu.ptir), [tile/schedule/npu.ptir](../../filetests/tile/schedule/npu.ptir).

### `seal-dispatches`

Outlines each group of blocks into a `tile.dispatch` that the function calls with
`host.dispatch`.

- Each `tile.group` becomes a dispatch, and so does each `tile.block` outside a group, as a
  group of its own.
- The dispatch takes the values its blocks read from outside the group, in the order they are
  first read, and gives what the group yields. When a block uses a dimension symbol that none
  of those values binds, the dispatch also takes the nearest region argument that does: one of
  the function, or of a region of control flow around the group.
- Its name is a hash of its text, so equal groups share one dispatch: layers that compute the
  same blocks call one.

Defined by `SealDispatches` in [crates/gpt-rs-ir/src/dialects/tile/passes/seal_dispatches.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/seal_dispatches.rs).

| Statistic | |
|---|---|
| `sealed` | groups outlined into dispatches |

Once-rewrites, which rewrite each root once, after the patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `Seal` | `tile.group`, `tile.block` | A group of a function, or a block of one outside a group, becomes a `tile.dispatch` that the function calls with `host.dispatch` in its place. Equal groups share one dispatch. | [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir), [tile/seal_dispatches/seal.ptir](../../filetests/tile/seal_dispatches/seal.ptir) |

Filetests: [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir), [tile/seal_dispatches/seal.ptir](../../filetests/tile/seal_dispatches/seal.ptir).

### `select-atoms`

Chooses how the first block of each dispatch computes its innermost step on the target that
the dispatch names.

- Vectors: the first block of a dispatch computes on vectors whose lanes run along the
  iterator that is the last coordinate of every write, when its body computes in `arith` ops
  that `isa` has a lane-wise form of, the verifier accepts it computing in them, and each block
  after it is its epilogue and computes in the same lanes ([`Dispatch::vector_lanes`]). A
  vector holds as many lanes as the target's widest register holds of the block's widest
  dtype. The block's loops walk the vectors of that iterator (`d by lanes`), unless its extent
  is one vector, and run each in the lanes (`d by lanes : lanes`).
- Any other dispatch computes on scalars.

Defined by `SelectAtoms` in [crates/gpt-rs-ir/src/dialects/tile/passes/select_atoms.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/select_atoms.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target the atoms are for. |

| Statistic | |
|---|---|
| `vectorized` | dispatches whose first block computes on vectors |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `SelectVectors` | `tile.dispatch` | A dispatch that names the pass's target, whose first block can compute on vectors: the block walks the vectors and runs each in the lanes. | [tile/select_atoms/vectors.ptir](../../filetests/tile/select_atoms/vectors.ptir) |

Filetests: [tile/name_target/contract.ptir](../../filetests/tile/name_target/contract.ptir), [tile/select_atoms/vectors.ptir](../../filetests/tile/select_atoms/vectors.ptir), [tile/split_lane_folds/lane_folds.ptir](../../filetests/tile/split_lane_folds/lane_folds.ptir), [tile/split_lane_folds/maxima.ptir](../../filetests/tile/split_lane_folds/maxima.ptir), [tile/split_lane_folds/padded.ptir](../../filetests/tile/split_lane_folds/padded.ptir).

### `split-lane-folds`

Splits the first block of a dispatch into the folds of the lanes of vectors and the block that
folds them. It applies where vectors along the block's result cannot compute it
([`Dispatch::vector_lanes`]) and the step of its fold is commutative and associative. The step
is associative exactly, as `maximum` and `minimum` are, or on floats where the dispatch's
numerics allow reassociation (`reassoc`), as `add` is.

- The block has no modes and folds one new result by `sum`, `max` or `min`, with its parallel
  iterators first. Its last iterator `k` is a `red` one of a static extent, along which its
  body computes in lanes ([`Block::body_lanes`]). The verifier accepts the lane folds computing
  in their lanes.
- Where the lanes do not fill the extent, each read that takes `k` takes it as a whole
  coordinate on an axis of that many elements. Each of those reads without a pad pads past the
  axis with the identity of the fold's step ([`CombineKind::identity`]), -0 for a float sum.
  At the pad of one of those reads, the body yields what changes nothing in the fold
  ([`Block::pad_skip`]). It does so exactly, or as products of the pad 0 where the dispatch's
  numerics skip them (`skip_pad`).
- The lane folds replace `k` by `j = k / lanes` in its place and by `l = k mod lanes`, a
  parallel iterator after the others. Their vectors run along `l`. Their reads and `tile.index`
  values take `lanes * j + l` for `k`, and they give the folds with the lanes as a last axis.
- The folding block folds the lanes of each element in halves, the upper half onto the lower,
  until one is left. The middle lane of an odd count stays for the next step, and the IR holds
  the order of the steps.

Defined by `SplitLaneFolds` in [crates/gpt-rs-ir/src/dialects/tile/passes/split_lane_folds.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/split_lane_folds.rs).

| Option | Type | Default | |
|---|---|---|---|
| `target` | `Option<Target>` | `none` | The target whose vectors the lane folds compute in. |

| Statistic | |
|---|---|
| `split` | dispatches whose first fold folds its points in the lanes of vectors |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `SplitIntoLanes` | `tile.dispatch` | A dispatch whose first block cannot compute on vectors along its result, and whose numerics let that block's fold regroup: the fold splits into the folds of its lanes and the block that folds them. | [tile/split_lane_folds/lane_folds.ptir](../../filetests/tile/split_lane_folds/lane_folds.ptir), [tile/split_lane_folds/maxima.ptir](../../filetests/tile/split_lane_folds/maxima.ptir), [tile/split_lane_folds/padded.ptir](../../filetests/tile/split_lane_folds/padded.ptir) |

Filetests: [tile/split_lane_folds/lane_folds.ptir](../../filetests/tile/split_lane_folds/lane_folds.ptir), [tile/split_lane_folds/maxima.ptir](../../filetests/tile/split_lane_folds/maxima.ptir), [tile/split_lane_folds/padded.ptir](../../filetests/tile/split_lane_folds/padded.ptir).

### `tensor-to-tile`

Converts the `tensor` ops that compute on elements to `tile.block`s that compute the same
values, bit for bit.

- Elementwise ops become a parallel block with one `arith` op. Contractions, reductions,
  `argmax`, `top_k` and windows become blocks with reduction iterators, declared in the order
  the tensor interpreter folds them. An op with an accumulator dtype casts into it in the body
  and back to its result dtype in a block of its own.
- Movement ops (`reshape`, `transpose`, `broadcast_to`, `slice`, `take`, `gather`,
  `dynamic_slice`, `extract_patches`) become copy blocks. `concat` selects among padded
  reads. `iota` is `tile.index`.
- `dynamic_update_slice` and `scatter_*` write into an operand, and `segment_*` into a tensor
  filled with the initial value of its fold. `rng_*` draws each element with `arith.rng_*`,
  and the quantization ops are their compositions in `arith`.
- `tensor.cond`, `tensor.while`, `tensor.scan` and `tensor.constant` stay, with the ops of
  their regions converted, and a `tensor.custom_call` is an error.

Defined by `TensorToTile` in [crates/gpt-rs-ir/src/conversion/tensor_to_tile/mod.rs](../../crates/gpt-rs-ir/src/conversion/tensor_to_tile/mod.rs).

| Statistic | |
|---|---|
| `converted` | tensor ops converted |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `UnaryToTile` | `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.log`, `tensor.tanh`, `tensor.erf`, `tensor.rsqrt`, `tensor.reciprocal`, `tensor.sin`, `tensor.cos`, `tensor.not` | A `tensor` op of one operand is a parallel block with the `arith` op it applies. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `BinaryToTile` | `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`, `tensor.shift_left`, `tensor.shift_right_logical`, `tensor.shift_right_arithmetic` | A `tensor` op of two operands is a parallel block with the `arith` op it applies. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `CompareToTile` | `tensor.less`, `tensor.less_equal`, `tensor.equal`, `tensor.greater_equal`, `tensor.greater`, `tensor.not_equal` | A `tensor` comparison is a parallel block with the `arith` comparison it applies. | [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `SelectToTile` | `tensor.select` | `tensor.select` is a parallel block with `arith.select`. | [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `CastToTile` | `tensor.cast` | `tensor.cast` is a parallel block with `arith.cast`. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `DotToTile` | `tensor.dot_general` | `tensor.dot_general` is a block with one reduction iterator per contracted pair, in `contract_lhs` order, that sums the products in the accumulator dtype; a parallel block casts the sums to the result dtype. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `ReduceToTile` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A `tensor.reduce_*` op is a block whose reduced axes are reduction iterators, in axis order, folding with the combine of its `arith` op in the accumulator dtype; a parallel block casts to the result dtype. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `ReshapeToTile` | `tensor.reshape` | `tensor.reshape` is a copy block that reads the input at the row-major position of each result element. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `TransposeToTile` | `tensor.transpose` | `tensor.transpose` is a copy block with permuted read coordinates. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `BroadcastToTile` | `tensor.broadcast_to` | `tensor.broadcast_to` is a copy block that reads coordinate 0 on each broadcast axis. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `SliceToTile` | `tensor.slice` | `tensor.slice` is a copy block that reads `start + k * stride` on each axis. | [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `BitcastToTile` | `tensor.bitcast` | `tensor.bitcast` of equal widths is a parallel block with `arith.bitcast`. Otherwise each result element ors together, in a `ui64`, the shifted bits of the input elements that overlap it in the row's bit stream, and a second block cuts them to the result's width. | [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `ConcatToTile` | `tensor.concat` | `tensor.concat` is a block that reads every input shifted by its offset, padded outside it, and selects the input whose range holds the point. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `IotaToTile` | `tensor.iota` | `tensor.iota` is a block that yields `tile.index` of its axis. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `TakeToTile` | `tensor.take` | `tensor.take` is a copy block that reads the table at the row its index operand names. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `DynamicSliceToTile` | `tensor.dynamic_slice` | `tensor.dynamic_slice` is a copy block from the clamped start, `min(max(start, 0), extent - size)`. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `DynamicUpdateSliceToTile` | `tensor.dynamic_update_slice` | `tensor.dynamic_update_slice` is a block that writes the patch into the base at the clamped start. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `ExtractPatchesToTile` | `tensor.extract_patches` | `tensor.extract_patches` is a copy block that reads channel `c` of window element `k` at position `o` into column `k * C + c`, padded outside the input. | [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir) |
| `ReduceWindowToTile` | `tensor.reduce_window_sum`, `tensor.reduce_window_max`, `tensor.reduce_window_min`, `tensor.reduce_window_and`, `tensor.reduce_window_or` | A `tensor.reduce_window_*` op is a block with the window offsets as reduction iterators after the positions, folding with the combine of its `arith` op. A window element in the padding, or between dilated input elements, fails one of the block's constraints, which hold for every dtype. `pad-with-identity` turns the constraints that keep the read inside the input into a pad of the fold's identity, where the dtype holds that identity. | [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir) |
| `StopGradientToTile` | `tensor.stop_gradient` | `tensor.stop_gradient` is its input: a forward computation has no gradients. | [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `ArgMaxToTile` | `tensor.argmax` | `tensor.argmax` is a block that folds its axis with `argmax` over each element and its index. | [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `SegmentToTile` | `tensor.segment_sum`, `tensor.segment_max`, `tensor.segment_min`, `tensor.segment_and`, `tensor.segment_or` | A `tensor.segment_*` op folds each row of `data` into a tensor filled with the initial value of its fold, at the row its id names, in the accumulator dtype; a parallel block casts back. | [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `SelectKToTile` | `tensor.top_k`, `tensor.bottom_k` | `tensor.top_k` and `tensor.bottom_k` are a block that folds their axis with `top_k`, which appends `k` as the last axis; for another axis, copy blocks move it back. | [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `GatherToTile` | `tensor.gather` | `tensor.gather` is a copy block whose coordinate on the axis is the index operand's element. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `ScatterToTile` | `tensor.scatter_add`, `tensor.scatter_max`, `tensor.scatter_min` | A `tensor.scatter_*` op that folds is a block that folds each update into the input at the element its index names, in row-major order of the updates, with the combine of its `arith` op, in the accumulator dtype between cast blocks. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `ScatterReplaceToTile` | `tensor.scatter_replace` | `tensor.scatter_replace` is a block that writes each update into the input at the element its index names, in row-major order of the updates, so the last one wins. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `RngToTile` | `tensor.rng_uniform`, `tensor.rng_normal` | A `tensor.rng_*` op is a block that draws element `i` of the state's stream with the `arith` op of the same name, and a block that advances the state's counter by the blocks read. | [tile/tensor_to_tile/random.ptir](../../filetests/tile/tensor_to_tile/random.ptir) |
| `DequantizeToTile` | `tensor.dequantize` | `tensor.dequantize` is a parallel block that multiplies each value by its block's scale and adds the offset, in f32. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `QuantizeToTile` | `tensor.quantize` | `tensor.quantize` is a parallel block that rounds `(x - offset) / scale` to the nearest value of the quantized dtype in `arith`. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `QuantizeStochasticToTile` | `tensor.quantize_stochastic` | `tensor.quantize_stochastic` is the block of `tensor.quantize` that rounds up when the element's draw is below the fraction of the way to the next value, compared in f64. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `QuantizeTableToTile` | `tensor.quantize_table` | `tensor.quantize_table` is a block that folds the table with `argmin` over the distances of its values to `(x - offset) / scale`, then a block that stores each index as a code. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `BlockScaleToTile` | `tensor.block_scale_amax`, `tensor.block_scale_pow2_floor`, `tensor.block_scale_pow2_ceil` | A `tensor.block_scale_*` op is a block that folds each block's range, then a block that computes each block's scale from its largest magnitude and stores it. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `BlockScaleRangeToTile` | `tensor.block_scale_range` | `tensor.block_scale_range` is a block that folds each block's range, then a block that computes each block's scale and offset from it. | [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |

Filetests: [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/copies.ptir](../../filetests/tile/canonicalize/copies.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir), [tile/tensor_to_tile/errors.ptir](../../filetests/tile/tensor_to_tile/errors.ptir), [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir), [tile/tensor_to_tile/random.ptir](../../filetests/tile/tensor_to_tile/random.ptir), [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir).

### `tile-to-loop`

Lowers each `tile.dispatch` to a `loop.func` of the same name that computes the same values, bit
for bit but for the rounding changes its `numerics` names, with the loops that its blocks'
modes, holdings, clips and placements say. A block without modes walks the points of every
iterator one at a time, in the order of its reference walk
([`Block::reference_modes`](crate::dialects::tile::Block::reference_modes)): the walk order of
the `tile.block` entry, or the order of the block when a fold updates an operand where its
points may collide. It decides nothing: each loop follows from the IR by one rule, and every
block that the verifier accepts lowers.

- The function takes each argument of the dispatch in an `in` buffer, and gives each result in
  an `out` buffer. A result that a block writes as an updated copy of an argument that nothing
  else in the dispatch reads updates the argument's buffer instead, an `inout` parameter. A
  result that a map writes at each point may share the buffer of an argument that only the
  map reads, and only at that point (`out<over = ..>`). A block result that no result gives
  lives in `private` scratch, and its extents must be static.
- The blocks lower in runs: a block, and the blocks after it that a read places in its loops
  (`at`). Each walk of the points of the run's first block is a loop nest of its own, which runs
  the reads and the body at each point. Consecutive `none` folds into new results share one walk
  ([`Block::fold_walks`](crate::dialects::tile::Block::fold_walks)), and every other fold walks
  alone. Each of the block's modes but those of its register tile is a loop over the tiles of
  its iterator, of the mode's size, outermost first: one after another, or one tile per worker
  of its level (`workers`). Worker modes of one level, each right inside the one before it,
  over another iterator and not clipped, are one `loop.forall`. It walks the tuples of their
  tiles, which a worker takes in turn. A mode of one point walks the points of its iterator's
  innermost tile, so every iterator that the register tile leaves out ends in such a loop, in
  the order of the modes. A fold into an operand copies the operand, unless it updates the
  operand's buffer, then walks every point in order.
- The `reg` and `lanes` modes are the register tile, innermost: one `loop.atom` over the
  innermost tile of its iterators, in vectors of the lanes. A loop over the tiles of an iterator
  of the register tile peels a tile that does not hold a whole tile's points: its whole tiles
  and its last tile each have a body of their own. So where the points of a tile are known
  before the run, the atom's extents are constants, and its tile holds those points, rounded up
  to whole vectors along the lanes. A loop over the tiles of any other iterator masks such a
  tile: one body serves every tile.
- A fold's state starts before the loop of the mode its `held scratch` names, or without a
  holding before its first loop over a reduced iterator, one element per point of the tiles
  the loops inside walk, and finishes after the loops inside. Without a holding, it lives in a
  `reg` buffer, the register tile's points, or in one element of private scratch for a block
  without a register tile. The scratch of a holding is private. Where the loops inside end with
  a run that walks one register tile and reduced iterators alone, that run folds into a copy, in
  a `reg` buffer, of the part of the scratch that the tile covers. The part is copied in before
  the run and back after. Otherwise each point folds into the scratch. A prefix fold walks the
  points of its prefix up to the element's innermost.
- A block placed `at reg` computes on each element that the block before it writes, where that
  block writes it. A block placed at a mode of the first block's loops computes on the elements
  of the piece that a tile of that mode's loop finishes: at the point where the block before it
  finishes an element, when it reads that element alone; else once the piece is finished,
  reading the piece's elements from private scratch.
- A read held in a copy reads from a copy in the buffer space that the dispatch's target gives
  the memory it is held for. The loop of its mode makes the copy in each of its tiles, or the
  function makes it before the nest. The copy holds the elements that the loops inside read, in
  the order they walk them ([`Walk::copy_axes`](crate::dialects::tile::Walk::copy_axes)). A
  loop inside over the tiles of an iterator that the read takes gives an axis for those tiles,
  but none when the block clips the iterator. A loop of one point over such an iterator gives an
  axis for the iterator's points. The iterators of the register tile that the read takes give
  the last axes, one for the points of each. A read held in a prefetch asks its cache in each
  tile of the loop of its mode for the element at the same place of the next tile.
- A loop over an iterator the block clips, around which the other iterators of a padded read's
  coordinate stand fixed, narrows to the points where the coordinate stays in range, when a
  point outside contributes nothing the fold keeps: the body yields the fold's identity at the
  pad, or products of a zero pad into a sum under `skip_pad`. A loop over tiles of an iterator
  of the register tile keeps its tiles and cuts each to its points in range instead, building
  the tile once for each count of points a cut tile can hold, under a check of the count, so
  that the register tile's shape stays known before the run. When the loops narrow every
  coordinate of a read this way, the read reads without its pad. The other reads keep their pad.
- The `loop.func` names the dispatch's target, and says whether the dispatch contracts, which
  `contract` then realizes.
- `tile.block` and `tile.group` outside a dispatch are errors: `seal-dispatches` outlines them.

Defined by `TileToLoop` in [crates/gpt-rs-ir/src/conversion/tile_to_loop/mod.rs](../../crates/gpt-rs-ir/src/conversion/tile_to_loop/mod.rs).

| Statistic | |
|---|---|
| `lowered` | dispatches lowered |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `DispatchToLoop` | `tile.dispatch` | `tile.dispatch` becomes the `loop.func` of its name, with a buffer for each argument and each result, and its blocks lowered in order.<br><br>- A result that may update an argument in place ([`Dispatch::updated_args`]) takes the   argument's buffer, an `inout` parameter. - A result that may share the buffer of an argument ([`Dispatch::shareable_args`]) says so on   its parameter (`out<over = ..>`). | [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [isa/loop_to_isa/once.ptir](../../filetests/isa/loop_to_isa/once.ptir), [isa/loop_to_isa/pad.ptir](../../filetests/isa/loop_to_isa/pad.ptir), [isa/loop_to_isa/tile.ptir](../../filetests/isa/loop_to_isa/tile.ptir), [loop/contract/products.ptir](../../filetests/loop/contract/products.ptir), [loop/tile_to_loop/clip.ptir](../../filetests/loop/tile_to_loop/clip.ptir), [loop/tile_to_loop/epilogue.ptir](../../filetests/loop/tile_to_loop/epilogue.ptir), [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/nest.ptir](../../filetests/loop/tile_to_loop/nest.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/reg.ptir](../../filetests/loop/tile_to_loop/reg.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir), [loop/tile_to_loop/stage.ptir](../../filetests/loop/tile_to_loop/stage.ptir) |

Filetests: [isa/loop_to_isa/forms.ptir](../../filetests/isa/loop_to_isa/forms.ptir), [isa/loop_to_isa/once.ptir](../../filetests/isa/loop_to_isa/once.ptir), [isa/loop_to_isa/pad.ptir](../../filetests/isa/loop_to_isa/pad.ptir), [isa/loop_to_isa/registers.ptir](../../filetests/isa/loop_to_isa/registers.ptir), [isa/loop_to_isa/tile.ptir](../../filetests/isa/loop_to_isa/tile.ptir), [loop/contract/products.ptir](../../filetests/loop/contract/products.ptir), [loop/tile_to_loop/clip.ptir](../../filetests/loop/tile_to_loop/clip.ptir), [loop/tile_to_loop/epilogue.ptir](../../filetests/loop/tile_to_loop/epilogue.ptir), [loop/tile_to_loop/errors.ptir](../../filetests/loop/tile_to_loop/errors.ptir), [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/nest.ptir](../../filetests/loop/tile_to_loop/nest.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/reg.ptir](../../filetests/loop/tile_to_loop/reg.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir), [loop/tile_to_loop/stage.ptir](../../filetests/loop/tile_to_loop/stage.ptir).

### `walk-order-layouts`

Lays out each value computed before the run, which a dispatch reads in its first block alone, in
the order that block's schedule walks it. The layout's axes are those of a copy of the read over
the whole nest ([`Walk::copy_axes`](crate::dialects::tile::Walk::copy_axes)): one for the tiles
of each loop over an iterator the read takes and one for the points of its innermost tile,
consecutive axes of one iterator merged. Each axis of the value whose coordinate is an iterator
splits into those axes, at sizes that divide the points around them. The dispatch that computes
the value computes it in that layout, and the reader takes it there, at `d floordiv t` and `d
mod t`. A dispatch that no call names any more is erased. A value of the checkpoint keeps its
layout: a copy of it would hold the weights in memory twice.

Defined by `WalkOrderLayouts` in [crates/gpt-rs-ir/src/dialects/tile/passes/walk_order_layouts.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/walk_order_layouts.rs).

| Statistic | |
|---|---|
| `laid_out` | values laid out in the order their reader walks them |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `LayOutInWalkOrder` | `host.dispatch` | A call of a dispatch whose first block reads a value computed before the run, which another dispatch computes for it alone in one block that writes each element once, in a layout other than the order the reader walks it: the value is computed in that order, and read there. | [tile/walk_order_layouts/weights.ptir](../../filetests/tile/walk_order_layouts/weights.ptir) |

Filetests: [tile/walk_order_layouts/weights.ptir](../../filetests/tile/walk_order_layouts/weights.ptir).

### `widen-narrow-floats`

Computes each op with [`NarrowInF32`] whose operands are bf16 or f16 in f32: it casts the
operands to f32, applies the op, and casts a float result back, which rounds it once, as the
op's semantics do. A backend with no arithmetic on those floats runs it before it decides how
to compute the ops. The casts are work of their own, so it is a pass, not part of
`canonicalize`.

Defined by `WidenNarrowFloats` in [crates/gpt-rs-ir/src/passes/widen_narrow_floats.rs](../../crates/gpt-rs-ir/src/passes/widen_narrow_floats.rs).

| Statistic | |
|---|---|
| `widened` | ops on bf16 or f16 computed in f32 |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ComputeInF32` | every op with `NarrowInF32` | An op on bf16 or f16 operands is the op on their f32 values, its float result cast back to the operands' dtype. | [tensor/widen_narrow_floats/elementwise.ptir](../../filetests/tensor/widen_narrow_floats/elementwise.ptir), [tile/widen_narrow_floats/body.ptir](../../filetests/tile/widen_narrow_floats/body.ptir) |

Filetests: [tensor/widen_narrow_floats/elementwise.ptir](../../filetests/tensor/widen_narrow_floats/elementwise.ptir), [tile/widen_narrow_floats/body.ptir](../../filetests/tile/widen_narrow_floats/body.ptir).

### `widen-reused-maps`

Computes the result of a map wide where the blocks that read it only widen what they read, one
of them at several points for each element. The map then widens each element once, where its
readers widened it at each read, as the projections that read one normalized activation in
bf16 do. The values are the same, but the map's result takes more bytes, so it is a pass rather
than a canonical form. A pipeline runs it when its target steps the readers' folds in the wide
dtype. On a target whose step takes the narrow dtype as it is, such as a dot product of bf16
pairs, the readers read the result narrow.

- The map runs at each call: a value known before the run that grows past the parameters it
  reads is no value `outline-static` stores, as a weight is not copied larger.
- The map writes each point of its iterators at an element of its own.
- Each reader reads it in one read, without a pad, holding or placement, and its body only
  casts what the read gives, all readers to one float wider than the result's; the
  coordinates of one reader's read leave out an iterator of more than one point.

Defined by `WidenReusedMaps` in [crates/gpt-rs-ir/src/dialects/tile/passes/widen_reused_maps.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/widen_reused_maps.rs).

| Statistic | |
|---|---|
| `widened` | map results widened once for readers that widen them |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `WidenInMap` | `tile.block` | A map whose readers each cast every value they read of it to one wider float, one of them at several points for each element: the map yields its value cast to that float, and its readers read it without the casts. | [tile/widen_reused_maps/activations.ptir](../../filetests/tile/widen_reused_maps/activations.ptir) |

Filetests: [tile/widen_reused_maps/activations.ptir](../../filetests/tile/widen_reused_maps/activations.ptir).

### `winograd-conv`

Computes each 3x3 convolution with stride 1 by Winograd's F(2x2, 3x3) where the compilation
allows its rounding (`numerics = <winograd = true>`). It transforms the weight and the input
into 16 matrices each, runs one contraction over the input channels for all 16 pairs, and
transforms their products into the convolution's result. That takes 16 products per tile of
2x2 outputs and channel pair instead of 36. A pipeline runs it only when asked to.

- A convolution here is a block whose one fold sums the products of a read padded with 0 and
  a read of the weight, where two coordinates of the padded read are each an output iterator
  plus a tap iterator of 3 points, with step 1.
- It convolves one image: its iterators of static extents are the rows, the columns, their
  taps, the input channels and one more, the output channels, so a batch of several images,
  whose iterator is one more, keeps its block.
- Its body multiplies two f32 elements. A convolution of another dtype keeps its block.
- The weight transform runs with the other blocks. It computes a value larger than the weight
  it reads, so no pass moves it to load time.
- The result is the convolution's, at its coordinates and in its layout, so the blocks that run
  on its elements still take it as their epilogue.

The sums are equal in real arithmetic, not bit for bit: the transforms round on their own.

Defined by `WinogradConv` in [crates/gpt-rs-ir/src/dialects/tile/passes/winograd_conv.rs](../../crates/gpt-rs-ir/src/dialects/tile/passes/winograd_conv.rs).

| Option | Type | Default | |
|---|---|---|---|
| `numerics` | `Numerics` | `<>` | The rounding changes the compilation allows: the pass rewrites only where they name `winograd`. |

| Statistic | |
|---|---|
| `rewritten` | convolutions computed by Winograd's F(2x2, 3x3) |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ConvToWinograd` | `tile.block` | A 3x3 convolution with stride 1, `y[p, q, k] = sum x[p + i - pad, q + j - pad, c] w[i, j, c, k]` over the taps `i`, `j` and the channels `c`, as Winograd's F(2x2, 3x3) computes it. Each 2x2 tile of the output comes from a 4x4 tile of the input. The weight transform `U = G w G^T` and the input transform `V = B^T d B` give 16 matrices each, in one block per row of `G` or `B^T`. One contraction over the input channels, `M[r, s, b, k] = sum V_b[r, s, c] U_b[c, k]`, picks each pair of matrices from lists by `b`. The output transform `Y = A^T M A` gives each output element at the convolution's coordinates and in its layout. | [tile/winograd_conv/conv.ptir](../../filetests/tile/winograd_conv/conv.ptir) |

Filetests: [tile/winograd_conv/allowed.ptir](../../filetests/tile/winograd_conv/allowed.ptir), [tile/winograd_conv/conv.ptir](../../filetests/tile/winograd_conv/conv.ptir).
