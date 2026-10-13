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
(`type Root = Has<Commutative>`). The example on this page compiles as a doc test. It is not a
pass of the crate.

```rust
use anyhow::Result;
use gpt_rs_ir::dialects::tensor::{Div, Mul, Reciprocal};
use gpt_rs_ir::ir::{Matched, Module};
use gpt_rs_ir::passes::{pass, run_patterns, Pass};
use gpt_rs_ir::require;
use gpt_rs_ir::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites};

/// `x / s` is `x * reciprocal(s)` for a static float `s`: the backend computes the reciprocal
/// once at load time, and a run multiplies, which costs less than dividing.
#[pattern]
struct DivideByStatic;

impl Pattern for DivideByStatic {
    type Root = Div;

    fn rewrite(&self, div: Matched<'_, Div>, rewriter: &mut Rewriter<'_>) -> Result<(), NoMatch> {
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

    fn run(&self, module: &mut Module) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::REWRITTEN)
    }
}
```

The pass:

- The doc comment is required. Its first paragraph is the pass's summary in the catalog.
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
- It reads the IR through its root, a `Matched` op with its attributes and an accessor per
  operand, result and region group, and through the `ValueRef`s those give: `defining_op::<T>()`,
  `uses()`, `has_one_use()`, `tensor_type()`, `constant()`, `splat()`, `known_before_run()` and
  `depends_on_parameter()`.
- It changes the IR only through the rewriter: `create`, `create_op`, `create_with_regions`,
  `replace_op`, `replace_all_uses`, `set_op`, `set_operands`, `erase` and `remove_arg`. New ops go
  before the root unless `set_insertion_point` moves them. The rewriter stages the changes, and
  the driver applies them when the pattern returns `Ok`.
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
        ins: TransposeOperands<Folding<'_>>,
        _cx: &FoldCx<'_>,
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
passes::populate_tensor_passes(&mut pm);
pm.add(PackWeights);
passes::populate_load_time_passes(&mut pm);
```

Some passes of a backend put parameters in the layouts its kernels read, such as the reshape and
transpose into the C GEMM's panels. They go before `populate_load_time_passes`, whose
`outline-static` moves their work to load time. Such a pass writes plain ops, so the module keeps
its meaning and the interpreter checks the pass. The catalog lists the pipeline of each backend.

Every module a backend compiles holds only `tensor` ops. The backend chooses its kernels when it
compiles: it recognizes op sequences, such as a convolution or a projection over a stored bf16
weight, and runs one kernel in their place.

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

## Folds

The passes that fold an op try its own `#[fold]` folder, then the folds of its traits. A fold replaces the op by an existing value or by a constant that repeats or moves constant bytes; it never computes on elements.

| Op | Folds | Fires in |
|---|---|---|
| `tensor.bitcast` | A bitcast to the input's own type is the input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.broadcast_to` | A broadcast to the input's own type is the input. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.cast` | A cast to the input's own type is the input. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `tensor.concat` | A concat of one input is that input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.dynamic_slice` | A slice as large as its input starts at 0 whatever `starts` holds, so it is the input. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
| `tensor.dynamic_update_slice` | An update as large as the base replaces all of it, so the result is the update. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
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
| `CanonicalIndices` | `tile.block` | Every index expression is in canonical form: sums with merged coefficients, `floordiv` and `mod` reduced where the iterators' extents allow, and `min` and `max` decided where the extents decide them. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `CollapseChain` | `tensor.reshape`, `tensor.broadcast_to`, `tensor.slice`, `tensor.transpose` | `t(t(x))` is one `t` of `x` for a movement op `t`: a reshape or `broadcast_to` to the last shape, the slice whose starts are `s1.starts + s2.starts * s1.strides` and whose strides are `s1.strides * s2.strides`, and the transpose `q[i] = p1[p2[i]]`. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `CompactInputs` | `tile.block` | A read whose value the body does not use is removed, with its body argument. Operands that hold the same value merge into the first, and operands the block no longer names are removed. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `CompareWithItself` | `tensor.less`, `tensor.less_equal`, `tensor.equal`, `tensor.greater_equal`, `tensor.greater`, `tensor.not_equal` | A comparison of an integer or `i1` value with itself holds everywhere or nowhere, as its relation does on equal operands. Equal constants are one value once the driver merges them. Floats are left alone: NaN is not equal to itself. | [tensor/canonicalize/constants.ptir](../../filetests/tensor/canonicalize/constants.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `ConcatOfConcat` | `tensor.concat` | A concat of a concat on the same axis is one concat of the inner concat's inputs. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `DotOfOneProduct` | `tensor.dot_general` | A dot whose contracted axes hold one element sums one product. It is the product of the operands, laid out along the result's axes and cast to the accumulator, plus 0: the sum starts from 0, so a product of `-0.0` gives `0.0`. The result is cast to `out`. | [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir) |
| `DropDefaultDtypes` | `tensor.dot_general`, `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or`, `tensor.reduce_window_sum`, `tensor.reduce_window_max`, `tensor.reduce_window_min`, `tensor.reduce_window_and`, `tensor.reduce_window_or` | An `out` equal to the input dtype, or an `accum` equal to its default accumulator, is left out. | [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir) |
| `DropUnitIters` | `tile.block` | The pattern removes an iterator of extent 1, which is always 0. It writes 0 for it in every expression and write, and moves the later iterators down one position. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `EraseIdentityBlock` | `tile.block` | A block that reads its operand at its own iterators, in the order it writes them, into a result of the operand's type, and yields what it reads, is that operand. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `Factor` | every op with `DistributesOver` | `g(f(a, b), f(a, c))` is `f(a, g(b, c))` when `f` distributes over `g`, which saves an `f`. For a commutative `f` the shared operand may be on either side. The pattern is rooted at either `f` and rewrites its only use, `g`. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `FoldIndexArith` | `arith.add`, `arith.sub`, `arith.mul`, `arith.minimum`, `arith.maximum`, `arith.neg` | Integer `arith` on `tile.index` values and constants is one `tile.index` of the combined expression, so that index math is in the index language. It folds where no value leaves the range of the dtype: always for si64, and for another dtype where the block's iterator extents bound the values. | [tile/canonicalize/index.ptir](../../filetests/tile/canonicalize/index.ptir) |
| `GroupStatic` | every op with `Associative` | `f(f(x, a), b)` is `f(x, f(a, b))` when `f` is associative, `a` and `b` are static, and one of them reads a parameter. The backend then computes `f(a, b)` once at load time, so a run computes one `f` where it computed two. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `HoistBroadcast` | every op with `Elementwise` | `f(broadcast_to(a), broadcast_to(b))` is `broadcast_to(f(a, b))` when `a` and `b` have the same shape, so `f` computes on fewer elements. The pattern splats a splat constant operand to that shape. | [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `IotaOfUnitAxis` | `tensor.iota` | A `tensor.iota` along an axis of extent 1 counts no further than 0: it is 0, broadcast. | [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir) |
| `MergeReductions` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | `reduce(reduce(x, a), b)` of one fold is one reduction of `x` over the axes of both, reshaped to the result, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `NonNegativeAxis` | `tensor.argmax`, `tensor.top_k`, `tensor.bottom_k`, `tensor.concat`, `tensor.iota`, `tensor.gather`, `tensor.scatter_add`, `tensor.scatter_max`, `tensor.scatter_min`, `tensor.scatter_replace` | A negative `axis` is written as the axis it counts back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `NonNegativeDotAxes` | `tensor.dot_general` | Negative batch and contract axes are written as the axes they count back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `ReduceOfBroadcast` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A reduction of `broadcast_to(x)` is `x` reshaped to the result when its fold is idempotent, the reduction covers every axis the broadcast repeats `x` along, and its other axes have extent 1. Each repeated axis must have a static extent above zero, since a reduction over no elements gives its initial value. | [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `ReduceOfConcat` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A reduction of `concat(x1, x2, ...)` that reduces the concatenated axis is its fold applied to the reductions of the parts, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `SliceOfConcat` | `tensor.slice` | A slice of a concat that reads one input along the concatenated axis is a slice of that input. A slice that spans several inputs stays: as a concat of slices, it would copy what it reads twice. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `SortedReduceAxes` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | The reduced axes are written non-negative and in increasing order. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `SplatReads` | `tile.block` | A read without `pad` of a constant whose elements are all equal is that element, an `arith.constant` in the body. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir) |
| `TransposeIntoDot` | `tensor.dot_general` | A transpose that feeds a dot folds into the dot's axes when it keeps the order of the operand's free axes: the dot reads the transpose's input and names its batch and contracted axes there. Each sum keeps its order, which follows the listed contracted axes. | [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `TransposeOfUnitAxes` | `tensor.transpose` | A transpose that moves only axes of extent 1 keeps the row-major order of the elements, so it is a reshape to its result shape. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `UnitStrides` | `tensor.slice` | Steps of 1 on every axis are written as no strides. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |

Filetests: [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir), [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir), [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/canonicalize/constants.ptir](../../filetests/tensor/canonicalize/constants.ptir), [tensor/canonicalize/dot.ptir](../../filetests/tensor/canonicalize/dot.ptir), [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/fuel.ptir](../../filetests/tensor/canonicalize/fuel.ptir), [tensor/canonicalize/group_static.ptir](../../filetests/tensor/canonicalize/group_static.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/masks.ptir](../../filetests/tensor/canonicalize/masks.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/canonicalize/index.ptir](../../filetests/tile/canonicalize/index.ptir).

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
| `ScaleIntoDot` | `tensor.mul` | `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are static and `s` varies only along result axes that come from `w`. `s'` is `s` laid out along those axes of `w` and broadcast to its shape. The scaled weight is static and reads a parameter, so `outline-static` moves it to load time.<br><br>A weight read through a cast is left alone: scaling it would store a widened copy. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/dot_scale_fold/scales.ptir](../../filetests/tensor/dot_scale_fold/scales.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |

Filetests: [tensor/dot_scale_fold/cast.ptir](../../filetests/tensor/dot_scale_fold/cast.ptir), [tensor/dot_scale_fold/scales.ptir](../../filetests/tensor/dot_scale_fold/scales.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `fold`

Replaces each op that folds by an existing value or a constant, until nothing changes. The
folds are those of `canonicalize`, without its patterns.

Defined by `Fold` in [crates/gpt-rs-ir/src/passes/fold.rs](../../crates/gpt-rs-ir/src/passes/fold.rs).

| Statistic | |
|---|---|
| `folded` | ops replaced by an existing value or a constant |

It applies every [fold](#folds).

Filetests: [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir).

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

### `outline-static`

Moves each static value that a run-time op reads into a load-time function, which the runtime
runs once when it loads the parameters. The function that reads the value gets a new argument
bound to the parameter the load-time function computes.

- What moves is the value before an op that only repeats its elements, such as a
  `broadcast_to`, so the runtime keeps the smaller tensor.
- A static value computed from no parameter stays: as a parameter, it would be stored once for
  each shape a program is compiled for.
- The values stored from a parameter take at most the parameter's bytes in total. The pass
  allocates those bytes in the order the values are defined. So the values a function stores
  never take more bytes than the parameters they read. A value that does not fit, such as a
  weight cast to a wider dtype, stays, and the static values it reads move instead. A value or
  parameter of unknown size never fits.
- The parameter's id is a hash of the load-time function's text, so equal computations share
  one parameter across functions and programs. A load-time function can read the result of
  one defined before it.

Defined by `OutlineStatic` in [crates/gpt-rs-ir/src/passes/outline_static.rs](../../crates/gpt-rs-ir/src/passes/outline_static.rs).

| Statistic | |
|---|---|
| `outlined` | values moved into load-time functions |

Filetests: [tensor/outline_static/budget.ptir](../../filetests/tensor/outline_static/budget.ptir), [tensor/outline_static/rerun.ptir](../../filetests/tensor/outline_static/rerun.ptir), [tensor/outline_static/static.ptir](../../filetests/tensor/outline_static/static.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `prune-params`

Removes from every function the arguments bound to a parameter that nothing reads. A run-time
argument stays even when nothing reads it, so the run-time arguments keep their positions.

Defined by `PruneParams` in [crates/gpt-rs-ir/src/passes/prune_params.rs](../../crates/gpt-rs-ir/src/passes/prune_params.rs).

| Statistic | |
|---|---|
| `pruned` | parameter arguments removed |

Filetests: [tensor/prune_params/prune_params.ptir](../../filetests/tensor/prune_params/prune_params.ptir).

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

Filetests: [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir), [tile/seal_dispatches/seal.ptir](../../filetests/tile/seal_dispatches/seal.ptir).

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
| `UnaryToTile` | `tensor.neg`, `tensor.abs`, `tensor.exp`, `tensor.log`, `tensor.tanh`, `tensor.erf`, `tensor.rsqrt`, `tensor.reciprocal`, `tensor.sin`, `tensor.cos`, `tensor.not` | A `tensor` op of one operand is a parallel block with the `arith` op it applies. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `BinaryToTile` | `tensor.add`, `tensor.sub`, `tensor.mul`, `tensor.div`, `tensor.maximum`, `tensor.minimum`, `tensor.and`, `tensor.or`, `tensor.xor`, `tensor.shift_left`, `tensor.shift_right_logical`, `tensor.shift_right_arithmetic` | A `tensor` op of two operands is a parallel block with the `arith` op it applies. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `CompareToTile` | `tensor.less`, `tensor.less_equal`, `tensor.equal`, `tensor.greater_equal`, `tensor.greater`, `tensor.not_equal` | A `tensor` comparison is a parallel block with the `arith` comparison it applies. | [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `SelectToTile` | `tensor.select` | `tensor.select` is a parallel block with `arith.select`. | [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir) |
| `CastToTile` | `tensor.cast` | `tensor.cast` is a parallel block with `arith.cast`. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `DotToTile` | `tensor.dot_general` | `tensor.dot_general` is a block with one reduction iterator per contracted pair, in `contract_lhs` order, that sums the products in the accumulator dtype; a parallel block casts the sums to the result dtype. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `ReduceToTile` | `tensor.reduce_sum`, `tensor.reduce_max`, `tensor.reduce_min`, `tensor.reduce_and`, `tensor.reduce_or` | A `tensor.reduce_*` op is a block whose reduced axes are reduction iterators, in axis order, folding with the combine of its `arith` op in the accumulator dtype; a parallel block casts to the result dtype. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir) |
| `ReshapeToTile` | `tensor.reshape` | `tensor.reshape` is a copy block that reads the input at the row-major position of each result element. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `TransposeToTile` | `tensor.transpose` | `tensor.transpose` is a copy block with permuted read coordinates. | [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `BroadcastToTile` | `tensor.broadcast_to` | `tensor.broadcast_to` is a copy block that reads coordinate 0 on each broadcast axis. | [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `SliceToTile` | `tensor.slice` | `tensor.slice` is a copy block that reads `start + k * stride` on each axis. | [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `BitcastToTile` | `tensor.bitcast` | `tensor.bitcast` of equal widths is a parallel block with `arith.bitcast`. Otherwise each result element ors together, in a `ui64`, the shifted bits of the input elements that overlap it in the row's bit stream, and a second block cuts them to the result's width. | [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `ConcatToTile` | `tensor.concat` | `tensor.concat` is a block that reads every input shifted by its offset, padded outside it, and selects the input whose range holds the point. | [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir) |
| `IotaToTile` | `tensor.iota` | `tensor.iota` is a block that yields `tile.index` of its axis. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `TakeToTile` | `tensor.take` | `tensor.take` is a copy block that reads the table at the row its index operand names. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir) |
| `DynamicSliceToTile` | `tensor.dynamic_slice` | `tensor.dynamic_slice` is a copy block from the clamped start, `min(max(start, 0), extent - size)`. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `DynamicUpdateSliceToTile` | `tensor.dynamic_update_slice` | `tensor.dynamic_update_slice` is a block that writes the update into the base at the clamped start. | [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir) |
| `ExtractPatchesToTile` | `tensor.extract_patches` | `tensor.extract_patches` is a copy block that reads channel `c` of window element `k` at position `o` into column `k * C + c`, padded outside the input. | [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir) |
| `ReduceWindowToTile` | `tensor.reduce_window_sum`, `tensor.reduce_window_max`, `tensor.reduce_window_min`, `tensor.reduce_window_and`, `tensor.reduce_window_or` | A `tensor.reduce_window_*` op is a block with the window offsets as reduction iterators after the positions, folding with the combine of its `arith` op. A window element in the padding or between dilated input elements fails one of the block's constraints. | [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir) |
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

Filetests: [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [tile/canonicalize/bodies.ptir](../../filetests/tile/canonicalize/bodies.ptir), [tile/tensor_to_tile/contraction.ptir](../../filetests/tile/tensor_to_tile/contraction.ptir), [tile/tensor_to_tile/control.ptir](../../filetests/tile/tensor_to_tile/control.ptir), [tile/tensor_to_tile/elementwise.ptir](../../filetests/tile/tensor_to_tile/elementwise.ptir), [tile/tensor_to_tile/errors.ptir](../../filetests/tile/tensor_to_tile/errors.ptir), [tile/tensor_to_tile/indexing.ptir](../../filetests/tile/tensor_to_tile/indexing.ptir), [tile/tensor_to_tile/layout.ptir](../../filetests/tile/tensor_to_tile/layout.ptir), [tile/tensor_to_tile/quant.ptir](../../filetests/tile/tensor_to_tile/quant.ptir), [tile/tensor_to_tile/random.ptir](../../filetests/tile/tensor_to_tile/random.ptir), [tile/tensor_to_tile/window.ptir](../../filetests/tile/tensor_to_tile/window.ptir).

### `tile-to-loop`

Lowers each `tile.dispatch` to a `loop.func` of the same name that computes the same values, bit
for bit, with the loops that its schedule record asks for. A dispatch whose schedule record has
no nest gets the reference schedule: one sequential loop per iterator, in the order of the
block.

- The function takes each argument of the dispatch in an `in` buffer, and gives each result in
  an `out` buffer. A block result that no result gives lives in `private` scratch, and its
  extents must be static.
- Each fold of a block is a loop nest of its own, which runs the reads and the body at each
  point. Without a nest, a fold into a new result walks the parallel iterators, then the
  reduction iterators inside them. It folds into private scratch as the combine kind's reference
  does. A fold into an operand copies the operand, then walks every point in order.
- `tile.block` and `tile.group` outside a dispatch are errors: `seal-dispatches` outlines them.

Defined by `TileToLoop` in [crates/gpt-rs-ir/src/conversion/tile_to_loop/mod.rs](../../crates/gpt-rs-ir/src/conversion/tile_to_loop/mod.rs).

| Statistic | |
|---|---|
| `lowered` | dispatches lowered |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `DispatchToLoop` | `tile.dispatch` | `tile.dispatch` becomes the `loop.func` of its name: an `in` buffer for each argument, an `out` buffer for each result, and its blocks lowered in order. | [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/nest.ptir](../../filetests/loop/tile_to_loop/nest.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir) |

Filetests: [loop/tile_to_loop/errors.ptir](../../filetests/loop/tile_to_loop/errors.ptir), [loop/tile_to_loop/fold.ptir](../../filetests/loop/tile_to_loop/fold.ptir), [loop/tile_to_loop/indexing.ptir](../../filetests/loop/tile_to_loop/indexing.ptir), [loop/tile_to_loop/map.ptir](../../filetests/loop/tile_to_loop/map.ptir), [loop/tile_to_loop/nest.ptir](../../filetests/loop/tile_to_loop/nest.ptir), [loop/tile_to_loop/pipeline.ptir](../../filetests/loop/tile_to_loop/pipeline.ptir), [loop/tile_to_loop/softmax.ptir](../../filetests/loop/tile_to_loop/softmax.ptir).
