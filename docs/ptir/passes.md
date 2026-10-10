<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# Passes

The graph arena optimizes every program it compiles with the backend's pipeline of these passes
before the backend runs it. The passes work on any dialect; the ones that only rewrite `tensor` ops
live in `crates/gpt-rs-ir/src/dialects/tensor/passes/`. The ops' semantics are their `Eval` impls,
which the filetests run before and after each case's pass list. The pass catalog,
`docs/ptir/passes.md`, is generated from the code. It lists the pipeline of each backend, and every
pass with its options, statistics, folds and patterns.

## Writing a pass

A pass is a struct with `#[pass]` and an `impl Pass`. This one walks the module itself: it
matches one op, checks it, rewrites it and counts what it did. The two full examples on this page
compile as doc tests. They are not passes of the crate.

```rust
use anyhow::Result;
use gpt_rs_ir::dialects::tensor::StopGradient;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::{pass, Pass};
use gpt_rs_ir::rewrite::Rewriter;

/// Replaces each `tensor.stop_gradient` by its input: a forward pass computes no gradients, so
/// the op only copies its input.
#[pass(
    name = "strip-stop-gradient",
    stats(stripped = "stop_gradient ops replaced by their input")
)]
pub struct StripStopGradient;

impl Pass for StripStopGradient {
    fn run(&self, module: &mut Module) -> Result<bool> {
        let mut stripped = 0;
        // `walk` lists every op, those in regions too, before the loop changes anything.
        for inst in module.walk(module.body()) {
            // Match: `op_as` gives the op with its attributes, operands and results by name.
            let Some(op) = module.op_as::<StopGradient>(inst) else {
                continue;
            };
            // Rewrite: the uses of the result read the input, and the op is erased.
            Rewriter::new(module).replace(inst, &[op.operands.input])?;
            stripped += 1;
        }
        Self::STRIPPED.add(stripped);
        Ok(stripped > 0)
    }
}
```

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

## Writing a pattern

Most rewrites are patterns: local rewrites that the `Greedy` driver applies to each op until
nothing changes. A pattern is a struct with `#[pattern]` and an `impl Pattern`, rooted at one op
type, or at every op with a trait (`type Root = Has<Commutative>`). It returns `Ok(false)` and
leaves the IR alone when it does not apply, and `Ok(true)` once it has rewritten.

```rust
use anyhow::Result;
use gpt_rs_ir::dialects::tensor::{Binary, BinaryKind, Unary, UnaryKind};
use gpt_rs_ir::module::Module;
use gpt_rs_ir::op::Matched;
use gpt_rs_ir::passes::{pass, Greedy, Pass, Rewrites};
use gpt_rs_ir::rewrite::{pattern, Pattern, PatternSet, Rewriter};

/// `x / s` is `x * reciprocal(s)` for a static float `s`: the backend computes the reciprocal
/// once at load time, and a run multiplies, which costs less than dividing.
#[pattern]
struct DivideByStatic;

impl Pattern for DivideByStatic {
    // `Binary` roots the whole family of binary ops; the pattern checks the kind.
    type Root = Binary;

    fn rewrite(&self, div: Matched<Binary>, rw: &mut Rewriter<'_>) -> Result<bool> {
        // Match: a division.
        if div.attrs.kind != BinaryKind::Div {
            return Ok(false);
        }
        let (x, s) = (div.operands.lhs, div.operands.rhs);
        // Check: floats only, as an integer division truncates; and a static divisor, computed
        // from constants and parameters alone, so that the reciprocal moves to load time.
        let float = rw.module.ty(s).as_tensor().is_some_and(|t| t.dtype.is_float());
        if !float || !rw.module.is_static(s) {
            return Ok(false);
        }
        // Rewrite: build the new ops before the root, then send the root's uses to the product.
        // The result types come from the ops' type rules.
        let at = div.inst;
        let r = rw.insert_before(at, Unary { kind: UnaryKind::Reciprocal }, s)?.output;
        let product = rw.insert_before(at, Binary { kind: BinaryKind::Mul }, (x, r))?.output;
        rw.replace(at, &[product])?;
        Ok(true)
    }
}

/// Adds the patterns that turn a division by a static value into a multiplication.
pub fn populate_divide_by_static_patterns(set: &mut PatternSet) {
    set.add(DivideByStatic);
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

impl Pass for MultiplyByReciprocal {
    // What the pass applies: `run` passes it to the driver, and the catalog lists it.
    fn rewrites(&self) -> Rewrites {
        let mut patterns = PatternSet::new();
        populate_divide_by_static_patterns(&mut patterns);
        Rewrites {
            folds: false,
            patterns,
        }
    }

    fn run(&self, module: &mut Module) -> Result<bool> {
        let report = Greedy::new(self.rewrites()).erase_dead().run(module)?;
        Self::REWRITTEN.add(report.rewritten);
        Ok(report.changed())
    }
}
```

- The doc comment of the pattern is required: it states the rewrite, and the catalog shows it.
- `rw.def_as::<T>(value)` matches the op that defines an operand, `rw.module` answers the queries
  (`ty`, `uses`, `constant`, `is_static`), and `rw.insert_before`, `rw.replace`, `rw.set_op` and
  `rw.erase` change the IR. The driver visits every op that a rewrite touched again.
- A pattern that does not keep every result bit for bit, or that chooses a layout for one backend,
  belongs to the passes that add it, through a `populate_*_patterns` function, so that several
  passes can share it.

## Canonical patterns and folds

A pattern that keeps every result bit for bit (apart from a NaN's sign and payload) and leads to
one normal form is canonical: `#[pattern(canonical)]` registers it, and `canonicalize` applies
every registered one. A pattern generic over its root op names the ops it registers for, as in
`#[pattern(canonical(ArgMax, Concat))]`.

```rust
/// `transpose(transpose(x, p1), p2)` is `transpose(x, q)` with `q[i] = p1[p2[i]]`.
#[pattern(canonical)]
struct CollapseTransposeChain;
```

An op's own folder is `#[fold]` on `impl Fold for <op>`, next to the op, with a doc comment that
says what the op folds to. The traits of an op fold it too, by the laws they state.

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
            output: Folded::Value(ins.input.value),
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
| `tensor.broadcast_to` | A broadcast to the input's own type is the input. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.cast` | A cast to the input's own type is the input. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `tensor.concat` | A concat of one input is that input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.dynamic_slice` | A slice as large as its input starts at 0 whatever `starts` holds, so it is the input. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
| `tensor.dynamic_update_slice` | An update as large as the base replaces all of it, so the result is the update. | [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir) |
| `tensor.reshape` | A reshape to the input's own type is the input, and a reshape of a constant is that constant with the new extents, its elements in the same row-major order. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.select` | `select(pred, on_true, on_false)` is `on_true` when `pred` is all true or both branches are the same value, and `on_false` when `pred` is all false. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `tensor.slice` | A slice with the input's extents covers the whole input, so it is the input. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `tensor.transpose` | A transpose by the identity `perm` is the input. | [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |

The traits whose folds fire in the filetests; [tensor.md](tensor.md#traits) describes every trait.

| Trait | | Fires in |
|---|---|---|
| `Absorbing` | `f(a, z) = z`, for the integers and `i1`. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `Commutative` | `f(a, b) = f(b, a)`. Folding puts a run-time operand first and a constant last (see `Module::is_static`), so the other folds, the patterns and CSE see a constant on the right. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir), [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |
| `Idempotent` | For a unary op `f(f(a)) = f(a)`, for a binary one `f(a, a) = a`. | [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `Involution` | `f(f(a)) = a`. | [tensor/canonicalize/fuel.ptir](../../filetests/tensor/canonicalize/fuel.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |
| `Movement` | Each result element is an element of one of the data operands: the first `data` operands, or all of them when `data` is `None`. The op moves elements without computing on them, so a splat stays a splat. The fold builds that splat only when it holds no more elements than the operands: a constant stores every element, so a broadcast splat stays a broadcast. | [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir) |
| `RightIdentity` | `f(a, e) = a`. A commutative op has `e` on the left too, since folding moves constants to the right. | [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir) |

## Passes

### `c-pack-weights`

Puts the weights of the dots that C kernels compute in the layout each kernel reads: GEMM
weights in the GEMM's panels, and grouped convolution weights tap by tap.

Each layout is a transpose of the weight, after a reshape for the GEMM's panels. The dot then
contracts the weight in that layout, so the IR keeps its meaning. When the weight is static,
`outline-static` moves the layout to load time, and the kernel reads the laid-out parameter.
GEMM weights that are not static, activations among them, stay as they are, and the GEMM packs
the panels it needs on each call.

Defined by `PackWeights` in [crates/gpt-rs-backend-c/src/optimizer/mod.rs](../../crates/gpt-rs-backend-c/src/optimizer/mod.rs).

| Statistic | |
|---|---|
| `packed` | weights laid out for a kernel |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `PackGemmWeight` | `tensor.dot_general` | Rewrites an f32 `x [.., K] . w`, which codegen sends to the GEMM (a convolution's patches among them), into `reshape(x . packed)`, where `packed` is `w` in panels of `PANEL_N` columns, `[N / PANEL_N, K, PANEL_N]`. `w` is `[K, N]` or `[N, K]`. The packed dot gives `[.., N / PANEL_N, PANEL_N]`, the same elements as `[.., N]`.<br><br>It packs only a static `w` whose packed form takes no more bytes than the parameters it is computed from, so `outline-static` moves the packing to load time. It skips a `w` whose `N` is not a multiple of `PANEL_N`, since the panels would pad it. A run-time `x` with one row, or with fewer than `LINEAR_PACK_MIN_ROWS` rows against an `[N, K]` weight, leaves `w` unpacked: the GEMM or the dot-product kernels read `w` in place. | [backend/c/pack.ptir](../../filetests/backend/c/pack.ptir), [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir) |
| `GroupedConvWeight` | `tensor.dot_general` | Lays out the weight of a grouped convolution for the grouped kernel. The dot of the patches `[N, OH, OW, KH, KW, G, C_in / G]` with the weight `[G, C_out / G, C_in / G, KH, KW]`, batched over the groups, becomes the same dot with the weight transposed to `[KH, KW, C_in / G, G, C_out / G]`, the order of `[KH, KW, C_in / G, C_out]` that the kernel reads. The result keeps its axes. `outline-static` moves the transpose of a static weight to load time. | [backend/c/pack.ptir](../../filetests/backend/c/pack.ptir) |

Filetests: [backend/c/pack.ptir](../../filetests/backend/c/pack.ptir).

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
| `CollapseBroadcastChain` | `tensor.broadcast_to` | `broadcast_to(broadcast_to(x, s1), s2)` is `broadcast_to(x, s2)`: every axis of `x` that broadcasts to `s1` broadcasts to `s2` too. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `CollapseReshapeChain` | `tensor.reshape` | `reshape(reshape(x))` is one reshape of `x`: both keep the row-major order. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `CollapseSliceChain` | `tensor.slice` | `slice(slice(x, s1), s2)` is one slice of `x`: element `k` of the second slice is element `s2.starts + k * s2.strides` of the first, which is element `s1.starts + (s2.starts + k * s2.strides) * s1.strides` of `x`. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |
| `CollapseTransposeChain` | `tensor.transpose` | `transpose(transpose(x, p1), p2)` is `transpose(x, q)` with `q[i] = p1[p2[i]]`. | [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir) |
| `DropDefaultDotDtypes` | `tensor.dot_general` | An `out` equal to the operands' dtype, or an `accum` equal to their default accumulator, is left out. | [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir) |
| `DropDefaultReduceDtypes` | `tensor.reduce_{kind}` (`Reduce`) | An `out` equal to the input's dtype, or an `accum` equal to its default accumulator, is left out. | [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir) |
| `DropDefaultWindowAccum` | `tensor.reduce_window_{kind}` (`ReduceWindow`) | An `accum` equal to the input's default accumulator is left out. | [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir) |
| `Factor` | every op with `DistributesOver` | `g(f(a, b), f(a, c))` is `f(a, g(b, c))` when `f` distributes over `g`, which saves an `f`. For a commutative `f` the shared operand may be on either side. The pattern is rooted at either `f` and rewrites its only use, `g`. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `GroupStatic` | every op with `Associative` | `f(f(x, a), b)` is `f(x, f(a, b))` when `f` is associative, `a` and `b` are static, and one of them reads a parameter. The backend then computes `f(a, b)` once at load time, so a run computes one `f` where it computed two. | [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir) |
| `HoistBroadcast` | every op with `Elementwise` | `f(broadcast_to(a), broadcast_to(b))` is `broadcast_to(f(a, b))` when `a` and `b` have the same shape, so `f` computes on fewer elements. The pattern rebuilds a splat constant operand in that shape. | [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir) |
| `MergeReductions` | `tensor.reduce_{kind}` (`Reduce`) | `reduce(reduce(x, a), b)` is one reduction of `x` over the axes of both, reshaped to the result, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `NonNegativeAxis` | `tensor.argmax`, `tensor.top_k`, `tensor.concat`, `tensor.iota`, `tensor.gather`, `tensor.scatter_{kind}` (`Scatter`) | A negative `axis` is written as the axis it counts back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `NonNegativeDotAxes` | `tensor.dot_general` | Negative batch and contract axes are written as the axes they count back to. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `ReduceOfBroadcast` | `tensor.reduce_{kind}` (`Reduce`) | A reduction of `broadcast_to(x)` is `x` reshaped to the result when the combining op is idempotent, the reduction covers every axis the broadcast repeats `x` along, and its other axes have extent 1. Each repeated axis must have a static extent above zero, since a reduction over no elements gives its initial value. | [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `ReduceOfConcat` | `tensor.reduce_{kind}` (`Reduce`) | A reduction of `concat(x1, x2, ...)` that reduces the concatenated axis is the combining op applied to the reductions of the parts, when the elements may combine in any order. | [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir) |
| `SortedReduceAxes` | `tensor.reduce_{kind}` (`Reduce`) | The reduced axes are written non-negative and in increasing order. | [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir) |
| `UnitStrides` | `tensor.slice` | Steps of 1 on every axis are written as no strides. | [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir) |

Filetests: [tensor/canonicalize/algebra.ptir](../../filetests/tensor/canonicalize/algebra.ptir), [tensor/canonicalize/axes.ptir](../../filetests/tensor/canonicalize/axes.ptir), [tensor/canonicalize/broadcast.ptir](../../filetests/tensor/canonicalize/broadcast.ptir), [tensor/canonicalize/commutative.ptir](../../filetests/tensor/canonicalize/commutative.ptir), [tensor/canonicalize/dtypes.ptir](../../filetests/tensor/canonicalize/dtypes.ptir), [tensor/canonicalize/fuel.ptir](../../filetests/tensor/canonicalize/fuel.ptir), [tensor/canonicalize/group_static.ptir](../../filetests/tensor/canonicalize/group_static.ptir), [tensor/canonicalize/integer_and_narrow.ptir](../../filetests/tensor/canonicalize/integer_and_narrow.ptir), [tensor/canonicalize/layout.ptir](../../filetests/tensor/canonicalize/layout.ptir), [tensor/canonicalize/reduce.ptir](../../filetests/tensor/canonicalize/reduce.ptir), [tensor/canonicalize/transpose.ptir](../../filetests/tensor/canonicalize/transpose.ptir), [tensor/pipeline/commutative.ptir](../../filetests/tensor/pipeline/commutative.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

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

Defined by `Dce` in [crates/gpt-rs-ir/src/passes/canonicalize.rs](../../crates/gpt-rs-ir/src/passes/canonicalize.rs).

| Statistic | |
|---|---|
| `erased` | dead ops erased |

Filetests: [tensor/cse/cse.ptir](../../filetests/tensor/cse/cse.ptir), [tensor/dce/dce.ptir](../../filetests/tensor/dce/dce.ptir), [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir), [tensor/outline/budget.ptir](../../filetests/tensor/outline/budget.ptir), [tensor/outline/rerun.ptir](../../filetests/tensor/outline/rerun.ptir), [tensor/outline/static.ptir](../../filetests/tensor/outline/static.ptir).

### `dot-scale-fold`

Moves the static scales of dot results into the weights, so a convolution absorbs the
BatchNorm after it and a run saves the multiply.

The two sides are equal in real arithmetic, not bit for bit: the scale rounds into every
weight before the sum, instead of into the sum once. So it is a pass of its own, not part of
`canonicalize`.

Defined by `DotScaleFold` in [crates/gpt-rs-ir/src/dialects/tensor/passes/dot_scale.rs](../../crates/gpt-rs-ir/src/dialects/tensor/passes/dot_scale.rs).

| Statistic | |
|---|---|
| `moved` | scales moved into a weight |

Patterns:

| Pattern | Root | Rewrite | Fires in |
|---|---|---|---|
| `ScaleIntoDot` | `tensor.{kind}` (`Binary`) | `dot_general(x, w) * broadcast_to(s)` becomes `dot_general(x, w * s')` when `w` and `s` are static and `s` varies only along result axes that come from `w`. `s'` is `s` laid out along those axes of `w` and broadcast to its shape. The scaled weight is static and reads a parameter, so `outline-static` moves it to load time.<br><br>A weight read through a cast is left alone: scaling it would store a widened copy. | [backend/c/pipeline.ptir](../../filetests/backend/c/pipeline.ptir), [tensor/dot_scale/dot_scale.ptir](../../filetests/tensor/dot_scale/dot_scale.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir) |

Filetests: [tensor/dot_scale/cast.ptir](../../filetests/tensor/dot_scale/cast.ptir), [tensor/dot_scale/dot_scale.ptir](../../filetests/tensor/dot_scale/dot_scale.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `fold`

Replaces each op that folds by an existing value or a constant, until nothing changes. The
folds are those of `canonicalize`, without its patterns.

Defined by `Fold` in [crates/gpt-rs-ir/src/passes/canonicalize.rs](../../crates/gpt-rs-ir/src/passes/canonicalize.rs).

| Statistic | |
|---|---|
| `folded` | ops replaced by an existing value or a constant |

It applies every [fold](#folds).

Filetests: [tensor/fold/elementwise.ptir](../../filetests/tensor/fold/elementwise.ptir), [tensor/fold/indexing.ptir](../../filetests/tensor/fold/indexing.ptir), [tensor/fold/layout.ptir](../../filetests/tensor/fold/layout.ptir).

### `merge-contracted-axes`

Puts static dot operands in the layout the dot reads. When the lhs of a dot splits one axis into
its contracted axes, the dot reads that axis whole. The pass transposes and reshapes the static
weight to match.

Defined by `MergeContractedAxes` in [crates/gpt-rs-ir/src/dialects/tensor/passes/contract.rs](../../crates/gpt-rs-ir/src/dialects/tensor/passes/contract.rs).

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

Filetests: [tensor/outline/budget.ptir](../../filetests/tensor/outline/budget.ptir), [tensor/outline/rerun.ptir](../../filetests/tensor/outline/rerun.ptir), [tensor/outline/static.ptir](../../filetests/tensor/outline/static.ptir), [tensor/pipeline/conv_bn.ptir](../../filetests/tensor/pipeline/conv_bn.ptir).

### `prune-params`

Removes from every function the arguments bound to a parameter that nothing reads. A run-time
argument stays even when nothing reads it, so the run-time arguments keep their positions.

Defined by `PruneParams` in [crates/gpt-rs-ir/src/passes/prune_params.rs](../../crates/gpt-rs-ir/src/passes/prune_params.rs).

| Statistic | |
|---|---|
| `pruned` | parameter arguments removed |

Filetests: [tensor/prune_params/prune_params.ptir](../../filetests/tensor/prune_params/prune_params.ptir).
