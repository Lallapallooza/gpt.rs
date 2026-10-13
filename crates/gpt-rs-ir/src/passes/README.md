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

```rust,no_run
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
# fn main() {}
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

```rust,ignore
/// `t(t(x))` is one `t` of `x` for a movement op `t`.
#[pattern(canonical(Reshape, BroadcastTo, Slice, Transpose))]
struct CollapseChain<T>(PhantomData<T>);
```

An op's own folder is `#[fold]` on `impl Fold for <op>`, in its dialect's `fold/`, with a doc
comment that says what the op folds to. The traits of an op fold it too, by the laws they state.

```rust,ignore
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

```rust,ignore
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
