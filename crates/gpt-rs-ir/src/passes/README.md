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

```rust,no_run
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
# fn main() {}
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

```rust,no_run
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
# fn main() {}
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

```rust,ignore
/// `transpose(transpose(x, p1), p2)` is `transpose(x, q)` with `q[i] = p1[p2[i]]`.
#[pattern(canonical)]
struct CollapseTransposeChain;
```

An op's own folder is `#[fold]` on `impl Fold for <op>`, next to the op, with a doc comment that
says what the op folds to. The traits of an op fold it too, by the laws they state.

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
