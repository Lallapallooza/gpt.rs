<!-- Generated from the code by `cargo run -p gpt-rs-docs`. Do not edit: change the code and run it again. -->

# The tile dialect

The `tile` dialect says what each dispatch computes, element by element, and nothing about how a
schedule runs it. `tensor-to-tile` converts the `tensor` ops that compute on elements into it. This
page states the rules that every block follows, and each op's own documentation defines that op.

## Conventions

- Each op is defined in `ir.rs`: its struct, its attributes and its groups. Its type rule
  (`OpDef::result_types`) is in `verify/`. Its semantics, its `Eval` impl, are in `interp/`. Each
  is in the file named after the op's section of `ir.rs`. Helpers are in `utils/`. Every pass is
  checked against the semantics.
- Control flow stays at the `tensor` level: `tensor.cond`, `tensor.while` and `tensor.scan` hold
  regions of `tile` blocks.
- Block bodies compute in the `arith` dialect, one scalar at a time, with the per-element
  functions of the `tensor` ops.
- Normative keywords: MUST / SHOULD / MAY.

## Blocks

A `tile.block` is a generalized reduction over an iteration domain:

```text
%out = tile.block(%x, %w) {iters = [par {M}, par 6, red 8], reads = [$0[d0, d2], $1[d1, d2]], folds = [sum -> [d0, d1]]} ({
^body(%a: f32, %b: bf16):
  %c = arith.cast(%b) {to = f32} : (bf16) -> f32
  %p = arith.mul(%a, %c) : (f32, f32) -> f32
  tile.yield(%p) : (f32) -> ()
}) : (tensor<{M}x8xf32>, tensor<6x8xbf16>) -> tensor<{M}x6xf32>
```

- Attributes name operands by position, `$i`, and iterators by position, `d0` for the first.
- The body sees no memory. Its arguments are the values of the reads, in order, and `tile.index`
  gives it the value of an index expression at the current point.

### Iterators

`kind extent`, where the extent is a constant or a dimension symbol, `{T}`. Iterator `i` is `di`
in index expressions. A block has at most 16.

- `par` iterators index the results.
- `red` iterators are folded by the folds. `red_ordered` iterators are folded in their order: no
  schedule may split or reorder them.
- A `prefix` iterator is folded, and every prefix of the fold is a result: `out[s] = fold(x[0..=s])`.
  A block has at most one, and none together with `red` iterators.

### Index expressions

One language serves reads, writes, `constraints` and `tile.index`:

```text
expr   := term (('+' | '-') term)*
term   := factor (('*' | 'floordiv' | 'mod') factor)*
factor := INT | 'd' INT | '{' SYM '}' | '$' INT '[' expr, ... ']'
        | 'min(' expr ',' expr ')' | 'max(' expr ',' expr ')' | '(' expr ')' | '-' factor
```

- Values are `i64`. `+`, `-` and `*` wrap as si64 arithmetic does, so a position that a body
  computes in si64 is the same index expression. `floordiv` and `mod` round toward minus
  infinity.
- Expressions are affine in the iterators: in a product at most one factor uses iterators, and no
  divisor does.
- `$i[e, ...]` reads an element of the integer operand `i` as an index. Index reads nest, and an
  index read MUST be in range. The expression of `tile.index` reads no operand.
- Integer `arith` on `tile.index` values is index math: the canonical form is one `tile.index`
  of the combined expression, wherever the values stay in the range of the dtype.

### Reads

`$i[e0, ..., e_{r-1}]`, one expression per axis, optionally followed by `pad v`.

- Without `pad`, the coordinates of every point that the block computes MUST be in range.
- With `pad v`, a point whose coordinates fall outside the operand reads `v`, which MUST be a value
  of the operand's dtype.
- `$a..$b[p][e, ...]` reads one of a run of operands of one type, the one that `p` picks at the
  point. Per-expert weights stay separate tensors this way.
- A point where one of the `constraints` fails reads nothing.

### Writes

Each result has one write, in the fold that gives it.

- A new tensor: `[c, ...]`, each coordinate a parallel or prefix iterator or `0`, which adds an
  axis of extent 1.
  - A result without a combine names every iterator once, so the block writes each element once.
  - A folded result names every iterator that is not folded, except the lanes of its combine kind,
    and the kind appends its own axes (`top_k` appends `k`).
- An updated operand: `$k[e, ...]`, any expressions, for `dynamic_update_slice`, scatters and
  segment reductions. The result is a copy of operand `k` with the written elements replaced, or
  with `sum`, `max`, `min`, `and` and `or` folded into them.
- Reads see every operand as it was before the block.
- Where points collide, a combine folds them in point order, and without one the last point wins.

### `constraints`

A list of constraints, all of which must hold at a point for it to count. Each is a comparison
(`<`, `<=`, `==`) of index expressions, or an `or` of comparisons. A point where one fails
contributes nothing. A block with constraints writes only folded or updated results, so that
every element of a result is defined.

### Folds

`folds` lists one fold per group of yielded values, in yield order: `kind -> write`, or
`kind -> (write, ...)` for a kind with several results, then `init $k`, or `init ($k, ...)`, when
the fold starts from given states. A fold takes the values its combine kind folds and has one write
per result of the kind, so every position in the list states what it is. `none -> write` writes
its one yielded value per point.

| Kind | Body yields | Results | Exact under reassociation |
|---|---|---|---|
| `none` | `x` | `x` per point | n/a |
| `sum` | `x` | the sum, from 0 | integers |
| `max`, `min` | `x` | IEEE `maximum`, `minimum`, from `-inf`, `+inf` or the integer bounds | yes |
| `and`, `or` | `x` | from all ones, from 0 | yes |
| `argmax`, `argmin` | `(x, i)` | the extreme value and its index; ties to the smaller index, NaN largest | yes |
| `top_k<k>`, `bottom_k<k>` | `(x, i)` | the `k` largest or smallest values and their indices, along a new last axis; equal values in index order | yes |
| `online_softmax<lanes>` | `(s, v)` | the state `(m, l, o)`: `m = max_j s_j`, `l = sum_j exp(s_j - m)`, `o = sum_j exp(s_j - m) v_j` | no |
| `online_softmax_merge<lanes>` | `(m, l, o)` | the states merged into one | no |
| `logsumexp` | `s` | the state `(m, l)` | no |
| `logsumexp_merge` | `(m, l)` | the states merged into one | no |

- Each kind is one computation: a kind that would switch its inputs or results on a parameter is
  two kinds instead, as `top_k` and `bottom_k`, or a kind and its merge.
- `online_softmax` lanes are parallel iterators, by position, that only the value depends on; the
  score MUST NOT depend on them. Its `m` and `l` results drop the lanes.
- The softmax kinds give their state. The weighted sum is `o / l` and the log-sum-exp is
  `m + log(l)`, which a following block computes.
- A fold's `init` names the operands that hold its starting state, one per state component, of
  the types of its results.

### Reference semantics

The interpreter walks the domain in lexicographic order of the iterators, the first declared
outermost. At each point that meets the `constraints`, it reads, runs the body, and folds each
group of yielded values into the accumulator of the element its write names, each step rounded
to the result's dtype. A folded element that no point reaches holds the combine's identity, or its
`init`. `tensor-to-tile` declares the reduction iterators in the order the `tensor` interpreter
folds, so the conversion is bit-exact.

`online_softmax` and `logsumexp` fold in two passes over the points of each element: the maximum,
then the sums of `exp(s - m)` and `exp(s - m) v`, each step rounded to f32. The merges fold the
states in point order, and a state with maximum `-inf` contributes nothing to a merge.

## Groups and dispatches

`tile.group` gathers blocks while dispatches are formed, and `tile.dispatch` is a sealed group: a
top-level symbol that `host.dispatch` calls. A dispatch runs on its arguments alone, so they bind
every dimension symbol its blocks use.

A dispatch carries its schedule, `#sched<...>`: how it runs, which changes no value it computes.
`gpt_rs_ir::schedule` defines its fields and the rules a schedule follows.

## Ops

- `tile.block`
- `tile.index`
- `tile.yield`
- `tile.group`
- `tile.dispatch`

### `tile.block`

A generalized reduction over the iteration domain `iters`, restricted by
`constraints`.

- Its iterators are named by their position: `d0` is the first of `iters`.
- At each point of the domain, every read gives the body one scalar: an element of an operand
  at index expressions of the iterators. Read `i` is the body's argument `i`. A read with
  `pad v` gives `v` where its coordinates fall outside the operand; a read without it MUST stay
  inside.
- The body computes scalars in `arith` and yields them. `tile.index` gives it the value of an
  index expression at the point.
- `folds` splits the yielded values into groups, in yield order: each fold takes the values its
  combine kind folds over the reduction iterators, and gives the results its writes place, in
  result order. A point where one of the `constraints` fails contributes nothing.
- Each write is `[c, ...]`, the coordinates of a new tensor, each a parallel or prefix iterator
  or `0`; or `$k[e, ...]`, an updated copy of operand `k` at those index expressions. Reads see
  every operand as it was before the block.
- The reference order folds the points in lexicographic order of `iters`, the first one
  outermost, rounding each step to the result dtype.

Defined by `Block` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | `ins`: tensors |
| Results | `outs`: tensors |
| Regions | `body` |
| Attributes | `iters`: `[Iter]`; `reads`: `[Read]`; `folds`: `[Fold]`; `constraints`: `[Constraint]`, optional |
| Traits | isolated from above |
| Effects | none |
| Patterns | `CanonicalIndices` (`canonicalize`); `CompactInputs` (`canonicalize`); `DropUnitIters` (`canonicalize`); `EraseIdentityBlock` (`canonicalize`); `SplatReads` (`canonicalize`) |

| `iters` | Text form |
|---|---|
| `Iter` | ` $kind $extent ` |

| `reads` | Text form |
|---|---|
| `Read` | ``` $source `` $coords (`pad` $pad^)? ``` |
| `SourceText` | `` $first (`..` $list^)? `` |
| `OperandList` | `` $last `[` $pick `]` `` |

| `folds` | Text form |
|---|---|
| `Fold` | `` $combine `->` tuple($writes) (`init` tuple($init)^)? `` |
| `Write` | ``` ($into^)? `` $coords ``` |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @experts(%input: tensor<{T}x8xf32>, %choices: tensor<{T}xsi32>, %first_expert: tensor<4x8xf32>, %second_expert: tensor<4x8xf32>, %third_expert: tensor<4x8xf32>) -> tensor<{T}x4xf32> {
  %output = tile.block(%input, %choices, %first_expert, %second_expert, %third_expert) {iters = [par {T}, par 4, red 8], reads = [$0[d0, d2], $2..$4[$1[d0]][d1, d2]], folds = [sum -> [d0, d1]]} ({
  ^body(%activation: f32, %expert_weight: f32):
    %product = arith.mul(%activation, %expert_weight) : (f32, f32) -> f32
    tile.yield(%product) : (f32) -> ()
  }) : (tensor<{T}x8xf32>, tensor<{T}xsi32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<{T}x4xf32>
  func.return(%output) : (tensor<{T}x4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tile.block: a block with a prefix iterator has no reduction iterators`
- `tile.block: read 0 gives 1 coordinates for tensor<4x4xf32>`
- `tile.block: read 0 uses d1, which is not an iterator of the block`
- `tile.block: read 0 multiplies or divides by an expression of iterators: d0 * d0`
- `tile.block: read 0 reads $1 as an index, but it is tensor<4xf32>`
- `tile.block: read 0 pads with 0.5, which is not a si32 value`
- `tile.block: the folds take 2 yielded values, but the body yields 1`
- `tile.block: a block with reduction or prefix iterators needs a fold other than none`
- `tile.block: write 0 must name each of the iterators [d0, d1] once, got [d0]`
- `tile.block: write 0 writes a new tensor at d0 + 1; its coordinates are iterators or 0`
- `tile.block: write 0 folds into an operand with argmax, but only none, sum, max, min, and and or fold into one`
- `tile.block: result 0 is a plain new tensor, which a block with constraints cannot write`
- `tile.block: lane d1 is not a parallel iterator of the block`

### `tile.index`

The value of `expr` at the current point of the enclosing block, converted to
`dtype` as `tensor.iota` converts its indices. The expression reads no operand: integer `arith`
on index values folds into it.

Defined by `Index` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `output`: scalar |
| Attributes | `expr`: `IndexExpr`; `dtype`: `DType` |
| Effects | none |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @folds(%scores: tensor<2x5xf32>, %flags: tensor<2x5xi1>) -> (tensor<2xf32>, tensor<2xsi32>, tensor<2xf32>, tensor<2xsi64>, tensor<2x3xf32>, tensor<2x3xsi32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xi1>, tensor<2xi1>) {
  %argmax_value, %argmax_index, %argmin_value, %argmin_index, %top_values, %top_indices = tile.block(%scores) {iters = [par 2, red 5], reads = [$0[d0, d1]], folds = [argmax -> ([d0], [d0]), argmin -> ([d0], [d0]), top_k<k = 3> -> ([d0], [d0])]} ({
  ^body(%score: f32):
    %index = tile.index() {expr = d1, dtype = si32} : () -> si32
    %wide_index = tile.index() {expr = d1, dtype = si64} : () -> si64
    tile.yield(%score, %index, %score, %wide_index, %score, %index) : (f32, si32, f32, si64, f32, si32) -> ()
  }) : (tensor<2x5xf32>) -> (tensor<2xf32>, tensor<2xsi32>, tensor<2xf32>, tensor<2xsi64>, tensor<2x3xf32>, tensor<2x3xsi32>)
  %sum, %max, %min = tile.block(%scores) {iters = [par 2, red_ordered 5], reads = [$0[d0, d1]], folds = [sum -> [d0], max -> [d0], min -> [d0]]} ({
  ^body(%score: f32):
    tile.yield(%score, %score, %score) : (f32, f32, f32) -> ()
  }) : (tensor<2x5xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %all, %any = tile.block(%flags) {iters = [par 2, red 5], reads = [$0[d0, d1]], folds = [and -> [d0], or -> [d0]]} ({
  ^body(%flag: i1):
    tile.yield(%flag, %flag) : (i1, i1) -> ()
  }) : (tensor<2x5xi1>) -> (tensor<2xi1>, tensor<2xi1>)
  func.return(%argmax_value, %argmax_index, %argmin_value, %argmin_index, %top_values, %top_indices, %sum, %max, %min, %all, %any) : (tensor<2xf32>, tensor<2xsi32>, tensor<2xf32>, tensor<2xsi64>, tensor<2x3xf32>, tensor<2x3xsi32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xi1>, tensor<2xi1>) -> ()
}
```

### `tile.yield`

Ends the body of a `tile.block` or `tile.group`. Its operands are
the values the region gives.

Defined by `Yield` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | `values`: values |
| Results | none |
| Traits | terminator |
| Effects | none |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
func.func @experts(%input: tensor<{T}x8xf32>, %choices: tensor<{T}xsi32>, %first_expert: tensor<4x8xf32>, %second_expert: tensor<4x8xf32>, %third_expert: tensor<4x8xf32>) -> tensor<{T}x4xf32> {
  %output = tile.block(%input, %choices, %first_expert, %second_expert, %third_expert) {iters = [par {T}, par 4, red 8], reads = [$0[d0, d2], $2..$4[$1[d0]][d1, d2]], folds = [sum -> [d0, d1]]} ({
  ^body(%activation: f32, %expert_weight: f32):
    %product = arith.mul(%activation, %expert_weight) : (f32, f32) -> f32
    tile.yield(%product) : (f32) -> ()
  }) : (tensor<{T}x8xf32>, tensor<{T}xsi32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<{T}x4xf32>
  func.return(%output) : (tensor<{T}x4xf32>) -> ()
}
```

### `tile.group`

Blocks that become one generated function, while dispatches are formed. Its
results are the tensors its body yields.

- The body holds `tile.block`s, which read values of the enclosing function, and ends with
  `tile.yield`.
- `grid` lists the parallel iterators, by position, that every block of the group shares and
  that a schedule may split across workers.

Defined by `Group` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | `outs`: tensors |
| Regions | `body` |
| Attributes | `grid`: `IterList`, optional |
| Effects | none |

Example, from `filetests/tile/roundtrip/block.ptir`:

```text
tile.dispatch @rms(%input: tensor<{T}x8xf32>) -> tensor<{T}xf32> attributes {grid = [d0], facts = {T = [16, 4096] % 16}} {
  %sum_of_squares = tile.block(%input) {iters = [par {T}, red 8], reads = [$0[d0, d1]], folds = [sum -> [d0]]} ({
  ^body(%element: f32):
    %square = arith.mul(%element, %element) : (f32, f32) -> f32
    tile.yield(%square) : (f32) -> ()
  }) : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  func.return(%sum_of_squares) : (tensor<{T}xf32>) -> ()
}
func.func @call(%input: tensor<{T}x8xf32>) -> (tensor<{T}xf32>, tensor<{T}x8xf32>) {
  %sum_of_squares = host.dispatch(%input) {callee = @rms} : (tensor<{T}x8xf32>) -> tensor<{T}xf32>
  %exponentials = tile.group() {grid = [d0]} ({
    %block_exponentials = tile.block(%input) {iters = [par {T}, par 8], reads = [$0[d0, d1]], folds = [none -> [d0, d1]]} ({
    ^body(%element: f32):
      %exponential = arith.exp(%element) : (f32) -> f32
      tile.yield(%exponential) : (f32) -> ()
    }) : (tensor<{T}x8xf32>) -> tensor<{T}x8xf32>
    tile.yield(%block_exponentials) : (tensor<{T}x8xf32>) -> ()
  }) : () -> tensor<{T}x8xf32>
  func.return(%sum_of_squares, %exponentials) : (tensor<{T}xf32>, tensor<{T}x8xf32>) -> ()
}
```

### `tile.dispatch`

One fused group of blocks, which becomes one generated function. A top-level
symbol, called by `host.dispatch`.

- Its body's block arguments are its arguments, and its `func.return` gives its results.
- `grid` lists the parallel iterators, by position, that its blocks share. `facts` gives a
  range and a divisor of each dimension symbol of its arguments.
- `schedule` says how it runs, which changes no value it computes.

Defined by `Dispatch` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Regions | `body` |
| Attributes | `name`: `String`; `args`: `[Type]`; `results`: `[Type]`; `grid`: `IterList`, optional; `facts`: `{name = SymFact}`, optional; `schedule`: `Schedule`, optional |
| Traits | isolated from above |
| Effects | none |
| Patterns | `DispatchToLoop` (`tile-to-loop`) |

| `facts` | Text form |
|---|---|
| `SymFact` | `` `[` $min `,` $max `]` (`%` $divisor^)? `` |

| `schedule` | Text form |
|---|---|
| `Schedule` | `` `#sched` `<` struct($nest) `>` `` |
| `Loop` | `` $iter `by` $extent `:` $parallelism `` |

Example, from `filetests/tile/roundtrip/schedule.ptir`:

```text
tile.dispatch @project(%input: tensor<4x8xf32>, %weight: tensor<6x8xf32>) -> tensor<4x6xf32> attributes {schedule = #sched<nest = [d0 by 2 : par<level = l3, units = 2>, d1 by 3 : par<level = core>, d2 by 4 : seq]>} {
  %product = tile.block(%input, %weight) {iters = [par 4, par 6, red 8], reads = [$0[d0, d2], $1[d1, d2]], folds = [sum -> [d0, d1]]} ({
  ^body(%element: f32, %scale: f32):
    %term = arith.mul(%element, %scale) : (f32, f32) -> f32
    tile.yield(%term) : (f32) -> ()
  }) : (tensor<4x8xf32>, tensor<6x8xf32>) -> tensor<4x6xf32>
  func.return(%product) : (tensor<4x6xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tile.dispatch: a block uses {T}, which no argument binds`
- `tile.dispatch: facts give {T} the empty range [8, 4]`
- `tile.dispatch: facts give {T} the divisor 0`
- `tile.dispatch: a nest schedules the one block of a dispatch, and this one holds 2`
- `tile.dispatch: loop 0 walks d2, which the block does not have`
- `tile.dispatch: loop 0 walks empty tiles`
- `tile.dispatch: loop 1 walks tiles of 4 points of d1, inside tiles of 2`
- `tile.dispatch: loop 0 runs on no units`
- `tile.dispatch: loop 1 runs at level l3, inside a loop at level core`
- `tile.dispatch: loop 0 runs tiles of d1, a reduction iterator, in parallel: that splits the reduction`
- `tile.dispatch: loop 0 walks d1, a prefix iterator, which a nest leaves whole`
- `tile.dispatch: a nest of a block that updates an operand would reorder its writes`
- `tile.dispatch: a nest of `online_softmax` or `logsumexp` needs their update of a tile`
- `tile.dispatch: the state of a fold keeps the tile of d0, whose extent {N} is not static`

## Traits

A trait states a fact about the ops that have it. The folds and patterns it implies apply to every one of them; [passes.md](passes.md) says which pass runs each.

### `IsolatedFromAbove`

Its regions read no value defined outside them.

Defined by `IsolatedFromAbove` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `tile.block`, `tile.dispatch`.

### `Terminator`

Ends a region: its operands are what the region yields, and it has no results.

Defined by `Terminator` in [crates/gpt-rs-ir/src/traits/structure.rs](../../crates/gpt-rs-ir/src/traits/structure.rs).

Ops: `tile.yield`.
