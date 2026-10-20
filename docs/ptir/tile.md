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
  - A folded result names every iterator that is not folded, except the value's iterators that
    its combine kind leaves out, and the kind appends its own axes (`top_k` appends `k`).
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
| `online_softmax<value_iters>` | `(s, v)` | the state `(m, l, o)`: `m = max_j s_j`, `l = sum_j exp(s_j - m)`, `o = sum_j exp(s_j - m) v_j` | no |
| `online_softmax_merge<value_iters>` | `(m, l, o)` | the states merged into one | no |
| `logsumexp` | `s` | the state `(m, l)` | no |
| `logsumexp_merge` | `(m, l)` | the states merged into one | no |

- Each kind is one computation: a kind that would switch its inputs or results on a parameter is
  two kinds instead, as `top_k` and `bottom_k`, or a kind and its merge.
- The `value_iters` of `online_softmax` are the value's iterators: parallel iterators, by
  position, that only the value depends on; the score MUST NOT depend on them. Its `m` and `l`
  results leave them out.
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

## Schedules

How a dispatch runs changes no value it computes, except by the rounding changes its
`numerics` names. The IR says it where it applies:

- Each block's `loops` are its modes, outermost first, one loop each: an iterator cut into pieces of
  a size, run one after another, on the workers of an execution level, or at once in registers
  (`reg`), the innermost along the lanes of a vector (`lanes`). Every iterator outside the register
  tile ends in a mode of one point, so the modes say every loop. A block without modes walks one
  point at a time, in the order that the op's entry gives. Its `clip` names the iterators whose
  loops skip the points where a padded read falls outside its operand.
- A fold `held` in scratch keeps its state in a scratch that it starts before the loop of a mode, in
  each piece of the modes around that mode, and finishes after that loop. One that holds none keeps
  the register tile's points, or one element, from its first loop over a reduced iterator on.
- A read placed `at` runs its block in the loops of the first block of its run, at a piece of
  them: the register tile (`reg`) or a piece of a mode (`at 2`). It runs on each element of the
  block before it as soon as that element is finished when it reads that element alone, and over
  the piece once the piece is finished when it reads across it.
- A read `held` keeps what it takes in a copy or asks a cache for it ahead (`prefetch`), at a mode
  of the block's loops.
- The dispatch names its `target`, by its fingerprint, and its `numerics`.

The passes that decide a schedule write it into the IR, and the verifier checks the rules it
keeps (`verify/schedule.rs` for a block, `verify/dispatch.rs` for placements and the target).
What follows from it, such as how a loop builds a piece that holds fewer points, is computed by
one function each (`utils/schedule.rs`), which the lowering below `tile` reads: it realizes the
schedule and decides nothing.

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
- `loops` says how the block runs, which changes no value it computes: its modes, outermost
  first, each cutting one iterator into pieces (see [`Mode`]). Every iterator outside the
  register tile ends in a mode of one point, so the modes say every loop. Left out, the block
  walks each iterator one point at a time, in this order:
  1. the iterators that are not reduced and that the coordinates of its first write take, in
     the order they take them;
  2. the other iterators that are not reduced, in the order of `iters`;
  3. the reduced ones, in the order of `iters`.

  When a fold updates an operand where its points may collide, all come in the order of
  `iters`.
- `clip` names the iterators whose loops skip the points where a padded read falls outside its
  operand, when such a point contributes nothing the fold keeps.

Defined by `Block` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | `ins`: tensors |
| Results | `outs`: tensors |
| Regions | `body` |
| Attributes | `iters`: `[Iter]`; `reads`: `[Read]`; `folds`: `[Fold]`; `constraints`: `[Constraint]`, optional; `loops`: `[Mode]`, optional; `clip`: `IterList`, optional |
| Traits | isolated from above |
| Effects | none |
| Patterns | `CanonicalIndices` (`canonicalize`); `CompactInputs` (`canonicalize`); `ComposeCopyReads` (`canonicalize`); `DelinearizeIters` (`canonicalize`); `DropUnitIters` (`canonicalize`); `EraseIdentityBlock` (`canonicalize`); `MergeEqualReads` (`canonicalize`); `OrderItersByWrite` (`canonicalize`); `SplatReads` (`canonicalize`); `WriteInCopyOrder` (`canonicalize`); `WriteIntoUpdate` (`canonicalize`); `GroupChain` (`form-groups`); `ComputeMapInReader` (`inline-static-maps`); `BoundsToPad` (`pad-with-identity`); `LanesLast` (`plan-layouts`); `Seal` (`seal-dispatches`); `WidenInMap` (`widen-reused-maps`); `ConvToWinograd` (`winograd-conv`) |

| `iters` | Text form |
|---|---|
| `Iter` | ` $kind $extent ` |

| `reads` | Text form |
|---|---|
| `Read` | ``` $source `` $coords (`pad` $pad^)? (`at` $at^)? (`held` $held^)? ``` |
| `SourceText` | `` $first (`..` $list^)? `` |
| `OperandList` | `` $last `[` $pick `]` `` |

| `folds` | Text form |
|---|---|
| `Fold` | `` $combine `->` tuple($writes) (`init` tuple($init)^)? (`held` $held^)? `` |
| `Write` | ``` ($into^)? `` $coords ``` |

| `loops` | Text form |
|---|---|
| `Mode` | `` $iter `by` $size (`:` $kind^)? (`in` $memory^)? `` |

Example, from `filetests/tile/roundtrip/loops.ptir`:

```text
tile.dispatch @every_change(%input: tensor<4xf32>) -> tensor<4xf32> attributes {numerics = <contract = true, skip_pad = true, reassoc = true, winograd = true, scale_fold = true, expand_math = true>} {
  %copied = tile.block(%input) {iters = [par 4], reads = [$0[d0]], folds = [none -> [d0]]} ({
  ^body(%element: f32):
    tile.yield(%element) : (f32) -> ()
  }) : (tensor<4xf32>) -> tensor<4xf32>
  func.return(%copied) : (tensor<4xf32>) -> ()
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
- `tile.block: the value's iterator d1 is not a parallel iterator of the block`
- `tile.block: a block has at most one prefix iterator`
- `tile.block: read 0 names operand $1, but the block has 1 operands`
- `tile.block: read 0 names operand $2, but the block has 2 operands`
- `tile.block: the operands of read 0 differ in type: tensor<4xf32> and tensor<8xf32>`
- `tile.block: fold 0 has 1 writes, but argmax gives 2 results`
- `tile.block: write 0 updates $1, but the block has 1 operands`
- `tile.block: write 0 writes f32 values into tensor<4xsi32>`
- `tile.block: init lists 2 operands, but sum starts from 1 states`
- `tile.block: read 0 has no pad, but coordinate 0 is 4 at the point [3], outside tensor<4xf32>`
- `tile.block: yielded value 0, which online_softmax keeps apart from the value's iterators, depends on d1`
- `tile.block: mode 0 walks d2, which the block does not have`
- `tile.block: mode 0 walks pieces of no points`
- `tile.block: mode 0 runs pieces of d1 on workers, which is not a parallel iterator: that splits a fold`
- `tile.block: mode 0 walks d1 in pieces of 3 points, a prefix iterator, which its loop walks one point at a time`
- `tile.block: mode 0 walks d1 in pieces of 3 points, a reduction in order, which its loop walks one point at a time`
- `tile.block: fold 0 starts its state outside the loop over d1, the prefix iterator, whose point each element folds up to`
- `tile.block: the register tile has no lanes mode`
- `tile.block: mode 1 holds d1 in the register tile, which is not a parallel iterator`
- `tile.block: mode 1 is a mode of the register tile, and walks the pieces of a memory`
- `tile.block: read 0 is held at mode 3, of loops of 3 modes`
- `tile.block: the block clips d2, which it does not have`
- `tile.block: mode 2 comes after the lanes, the innermost mode`
- `tile.block: mode 1 walks pieces of d1 inside the register tile, whose modes are the innermost`
- `tile.block: read 0 is held at mode 2, a mode of the register tile, which runs its points at once`
- `tile.block: mode 1 walks pieces of 4 points of d1, inside pieces of 2`
- `tile.block: modes of a block that updates an operand where its points may collide would reorder its writes`
- `tile.block: a block that folds with `online_softmax` or `logsumexp` takes no modes: its fold has no update of a piece`
- `tile.block: the state of fold 0 keeps the piece of d0, whose extent {N} is not static`
- `tile.block: the register tile holds d1 twice`
- `tile.block: a register tile holds at most 3 iterators, not 4`
- `tile.block: the register tile holds 16 points of d1, whose innermost piece has 8`
- `tile.block: the register tile holds 12 points of d1, which vectors of 8 lanes do not fill`
- `tile.block: a register tile of a block with constraints would guard each point`
- `tile.block: a block with a prefix iterator has no register tile`
- `tile.block: a fold in a register tile keeps one element per point`
- `tile.block: the register tile holds d0, which a write leaves out: its points would write one element`
- `tile.block: read 1 holds $0, which another read holds`
- `tile.block: $0 is held in a copy, and the block uses it other than in one read of it alone`
- `tile.block: the copy of $0 has an axis of {N} elements, which is not static`
- `tile.block: the register tile holds d0, which the read of $0 takes other than by a multiple of it`
- `tile.block: the register tile holds d1, which the read of $0 takes other than by a multiple of it`
- `tile.block: the register tile holds d1, which the read of $0 takes other than by a multiple of it`
- `tile.block: the lanes run along d1, which the read of $0 takes other than in whole vectors of consecutive elements`
- `tile.block: the register tile holds d0, by which a read picks its operand of a list`
- `tile.block: $0 is copied where d0, by which its read picks the operand, does not stand at one point`
- `tile.block: $1 is prefetched at a mode over an iterator its read does not take`
- `tile.block: $0 is prefetched, and the block uses it other than in one read of it alone`
- `tile.block: $0 is prefetched, and its read pads or picks it from a list`
- `tile.block: no mode walks d1`
- `tile.block: the modes walk d0 in pieces of 2 points, and none walks it one point at a time`
- `tile.block: fold 0 holds its state at a mode, and the block has no modes`
- `tile.block: fold 0 holds a state, and a fold of `none` keeps none`
- `tile.block: fold 0 holds its state at mode 2, of loops of 2 modes`
- `tile.block: fold 0 starts its state inside a loop over d1, a reduced iterator: it would finish each element more than once`
- `tile.block: fold 0 holds no state, and its loops walk the elements of more than the register tile inside its first loop over a reduced iterator`
- `tile.block: fold 0 holds its state at mode 2, a mode of the register tile, which runs its points at once`
- `tile.block: the state of fold 1 keeps the piece of d1, whose extent {N} is not static`
- `tile.block: the lanes run along d1, which the read of $0 takes other than in whole vectors of consecutive elements`
- `tile.block: the lanes run along d1, which write 0 takes other than by 1 in its last coordinate`

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

Example, from `filetests/tile/roundtrip/loops.ptir`:

```text
tile.dispatch @every_change(%input: tensor<4xf32>) -> tensor<4xf32> attributes {numerics = <contract = true, skip_pad = true, reassoc = true, winograd = true, scale_fold = true, expand_math = true>} {
  %copied = tile.block(%input) {iters = [par 4], reads = [$0[d0]], folds = [none -> [d0]]} ({
  ^body(%element: f32):
    tile.yield(%element) : (f32) -> ()
  }) : (tensor<4xf32>) -> tensor<4xf32>
  func.return(%copied) : (tensor<4xf32>) -> ()
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
| Patterns | `Seal` (`seal-dispatches`) |

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
- `target` names the target its blocks' loops are for, and `numerics` the rounding changes the
  lowerings below `tile` may make: how it runs changes no value it computes, except by those.

Defined by `Dispatch` in [crates/gpt-rs-ir/src/dialects/tile/ir.rs](../../crates/gpt-rs-ir/src/dialects/tile/ir.rs).

| | |
|---|---|
| Operands | none |
| Results | none |
| Regions | `body` |
| Attributes | `name`: `String`; `args`: `[Type]`; `results`: `[Type]`; `grid`: `IterList`, optional; `facts`: `{name = SymFact}`, optional; `target`: `TargetRef`, optional; `numerics`: `Numerics`, optional |
| Traits | isolated from above |
| Effects | none |
| Patterns | `NameDispatchTarget` (`name-target`); `PlaceAtRegisterTile` (`place-epilogues`); `ScheduleFirstBlock` (`schedule`); `SelectVectors` (`select-atoms`); `SplitIntoLanes` (`split-lane-folds`); `DispatchToLoop` (`tile-to-loop`) |

| `facts` | Text form |
|---|---|
| `SymFact` | `` `[` $min `,` $max `]` (`%` $divisor^)? `` |

| `numerics` | Text form |
|---|---|
| `Numerics` | `` `<` struct($contract, $skip_pad, $reassoc, $winograd, $scale_fold, $expand_math) `>` `` |

Example, from `filetests/tile/roundtrip/loops.ptir`:

```text
tile.dispatch @every_change(%input: tensor<4xf32>) -> tensor<4xf32> attributes {numerics = <contract = true, skip_pad = true, reassoc = true, winograd = true, scale_fold = true, expand_math = true>} {
  %copied = tile.block(%input) {iters = [par 4], reads = [$0[d0]], folds = [none -> [d0]]} ({
  ^body(%element: f32):
    tile.yield(%element) : (f32) -> ()
  }) : (tensor<4xf32>) -> tensor<4xf32>
  func.return(%copied) : (tensor<4xf32>) -> ()
}
```

Rejected, as the verify filetests pin it:

- `tile.dispatch: a block uses {T}, which no argument binds`
- `tile.dispatch: facts give {T} the empty range [8, 4]`
- `tile.dispatch: facts give {T} the divisor 0`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: its read 0 takes axis 0 of the result at d1, where the block before it writes d0`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: it folds its points with sum`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: it has 3 iterators, and the block before it 2`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: its d0 is not a parallel iterator of the same extent in both blocks`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: its read 0 takes the result of the block before it from a list or with a pad`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: an index expression of it reads the result of the block before it`
- `tile.dispatch: block 1 runs in the loops of the block before it, whose result has another use`
- `tile.dispatch: block 2 is placed at 0, and the block before it at reg: the blocks of a run are placed at one piece`
- `tile.dispatch: block 0 is placed, and no block comes before it`
- `tile.dispatch: block 1 is placed by read 1, which does not read the result of the block before it`
- `tile.dispatch: block 1 runs in the loops of the block before it, and has modes of its own`
- `tile.dispatch: block 1 is placed by two of its reads, where it runs in one block's loops`
- `tile.dispatch: block 1 is placed at mode 3 of the first block of its run, of 3 modes`
- `tile.dispatch: block 1 is placed at mode 2, a mode of the register tile, where a block is placed at `reg``
- `tile.dispatch: block 1 reads the result of the block before it across the piece where it is placed, whose extent along d0 is not static`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: its read 0 takes axis 2 of the result at 0, a constant along d2, which the piece where it is placed does not hold whole`
- `tile.dispatch: block 1 does not run on the elements that the block before it finishes where it is placed: its read 1 takes axis 2 of the result at d0 mod 16, where the block before it writes d2`
- `tile.dispatch: block 1 runs in the loops of the block before it, whose result has another use`
- `tile.dispatch: op 0 is tensor.add, and a dispatch holds blocks`
- `tile.dispatch: block 0 gives a result of tensor<{N}xf32> that the dispatch does not give, which lives in scratch of its own, whose extents must be static`
- `tile.dispatch: block 1 runs in the loops of the block before it, and holds a read of its own`
- `tile.dispatch: block 1: the lanes run along d1, which the read of $1 takes other than in whole vectors of consecutive elements`
- `tile.dispatch: block 0: mode 0 runs on the workers of cluster, which x86-64-v4 does not have`
- `tile.dispatch: block 0: the read of $0 is held for smem, a memory x86-64-v4 does not have`
- `tile.dispatch: block 0: mode 0 walks the pieces of dram, which is not a cache or scratch of x86-64-v4`
- `tile.dispatch: block 0: the read of $0 is held for l2, for which %S/bare-l2.toml has no buffer space that holds a copy`
- `tile.dispatch: block 0: the read of $0 is held for l2, for which %S/bare-l2.toml has no prefetch that asks it for lines`
- `tile.dispatch: block 0: 16 lanes of f64 take 1024 bits, and x86-64-v4 has registers of [128, 256, 512] bits`

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
