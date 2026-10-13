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
`Schedule` in `ir.rs` declares its fields, and `verify/dispatch.rs` checks the rules a schedule
follows.
