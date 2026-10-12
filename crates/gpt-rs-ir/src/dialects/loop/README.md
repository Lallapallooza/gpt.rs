# The loop dialect

The `loop` dialect is one launch of a dispatch: loops over index values, buffers in memory spaces,
and loads and stores of their elements. `tile-to-loop` lowers each `tile.dispatch` to a
`loop.func`. In this dialect, values do not stand for tensors: a dispatch reads its arguments from
buffers and writes its results into buffers, one element at a time. Bodies compute in `arith`.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- Index values have type `index`, an `i64`. `loop.index` and `loop.predicate` compute with the index
  language of `tile`, in which `di` is the op's operand `i`. Their expressions read no operand
  element: a load gives such an element, and `arith.to_index` makes it an index.
- Normative keywords: MUST / SHOULD / MAY.

## Buffers

`!loop.buf<4x{T}xf32, global>` holds elements of one dtype at the coordinates of its extents,
row-major, in one memory space:

- `global`: the parameters of a `loop.func`, which its caller owns. Their extents MAY be dimension
  symbols, which the caller's arguments bind.
- `private`: scratch of one unit of work. `loop.alloc` allocates it, with static extents, and it
  lives as long as the region that allocates it: an allocation in the body of a loop is a new
  buffer in each of its iterations.

A load reads an element that a store wrote before it. Reading an element that no store wrote, or a
position outside the buffer, is an error, unless the load gives a `pad` value for positions outside
the buffer.

## Launches

A `loop.func` takes buffers, each with an access: an `in` parameter it only reads, and an `out`
parameter whose every element it writes. `host.dispatch` calls it with tensors: each `in`
parameter holds one argument, in order, and each `out` parameter gives one result, in order.

## Units of work

`loop.forall` runs its body as units of work at an execution level, which may run at the same time
and in any order. Two units of one forall MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values. Accesses before a forall happen before each of its
units, and those after it happen after all of them.

## Reference semantics

The interpreter runs the ops of a region in order. `loop.for` runs its body once for each index
from its start up to its end, in increasing order, and `loop.if` runs its body when its condition
holds. `loop.forall` runs its units in order of their index, each to its end, and fails on two
accesses of its units that race: each element remembers the unit of its last write and the units
that read it since.
