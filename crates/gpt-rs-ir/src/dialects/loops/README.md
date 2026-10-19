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
- `read_only`: memory the caller owns and nothing writes. An `in` parameter may be bound to it.
- `private`: scratch of one unit of work. `loop.alloc` allocates it, with static extents, and it
  lives as long as the region that allocates it: an allocation in the body of a loop is a new
  buffer in each of its iterations.
- `reg`: scratch as `private` is, which `loop-to-isa` keeps in vector registers: the accumulators
  of a register tile. `loop-to-isa` accepts a `reg` buffer only when the vector code reaches the
  buffer in whole registers, at positions fixed before the run, with no pad. That is, each load
  and store of the buffer:
  - has the buffer's dtype;
  - uses registers of one lane count, which divides the last extent;
  - stays inside the buffer;
  - starts at a multiple of that lane count.

A load reads an element that holds a value: the caller's, in an `in` or `inout` parameter, or one a
store wrote. Reading an element that holds none, or a position outside the buffer, is an error,
unless the load gives a `pad` value for positions outside the buffer.

## Launches

A `loop.func` takes buffers, each with an access: an `in` parameter it only reads, an `out`
parameter whose every element it writes, and an `inout` parameter it updates in place, which holds
the argument's elements when the launch starts and the result's when it ends. An `out<over = k>`
parameter may share the buffer of the `in` parameter `k`. No other parameter may share that buffer,
and the launch MUST read its elements only at positions that it has not written yet. `host.dispatch`
calls it with tensors: each `in` or `inout` parameter holds one argument, in order, and each
`inout` or `out` parameter gives one result, in order. `host.launch` calls it with buffers: one for
each parameter, its `in` parameters first, then its `out` and its `inout` parameters, each in order.

## Register tiles

`loop.atom` runs its body once for each point of a static box, the register tile of a dispatch,
below its extents. `loop-to-isa` computes the points at once, in vectors along the tile's last
axis, for the target that the `loop.func` names. Each point is a unit of work: two points MUST NOT
access one element when one of them writes it, so every order of the points gives the same values.

## Units of work

`loop.forall` runs its body as units of work at an execution level, which may run at the same time
and in any order. Two units of one forall MUST NOT access one element of a buffer when one of them
writes it, so every order gives the same values. Accesses before a forall happen before each of its
units, and those after it happen after all of them.

## Reference semantics

The interpreter runs the ops of a region in order. `loop.for` runs its body once for each index
from its start up to its end, in increasing order, and `loop.if` runs its body when its condition
holds. `loop.forall` runs its units in order of their index, and `loop.atom` runs its points in
row-major order. Each unit runs to its end. The interpreter fails when two units race on one
element: each element remembers the unit of its last write and the units that read it since.
