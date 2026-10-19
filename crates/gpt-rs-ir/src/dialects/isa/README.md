# The isa dialect

The `isa` dialect is vector code: the register tile of a dispatch, computed in vector registers
inside the loops of the `loop` dialect. `loop-to-isa` writes it from each `loop.atom`, with fixed
rules and no choice of its own. Its ops apply one operation to every lane of a group of registers,
and move lanes between registers and `loop` buffers.

## Conventions

- Each op is defined in `ir.rs`. Its type rule is in `verify.rs`, and its semantics, its `Eval`
  impl, are in `interp.rs`.
- A value of type `!isa.vec<6x4x16xf32>` is a group of 6x4 registers that each hold 16 lanes of
  `f32`. The last extent is the lanes of one register, and the extents before it, at most
  `MAX_GROUP_RANK` (3), count registers. Lane `l` of register `[i, j]` is element `[i, j, l]`. A
  vector of `i1` is a mask: one flag per lane.
- Lane-wise ops take operands of equal extents. All are of one type, except the mask of
  `isa.select`. Each applies the `arith` op of the same name to each lane, as the `tensor` op of
  that name applies it to each element: `isa.add` is `arith.add` per lane. `isa.expand` repeats a
  register along an axis of extent 1.
- Index values have type `index`, as in `loop`.
- Normative keywords: MUST / SHOULD / MAY.

## Memory

`isa.load`, `isa.broadcast` and `isa.store` move lanes between registers and a `loop` buffer. Each
names one base position in the buffer, and `steps`. `steps` has one entry per axis of the group.
Each entry gives, for each axis of the buffer, how far the position moves from one register to the
next along that group axis. Register `g` starts at `base + g[0] * steps[0] + ...`. `isa.load` and
`isa.store` take its lanes from there along the last axis of the buffer, and `isa.broadcast`
reads one element into every lane.

`isa.masked_load`, `isa.masked_broadcast` and `isa.masked_store` take a mask of the extents of
their registers first. A lane whose flag is clear is not accessed: a load gives it 0, and a store
leaves its element as it was. Every lane that is accessed MUST be inside the buffer, and a load
MUST read elements that hold a value: the caller's, in an `in` or `inout` parameter, or ones that
a store wrote.

A read with `pad v` gives `v` for a lane outside the buffer. Its `checked` attribute names, per
register, the axes on which the register's lanes may fall outside, so the code checks those
alone. An empty `checked` allows every axis. A lane outside on any other axis makes the read
wrong, and the interpreter rejects it. `loop-to-isa` writes `checked` from the ranges that the
loops around a read give its position. `loop-to-isa` gives no pad to a read whose lanes all stay
inside.

## Targets

A register holds its lanes times the bits of its dtype. These bits fill one of the target's
vector widths, or the low bits of one. A mask holds one flag per lane. The rules of a target's
architecture, `isa::Rules`, say for each op whether the target has code for it on the registers it
computes. They also say what that code needs: the extensions of the instruction set, the mask and
vector registers it holds besides its operands and its result, and the operands it can read from
memory.

## Reference semantics

The interpreter holds the lanes of a vector as a tensor of its extents, and computes each lane as
the `tensor` op of the same name computes an element. Memory ops read and write the buffers of
the `loop` interpreter, which checks every access.
