# The x86 target

How x86-64 computes the `isa` dialect: the C form of each `isa` op on one AVX-512 register,
written with the compiler's intrinsics. Each form also gives the extensions of the instruction set
that its intrinsics need, the mask and vector registers it holds besides its operands and its
result, and the operands it can read from memory.

`loop-to-isa` checks every `isa` op it writes against these rules and the features of the
`loop.func`'s target, so an op that x86 cannot compute is an error there, before any C is written.
The x86 backend prints each op in its form.

## Conventions

- A register of computed lanes holds 512 bits: 16 lanes of `f32`, 8 of `f64` or of 64-bit
  integers, 16 of 32-bit integers. A register of `i1` lanes is a mask register of 8, 16, 32 or 64
  flags.
- Lanes of a float narrower than f32 (bf16, f16, FP8) are held as their bits, in a register of
  128, 256 or 512 bits, which x86 moves and converts but computes on only in f32. In their forms,
  `$H` is the intrinsics' prefix (`_mm`, `_mm256`, `_mm512`), `$S` the register's suffix
  (`si128`, ...), and `$E` the lanes' suffix (`epi16` or `epi8`).
- 16 lanes of a 4-bit integer sit two to a byte in the lower 64 bits of a 128-bit register, which
  x86 loads and widens to 32 bits but computes on in no other way.
- A form names the operands' registers at the same position `$0`, `$1`, ..., and is spelled for
  the class of its lanes: `$P` is `ps` or `pd` for floats, `$I` the integer width, `epi32` or
  `epi64`, and `$M` that or `epu32`, `epu64` for unsigned lanes, `$K` the C type of a mask.
- A memory form names the address of a register's first lane `$A`, its mask `$Q`, the stored
  register `$V`, the register variable of a `reg` buffer `$R`, a pad `$D` and the flags of the lanes
  inside the buffer `$F`; `$B` is one element in every lane, read only where a condition holds.
