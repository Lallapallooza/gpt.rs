# The arith dialect

The `arith` dialect computes on scalars: the bodies of `tile.block` are written in it. Each op is
the per-element function of the `tensor` op of the same name, applied to one value, so the two
dialects agree bit for bit.

## Conventions

- Each op is defined in `ir.rs`, and its type rule is in `verify.rs`. The ops of a family
  (`Unary`, `Binary`, `Compare`, `Rng`) share the family's type rule and `Eval`.
- Each op's function is the one copy of its per-element semantics, which the `tensor` op of the
  same name applies to every element. `interp.rs` implements `UnaryMath`, `BinaryMath` and
  `CompareMath` of `crate::scalar`, and `tensor/interp/random.rs` implements `Draw`.
- Operands of one op have one dtype, as in `tensor`: there is no implicit promotion.
- bf16 and f16 compute in f32 and round once per op to their dtype, as the `tensor` ops do.
- `arith.constant` holds its value as a rank-0 constant, so constant queries, folds and trait
  rewrites see it as they see `tensor.constant`.
- Each op lists its traits, as the `tensor` op of the same name does.
