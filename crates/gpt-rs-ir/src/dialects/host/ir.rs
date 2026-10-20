//! The ops of the `host` dialect: `host.dispatch` in the value form, and the ops of the buffer
//! form.

use crate::ir::{DType, Ranked, SymbolRef};
use crate::traits::OnlyUse;

// ------------------------------------------------------------------------------------------------
// The value form: `host.dispatch`.

/// `host.dispatch`: calls the dispatch named `callee` on `args` and gives its results. The callee
/// is a `tile.dispatch`, or the `loop.func` it lowers to.
///
/// - The arguments have the callee's argument types, with each of its dimension symbols bound to
///   one extent across all of them. The results have its result types under that binding.
/// - A `loop.func` takes each argument in an `in` or `inout` parameter, in order, and gives each
///   `inout` or `out` parameter as a result, in order. An `inout` parameter starts as a copy of its
///   argument.
/// - It is pure: a call computes its results from its arguments only.
#[crate::op(
    "host.dispatch",
    operands(args(Variadic(Tensor))),
    results(outs(Variadic(Tensor)))
)]
pub struct Dispatch {
    pub callee: SymbolRef,
}

// ------------------------------------------------------------------------------------------------
// The buffer form: `host.arena`, `host.slice`, `host.alloc`, `host.to_buffer`, `host.take`,
// `host.launch`, `host.copy` and `host.to_tensor`.

/// `host.arena`: `bytes` bytes of memory that buffers of the function's values live in, which a
/// runtime keeps from one call to the next. They start at a multiple of `alignment` bytes, a power
/// of two, and their buffer is of the space `arena` with that alignment, which no other op gives.
/// No byte holds a value when a call starts.
#[crate::op("host.arena", results(output(Buffer)), effects(alloc))]
pub struct Arena {
    pub bytes: u64,
    pub alignment: u64,
}

/// `host.slice`: the buffer of `dtype` with the extents `shape` at byte `offset` of `arena`, whose
/// bytes it fits in. The offset is a multiple of the element size and of the arena's alignment.
/// Two slices are two buffers, even of the same bytes. Slices whose bytes overlap hold values at
/// different times: a buffer holds its elements from the launch or the copy that writes it, until
/// another buffer writes one of its bytes.
#[crate::op(
    "host.slice",
    operands(arena(Buffer)),
    results(output(Buffer)),
    effects(alloc)
)]
pub struct Slice {
    pub offset: u64,
    pub shape: Ranked<usize>,
    pub dtype: DType,
}

/// `host.alloc`: a buffer of `dtype` with the extents `shape`, in memory of its own that starts at
/// a multiple of `alignment` bytes, a power of two, for a value that the function returns or that
/// an op other than a launch reads. No element holds a value until a launch or a copy writes it.
#[crate::op("host.alloc", results(output(Buffer)), effects(alloc))]
pub struct Alloc {
    pub shape: Ranked<usize>,
    pub dtype: DType,
    pub alignment: u64,
}

/// `host.to_buffer`: a buffer that holds the elements of `tensor`, which no launch and no copy
/// writes.
#[crate::op("host.to_buffer", operands(tensor(Tensor)), results(output(Buffer)))]
pub struct ToBuffer;

/// `host.take`: a buffer that holds the elements of `tensor`, which launches and copies may write.
/// It is the tensor's only use, so a runtime may give the tensor's own memory when nothing else
/// holds the tensor, and gives a copy otherwise.
#[crate::op(
    "host.take",
    operands(tensor(Tensor)),
    results(output(Buffer)),
    effects(alloc),
    traits(OnlyUse)
)]
pub struct Take;

/// `host.launch`: one launch of the `loop.func` named `callee`, with `ins` for its `in`
/// parameters, `outs` for its `out` parameters and `inouts` for its `inout` parameters, each in
/// order. It starts after every launch and copy before it in the function has ended.
///
/// - Each buffer has the type of its parameter, with each of the callee's dimension symbols bound
///   to one extent across all of them.
/// - A buffer that the launch writes goes to no other parameter, except that an `out<over = k>`
///   parameter may take the buffer of the `in` parameter at position `k`. No launch writes a
///   buffer of `host.to_buffer`.
#[crate::op(
    "host.launch",
    operands(
        ins(Variadic(Buffer)),
        outs(Variadic(Buffer)),
        inouts(Variadic(Buffer))
    ),
    effects(read(ins), write(outs), read(inouts), write(inouts))
)]
pub struct Launch {
    pub callee: SymbolRef,
}

/// `host.copy`: writes the elements of `from` into `to`, a buffer of the same type. It starts after
/// every launch and copy before it in the function has ended.
#[crate::op(
    "host.copy",
    operands(from(Buffer), to(Buffer)),
    effects(read(from), write(to))
)]
pub struct Copy;

/// `host.to_tensor`: the tensor of the elements that `buffer` holds when it runs.
#[crate::op(
    "host.to_tensor",
    operands(buffer(Buffer)),
    results(output(Tensor)),
    effects(read(buffer))
)]
pub struct ToTensor;

#[derive(crate::Dialect, Debug, Clone, PartialEq, Eq, Hash)]
#[dialect(name = "host", op = Host)]
pub enum HostOps {
    Dispatch(Dispatch),
    Arena(Arena),
    Slice(Slice),
    Alloc(Alloc),
    ToBuffer(ToBuffer),
    Take(Take),
    Launch(Launch),
    Copy(Copy),
    ToTensor(ToTensor),
}
