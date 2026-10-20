//! The type rules of the `host` ops: the types each op gives, and what it rejects.

use anyhow::{anyhow, bail, ensure, Result};

use crate::dialects::host::ir::*;
use crate::dialects::loops::{self, Access};
use crate::dialects::tile;
use crate::ir::{
    BufferType, Dim, NoRegions, OpDef, OwnedTypes, Space, SymBindings, TensorType, Type, TypeCx,
    Types,
};

/// The argument and result types of the dispatch named `name`: a `tile.dispatch`, or a
/// `loop.func` as a call with tensors sees it.
fn callee_types<R>(cx: &TypeCx<R>, name: &str) -> Result<(Vec<TensorType>, Vec<TensorType>)> {
    let op = cx
        .symbol(name)
        .ok_or_else(|| anyhow!("no symbol @{name}"))?;
    if let Some(func) = op.get::<loops::Func>() {
        return func.call_types();
    }
    let dispatch = op
        .get::<tile::Dispatch>()
        .ok_or_else(|| anyhow!("@{name} is not a tile.dispatch or a loop.func"))?;
    let tensors = |types: &[Type], what: &str| -> Result<Vec<TensorType>> {
        types
            .iter()
            .map(|ty| {
                ty.as_tensor()
                    .cloned()
                    .ok_or_else(|| anyhow!("@{name} {what} {ty}, not a tensor"))
            })
            .collect()
    };
    Ok((
        tensors(&dispatch.args, "takes")?,
        tensors(&dispatch.results, "gives")?,
    ))
}

impl OpDef for Dispatch {
    fn result_types(
        &self,
        ins: DispatchOperands<Types>,
        cx: &TypeCx<NoRegions>,
    ) -> Result<DispatchResults<OwnedTypes>> {
        let name = &self.callee.0;
        let (params, results) = callee_types(cx, name)?;
        ensure!(
            params.len() == ins.args.len(),
            "@{name} takes {} arguments, got {}",
            params.len(),
            ins.args.len()
        );
        let mut bound = SymBindings::default();
        for (index, (param, &arg)) in params.iter().zip(&ins.args).enumerate() {
            bound
                .bind(param, arg)
                .map_err(|err| anyhow!("argument {index}: {err}"))?;
        }
        let outs = results
            .iter()
            .map(|tensor| {
                let unbound = tensor.dims.iter().find(|dim| match dim {
                    Dim::Dynamic(sym) => bound.get(sym).is_none(),
                    Dim::Static(_) => false,
                });
                if let Some(dim) = unbound {
                    bail!("no argument binds {dim}");
                }
                Ok(bound.resolve(tensor))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DispatchResults { outs })
    }
}

impl OpDef for Arena {
    fn result_types(
        &self,
        _ins: ArenaOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ArenaResults<OwnedTypes>> {
        ensure_alignment(self.alignment)?;
        Ok(ArenaResults {
            output: self.buffer_type()?,
        })
    }
}

impl OpDef for Slice {
    fn result_types(
        &self,
        ins: SliceOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<SliceResults<OwnedTypes>> {
        let arena = ins.arena;
        let (arena_bytes, alignment) = match (&arena.dims[..], arena.space) {
            ([Dim::Static(bytes)], Space::Arena { alignment }) => (*bytes as u64, alignment),
            _ => bail!("{arena} is not an arena, the buffer of host.arena"),
        };
        let output = self.buffer_type();
        let bytes = output
            .tensor_type()
            .element_count()
            .and_then(|count| output.dtype.bytes_for(count))
            .ok_or_else(|| anyhow!("{output} takes more bytes than fit"))?;
        ensure!(
            self.offset
                .checked_add(bytes as u64)
                .is_some_and(|end| end <= arena_bytes),
            "{output} at byte {} does not fit in the {arena_bytes} bytes of the arena",
            self.offset
        );
        if let Some(size) = self.dtype.size_in_bytes() {
            ensure!(
                self.offset.is_multiple_of(size as u64),
                "byte {} is not a multiple of the {size} bytes of a {} element",
                self.offset,
                self.dtype
            );
        }
        ensure!(
            self.offset.is_multiple_of(alignment),
            "byte {} is not a multiple of the {alignment} bytes that the arena aligns to",
            self.offset
        );
        Ok(SliceResults { output })
    }
}

impl OpDef for Alloc {
    fn result_types(
        &self,
        _ins: AllocOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<AllocResults<OwnedTypes>> {
        ensure_alignment(self.alignment)?;
        Ok(AllocResults {
            output: self.buffer_type(),
        })
    }
}

/// Fails unless `alignment`, the bytes that memory starts at a multiple of, is a power of two.
fn ensure_alignment(alignment: u64) -> Result<()> {
    ensure!(
        alignment.is_power_of_two(),
        "an alignment of {alignment} bytes is not a power of two"
    );
    Ok(())
}

impl OpDef for ToBuffer {
    fn result_types(
        &self,
        ins: ToBufferOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ToBufferResults<OwnedTypes>> {
        Ok(ToBufferResults {
            output: BufferType::of(ins.tensor, Space::ReadOnly),
        })
    }
}

impl OpDef for Take {
    fn result_types(
        &self,
        ins: TakeOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<TakeResults<OwnedTypes>> {
        Ok(TakeResults {
            output: BufferType::of(ins.tensor, Space::Global),
        })
    }
}

impl OpDef for Launch {
    fn result_types(
        &self,
        ins: LaunchOperands<Types>,
        cx: &TypeCx<NoRegions>,
    ) -> Result<LaunchResults<OwnedTypes>> {
        let name = &self.callee.0;
        let func = cx
            .symbol(name)
            .and_then(|op| op.get::<loops::Func>())
            .ok_or_else(|| anyhow!("@{name} is not a loop.func"))?;
        let buffers = ins.params(func)?;
        let mut bound = SymBindings::default();
        let params = func.args.iter().zip(&func.access).zip(buffers);
        for (index, ((param_type, &access), buffer)) in params.enumerate() {
            let param = param_type
                .as_buffer()
                .ok_or_else(|| anyhow!("@{name} takes {param_type}, not a buffer"))?;
            // Only an `in` parameter reads a read-only buffer.
            let readable = access == Access::In && buffer.space == Space::ReadOnly;
            ensure!(
                buffer.space == Space::Global || readable,
                "parameter {index} gets {buffer}, not a global buffer"
            );
            bound
                .bind(&param.tensor_type(), &buffer.tensor_type())
                .map_err(|err| anyhow!("parameter {index}: {err}"))?;
        }
        Ok(LaunchResults::default())
    }
}

impl OpDef for Copy {
    fn result_types(
        &self,
        ins: CopyOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<CopyResults<OwnedTypes>> {
        ensure!(
            ins.from.tensor_type() == ins.to.tensor_type(),
            "it copies {} into {}, a buffer of another type",
            ins.from,
            ins.to
        );
        ensure!(
            ins.to.space == Space::Global,
            "it copies into {}, which is not a global buffer",
            ins.to
        );
        Ok(CopyResults::default())
    }
}

impl OpDef for ToTensor {
    fn result_types(
        &self,
        ins: ToTensorOperands<Types>,
        _cx: &TypeCx<NoRegions>,
    ) -> Result<ToTensorResults<OwnedTypes>> {
        Ok(ToTensorResults {
            output: ins.buffer.tensor_type(),
        })
    }
}
