use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};

use super::block::{copy, BlockLowering};
use super::utils::alloc;
use crate::dialects::r#loop::{self, Access, Return};
use crate::dialects::tile::{Block, Dispatch};
use crate::module::{Module, Value};
use crate::op::Matched;
use crate::rewrite::{pattern, Builder, NoMatch, Pattern, Rewriter};
use crate::schedule::Schedule;
use crate::traits::Terminator;
use crate::types::{BufferType, Dim, Space, Type};

/// `tile.dispatch` becomes the `loop.func` of its name: an `in` buffer for each argument, an `out`
/// buffer for each result, and its blocks lowered in order.
#[pattern]
pub(super) struct DispatchToLoop;

impl Pattern for DispatchToLoop {
    type Root = Dispatch;

    fn rewrite(
        &self,
        dispatch: Matched<'_, Dispatch>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        let module = rewriter.module();
        let attrs = dispatch.attrs();
        let mut args = Vec::new();
        let mut access = Vec::new();
        let params = [(&attrs.args, Access::In), (&attrs.results, Access::Out)];
        for (types, kind) in params {
            for ty in types {
                let Type::Tensor(tensor) = ty else {
                    return Err(NoMatch::because(format_args!(
                        "a dispatch takes tensors, not {ty}"
                    )));
                };
                args.push(Type::Buffer(BufferType::of(tensor, Space::Global)));
                access.push(kind);
            }
        }
        let func = r#loop::Func {
            name: attrs.name.clone(),
            args,
            access,
        };
        let body = dispatch.body().id();
        let schedule = &attrs.schedule;
        let mut outcome = Ok(());
        rewriter.create_with_regions(func, (), |_, builder, params| {
            outcome = lower_body(module, body, schedule, builder, params);
            builder.create(Return, ());
        });
        outcome.map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        rewriter.erase(&dispatch);
        Ok(())
    }
}

/// The extents of the tensor `value`.
fn dims(module: &Module, value: Value) -> Result<Vec<Dim>> {
    let ty = module.ty(value);
    let tensor = ty
        .as_tensor()
        .ok_or_else(|| anyhow!("a dispatch computes tensors, not {ty}"))?;
    Ok(tensor.dims.to_vec())
}

/// Lowers the blocks of the dispatch body `body` with `schedule` into the `loop.func` whose
/// parameters are `params`: its arguments, then its results.
fn lower_body(
    module: &Module,
    body: crate::module::Region,
    schedule: &Schedule,
    builder: &mut Builder<'_>,
    params: &[Value],
) -> Result<()> {
    let args = module.args(body);
    let returned = module.yielded(body);
    let (inputs, outputs) = params.split_at(args.len());
    let mut buffers: HashMap<Value, Value> =
        args.iter().copied().zip(inputs.iter().copied()).collect();
    // A block writes a result where the dispatch returns it first.
    let mut returns: HashMap<Value, Value> = HashMap::new();
    for (&value, &output) in returned.iter().zip(outputs) {
        if module.def_inst(value).is_some() {
            returns.entry(value).or_insert(output);
        }
    }
    for inst in module.insts(body) {
        let op = module.op(inst);
        if op.has_trait::<Terminator>() {
            break;
        }
        let Some(block) = op.get::<Block>() else {
            bail!("{} has no loop form: a dispatch holds blocks", op.name());
        };
        let mut results = Vec::new();
        for &result in module.results(inst) {
            let buffer = match returns.get(&result) {
                Some(&output) => output,
                None => {
                    let ty = module.ty(result);
                    let tensor = ty
                        .as_tensor()
                        .ok_or_else(|| anyhow!("a block gives tensors, not {ty}"))?;
                    let shape = tensor.static_dims().ok_or_else(|| {
                        anyhow!(
                            "{tensor} crosses blocks in private scratch, whose extents are static"
                        )
                    })?;
                    alloc(builder, &shape, tensor.dtype)
                }
            };
            buffers.insert(result, buffer);
            results.push(buffer);
        }
        let mut operands = Vec::new();
        for &operand in module.operands(inst) {
            operands.push((buffers[&operand], dims(module, operand)?));
        }
        let lowering = BlockLowering {
            module,
            block,
            body: module.regions(inst)[0],
            operands,
            results,
            schedule,
        };
        lowering.lower(builder)?;
    }
    // A result that is an argument, or a value returned twice, is a copy.
    for (&value, &output) in returned.iter().zip(outputs) {
        let buffer = buffers[&value];
        if buffer != output {
            copy(builder, buffer, output, &dims(module, value)?)?;
        }
    }
    Ok(())
}
