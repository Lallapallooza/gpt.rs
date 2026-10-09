//! Moving `tensor.broadcast_to` after the elementwise ops that read it.

use anyhow::Result;
use smallvec::SmallVec;

use crate::dense::Dense;
use crate::dialects::tensor::BroadcastTo;
use crate::module::{Module, Value};
use crate::rewrite::{pattern, Has, Pattern, Rewriter, WithTrait};
use crate::traits::Elementwise;
use crate::types::Dim;

/// `f(broadcast_to(a), broadcast_to(b))` is `broadcast_to(f(a, b))` when `a` and `b` have the
/// same shape, so `f` computes on fewer elements. The pattern rebuilds a splat constant operand
/// in that shape.
#[pattern(canonical)]
struct HoistBroadcast;

/// What an operand of the elementwise op becomes in the smaller shape.
enum Narrowed {
    /// The input of the `broadcast_to` that defines the operand.
    Input(Value),
    /// The bytes of one element of a splat constant operand.
    Splat(Vec<u8>),
}

fn narrowed(module: &Module, operand: Value) -> Option<Narrowed> {
    match module.def_as::<BroadcastTo>(operand) {
        Some(broadcast) => Some(Narrowed::Input(broadcast.operands.input)),
        None => Some(Narrowed::Splat(
            module.constant(operand)?.splat_bytes()?.to_vec(),
        )),
    }
}

/// The shape of the broadcast inputs, when there is at least one and they all have it.
fn input_dims(module: &Module, operands: &[Narrowed]) -> Option<SmallVec<[Dim; 4]>> {
    let mut dims = operands.iter().filter_map(|n| match n {
        Narrowed::Input(input) => Some(module.ty(*input).as_tensor().map(|t| &t.dims)),
        Narrowed::Splat(_) => None,
    });
    let first = dims.next()??;
    dims.all(|d| d == Some(first)).then(|| first.clone())
}

impl Pattern for HoistBroadcast {
    type Root = Has<Elementwise>;
    fn rewrite(&self, root: WithTrait<Elementwise>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let module = &*rw.module;
        let operands = module.operands(root.inst).to_vec();
        let Some(narrow) = operands
            .iter()
            .map(|&v| narrowed(module, v))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(false);
        };
        let Some(dims) = input_dims(module, &narrow) else {
            return Ok(false);
        };
        let wide = module
            .ty(module.results(root.inst)[0])
            .as_tensor()
            .expect("a tensor")
            .clone();
        if wide.dims == dims {
            return Ok(false);
        }
        let splat_dims: Option<Vec<usize>> = dims.iter().map(Dim::as_static).collect();
        let mut narrow_operands = Vec::with_capacity(operands.len());
        for (n, &operand) in narrow.into_iter().zip(&operands) {
            let value = match n {
                Narrowed::Input(input) => input,
                Narrowed::Splat(element) => {
                    let dtype = rw.module.ty(operand).as_tensor().expect("a tensor").dtype;
                    let dims = splat_dims.clone().expect("static when a splat is present");
                    rw.constant_before(root.inst, Dense::splat(dtype, dims, &element))?
                }
            };
            narrow_operands.push(value);
        }
        let inst = rw.insert_op_before(root.inst, root.op.clone(), narrow_operands)?;
        let result = rw.module.results(inst)[0];
        let broadcast = rw.insert_before(root.inst, BroadcastTo { shape: wide.dims }, result)?;
        rw.replace(root.inst, &[broadcast.output])?;
        Ok(true)
    }
}
