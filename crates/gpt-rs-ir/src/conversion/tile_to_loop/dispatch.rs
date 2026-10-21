use std::collections::HashMap;

use anyhow::Result;

use super::block::{copy, BlockLowering, Buffers, OperandBuffer, Place, RunFacts};
use super::utils::alloc;
use crate::dialects::loops::{self, Access, Return};
use crate::dialects::tile::{Block, Dispatch, DispatchOp};
use crate::ir::{BufferType, Dim, Inst, Module, Region, Space, Type, Value};
use crate::rewrite::{pattern, Build, Builder, NoMatch, Pattern, Rewriter};
use crate::targets::{TargetDesc, TargetRef};

/// `tile.dispatch` becomes the `loop.func` of its name, with a buffer for each argument and each
/// result, and its blocks lowered in order.
///
/// - A result that may update an argument in place ([`Dispatch::updated_args`]) takes the
///   argument's buffer, an `inout` parameter.
/// - A result that may share the buffer of an argument ([`Dispatch::shareable_args`]) says so on
///   its parameter (`out<over = ..>`).
#[pattern]
pub(super) struct DispatchToLoop;

impl Pattern for DispatchToLoop {
    type Root = Dispatch;

    fn rewrite(&self, dispatch: DispatchOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = rewriter.module();
        let attrs = dispatch.attrs();
        let body = dispatch.body().id();
        let args = buffer_types(&attrs.args);
        let results = buffer_types(&attrs.results);
        let params = Params::of(module, body, args, results);
        let func = loops::Func {
            name: attrs.name.clone(),
            args: params.types.clone(),
            access: params.access.clone(),
            target: attrs.target.clone(),
            contract: attrs.numerics.contract,
        };
        let target = attrs
            .target
            .as_ref()
            .map(TargetRef::target)
            .transpose()
            .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        let facts = (attrs.numerics.skip_pad, target.as_deref());
        rewriter
            .create_with_body(func, (), |builder, values| {
                lower_body(module, body, facts, builder, &params, values)?;
                builder.create(Return, ());
                Ok(())
            })
            .map_err(|err| NoMatch::because(format_args!("{err:#}")))?;
        rewriter.erase(&dispatch);
        Ok(())
    }
}

/// The `global` buffer of each of `types`, the tensors that a dispatch takes or gives.
fn buffer_types(types: &[Type]) -> Vec<Type> {
    let mut buffers = Vec::with_capacity(types.len());
    for ty in types {
        let tensor = ty
            .as_tensor()
            .expect("the verifier keeps the arguments and results of a dispatch tensors");
        buffers.push(Type::Buffer(BufferType::of(tensor, Space::Global)));
    }
    buffers
}

/// The parameters of the `loop.func` of a dispatch: the position of the parameter of each argument
/// and of each result, and the type and the access of each parameter.
struct Params {
    of_args: Vec<usize>,
    of_results: Vec<usize>,
    types: Vec<Type>,
    access: Vec<Access>,
}

impl Params {
    /// The parameters of the dispatch whose body is `body`, with the buffer types `args` and
    /// `results`. The `in` and `inout` parameters take the arguments in order, and the `inout` and
    /// `out` parameters give the results in order. So an `out` result goes after every argument,
    /// unless a later result updates one ([`Dispatch::updated_args`]). Then it goes before the
    /// argument that the first such result updates, and before the arguments after it.
    fn of(module: &Module, body: Region, args: Vec<Type>, results: Vec<Type>) -> Params {
        let mut params = Params {
            of_args: vec![0; args.len()],
            of_results: vec![0; results.len()],
            types: Vec::with_capacity(args.len() + results.len()),
            access: Vec::with_capacity(args.len() + results.len()),
        };
        let mut next_arg = 0;
        let mut waiting = Vec::new();
        for (result, tied) in Dispatch::updated_args(module, body).into_iter().enumerate() {
            let Some(arg) = tied else {
                waiting.push(result);
                continue;
            };
            for (earlier, ty) in args.iter().enumerate().take(arg).skip(next_arg) {
                params.of_args[earlier] = params.push(ty, Access::In);
            }
            for out in waiting.drain(..) {
                params.of_results[out] = params.push(&results[out], Access::Out { over: None });
            }
            let position = params.push(&args[arg], Access::Inout);
            params.of_args[arg] = position;
            params.of_results[result] = position;
            next_arg = arg + 1;
        }
        for (arg, ty) in args.iter().enumerate().skip(next_arg) {
            params.of_args[arg] = params.push(ty, Access::In);
        }
        for out in waiting {
            params.of_results[out] = params.push(&results[out], Access::Out { over: None });
        }
        for (result, over) in Dispatch::shareable_args(module, body)
            .into_iter()
            .enumerate()
        {
            let Some(arg) = over else {
                continue;
            };
            let (position, shared) = (params.of_results[result], params.of_args[arg]);
            if params.access[position] == (Access::Out { over: None })
                && params.access[shared] == Access::In
            {
                params.access[position] = Access::Out { over: Some(shared) };
            }
        }
        params
    }

    /// Adds a parameter and gives its position.
    fn push(&mut self, ty: &Type, access: Access) -> usize {
        self.types.push(ty.clone());
        self.access.push(access);
        self.types.len() - 1
    }
}

/// The extents of the tensor `value`, an argument of a dispatch or a result of one of its blocks.
fn dims(module: &Module, value: Value) -> Vec<Dim> {
    let tensor = module
        .ty(value)
        .as_tensor()
        .expect("the verifier keeps the values of a dispatch tensors");
    tensor.dims.to_vec()
}

/// The operand of the block of `after` that the one result of the block of `before` is.
fn input_of(module: &Module, before: Inst, after: Inst) -> usize {
    Block::epilogue_input(module, before, after)
        .expect("the verifier places a block by its read of the result of the block before it")
}

/// The buffer of each operand of the block `inst`, with its extents, from `buffers`. A placed
/// block never reads its operand `input`, whose elements the block before it gives instead: that
/// operand has no buffer.
fn block_operands(
    module: &Module,
    inst: Inst,
    buffers: &HashMap<Value, Value>,
    input: Option<usize>,
) -> Vec<OperandBuffer> {
    let mut operands = Vec::new();
    for (position, &operand) in module.operands(inst).iter().enumerate() {
        let buffer = match input == Some(position) {
            true => None,
            false => Some(buffers[&operand]),
        };
        operands.push(OperandBuffer {
            buffer,
            dims: dims(module, operand),
        });
    }
    operands
}

/// The runs of `blocks` that lower together, in order: each block, with the blocks after it that a
/// read places in the loops of the block before them.
fn runs<'m>(blocks: Vec<(Inst, &'m Block)>) -> Vec<Vec<(Inst, &'m Block)>> {
    let mut runs: Vec<Vec<(Inst, &'m Block)>> = Vec::new();
    for (inst, block) in blocks {
        let placed = block.reads.iter().any(|read| read.at.is_some());
        match (placed, runs.last_mut()) {
            (true, Some(run)) => run.push((inst, block)),
            _ => runs.push(vec![(inst, block)]),
        }
    }
    runs
}

/// The buffers of the results of the block `last`, the last of the blocks that lower together,
/// which `buffers` gains: for each result, the first output that the dispatch returns it in, or
/// scratch ([`Dispatch::SCRATCH_SPACE`]).
fn run_results(
    module: &Module,
    builder: &mut Builder,
    last: Inst,
    returns: &HashMap<Value, Value>,
    buffers: &mut HashMap<Value, Value>,
) -> Vec<Value> {
    let mut results = Vec::new();
    for &result in module.results(last) {
        let buffer = match returns.get(&result) {
            Some(&output) => output,
            None => {
                let tensor = module
                    .ty(result)
                    .as_tensor()
                    .expect("the verifier keeps the results of a block tensors");
                let shape = tensor
                    .static_dims()
                    .expect("the dispatch verifier keeps a result between blocks static");
                alloc(builder, &shape, tensor.dtype, Dispatch::SCRATCH_SPACE)
            }
        };
        buffers.insert(result, buffer);
        results.push(buffer);
    }
    results
}

/// The lowering of the blocks `run`, which lower together in the loops of the first and write the
/// last one's `results`: the first block, with each block after it placed in those loops. The walk
/// of the first block is in `facts`.
fn run_lowering<'m>(
    module: &'m Module,
    run: &[(Inst, &'m Block)],
    facts: RunFacts<'m>,
    buffers: &HashMap<Value, Value>,
    results: Vec<Value>,
) -> BlockLowering<'m> {
    let last = run.len() - 1;
    let mut placed = Vec::with_capacity(last);
    for index in 1..run.len() {
        let ((before, earlier), (inst, block)) = (run[index - 1], run[index]);
        let input = input_of(module, before, inst);
        let block_buffers = Buffers {
            operands: block_operands(module, inst, buffers, Some(input)),
            results: match index == last {
                true => results.clone(),
                false => Vec::new(),
            },
        };
        let place = Place::Placed {
            input,
            at_point: earlier.read_at_point(block, input),
        };
        placed.push(BlockLowering::new(
            module,
            (inst, block),
            block_buffers,
            facts,
            place,
        ));
    }
    let (inst, block) = run[0];
    let first_buffers = Buffers {
        operands: block_operands(module, inst, buffers, None),
        results: match last {
            0 => results,
            _ => Vec::new(),
        },
    };
    BlockLowering::new(
        module,
        (inst, block),
        first_buffers,
        facts,
        Place::First { placed },
    )
}

/// Lowers the blocks of the dispatch body `body` into the `loop.func` whose parameters, laid out
/// as `params` says, are `values`: with `skip_pad`, whether the dispatch's numerics let a loop skip
/// the products of a pad 0, and `target`, the dispatch's target.
fn lower_body(
    module: &Module,
    body: Region,
    (skip_pad, target): (bool, Option<&TargetDesc>),
    builder: &mut Builder,
    params: &Params,
    values: &[Value],
) -> Result<()> {
    let args = module.args(body);
    let returned = module.yielded(body);
    let mut buffers: HashMap<Value, Value> = HashMap::new();
    for (&arg, &position) in args.iter().zip(&params.of_args) {
        buffers.insert(arg, values[position]);
    }
    let mut outputs = Vec::with_capacity(returned.len());
    for &position in &params.of_results {
        outputs.push(values[position]);
    }
    // A block writes a result where the dispatch returns it first.
    let mut returns: HashMap<Value, Value> = HashMap::new();
    for (&value, &output) in returned.iter().zip(&outputs) {
        if module.def_inst(value).is_some() {
            returns.entry(value).or_insert(output);
        }
    }
    let blocks = Dispatch::blocks(module, body);
    for run in runs(blocks) {
        let &(last, _) = run.last().expect("a run holds a block");
        let results = run_results(module, builder, last, &returns, &mut buffers);
        let walk = run[0].1.walk();
        let facts = RunFacts {
            walk: &walk,
            skip_pad,
            target,
        };
        let first = run_lowering(module, &run, facts, &buffers, results);
        // The loops of a run take the location of its first block.
        let outer = builder.location().cloned();
        builder.set_origin(module, run[0].0);
        first.lower(builder)?;
        builder.set_location(outer);
    }
    // A result that is an argument, or a value returned twice, is a copy.
    for (&value, &output) in returned.iter().zip(&outputs) {
        let buffer = buffers[&value];
        if buffer != output {
            copy(builder, buffer, output, &dims(module, value))?;
        }
    }
    Ok(())
}
