//! The step function of a function in the buffer form, `gptrs_step`: it runs the function's
//! launches in order, on the memory its runtime gives it. Each op of the buffer form is C: an arena
//! or a slice of one is an address, a launch a call of its dispatch, a copy a `memmove`, and a
//! result that an arena holds a `memmove` into memory of its own. A constant is memory that the
//! runtime holds. Any other op outside a dispatch is an error: the backend computes every op in
//! the C of a dispatch.
//!
//! The runtime gives the step the address of each entry of its [`StepPlan::memory`], in order, and
//! of each of its arenas; and, while it profiles, a function that the step calls before and after
//! each launch.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, ensure, Context, Result};
use gpt_rs_ir::dialects::{host, tensor};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::ir::{Inst, Module, TensorType, Type, Value};
use gpt_rs_ir::traits::Terminator;

use super::symbol;

/// What a step function takes, and where the results of its function come from.
#[derive(Default)]
pub(crate) struct StepPlan {
    /// The memory of each entry of the step's `memory` array, which the runtime gives it.
    pub memory: Vec<Memory>,
    /// Each of the step's arenas, in order.
    pub arenas: Vec<Arena>,
    /// Where each result of the function comes from.
    pub results: Vec<Produced>,
    /// The dispatch of each launch, in order, which a profile names.
    pub launches: Vec<String>,
}

/// An arena of a step, `host.arena`: its bytes, and the bytes its address is a multiple of.
#[derive(Clone, Copy)]
pub(crate) struct Arena {
    pub bytes: usize,
    pub alignment: usize,
}

/// An entry of a step's memory.
pub(crate) enum Memory {
    /// The elements of argument `index`, which the step reads in place.
    Read(usize),
    /// The elements of argument `index`, which the step may write: the argument's own memory when
    /// nothing else holds it, else a copy.
    Taken(usize),
    /// New memory for a buffer of the function's that it returns, `host.alloc`, at a multiple of
    /// `alignment` bytes.
    Alloc { ty: TensorType, alignment: usize },
    /// New memory that a result held in an arena is copied to, at its `host.to_tensor`.
    Copy(TensorType),
    /// A constant of the function.
    Constant(HostTensor),
}

/// Where a result of the function comes from.
#[derive(Clone, Copy)]
pub(crate) enum Produced {
    /// Argument `index`, as the call gives it.
    Arg(usize),
    /// The elements that memory entry `index` holds after the step.
    Memory(usize),
}

/// Lines of C, each with the op it comes from.
pub(crate) type Lines = Vec<(String, Option<Inst>)>;

/// The C of the step function `function` of the function `func` of `module`, line by line with the
/// op each line comes from, and what the step takes.
pub(crate) fn step(module: &Module, func: Inst, function: &str) -> Result<(Lines, StepPlan)> {
    let mut printer = StepPrinter {
        module,
        out: Vec::new(),
        names: HashMap::new(),
        tensors: HashMap::new(),
        buffers: HashMap::new(),
        results_held: HashSet::new(),
        plan: StepPlan::default(),
    };
    printer
        .function(func, function)
        .with_context(|| format!("writing C for {function}"))?;
    Ok((printer.out, printer.plan))
}

/// The C of one step function, written op by op.
struct StepPrinter<'m> {
    module: &'m Module,
    out: Lines,
    /// The C name of each buffer's address.
    names: HashMap<Value, String>,
    /// Where each tensor value is: an argument, or the memory entry that holds its elements.
    tensors: HashMap<Value, Produced>,
    /// The memory entry of each buffer that is one: of an argument, an allocation or a constant.
    buffers: HashMap<Value, usize>,
    /// The buffers whose memory is a result already, which nothing may write after.
    results_held: HashSet<Value>,
    plan: StepPlan,
}

impl StepPrinter<'_> {
    fn function(&mut self, func: Inst, function: &str) -> Result<()> {
        self.header(format!(
            "void {function}(void *const *memory, uint8_t *const *arenas, gptrs_mark mark, void \
             *context) {{"
        ));
        let body = self.module.regions(func)[0];
        for (index, &arg) in self.module.args(body).iter().enumerate() {
            self.tensors.insert(arg, Produced::Arg(index));
        }
        for inst in self.module.insts(body) {
            self.op(inst)?;
        }
        self.header("}".to_string());
        Ok(())
    }

    fn op(&mut self, inst: Inst) -> Result<()> {
        let module = self.module;
        let op = module.op(inst);
        if op.has_trait::<Terminator>() {
            return self.results(inst);
        }
        if let Some(constant) = op.get::<tensor::Constant>() {
            let value = &constant.value;
            let tensor = HostTensor::from_le_bytes(&value.tensor_type(), &value.bytes)?;
            let entry = self.entry(Memory::Constant(tensor));
            self.tensors
                .insert(module.results(inst)[0], Produced::Memory(entry));
            return Ok(());
        }
        if let Some(arena) = op.get::<host::Arena>() {
            let index = self.plan.arenas.len();
            self.plan.arenas.push(Arena {
                bytes: usize::try_from(arena.bytes)?,
                alignment: usize::try_from(arena.alignment)?,
            });
            let name = self.define(module.results(inst)[0]);
            self.line(inst, format!("uint8_t *{name} = arenas[{index}];"));
            return Ok(());
        }
        if let Some(slice) = op.get::<host::Slice>() {
            let arena = self.name(module.operands(inst)[0])?.to_string();
            let name = self.define(module.results(inst)[0]);
            let offset = slice.offset;
            self.line(inst, format!("uint8_t *{name} = {arena} + {offset};"));
            return Ok(());
        }
        if let Some(alloc) = op.get::<host::Alloc>() {
            let buffer = module.results(inst)[0];
            let entry = self.entry(Memory::Alloc {
                ty: buffer_tensor(module, buffer)?,
                alignment: usize::try_from(alloc.alignment)?,
            });
            return self.memory_buffer(inst, buffer, entry);
        }
        if op.get::<host::ToBuffer>().is_some() {
            let entry = match self.tensor(module.operands(inst)[0])? {
                Produced::Arg(index) => self.entry(Memory::Read(index)),
                Produced::Memory(entry) => entry,
            };
            return self.memory_buffer(inst, module.results(inst)[0], entry);
        }
        if op.get::<host::Take>().is_some() {
            let Produced::Arg(index) = self.tensor(module.operands(inst)[0])? else {
                bail!(
                    "host.take of a value that the step computes: the x86 backend updates in place \
                     only a tensor that the step takes as an argument"
                );
            };
            let entry = self.entry(Memory::Taken(index));
            return self.memory_buffer(inst, module.results(inst)[0], entry);
        }
        if let Some(launch) = module.op_as::<host::Launch>(inst) {
            return self.launch(inst, &launch);
        }
        if op.get::<host::Copy>().is_some() {
            let operands = module.operands(inst);
            self.check_writable(operands[1])?;
            let bytes = byte_size(&buffer_tensor(module, operands[0])?)?;
            let from = self.name(operands[0])?.to_string();
            let to = self.name(operands[1])?.to_string();
            self.line(inst, format!("memmove({to}, {from}, {bytes});"));
            return Ok(());
        }
        if op.get::<host::ToTensor>().is_some() {
            return self.result_tensor(inst);
        }
        bail!(
            "{} is outside a dispatch, and the x86 backend computes every op in the C of a \
             dispatch",
            op.name()
        )
    }

    /// A buffer whose address is memory entry `entry`.
    fn memory_buffer(&mut self, inst: Inst, buffer: Value, entry: usize) -> Result<()> {
        self.buffers.insert(buffer, entry);
        let name = self.define(buffer);
        self.line(
            inst,
            format!("uint8_t *{name} = (uint8_t *)memory[{entry}];"),
        );
        Ok(())
    }

    /// A call of the dispatch of `launch` on the buffers of its parameters, in order, between the
    /// marks of a profile.
    fn launch(&mut self, inst: Inst, launch: &host::LaunchOp) -> Result<()> {
        for buffer in launch.outs().iter().chain(launch.inouts().iter()) {
            self.check_writable(buffer.id())?;
        }
        let index = self.plan.launches.len();
        self.plan.launches.push(launch.callee.0.clone());
        let mut params = Vec::new();
        for buffer in launch.params()? {
            params.push(self.name(buffer.id())?.to_string());
        }
        let stem = symbol(&launch.callee.0);
        self.line(inst, format!("if (mark) mark(context, {index}, 1);"));
        self.line(
            inst,
            format!(
                "{{ void *params[] = {{{}}}; gptrs_run_{stem}(params); }}",
                params.join(", ")
            ),
        );
        self.line(inst, format!("if (mark) mark(context, {index}, 0);"));
        Ok(())
    }

    /// The tensor of the elements a buffer holds now: those of its memory entry, or, for a buffer
    /// of an arena, a copy into memory of its own.
    fn result_tensor(&mut self, inst: Inst) -> Result<()> {
        let module = self.module;
        let buffer = module.operands(inst)[0];
        let produced = match self.buffers.get(&buffer) {
            Some(&entry) => match &self.plan.memory[entry] {
                Memory::Read(index) => Produced::Arg(*index),
                _ => {
                    self.results_held.insert(buffer);
                    Produced::Memory(entry)
                }
            },
            None => {
                let ty = buffer_tensor(module, buffer)?;
                let bytes = byte_size(&ty)?;
                let entry = self.entry(Memory::Copy(ty));
                let from = self.name(buffer)?.to_string();
                self.line(inst, format!("memmove(memory[{entry}], {from}, {bytes});"));
                Produced::Memory(entry)
            }
        };
        self.tensors.insert(module.results(inst)[0], produced);
        Ok(())
    }

    /// Where each result of the function comes from: the operands of its terminator.
    fn results(&mut self, inst: Inst) -> Result<()> {
        for &operand in self.module.operands(inst) {
            let produced = self.tensor(operand)?;
            self.plan.results.push(produced);
        }
        Ok(())
    }

    /// Fails when a result holds the memory of `buffer` already: a write now would change it.
    fn check_writable(&self, buffer: Value) -> Result<()> {
        ensure!(
            !self.results_held.contains(&buffer),
            "an op writes a buffer whose elements a result of the function holds already"
        );
        Ok(())
    }

    fn tensor(&self, value: Value) -> Result<Produced> {
        self.tensors
            .get(&value)
            .copied()
            .ok_or_else(|| anyhow!("a tensor of the step comes from an op outside the buffer form"))
    }

    fn entry(&mut self, memory: Memory) -> usize {
        self.plan.memory.push(memory);
        self.plan.memory.len() - 1
    }

    fn define(&mut self, value: Value) -> String {
        let name = format!("v{}", self.names.len());
        self.names.insert(value, name.clone());
        name
    }

    fn name(&self, value: Value) -> Result<&str> {
        self.names
            .get(&value)
            .map(String::as_str)
            .ok_or_else(|| anyhow!("a buffer is used before the step defines it"))
    }

    fn line(&mut self, inst: Inst, text: String) {
        self.out.push((format!("    {text}"), Some(inst)));
    }

    fn header(&mut self, text: String) {
        self.out.push((text, None));
    }
}

/// The tensor type of the elements of the buffer `value`.
pub(crate) fn buffer_tensor(module: &Module, value: Value) -> Result<TensorType> {
    let ty = module.ty(value);
    let buffer = ty
        .as_buffer()
        .ok_or_else(|| anyhow!("{ty} is not a buffer"))?;
    Ok(buffer.tensor_type())
}

/// The bytes the elements of `ty` take.
pub(crate) fn byte_size(ty: &TensorType) -> Result<usize> {
    let dims = ty
        .static_dims()
        .ok_or_else(|| anyhow!("the x86 backend runs static extents only, not {ty}"))?;
    ty.dtype
        .bytes_for(dims.iter().product())
        .ok_or_else(|| anyhow!("{ty} takes more bytes than fit"))
}

/// The tensor types of the arguments of the function `func`, which each run checks its arguments
/// against.
pub(crate) fn arg_types(module: &Module, func: Inst) -> Result<Vec<TensorType>> {
    let body = module.regions(func)[0];
    let mut types = Vec::new();
    for &arg in module.args(body) {
        let Type::Tensor(tensor) = module.ty(arg) else {
            bail!(
                "an argument of the function is {}, not a tensor",
                module.ty(arg)
            );
        };
        ensure!(
            tensor.static_dims().is_some(),
            "the x86 backend runs static extents only, not {tensor}"
        );
        types.push(tensor.clone());
    }
    Ok(types)
}
