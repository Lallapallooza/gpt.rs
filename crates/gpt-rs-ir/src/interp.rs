//! The interpreter: runs a function by calling each op's `eval`. The filetests check every pass
//! against it.
//!
//! A run binds each dynamic dimension (`{B}`) to the extent of the first value that has it. It
//! checks every argument and every value an op computes against its type under those bindings.

use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{anyhow, bail, ensure, Context, Result};

use crate::host::HostValue;
use crate::module::{Inst, Module, Region, Value};
use crate::op::{OpDyn, RegionRunner};
use crate::traits::Terminator;
use crate::types::{Dim, Sym, Type};

struct Interp<'m> {
    module: &'m Module,
    env: HashMap<Value, Rc<HostValue>>,
    syms: HashMap<Sym, usize>,
}

impl<'m> Interp<'m> {
    fn new(module: &'m Module) -> Self {
        Self {
            module,
            env: HashMap::new(),
            syms: HashMap::new(),
        }
    }

    /// Checks `value` against `ty`, binding the dynamic dimensions it fixes.
    fn bind(&mut self, ty: &Type, value: &HostValue) -> Result<()> {
        let (Type::Tensor(tensor), HostValue::Tensor(host)) = (ty, value) else {
            let actual = value.ty();
            ensure!(actual == *ty, "the value has type {actual}, not {ty}");
            return Ok(());
        };
        ensure!(
            tensor.dtype == host.dtype() && tensor.rank() == host.dims.len(),
            "a {} tensor of shape {:?} is not a {tensor}",
            host.dtype(),
            host.dims
        );
        for (dim, &extent) in tensor.dims.iter().zip(&host.dims) {
            let bound = match dim {
                Dim::Static(n) => *n,
                Dim::Dynamic(sym) => *self.syms.entry(sym.clone()).or_insert(extent),
            };
            ensure!(
                bound == extent,
                "a tensor of shape {:?} is not a {tensor} with {dim} = {bound}",
                host.dims
            );
        }
        Ok(())
    }

    /// `ty` with every bound dynamic dimension replaced by its extent.
    fn resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Tensor(tensor) => Type::Tensor(
                tensor.with_dims(
                    tensor
                        .dims
                        .iter()
                        .map(|dim| match dim {
                            Dim::Dynamic(sym) => self
                                .syms
                                .get(sym)
                                .map_or_else(|| dim.clone(), |&n| Dim::Static(n)),
                            Dim::Static(_) => dim.clone(),
                        })
                        .collect(),
                ),
            ),
            other => other.clone(),
        }
    }

    /// The values `inst` computes from `inputs`, its operands in order, checked against its
    /// result types.
    fn eval(&mut self, inst: Inst, inputs: &[&HostValue]) -> Result<Vec<HostValue>> {
        let module = self.module;
        let op = module.op(inst);
        let results = module.results(inst);
        let result_types: Vec<Type> = results
            .iter()
            .map(|&v| self.resolve(module.ty(v)))
            .collect();
        let outputs = op
            .dyn_eval(
                inputs,
                module.operand_segments(inst),
                module.regions(inst),
                &result_types,
                self,
            )
            .with_context(|| op.name().into_owned())?;
        ensure!(
            outputs.len() == results.len(),
            "{} gave {} values for {} results",
            op.name(),
            outputs.len(),
            results.len()
        );
        for (index, (&result, output)) in results.iter().zip(&outputs).enumerate() {
            self.bind(module.ty(result), output)
                .with_context(|| format!("{} result {index}", op.name()))?;
        }
        Ok(outputs)
    }
}

impl RegionRunner for Interp<'_> {
    fn run_region(&mut self, region: Region, args: Vec<HostValue>) -> Result<Vec<HostValue>> {
        let module = self.module;
        let block_args = module.args(region);
        ensure!(
            block_args.len() == args.len(),
            "the region takes {} arguments, got {}",
            block_args.len(),
            args.len()
        );
        for (index, (&block_arg, arg)) in block_args.iter().zip(args).enumerate() {
            self.bind(module.ty(block_arg), &arg)
                .with_context(|| format!("argument {index}"))?;
            self.env.insert(block_arg, Rc::new(arg));
        }
        for inst in module.insts(region) {
            let inputs: Vec<Rc<HostValue>> = module
                .operands(inst)
                .iter()
                .map(|v| {
                    self.env
                        .get(v)
                        .cloned()
                        .ok_or_else(|| anyhow!("{v:?} has no value"))
                })
                .collect::<Result<_>>()?;
            if module.op(inst).has_trait::<Terminator>() {
                return Ok(inputs.iter().map(|v| (**v).clone()).collect());
            }
            let refs: Vec<&HostValue> = inputs.iter().map(|v| &**v).collect();
            let outputs = self.eval(inst, &refs)?;
            for (&result, output) in module.results(inst).iter().zip(outputs) {
                self.env.insert(result, Rc::new(output));
            }
        }
        bail!("the region ended without a terminator")
    }
}

/// Runs `func` with `args` (parameters included, in argument order) and returns its results.
pub fn run(module: &Module, func: &str, args: Vec<HostValue>) -> Result<Vec<HostValue>> {
    let inst = module
        .func(func)
        .ok_or_else(|| anyhow!("no function @{func}"))?;
    let body = module.regions(inst)[0];
    Interp::new(module).run_region(body, args)
}

/// Evaluates the op of `inst`, which has no regions, on `inputs`, its operands in order. Checks
/// the operands and the results against the types `module` gives them, binding a dynamic
/// dimension to the extent of the first operand that has it. A backend runs an op it has no
/// kernel for this way.
pub fn eval_inst(module: &Module, inst: Inst, inputs: &[&HostValue]) -> Result<Vec<HostValue>> {
    let op = module.op(inst);
    ensure!(
        module.regions(inst).is_empty(),
        "{} has regions, which only a run of its function evaluates",
        op.name()
    );
    let operands = module.operands(inst);
    ensure!(
        inputs.len() == operands.len(),
        "{} takes {} operands, got {}",
        op.name(),
        operands.len(),
        inputs.len()
    );
    let mut interp = Interp::new(module);
    for (index, (&operand, input)) in operands.iter().zip(inputs).enumerate() {
        interp
            .bind(module.ty(operand), input)
            .with_context(|| format!("{} operand {index}", op.name()))?;
    }
    interp.eval(inst, inputs)
}
