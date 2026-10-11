//! [`Builder`]: builds IR into a module directly, at the end of a region or before an
//! instruction.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::func::{Func, ParamId, Return};
use crate::fold::{fold_op, FoldResult};
use crate::location::Location;
use crate::module::{Inst, Module, NewInst, Region, Value};
use crate::op::{Created, Folded, Groups, InOp, OpDyn, OpKind, Operands, Values};
use crate::rewrite::build::materialize;
use crate::rewrite::Build;
use crate::types::{TensorType, Type};
use crate::Op;

/// Where a builder places new instructions.
#[derive(Debug, Clone, Copy)]
enum InsertPoint {
    End(Region),
    Before(Inst),
}

/// Builds ops into a module. It takes result types from each op's type rule, never from its
/// caller, and panics when the rule rejects the op: building an invalid op is a bug in the code
/// that builds it.
pub struct Builder<'m> {
    pub module: &'m mut Module,
    point: InsertPoint,
    /// Instructions this builder created, in order, those inside new regions too.
    created: Vec<Inst>,
    /// The location of the ops it creates.
    location: Option<Location>,
}

impl<'m> Builder<'m> {
    pub fn at_end(module: &'m mut Module, region: Region) -> Self {
        Self {
            module,
            point: InsertPoint::End(region),
            created: Vec::new(),
            location: None,
        }
    }

    pub fn before(module: &'m mut Module, inst: Inst) -> Self {
        Self {
            module,
            point: InsertPoint::Before(inst),
            created: Vec::new(),
            location: None,
        }
    }

    /// Gives the ops created from here on the location `location`.
    pub fn set_location(&mut self, location: Option<Location>) {
        self.location = location;
    }

    /// The instructions this builder created, in order.
    pub(crate) fn into_created(self) -> Vec<Inst> {
        self.created
    }

    fn place(&mut self, inst: Inst) {
        match self.point {
            InsertPoint::End(region) => self.module.append(region, inst),
            InsertPoint::Before(anchor) => self.module.insert_before(anchor, inst),
        }
        self.created.push(inst);
    }

    /// Creates the op value `op` on `operands`, in the groups `segments`, with the regions
    /// `regions` already built.
    #[track_caller]
    pub fn create_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
        regions: Vec<Region>,
    ) -> Inst {
        let (result_types, result_segments) = self
            .module
            .infer_results(&op, &operands, segments.as_deref(), &regions)
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let inst = self.module.create_inst(NewInst {
            op,
            operands,
            operand_segments: segments,
            result_types: result_types.into_vec(),
            result_segments,
            regions,
            location: self.location.clone(),
        });
        self.place(inst);
        inst
    }

    /// Creates `op`, which has no regions, on `operands`, and gives its results.
    #[track_caller]
    pub fn create<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
    ) -> Created<T> {
        assert!(
            T::Groups::REGIONS.is_empty(),
            "{} has regions; create it with create_with_regions",
            op.name()
        );
        let (operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.create_op(op.into(), operands.into_vec(), segments, Vec::new());
        self.created_results::<T>(inst)
    }

    /// Creates `op` on `operands`. `body(index, builder, args)` fills region `index`, whose block
    /// arguments are `args`, and ends it with its terminator.
    #[track_caller]
    pub fn create_with_regions<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
        body: impl FnMut(usize, &mut Builder<'_>, &[Value]),
    ) -> Created<T> {
        let (operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.create_op_with_regions(op.into(), operands.into_vec(), segments, body);
        self.created_results::<T>(inst)
    }

    /// Creates the op value `op` on `operands`, in the groups `segments`, filling its regions as
    /// [`Builder::create_with_regions`] does.
    #[track_caller]
    pub(crate) fn create_op_with_regions(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
        mut body: impl FnMut(usize, &mut Builder<'_>, &[Value]),
    ) -> Inst {
        let types: Vec<&Type> = operands.iter().map(|&v| self.module.ty(v)).collect();
        let signatures = op
            .dyn_region_signatures(&types, segments.as_deref())
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let mut regions = Vec::new();
        for (index, signature) in signatures.into_iter().enumerate() {
            let region = self.module.new_region(signature.args);
            let args = self.module.args(region).to_vec();
            let mut inner = Builder::at_end(self.module, region);
            inner.set_location(self.location.clone());
            body(index, &mut inner, &args);
            self.created.extend(inner.created);
            regions.push(region);
        }
        self.create_op(op, operands, segments, regions)
    }

    fn created_results<T: OpKind>(&self, inst: Inst) -> Created<T> {
        let results = T::Groups::split_results::<Values>(
            self.module.results(inst).iter().copied(),
            self.module.result_segments(inst),
        )
        .expect("the type rule gave the result groups");
        T::Groups::created(results)
    }

    /// The values of folded results of types `types`, creating a constant for each
    /// `Folded::Constant`.
    pub(crate) fn materialize(&mut self, folded: Vec<Folded>, types: &[Type]) -> Vec<Value> {
        materialize(folded, types, |constant| {
            let inst = self.create_op(constant, Vec::new(), None, Vec::new());
            self.module.results(inst)[0]
        })
    }

    /// Creates `op`, a copy of an op with operands `operands` in the groups `segments` and
    /// results `results`, on the values that `map` gives for its operands, and maps each of
    /// `results` to the new op's result.
    pub fn create_mapped(
        &mut self,
        op: Op,
        operands: &[Value],
        segments: Option<Vec<usize>>,
        results: &[Value],
        map: &mut HashMap<Value, Value>,
    ) -> Result<Inst> {
        let operands = operands
            .iter()
            .map(|v| {
                map.get(v)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("an operand of {} is not mapped", op.name()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inst = self.create_op(op, operands, segments, Vec::new());
        map.extend(
            results
                .iter()
                .copied()
                .zip(self.module.results(inst).to_vec()),
        );
        Ok(inst)
    }

    /// The results of `inst` and their groups.
    fn results_of(&self, inst: Inst) -> (Vec<Value>, Option<Vec<usize>>) {
        (
            self.module.results(inst).to_vec(),
            self.module.result_segments(inst).map(<[usize]>::to_vec),
        )
    }

    /// Ends the current function body.
    pub fn ret(&mut self, values: &[Value]) {
        self.create(Return, values.to_vec());
    }
}

impl Module {
    /// Adds `func.func @name`. `body(builder, args)` builds the body and returns the values to
    /// return. The function's result types are the types of those values.
    pub fn build_func(
        &mut self,
        name: &str,
        args: Vec<Type>,
        params: Vec<Option<ParamId>>,
        body: impl FnOnce(&mut Builder<'_>, &[Value]) -> Result<Vec<Value>>,
    ) -> Result<Inst> {
        let region = self.new_region(args.clone());
        let arg_values = self.args(region).to_vec();
        let mut builder = Builder::at_end(self, region);
        let returned = body(&mut builder, &arg_values)?;
        builder.ret(&returned);
        let results = returned.iter().map(|&v| self.ty(v).clone()).collect();
        let func = Func {
            name: name.to_string(),
            args,
            results,
            params: bound_params(params),
            result_params: Vec::new(),
        };
        let top = self.body();
        let mut builder = Builder::at_end(self, top);
        Ok(builder.create_op(func.into(), Vec::new(), None, vec![region]))
    }
}

/// `params`, or an empty list when none of them is bound. A `Func` stores its `params` this way.
pub(crate) fn bound_params(params: Vec<Option<ParamId>>) -> Vec<Option<ParamId>> {
    if params.iter().any(Option::is_some) {
        params
    } else {
        Vec::new()
    }
}

impl Build for Builder<'_> {
    /// Folds by the op's own folder and by its traits, since the operands are in the module.
    #[track_caller]
    fn create_or_fold_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
    ) -> (Vec<Value>, Option<Vec<usize>>) {
        let (types, result_segments) = self
            .module
            .infer_results(&op, &operands, segments.as_deref(), &[])
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let types = types.into_vec();
        let canonical = match fold_op(self.module, &op, &operands, segments.as_deref(), &types) {
            Some(FoldResult::Results(folded)) => {
                return (self.materialize(folded, &types), result_segments);
            }
            Some(FoldResult::Operands(canonical)) => canonical,
            None => operands,
        };
        let inst = self.create_op(op, canonical, segments, Vec::new());
        self.results_of(inst)
    }

    fn tensor_type(&self, value: Value) -> &TensorType {
        let ty = self.module.ty(value);
        ty.as_tensor()
            .unwrap_or_else(|| panic!("{value:?} is {ty}, not a tensor"))
    }
}
