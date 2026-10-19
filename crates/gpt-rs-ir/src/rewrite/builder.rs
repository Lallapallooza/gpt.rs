//! [`Builder`]: builds IR into a module directly, at the end of a region or before an
//! instruction.

use std::collections::HashMap;

use anyhow::Result;

use crate::dialects::func::{Func, ParamId, Return};
use crate::ir::{
    AsInst, Created, Folded, Groups, InOp, Inst, Location, Module, NewInst, OpDyn, OpKind,
    Operands, Region, Type, Value, Values,
};
use crate::rewrite::build::materialize;
use crate::rewrite::rewriter::Move;
use crate::rewrite::{fold_op, Build, FoldResult, KEY};
use crate::text::any_items;
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
    /// What the builder records for the module that a rewrite reads, when it builds a region that
    /// the rewrite creates.
    staged: Option<StagedRegions<'m>>,
}

/// When a rewrite creates a region, the region's builder records two things beside the ops. The
/// first is the ops that move into the region from the module that the rewrite reads. The second
/// is the scratch values that stand for their results. The rewrite imports those values as it
/// imports the values its staged ops read.
pub(crate) struct StagedRegions<'s> {
    pub(crate) source: &'s Module,
    pub(crate) moves: &'s mut Vec<Move>,
    pub(crate) imports: &'s mut HashMap<Value, Value>,
    pub(crate) imports_region: Region,
}

impl StagedRegions<'_> {
    fn reborrow(&mut self) -> StagedRegions<'_> {
        StagedRegions {
            source: self.source,
            moves: &mut *self.moves,
            imports: &mut *self.imports,
            imports_region: self.imports_region,
        }
    }
}

impl<'m> Builder<'m> {
    pub fn at_end(module: &'m mut Module, region: Region) -> Self {
        Self {
            module,
            point: InsertPoint::End(region),
            created: Vec::new(),
            location: None,
            staged: None,
        }
    }

    pub fn before(module: &'m mut Module, op: impl AsInst) -> Self {
        Self {
            module,
            point: InsertPoint::Before(op.inst()),
            created: Vec::new(),
            location: None,
            staged: None,
        }
    }

    /// The builder of the staged ops of a rewrite, at the end of `region` of the rewrite's
    /// scratch module, whose region builders record the ops they move in `staged`.
    pub(crate) fn staged(
        module: &'m mut Module,
        region: Region,
        staged: StagedRegions<'m>,
    ) -> Self {
        Self {
            staged: Some(staged),
            ..Self::at_end(module, region)
        }
    }

    /// The module the rewrite reads, for the builder of its staged ops.
    fn rewritten(&self) -> Option<&Module> {
        self.staged.as_ref().map(|staged| staged.source)
    }

    /// Gives the ops created from here on the location `location`.
    pub fn set_location(&mut self, location: Option<Location>) {
        self.location = location;
    }

    /// The ops created from here on are made from `op` of `source`, the module a transform
    /// reads: they take its location, with the running pass on its trail.
    pub fn set_origin(&mut self, source: &Module, op: impl AsInst) {
        self.location = source.made_from(op);
    }

    /// Moves `op`, an op of the module that the rewrite reads, here, at the end of a region that
    /// the rewrite creates, when the rewrite applies, and gives the values that stand for its
    /// results in the region. Only the builder of such a region moves ops.
    pub fn move_here(&mut self, op: impl AsInst) -> Vec<Value> {
        let InsertPoint::End(region) = self.point else {
            panic!("ops move to the end of a region a rewrite creates");
        };
        let position = self.module.insts(region).count();
        let staged = self
            .staged
            .as_mut()
            .expect("only the builder of a region that a rewrite creates moves ops into it");
        let inst = op.inst();
        staged.moves.push(Move {
            op: inst,
            region,
            position,
        });
        let mut results = Vec::new();
        for &result in staged.source.results(inst) {
            let ty = staged.source.ty(result).clone();
            let imported = *staged
                .imports
                .entry(result)
                .or_insert_with(|| self.module.append_arg(&KEY, staged.imports_region, ty));
            results.push(imported);
        }
        results
    }

    /// The location of the ops it creates.
    pub fn location(&self) -> Option<&Location> {
        self.location.as_ref()
    }

    /// The instructions this builder created, in order.
    pub(crate) fn into_created(self) -> Vec<Inst> {
        self.created
    }

    fn place(&mut self, inst: Inst) {
        match self.point {
            InsertPoint::End(region) => self.module.append(&KEY, region, inst),
            InsertPoint::Before(anchor) => self.module.insert_before(&KEY, anchor, inst),
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
            .infer_results(
                &op,
                &operands,
                segments.as_deref(),
                &regions,
                self.rewritten(),
            )
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

    /// Creates `new`, an op whose result types it states, such as one read from text, and gives
    /// its instruction.
    pub(crate) fn create_typed(&mut self, new: NewInst) -> Inst {
        let inst = self.module.create_inst(new);
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
        let (flat_operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.create_op(op.into(), flat_operands.into_vec(), segments, Vec::new());
        self.created_results::<T>(inst)
    }

    /// Creates `op` on `operands`. `body(index, builder, args)` fills region `index`, whose block
    /// arguments are `args`, and ends it with its terminator.
    #[track_caller]
    pub fn create_with_regions<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
        body: impl FnMut(usize, &mut Builder, &[Value]),
    ) -> Created<T> {
        let (flat_operands, segments) = T::Groups::flatten_operands::<Values>(operands.into());
        let inst = self.create_op_with_regions(op.into(), flat_operands.into_vec(), segments, body);
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
        mut body: impl FnMut(usize, &mut Builder, &[Value]),
    ) -> Inst {
        let types: Vec<&Type> = operands
            .iter()
            .map(|&operand| self.module.ty(operand))
            .collect();
        let signatures = op
            .dyn_region_signatures(&types, segments.as_deref())
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let mut regions = Vec::new();
        for (index, signature) in signatures.into_iter().enumerate() {
            let region = self.module.new_region(signature.args);
            let args = self.module.args(region).to_vec();
            let mut inner = Builder::at_end(self.module, region);
            inner.staged = self.staged.as_mut().map(StagedRegions::reborrow);
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
        let mapped_operands = operands
            .iter()
            .map(|operand| {
                map.get(operand)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("an operand of {} is not mapped", op.name()))
            })
            .collect::<Result<Vec<_>>>()?;
        let inst = self.create_op(op, mapped_operands, segments, Vec::new());
        map.extend(
            results
                .iter()
                .copied()
                .zip(self.module.results(inst).to_vec()),
        );
        Ok(inst)
    }

    /// Creates a copy of `inst` of `source`, with its regions, as [`Module::copy_inst`] makes it:
    /// its operands are the values `map` gives for its own, and `map` gains its results and
    /// region arguments. Each op of the copy is made from its original.
    pub fn create_copy(
        &mut self,
        source: &Module,
        op: impl AsInst,
        map: &mut HashMap<Value, Value>,
    ) -> Result<Inst> {
        let inst = op.inst();
        let copy = self.module.copy_inst(source, inst, map)?;
        self.place(copy);
        let copied = self.module.walk_inst(copy);
        for (&copied_op, original) in copied.iter().zip(source.walk_inst(inst)) {
            self.module
                .set_location(&KEY, copied_op, source.made_from(original));
        }
        self.created.extend(copied.into_iter().skip(1));
        Ok(copy)
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
    /// Adds `func.func @name`, whose arguments bind `params` and whose results are the parameters
    /// `result_params` when it runs at load time. `body(builder, args)` builds the body and
    /// returns the values to return. The function's result types are the types of those values.
    pub fn build_func(
        &mut self,
        name: &str,
        args: Vec<Type>,
        params: Vec<Option<ParamId>>,
        result_params: Vec<ParamId>,
        body: impl FnOnce(&mut Builder, &[Value]) -> Result<Vec<Value>>,
    ) -> Result<Inst> {
        let region = self.new_region(args.clone());
        let arg_values = self.args(region).to_vec();
        let mut body_builder = Builder::at_end(self, region);
        let returned = body(&mut body_builder, &arg_values)?;
        body_builder.ret(&returned);
        let results = returned
            .iter()
            .map(|&value| self.ty(value).clone())
            .collect();
        let func = Func {
            name: name.to_string(),
            args,
            results,
            params: any_items(params),
            result_params,
        };
        let top = self.body();
        let mut builder = Builder::at_end(self, top);
        Ok(builder.create_op(func.into(), Vec::new(), None, vec![region]))
    }
}

impl Build for Builder<'_> {
    fn create_with_regions<T: InOp>(
        &mut self,
        op: T,
        operands: impl Into<Operands<T, Values>>,
        body: impl FnMut(usize, &mut Builder, &[Value]),
    ) -> Created<T> {
        Builder::create_with_regions(self, op, operands, body)
    }

    /// Folds by the op's own folder and by its traits, since the operands are in the module.
    #[track_caller]
    fn create_or_fold_op(
        &mut self,
        op: Op,
        operands: Vec<Value>,
        segments: Option<Vec<usize>>,
    ) -> (Vec<Value>, Option<Vec<usize>>) {
        let (inferred, result_segments) = self
            .module
            .infer_results(&op, &operands, segments.as_deref(), &[], self.rewritten())
            .unwrap_or_else(|err| panic!("building {}: {err:#}", op.name()));
        let types = inferred.into_vec();
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

    #[track_caller]
    fn create<T: InOp>(&mut self, op: T, operands: impl Into<Operands<T, Values>>) -> Created<T> {
        Builder::create(self, op, operands)
    }

    fn value_type(&self, value: Value) -> &Type {
        self.module.ty(value)
    }
}
