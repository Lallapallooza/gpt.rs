//! The signature of `loop.func` as a call with tensors sees it, the bounds of its loops, and the
//! ranges and the affine forms of its indices.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use super::ir::*;
use crate::ir::{AtOrigin, BinaryOp, IndexExpr, Inst, Module, Root, TensorType, Value, ValueDef};

impl Access {
    /// Whether a call passes the parameter an argument: an `in` or an `inout` parameter.
    pub fn takes_argument(self) -> bool {
        matches!(self, Access::In | Access::Inout)
    }

    /// Whether a call gets a result of the parameter: an `out` or an `inout` parameter.
    pub fn gives_result(self) -> bool {
        matches!(self, Access::Out { .. } | Access::Inout)
    }
}

impl Func {
    /// The tensor types of the arguments and of the results of a call with tensors: those of the
    /// parameters that take an argument, and those of the parameters that give a result, each in
    /// order.
    pub fn call_types(&self) -> Result<(Vec<TensorType>, Vec<TensorType>)> {
        let mut args = Vec::new();
        let mut results = Vec::new();
        for (index, (ty, &access)) in self.args.iter().zip(&self.access).enumerate() {
            let buffer = ty
                .as_buffer()
                .ok_or_else(|| anyhow!("parameter {index} of @{} is not a buffer", self.name))?;
            if access.takes_argument() {
                args.push(buffer.tensor_type());
            }
            if access.gives_result() {
                results.push(buffer.tensor_type());
            }
        }
        Ok((args, results))
    }
}

/// The start and the end of the `loop.for` or the `loop.forall` of `inst`, when it is one.
pub(crate) fn loop_bounds(module: &Module, inst: Inst) -> Option<(Value, Value)> {
    if let Some(looped) = module.op_as::<For>(inst) {
        return Some((looped.start().id(), looped.end().id()));
    }
    let looped = module.op_as::<Forall>(inst)?;
    Some((looped.start().id(), looped.end().id()))
}

/// The least and the greatest value of each index that the loops around it bound: a constant, an
/// expression of bounded indices, the variable of a `loop.for` or a `loop.forall` whose start and
/// end are bounded, or a coordinate of a `loop.atom`, below its tile. Each range is computed once.
pub struct IndexRanges<'m> {
    module: &'m Module,
    known: HashMap<Value, Option<(i64, i64)>>,
    forms: AffineForms<'m>,
}

/// The step budget of [`IndexRanges::least`]: each `min` or `max` that it takes apart doubles the
/// forms it bounds.
const LEAST_STEPS: usize = 256;

impl<'m> IndexRanges<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self {
            module,
            known: HashMap::new(),
            forms: AffineForms::new(module),
        }
    }

    /// The affine form of the index `value` ([`AffineForms`]).
    pub fn form(&mut self, value: Value) -> Affine {
        self.forms.of(value)
    }

    /// The least value that the index `form` takes at any point where the loops around its
    /// values run, when the loops bound it. `least` finds it in three steps:
    /// 1. Take apart each `min` and `max` among its values, and add the rest of the form to each
    ///    side.
    /// 2. Put the variable of the innermost loop at the bound of its loop that lowers the form.
    ///    When that bound names the variable of an outer loop, the form keeps that variable, so
    ///    that its terms can cancel against those already in the form.
    /// 3. Put each other value at the end of its range that lowers the form.
    pub fn least(&mut self, form: &Affine) -> Option<i64> {
        let mut budget = LEAST_STEPS;
        self.least_within(form, &mut budget)
    }

    fn least_within(&mut self, form: &Affine, budget: &mut usize) -> Option<i64> {
        *budget = budget.checked_sub(1)?;
        for &(value, coeff) in &form.terms {
            let Some((op, sides)) = self.extremum(value) else {
                continue;
            };
            let mut rest = form.clone();
            rest.add_scaled(&Affine::opaque(value), -coeff);
            let mut least_sides = [0; 2];
            for (least_side, side) in least_sides.iter_mut().zip(&sides) {
                let mut taken = rest.clone();
                taken.add_scaled(side, coeff);
                *least_side = self.least_within(&taken, budget)?;
            }
            // A positive multiple of the larger side is at least the larger of the two, and a
            // negative one at least the smaller.
            let larger = (op == BinaryOp::Max) == (coeff > 0);
            return Some(match larger {
                true => least_sides[0].max(least_sides[1]),
                false => least_sides[0].min(least_sides[1]),
            });
        }
        let mut innermost: Option<(usize, Value, i64)> = None;
        for &(value, coeff) in &form.terms {
            let Some(depth) = self.loop_depth(value) else {
                continue;
            };
            if innermost.is_none_or(|(deepest, _, _)| depth > deepest) {
                innermost = Some((depth, value, coeff));
            }
        }
        if let Some((_, value, coeff)) = innermost {
            let ValueDef::Arg(region, _) = self.module.def(value) else {
                return None;
            };
            let (start, end) = loop_bounds(self.module, self.module.parent_op(region)?)?;
            let mut bounded = form.clone();
            bounded.add_scaled(&Affine::opaque(value), -coeff);
            match coeff > 0 {
                true => bounded.add_scaled(&self.forms.of(start), coeff),
                false => {
                    // The variable is at most its end less one.
                    bounded.add_scaled(&self.forms.of(end), coeff);
                    bounded.constant -= coeff;
                }
            }
            return self.least_within(&bounded, budget);
        }
        let mut least = form.constant;
        for &(value, coeff) in &form.terms {
            let (low, high) = self.of(value)?;
            least += match coeff > 0 {
                true => coeff * low,
                false => coeff * high,
            };
        }
        Some(least)
    }

    /// The `min` or `max` that the index `value` is, and its two sides as affine forms, when it is
    /// a `loop.index` whose expression is a `min` or a `max` of two affine sides.
    fn extremum(&mut self, value: Value) -> Option<(BinaryOp, [Affine; 2])> {
        let index = self.module.value_ref(value).defining_op::<Index>()?;
        let Root::Binary(op @ (BinaryOp::Min | BinaryOp::Max), lhs, rhs) = index.expr.root() else {
            return None;
        };
        let mut operand_forms = Vec::new();
        for operand in index.ins() {
            operand_forms.push(self.forms.of(operand.id()));
        }
        let sides = [
            Affine::of_expr(&lhs, &operand_forms)?,
            Affine::of_expr(&rhs, &operand_forms)?,
        ];
        Some((op, sides))
    }

    /// The depth of the loop whose variable is `value`: the number of ops around the loop, the loop
    /// included. None when `value` is not the variable of a `loop.for` or a `loop.forall`.
    fn loop_depth(&self, value: Value) -> Option<usize> {
        let module = self.module;
        let ValueDef::Arg(region, _) = module.def(value) else {
            return None;
        };
        let looped = module
            .parent_op(region)
            .filter(|&looped| loop_bounds(module, looped).is_some())?;
        let mut depth = 1;
        let mut around = module.parent_region(looped);
        while let Some(outer) = around.and_then(|parent| module.parent_op(parent)) {
            depth += 1;
            around = module.parent_region(outer);
        }
        Some(depth)
    }

    /// The range of the index `value`, when the loops around it bound it.
    pub fn of(&mut self, value: Value) -> Option<(i64, i64)> {
        if let Some(&known) = self.known.get(&value) {
            return known;
        }
        let range = self.compute(value);
        self.known.insert(value, range);
        range
    }

    fn compute(&mut self, value: Value) -> Option<(i64, i64)> {
        let module = self.module;
        match module.def(value) {
            ValueDef::Result(inst, 0) => {
                let index = module.op_as::<Index>(inst)?;
                let mut ranges = Vec::new();
                for operand in index.ins() {
                    ranges.push(self.of(operand.id()));
                }
                index.expr.bounds(&|position| ranges[position])
            }
            ValueDef::Arg(region, axis) => {
                let looped = module.parent_op(region)?;
                if let Some(atom) = module.op(looped).get::<Atom>() {
                    return Some((0, *atom.tile.get(axis)? as i64 - 1));
                }
                let (start, end) = loop_bounds(module, looped)?;
                let (lowest, _) = self.of(start)?;
                let (_, highest) = self.of(end)?;
                Some((lowest, highest - 1))
            }
            _ => None,
        }
    }
}

/// An index as a constant plus multiples of values that no index expression of others defines:
/// loop variables, parameters and values loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Affine {
    pub constant: i64,
    pub terms: Vec<(Value, i64)>,
}

impl Affine {
    /// `value` as a term of its own.
    pub fn opaque(value: Value) -> Self {
        Self {
            constant: 0,
            terms: vec![(value, 1)],
        }
    }

    /// The constant `value`.
    pub fn constant(value: i64) -> Self {
        Self {
            constant: value,
            terms: Vec::new(),
        }
    }

    /// The form times -1.
    pub fn negated(&self) -> Self {
        let mut negated = Affine::constant(0);
        negated.add_scaled(self, -1);
        negated
    }

    /// The form of `expr`, whose operands have the forms `operands`, when it is a constant plus
    /// multiples of them.
    pub fn of_expr(expr: &IndexExpr, operands: &[Affine]) -> Option<Self> {
        let mut form = Affine::constant(expr.eval(&AtOrigin).ok()?);
        for (position, operand) in operands.iter().enumerate() {
            let coeff = expr.coefficient(position)?;
            if coeff != 0 {
                form.add_scaled(operand, coeff);
            }
        }
        Some(form)
    }

    /// Adds `factor` times `other`.
    fn add_scaled(&mut self, other: &Affine, factor: i64) {
        self.constant += factor * other.constant;
        for &(value, coeff) in &other.terms {
            match self.terms.iter_mut().find(|(term, _)| *term == value) {
                Some((_, sum)) => *sum += factor * coeff,
                None => self.terms.push((value, factor * coeff)),
            }
        }
        self.terms.retain(|&(_, coeff)| coeff != 0);
    }

    /// The value of the index, when it is a constant.
    pub fn as_constant(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.constant)
    }
}

/// The affine form of each index: a `loop.index` whose expression is a constant plus multiples of
/// its operands is that sum of their forms, so that terms cancel, and any other value is a term of
/// its own. Each form is computed once.
pub struct AffineForms<'m> {
    module: &'m Module,
    known: HashMap<Value, Affine>,
}

impl<'m> AffineForms<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self {
            module,
            known: HashMap::new(),
        }
    }

    /// The affine form of the index `value`.
    pub fn of(&mut self, value: Value) -> Affine {
        if let Some(known) = self.known.get(&value) {
            return known.clone();
        }
        let form = self.compute(value).unwrap_or_else(|| Affine::opaque(value));
        self.known.insert(value, form.clone());
        form
    }

    fn compute(&mut self, value: Value) -> Option<Affine> {
        let index = self.module.value_ref(value).defining_op::<Index>()?;
        let mut operand_forms = Vec::new();
        for operand in index.ins() {
            operand_forms.push(self.of(operand.id()));
        }
        Affine::of_expr(&index.expr, &operand_forms)
    }
}
