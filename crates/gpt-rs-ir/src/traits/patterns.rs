//! Patterns rooted at a trait: rewrites that follow from the trait, for every op that has it.

use super::{Associative, Commutative, DistributesOver};
use crate::ir::{DType, Value, ValueRef};
use crate::require;
use crate::rewrite::{pattern, Has, NoMatch, Pattern, Rewriter, WithTrait};

/// `f(f(x, a), b)` is `f(x, f(a, b))` when `f` is associative, `a` and `b` are static, and one of
/// them reads a parameter. The backend then computes `f(a, b)` once at load time, so a run
/// computes one `f` where it computed two.
#[pattern(canonical)]
struct GroupStatic;

impl Pattern for GroupStatic {
    type Root = Has<Associative>;

    fn rewrite(
        &self,
        outer: WithTrait<'_, Associative>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        require!(let &[inner_result, outer_static] = outer.operands.as_slice(), "the op takes another number of operands");
        let associative =
            operands_dtype(&outer.operands).is_some_and(|dtype| outer.fact.covers(dtype));
        require!(associative, "the op is not associative for its dtype");
        require!(let Some(inner) = inner_result.defining_inst(), "the lhs is an argument");
        let module = inner_result.module();
        require!(
            module.op(inner) == outer.op,
            "the lhs comes from another op"
        );
        require!(let &[dynamic, inner_static] = module.operands(inner), "the inner op takes another number of operands");
        let (dynamic, inner_static) = (module.value_ref(dynamic), module.value_ref(inner_static));
        require!(inner_result.has_one_use(), "the inner op has other users");
        require!(
            inner_static.known_before_run(),
            "the inner operand changes at run time"
        );
        require!(
            outer_static.known_before_run(),
            "the outer operand changes at run time"
        );
        require!(
            inner_static.depends_on_parameter() || outer_static.depends_on_parameter(),
            "neither operand depends on a parameter"
        );
        require!(
            !dynamic.known_before_run(),
            "the whole expression is known before the run"
        );
        let grouped = rewriter.create_op(outer.op.clone(), [inner_static, outer_static])[0];
        let result = rewriter.create_op(outer.op.clone(), [dynamic.id(), grouped])[0];
        rewriter.replace_op(&outer, result);
        Ok(())
    }
}

/// `g(f(a, b), f(a, c))` is `f(a, g(b, c))` when `f` distributes over `g`, which saves an `f`.
/// For a commutative `f` the shared operand may be on either side. The pattern is rooted at either
/// `f` and rewrites its only use, `g`.
#[pattern(canonical)]
struct Factor;

/// The operand that `f(u)` and `f(v)` share and their other operands, `a` and `(b, c)` for
/// `f(a, b)` and `f(a, c)`.
fn shared(u: &[Value], v: &[Value], commutative: bool) -> Option<(Value, Value, Value)> {
    let (&[u0, u1], &[v0, v1]) = (u, v) else {
        return None;
    };
    let orders: &[(Value, Value, Value, Value)] = if commutative {
        &[
            (u0, u1, v0, v1),
            (u0, u1, v1, v0),
            (u1, u0, v0, v1),
            (u1, u0, v1, v0),
        ]
    } else {
        &[(u0, u1, v0, v1)]
    };
    orders
        .iter()
        .find(|(a, _, a2, _)| a == a2)
        .map(|&(a, b, _, c)| (a, b, c))
}

impl Pattern for Factor {
    type Root = Has<DistributesOver>;

    fn rewrite(
        &self,
        first: WithTrait<'_, DistributesOver>,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), NoMatch> {
        require!(let &[first_result] = first.results.as_slice(), "the op gives another number of results");
        let distributes =
            operands_dtype(&first.operands).is_some_and(|dtype| first.fact.covers(dtype));
        require!(distributes, "the op does not distribute for its dtype");
        let module = first_result.module();
        require!(let [only_use] = first_result.uses(), "the op has another number of uses");
        let combine = only_use.inst;
        require!(
            *module.op(combine) == (first.fact.over)(),
            "its user is not the op it distributes over"
        );
        require!(let &[combine_lhs, combine_rhs] = module.operands(combine), "the user takes another number of operands");
        let first_on_left = combine_lhs == first_result.id();
        let other = module.value_ref(if first_on_left {
            combine_rhs
        } else {
            combine_lhs
        });
        require!(
            other != first_result,
            "the user combines the op with itself"
        );
        require!(let Some(second) = other.defining_inst(), "the other operand is an argument");
        require!(
            module.op(second) == first.op,
            "the other operand comes from another op"
        );
        require!(other.has_one_use(), "the other op has other users");
        let commutative = first.op.has_trait::<Commutative>();
        let first_operands: Vec<Value> =
            first.operands.iter().map(|operand| operand.id()).collect();
        require!(
            let Some((shared_operand, first_rest, second_rest)) =
                shared(&first_operands, module.operands(second), commutative),
            "the two ops share no operand"
        );
        // The user keeps its operand order: `g(f(a, c), f(a, b))` becomes `f(a, g(c, b))`.
        let rests = if first_on_left {
            [first_rest, second_rest]
        } else {
            [second_rest, first_rest]
        };
        rewriter.set_insertion_point(combine);
        let combined = rewriter.create_op((first.fact.over)(), rests)[0];
        let result = rewriter.create_op(first.op.clone(), [shared_operand, combined])[0];
        rewriter.replace_op(combine, result);
        Ok(())
    }
}

/// The dtype of the first of `operands`.
fn operands_dtype(operands: &[ValueRef<'_>]) -> Option<DType> {
    operands.first()?.value_type().dtype()
}
