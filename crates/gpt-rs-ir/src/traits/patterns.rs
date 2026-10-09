//! Patterns rooted at a trait: rewrites that follow from the trait, for every op that has it.

use anyhow::Result;

use super::{first_dtype, Associative, Commutative, DistributesOver};
use crate::module::Value;
use crate::rewrite::{pattern, Has, Pattern, Rewriter, WithTrait};

/// `f(f(x, a), b)` is `f(x, f(a, b))` when `f` is associative, `a` and `b` are static, and one of
/// them reads a parameter. The backend then computes `f(a, b)` once at load time, so a run
/// computes one `f` where it computed two.
#[pattern(canonical)]
struct GroupStatic;

impl Pattern for GroupStatic {
    type Root = Has<Associative>;
    fn rewrite(&self, root: WithTrait<Associative>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let module = &*rw.module;
        let &[inner, b] = module.operands(root.inst) else {
            return Ok(false);
        };
        let Some(def) = module.def_inst(inner).filter(|&d| *module.op(d) == root.op) else {
            return Ok(false);
        };
        let &[x, a] = module.operands(def) else {
            return Ok(false);
        };
        let applies = first_dtype(module, module.operands(root.inst))
            .is_some_and(|d| root.fact.covers(d))
            && module.uses(inner).len() == 1
            && module.is_static(a)
            && module.is_static(b)
            && (module.reads_param(a) || module.reads_param(b))
            && !module.is_static(x);
        if !applies {
            return Ok(false);
        }
        let ab = rw.insert_op_before(root.inst, root.op.clone(), vec![a, b])?;
        let ab = rw.module.results(ab)[0];
        let out = rw.insert_op_before(root.inst, root.op.clone(), vec![x, ab])?;
        let out = rw.module.results(out)[0];
        rw.replace(root.inst, &[out])?;
        Ok(true)
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
    fn rewrite(&self, root: WithTrait<DistributesOver>, rw: &mut Rewriter<'_>) -> Result<bool> {
        let module = &*rw.module;
        let &[u] = module.results(root.inst) else {
            return Ok(false);
        };
        let [only_use] = module.uses(u) else {
            return Ok(false);
        };
        let g = only_use.inst;
        if *module.op(g) != (root.fact.over)() {
            return Ok(false);
        }
        let &[g_lhs, g_rhs] = module.operands(g) else {
            return Ok(false);
        };
        let v = if g_lhs == u { g_rhs } else { g_lhs };
        let Some(v_def) = module.def_inst(v).filter(|&d| *module.op(d) == root.op) else {
            return Ok(false);
        };
        let covered =
            first_dtype(module, module.operands(root.inst)).is_some_and(|d| root.fact.covers(d));
        if !covered || v == u || module.uses(v).len() != 1 {
            return Ok(false);
        }
        let commutative = root.op.has_trait::<Commutative>();
        let Some((a, b, c)) = shared(
            module.operands(root.inst),
            module.operands(v_def),
            commutative,
        ) else {
            return Ok(false);
        };
        // `g` keeps its operand order: `g(f(a, c), f(a, b))` becomes `f(a, g(c, b))`.
        let terms = if g_lhs == u { vec![b, c] } else { vec![c, b] };
        let g_bc = rw.insert_op_before(g, (root.fact.over)(), terms)?;
        let g_bc = rw.module.results(g_bc)[0];
        let out = rw.insert_op_before(g, root.op.clone(), vec![a, g_bc])?;
        let out = rw.module.results(out)[0];
        rw.replace(g, &[out])?;
        Ok(true)
    }
}
