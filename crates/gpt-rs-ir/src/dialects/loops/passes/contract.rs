//! One rounding for a product and the sum it enters, where the function allows it.

use anyhow::Result;

use crate::dialects::arith::{self, Add, AddOp, Fma};
use crate::dialects::loops::Func;
use crate::ir::{AsInst, Module};
use crate::passes::{pass, run_patterns, Pass};
use crate::require;
use crate::rewrite::{pattern, NoMatch, Pattern, PatternSet, Rewriter, Rewrites, Rewriting};

/// In each `loop.func` that contracts, an `arith.add` of an f32 or f64 `arith.mul` that nothing
/// else takes is one `arith.fma`, which rounds the product and the sum once: the product the add
/// takes second, else the first.
#[pass(
    name = "contract",
    stats(contracted = "products and sums written as one arith.fma")
)]
pub struct Contract;

/// Whether the `loop.func` around `op` contracts.
fn contracts(module: &Module, op: impl AsInst) -> bool {
    let mut inst = op.inst();
    while let Some(region) = module.parent_region(inst) {
        let Some(parent) = module.parent_op(region) else {
            return false;
        };
        if let Some(func) = module.op_as::<Func>(parent) {
            return func.contract;
        }
        inst = parent;
    }
    false
}

/// `arith.add(x, arith.mul(a, b))` or `arith.add(arith.mul(a, b), x)`, in a function that
/// contracts, is `arith.fma(a, b, x)`, when nothing else takes the product and it is f32 or f64:
/// the product the add takes second, when it takes two.
#[pattern]
struct FuseProduct;

impl Pattern for FuseProduct {
    type Root = Add;

    fn rewrite(&self, add: AddOp, rewriter: &mut Rewriter) -> Result<(), NoMatch> {
        let module = add.module();
        require!(
            contracts(module, &add),
            "the function rounds each product and each sum"
        );
        let (lhs, rhs) = (add.lhs().id(), add.rhs().id());
        let fused = match arith::lone_product(module, rhs) {
            Some(product) => Some((product, lhs)),
            None => arith::lone_product(module, lhs).map(|product| (product, rhs)),
        };
        require!(
            let Some((product, other)) = fused,
            "no operand is an f32 or f64 product that only the add takes"
        );
        let fma = rewriter.create(Fma, (product.lhs().id(), product.rhs().id(), other));
        rewriter.replace_op(&add, fma);
        Ok(())
    }
}

fn populate_contract_patterns(set: &mut PatternSet) {
    set.add(FuseProduct);
}

impl Pass for Contract {
    fn rewrites(&self) -> Rewrites {
        Rewrites::patterns(populate_contract_patterns)
    }

    fn run(&self, module: &mut Rewriting) -> Result<bool> {
        run_patterns(module, self.rewrites(), &Self::CONTRACTED)
    }
}
