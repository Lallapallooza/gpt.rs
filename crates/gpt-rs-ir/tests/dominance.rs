//! Where values are defined, as `Module::dominates` answers, and the order of instructions that
//! `Module::is_before` keeps as a builder inserts them.

use gpt_rs_ir::dialects::tensor::Constant;
use gpt_rs_ir::ir::{DType, Dense, Inst, Module, Region};
use gpt_rs_ir::rewrite::Builder;

const TEXT: &str =
    "func.func @f(%0: tensor<i1>, %1: tensor<si32>) -> (tensor<si32>, tensor<si32>) {
  %2 = tensor.neg(%1) : (tensor<si32>) -> tensor<si32>
  %3 = tensor.cond(%0, %2) {results = [tensor<si32>]} ({
  ^then_branch(%4: tensor<si32>):
    %5 = tensor.add(%4, %2) : (tensor<si32>, tensor<si32>) -> tensor<si32>
    tensor.yield(%5) : (tensor<si32>) -> ()
  }, {
  ^else_branch(%6: tensor<si32>):
    tensor.yield(%6) : (tensor<si32>) -> ()
  }) : (tensor<i1>, tensor<si32>) -> tensor<si32>
  %7 = tensor.abs(%3) : (tensor<si32>) -> tensor<si32>
  func.return(%2, %7) : (tensor<si32>, tensor<si32>) -> ()
}

func.func @g(%0: tensor<si32>) -> tensor<si32> {
  func.return(%0) : (tensor<si32>) -> ()
}";

fn insts(module: &Module, region: Region) -> Vec<Inst> {
    module.insts(region).collect()
}

#[test]
fn values_dominate_what_comes_after_them_in_the_regions_they_hold() {
    let module = Module::from_text(TEXT).expect("the case parses");
    let funcs = module.funcs();
    let [func_f, func_g] = &funcs[..] else {
        panic!("two functions")
    };
    let body = module.regions(func_f)[0];
    let &[neg, cond, abs, ret] = &insts(&module, body)[..] else {
        panic!("four ops")
    };
    let then_branch = module.regions(cond)[0];
    let else_branch = module.regions(cond)[1];
    let &[add, then_yield] = &insts(&module, then_branch)[..] else {
        panic!("two ops")
    };
    let else_yield = insts(&module, else_branch)[0];
    let (pred, negated) = (module.args(body)[0], module.results(neg)[0]);
    let (branch_arg, sum, cond_result) = (
        module.args(then_branch)[0],
        module.results(add)[0],
        module.results(cond)[0],
    );

    // A function's argument and an earlier result reach into the regions of later ops.
    assert!(module.dominates(pred, add));
    assert!(module.dominates(negated, add));
    assert!(module.dominates(negated, ret));
    // A result is not defined before its own op, nor inside its regions, nor before it.
    assert!(!module.dominates(negated, neg));
    assert!(!module.dominates(cond_result, add));
    assert!(module.dominates(cond_result, abs));
    // A branch's values stay in the branch.
    assert!(module.dominates(branch_arg, then_yield));
    assert!(module.dominates(sum, then_yield));
    assert!(!module.dominates(sum, abs));
    assert!(!module.dominates(branch_arg, else_yield));
    // A function, isolated from above, sees nothing of another.
    let g_return = insts(&module, module.regions(func_g)[0])[0];
    assert!(!module.dominates(negated, g_return));
    assert!(!module.dominates(module.args(module.regions(func_g)[0])[0], ret));
}

#[test]
fn inserts_keep_the_order_of_the_region() {
    let mut module = Module::from_text(TEXT).expect("the case parses");
    let func_f = module.funcs()[0].inst();
    let body = module.regions(func_f)[0];
    let &[neg, cond, abs, ret] = &insts(&module, body)[..] else {
        panic!("four ops")
    };
    assert!(module.is_before(neg, cond) && module.is_before(abs, ret));
    // Inserting before the same instruction again and again halves the room between the keys of
    // the last insert and that instruction until none is left. The region then gets new keys, in
    // the same order.
    let mut inserted = Vec::new();
    for _ in 0..40 {
        let value = Dense {
            dtype: DType::Si32,
            dims: Vec::new(),
            bytes: vec![0u8; 4].into(),
        };
        let constant = Builder::before(&mut module, cond).create(Constant { value }, ());
        inserted.push(module.def_inst(constant).expect("a constant's result"));
    }
    let order = insts(&module, body);
    assert_eq!(order, [&[neg][..], &inserted, &[cond, abs, ret]].concat());
    for pair in order.windows(2) {
        assert!(module.is_before(pair[0], pair[1]), "{pair:?} out of order");
    }
}
