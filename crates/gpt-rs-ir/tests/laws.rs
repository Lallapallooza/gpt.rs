//! Checks every law an op's traits claim against the op's `Eval`.
//!
//! The op instances come from the roundtrip filetests. The test checks each instance with its
//! operands of one tensor type in each of `DTYPES` that the op accepts. Both sides of every
//! law of every trait must agree, as `gpt_rs_filetests::bits_eq` compares them, on random inputs
//! that start with the special values of `Specials::All`.

use std::fs;

use gpt_rs_filetests::{
    bits_eq, collect, filetests_root, random_value, rotation, run_dims, split_cases, Specials,
    ROUNDS,
};
use gpt_rs_ir::dense::Dense;
use gpt_rs_ir::interp;
use gpt_rs_ir::module::{Module, Value};
use gpt_rs_ir::op::OpDyn;
use gpt_rs_ir::rewrite::Builder;
use gpt_rs_ir::traits::{Law, Term};
use gpt_rs_ir::types::{DType, Dim, TensorType, Type};
use gpt_rs_ir::Op;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Each op of the roundtrip filetests once, with its operand types.
fn instances() -> Vec<(Op, Vec<Type>)> {
    let mut files = Vec::new();
    collect(&filetests_root(), &mut files);
    let mut found: Vec<(Op, Vec<Type>)> = Vec::new();
    let roundtrip = files
        .iter()
        .filter(|p| p.parent().is_some_and(|d| d.ends_with("roundtrip")));
    for path in roundtrip {
        let text = fs::read_to_string(path).expect("read filetest");
        for case in split_cases(&text).1 {
            let module = Module::from_text(&case.text)
                .unwrap_or_else(|err| panic!("{}:{}: {err:#}", path.display(), case.line));
            for inst in module.walk(module.body()) {
                let op = module.op(inst);
                if found.iter().all(|(seen, _)| seen != op) {
                    let types = module.operand_types(inst).into_iter().cloned().collect();
                    found.push((op.clone(), types));
                }
            }
        }
    }
    found
}

/// Builds `term` of a law of `op`, with the law's variables `vars`, all of type `ty`.
fn build(b: &mut Builder<'_>, op: &Op, ty: &TensorType, vars: &[Value], term: &Term) -> Value {
    let (head, args) = match term {
        Term::Var(index) => return vars[*index],
        Term::Splat(element) => {
            let dims = ty.static_dims().expect("static");
            return b
                .constant(Dense::splat(ty.dtype, dims, element))
                .expect("a constant");
        }
        Term::Apply(args) => (op.clone(), args),
        Term::Other(other, args) => ((**other).clone(), args),
    };
    let operands = args.iter().map(|t| build(b, op, ty, vars, t)).collect();
    let inst = b
        .build_op(head, operands, None, Vec::new())
        .expect("the law's terms build");
    b.module.results(inst)[0]
}

const DTYPES: [DType; 8] = [
    DType::F32,
    DType::Bf16,
    DType::F64,
    DType::Si8,
    DType::Ui8,
    DType::Si32,
    DType::Ui64,
    DType::I1,
];

/// The number of variables a law may use.
const VARS: usize = 3;

/// Checks `law` of `op` with variables of type `ty`.
fn check(op: &Op, ty: &TensorType, law: &Law) -> Result<(), String> {
    let mut module = Module::new();
    module
        .build_func(
            "law",
            vec![Type::Tensor(ty.clone()); VARS],
            Vec::new(),
            |b, args| {
                let lhs = build(b, op, ty, args, &law.lhs);
                let rhs = build(b, op, ty, args, &law.rhs);
                Ok(vec![lhs, rhs])
            },
        )
        .map_err(|err| format!("{err:#}"))?;
    for round in 0..ROUNDS {
        let mut rng = StdRng::seed_from_u64(round as u64);
        let args = (0..VARS)
            .map(|input| {
                random_value(
                    &Type::Tensor(ty.clone()),
                    rotation(input, round),
                    Specials::All,
                    &mut rng,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let results =
            interp::run(&module, "law", args.clone()).map_err(|err| format!("{err:#}"))?;
        if !bits_eq(&results[0], &results[1]) {
            return Err(format!(
                "fails on {args:?}: {:?} is not {:?}",
                results[0], results[1]
            ));
        }
    }
    Ok(())
}

#[test]
fn laws() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for (op, types) in instances() {
        let Some(first) = types.first().and_then(Type::as_tensor) else {
            continue;
        };
        for dtype in DTYPES {
            let dims = run_dims(&first.dims).into_iter().map(Dim::Static);
            let ty = TensorType::new(dtype, dims);
            let operands = vec![Type::Tensor(ty.clone()); types.len()];
            let refs: Vec<&Type> = operands.iter().collect();
            if op
                .dyn_result_types(&refs, None, &[], &Module::new())
                .is_err()
            {
                continue;
            }
            for t in op.traits().iter() {
                for law in t.laws(types.len(), dtype) {
                    checked += 1;
                    if let Err(err) = check(&op, &ty, &law) {
                        failures.push(format!(
                            "{} on {dtype}: {:?}: {}: {err}",
                            op.name(),
                            t,
                            law.name
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no laws found");
    assert!(
        failures.is_empty(),
        "{} of {checked} laws fail:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
