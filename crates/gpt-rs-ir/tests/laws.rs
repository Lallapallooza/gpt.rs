//! Checks every law an op's traits claim against the op's `Eval`, and every law a combine kind
//! claims against its fold.
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
use gpt_rs_ir::dialects::tensor::Constant;
use gpt_rs_ir::dialects::tile::{Acc, CombineKind, KBest};
use gpt_rs_ir::host::{HostScalar, HostValue};
use gpt_rs_ir::module::{Module, Value};
use gpt_rs_ir::op::OpDyn;
use gpt_rs_ir::rewrite::Builder;
use gpt_rs_ir::scalar::{self, Decoded};
use gpt_rs_ir::traits::{Law, Term};
use gpt_rs_ir::types::{DType, Dim, TensorType, Type};
use gpt_rs_ir::{interp, Op};
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
            let value = Dense::splat(ty.dtype, dims, element);
            return b.create(Constant { value }, ());
        }
        Term::Apply(args) => (op.clone(), args),
        Term::Other(other, args) => ((**other).clone(), args),
    };
    let operands = args.iter().map(|t| build(b, op, ty, vars, t)).collect();
    let inst = b.create_op(head, operands, None, Vec::new());
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

/// The combine kinds whose folds the laws below check.
fn combine_kinds() -> Vec<CombineKind> {
    let mut kinds = vec![
        CombineKind::Sum,
        CombineKind::Max,
        CombineKind::Min,
        CombineKind::And,
        CombineKind::Or,
        CombineKind::Argmax,
        CombineKind::Argmin,
    ];
    kinds.push(CombineKind::TopK(KBest { k: 3 }));
    kinds.push(CombineKind::BottomK(KBest { k: 3 }));
    kinds
}

/// The results of folding `points` with `kind`, from its identity, in their order.
fn fold(kind: &CombineKind, dtypes: &[DType], points: &[Vec<HostScalar>]) -> Vec<HostScalar> {
    let mut acc = Acc::new(kind, dtypes, &[]).expect("an accumulator");
    for point in points {
        acc.push(point).expect("a fold step");
    }
    acc.finish().expect("finished results")
}

/// The points of a fold result. A `top_k` or `bottom_k` result is `k` values then `k` indices:
/// its points are its `k` (value, index) pairs, without the padding at index -1 that a fold of
/// fewer than `k` points adds. Any other result is its own point.
fn as_points(kind: &CombineKind, results: Vec<HostScalar>) -> Vec<Vec<HostScalar>> {
    let appended = kind.rule().extra_dims(0);
    let &[Dim::Static(k)] = &appended[..] else {
        return vec![results];
    };
    let mut points = Vec::new();
    for rank in 0..k {
        let point = vec![results[rank], results[k + rank]];
        if point[1].as_int() != Some(-1) {
            points.push(point);
        }
    }
    points
}

/// `scalars` as a tensor of `dtype`, which `bits_eq` compares.
fn as_tensor(dtype: DType, scalars: &[HostScalar]) -> HostValue {
    HostValue::Tensor(
        scalar::tensor(&TensorType::new(dtype, [scalars.len()]), scalars).expect("a tensor"),
    )
}

/// Checks the laws of every combine kind wherever `exact` claims them. Folding the points in any
/// order, or folding chunks of them and then the chunks' results, gives the same bits. A fold of
/// no points gives the identity of its reduction.
#[test]
fn combine_laws() {
    const POINTS: usize = 11;
    let mut failures = Vec::new();
    let mut checked = 0;
    for kind in combine_kinds() {
        for dtype in DTYPES {
            let index = DType::Si64;
            let dtypes: Vec<DType> = match kind.rule().yields() {
                2 => vec![dtype, index],
                _ => vec![dtype],
            };
            if kind.rule().result_dtypes(&dtypes).is_err() || !kind.rule().exact(dtype) {
                continue;
            }
            checked += 1;
            for round in 0..ROUNDS {
                let mut rng = StdRng::seed_from_u64(round as u64);
                let ty = Type::Tensor(TensorType::new(dtype, [POINTS]));
                let value = random_value(&ty, rotation(0, round), Specials::All, &mut rng)
                    .expect("random values");
                let values = Decoded::new(&value.as_tensor().expect("a tensor").data);
                let points: Vec<Vec<HostScalar>> = (0..POINTS)
                    .map(|i| {
                        let mut point = vec![values.get(i)];
                        if dtypes.len() == 2 {
                            point.push(HostScalar::int(index, i as i128).expect("an index"));
                        }
                        point
                    })
                    .collect();
                let expected = fold(&kind, &dtypes, &points);
                let reversed: Vec<_> = points.iter().rev().cloned().collect();
                let rotated: Vec<_> = points
                    .iter()
                    .cycle()
                    .skip(round % POINTS)
                    .take(POINTS)
                    .cloned()
                    .collect();
                let mut chunked = Vec::new();
                for chunk in points.chunks(1 + round % 4) {
                    chunked.extend(as_points(&kind, fold(&kind, &dtypes, chunk)));
                }
                for (how, order) in [
                    ("reversed", reversed),
                    ("rotated", rotated),
                    ("chunked", chunked),
                ] {
                    let got = fold(&kind, &dtypes, &order);
                    let same = expected.len() == got.len()
                        && (0..dtypes.len()).all(|r| {
                            let n = expected.len() / dtypes.len();
                            let range = r * n..(r + 1) * n;
                            bits_eq(
                                &as_tensor(dtypes[r], &expected[range.clone()]),
                                &as_tensor(dtypes[r], &got[range]),
                            )
                        });
                    if !same {
                        failures.push(format!(
                            "{kind} on {dtype}, {how}, round {round}: {got:?} is not {expected:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 0, "no combine laws found");
    assert!(
        failures.is_empty(),
        "{} combine laws fail:\n\n{}",
        failures.len(),
        failures.join("\n")
    );
}
