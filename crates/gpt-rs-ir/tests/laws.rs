//! Checks every law an op's traits claim against the op's `Eval`, and every law a combine kind
//! claims against its fold.
//!
//! The op instances come from the roundtrip filetests. The test checks each instance with its
//! operands of one type in each of `DTYPES` that the op accepts: a tensor type for an op on
//! tensors, else a scalar type. Both sides of every law of every trait must agree, as
//! `gpt_rs_filetests::bits_eq` compares them, on random inputs that start with the special values
//! of `Specials::All`.

use std::fs;

use gpt_rs_filetests::{
    bits_eq, collect, filetests_root, random_value, rotation, run_dims, split_cases, Specials,
    ROUNDS,
};
use gpt_rs_ir::dialects::tile::{Acc, CombineKind, KBest};
use gpt_rs_ir::dialects::{arith, constant_op, tensor};
use gpt_rs_ir::host::{self, Decoded, HostScalar};
use gpt_rs_ir::interp::HostValue;
use gpt_rs_ir::ir::{DType, Dense, Dim, Module, OpDyn, Ranked, TensorType, Type, Value};
use gpt_rs_ir::rewrite::Builder;
use gpt_rs_ir::traits::{Law, Term};
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
        .filter(|file| file.parent().is_some_and(|dir| dir.ends_with("roundtrip")));
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
fn build(builder: &mut Builder, op: &Op, ty: &Type, vars: &[Value], term: &Term) -> Value {
    let (head, args) = match term {
        Term::Var(index) => return vars[*index],
        Term::Splat(element) => {
            let dtype = ty.dtype().expect("a tensor or a scalar");
            let dims = ty.as_tensor().map_or(Ranked::new(), |tensor| {
                tensor.static_dims().expect("static")
            });
            let constant = constant_op(Dense::splat(dtype, dims, element), ty).expect("a constant");
            let inst = builder.create_op(constant, Vec::new(), None, Vec::new());
            return builder.module.results(inst)[0];
        }
        Term::Apply(args) => (op.clone(), args.as_slice()),
        Term::Other(other, args) => ((**other).clone(), args.as_slice()),
        Term::Cast(to, operand) => {
            let cast: Op = match ty {
                Type::Scalar(_) => arith::Cast { to: *to }.into(),
                _ => tensor::Cast { to: *to }.into(),
            };
            (cast, std::slice::from_ref(&**operand))
        }
    };
    let operands = args
        .iter()
        .map(|subterm| build(builder, op, ty, vars, subterm))
        .collect();
    let inst = builder.create_op(head, operands, None, Vec::new());
    builder.module.results(inst)[0]
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

/// The values each variable of a law on scalars takes in a round, one run each.
const SCALAR_POINTS: usize = 8;

/// The type of the operands of an instance whose first operand has type `first`, of dtype
/// `dtype`: a tensor type of the extents a run gives `first` when it is a tensor, else a scalar.
fn operand_type(first: &Type, dtype: DType) -> Type {
    let Some(tensor) = first.as_tensor() else {
        return Type::Scalar(dtype);
    };
    let dims = run_dims(&tensor.dims).into_iter().map(Dim::Static);
    Type::Tensor(TensorType::new(dtype, dims))
}

/// The results of the function `law` of `module` on `args`, which hold the values of variables of
/// type `ty`: one run on tensors, or a run on each point of scalars, whose results it gathers into
/// tensors.
fn run_law(module: &Module, ty: &Type, args: &[HostValue]) -> Result<Vec<HostValue>, String> {
    if !matches!(ty, Type::Scalar(_)) {
        return interp::run(module, "law", args.to_vec()).map_err(|err| format!("{err:#}"));
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        values.push(Decoded::new(
            &arg.as_tensor().expect("a tensor of points").data,
        ));
    }
    let mut sides = [Vec::new(), Vec::new()];
    for point in 0..SCALAR_POINTS {
        let mut scalars = Vec::with_capacity(values.len());
        for value in &values {
            scalars.push(HostValue::Scalar(value.get(point)));
        }
        let results = interp::run(module, "law", scalars).map_err(|err| format!("{err:#}"))?;
        for (side, result) in sides.iter_mut().zip(results) {
            let HostValue::Scalar(scalar) = result else {
                return Err(format!("a law on scalars gives {result:?}"));
            };
            side.push(scalar);
        }
    }
    Ok(vec![
        as_tensor(sides[0][0].dtype(), &sides[0]),
        as_tensor(sides[1][0].dtype(), &sides[1]),
    ])
}

/// Checks `law` of `op` with variables of type `ty`.
fn check(op: &Op, ty: &Type, law: &Law) -> Result<(), String> {
    let input_type = match ty {
        Type::Scalar(dtype) => Type::Tensor(TensorType::new(*dtype, [SCALAR_POINTS])),
        _ => ty.clone(),
    };
    let mut module = Module::new();
    module
        .build_func(
            "law",
            vec![ty.clone(); VARS],
            Vec::new(),
            Vec::new(),
            |builder, args| {
                let lhs = build(builder, op, ty, args, &law.lhs);
                let rhs = build(builder, op, ty, args, &law.rhs);
                Ok(vec![lhs, rhs])
            },
        )
        .map_err(|err| format!("{err:#}"))?;
    for round in 0..ROUNDS {
        let mut rng = StdRng::seed_from_u64(round as u64);
        let args = (0..VARS)
            .map(|input| random_value(&input_type, rotation(input, round), Specials::All, &mut rng))
            .collect::<Result<Vec<_>, _>>()?;
        let results = run_law(&module, ty, &args)?;
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
        let Some(first) = types.first() else {
            continue;
        };
        for dtype in DTYPES {
            let ty = operand_type(first, dtype);
            let operands = vec![ty.clone(); types.len()];
            let refs: Vec<&Type> = operands.iter().collect();
            if op
                .dyn_result_types(&refs, None, &[], &Module::new(), None)
                .is_err()
            {
                continue;
            }
            for op_trait in op.traits().iter() {
                for law in op_trait.laws(types.len(), dtype) {
                    checked += 1;
                    if let Err(err) = check(&op, &ty, &law) {
                        failures.push(format!(
                            "{} on {dtype}: {:?}: {}: {err}",
                            op.name(),
                            op_trait,
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
    let &[Dim::Static(kept)] = &appended[..] else {
        return vec![results];
    };
    let mut points = Vec::new();
    for rank in 0..kept {
        let point = vec![results[rank], results[kept + rank]];
        if point[1].as_int() != Some(-1) {
            points.push(point);
        }
    }
    points
}

/// `scalars` as a tensor of `dtype`, which `bits_eq` compares.
fn as_tensor(dtype: DType, scalars: &[HostScalar]) -> HostValue {
    HostValue::Tensor(
        host::tensor(&TensorType::new(dtype, [scalars.len()]), scalars).expect("a tensor"),
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
                    .map(|point_index| {
                        let mut point = vec![values.get(point_index)];
                        if dtypes.len() == 2 {
                            point.push(
                                HostScalar::int(index, point_index as i128).expect("an index"),
                            );
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
                        && (0..dtypes.len()).all(|result_index| {
                            let results_per_dtype = expected.len() / dtypes.len();
                            let range = result_index * results_per_dtype
                                ..(result_index + 1) * results_per_dtype;
                            bits_eq(
                                &as_tensor(dtypes[result_index], &expected[range.clone()]),
                                &as_tensor(dtypes[result_index], &got[range]),
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
