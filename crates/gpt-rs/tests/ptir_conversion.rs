use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use gpt_rs::backend::conversion::{
    check_program_legality, default_entrypoint_name, get_conversion_target, hash_program,
    plan_buffers, plan_buffers_with, register_conversion_target, sanitize_symbol, walk_program,
    BufferizeOptions, ConversionCache, ConversionCacheKey, ConversionOptions, ConversionTarget,
    ConvertedEntrypoint, ConvertedIr, LegalitySpec, OperationKind, ProgramVisitor,
};
use gpt_rs::backend::ptir_utils::{tensor_spec_static, value_type_tensor};
use gpt_rs::backend::spec::{
    DType, DimSymbol, Dimension, Instruction, Operand, Operation, Program, ProgramBuilder, Region,
    RegionId, Shape, TensorSpec, TopKSpec, ValueId, ValueType,
};
use gpt_rs::ptir_program;

fn sample_program() -> Program {
    ptir_program!(
        r#"
func @single_result(%x: tensor<f32, 1x1>) -> tensor<f32, 1x1> {
  %sg = stop_gradient %x -> tensor<f32, 1x1>
  return %sg
}
"#
    )
}

fn sample_program_builder() -> (Program, ValueId, ValueId) {
    let spec = tensor_spec_static(DType::F32, &[2, 3]);
    let mut builder = ProgramBuilder::new();
    let param = builder.add_parameter(value_type_tensor(spec.clone()));
    let out = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(param)],
        value_type_tensor(spec),
    );
    let function = builder.finish("main", vec![out]);
    let program = Program::new("main").with_functions(vec![function]);
    (program, param, out)
}

fn renumber_function_values(program: &Program, delta: u32) -> Program {
    let mut out = program.clone();
    let Some(function) = out.functions.first_mut() else {
        return out;
    };

    for id in &mut function.parameter_ids {
        id.0 = id.0.saturating_add(delta);
    }

    for instruction in &mut function.body {
        instruction.id.0 = instruction.id.0.saturating_add(delta);
        for operand in &mut instruction.operands {
            match operand {
                Operand::Value(value) => value.0 = value.0.saturating_add(delta),
                Operand::TupleElement { tuple, .. } => {
                    tuple.0 = tuple.0.saturating_add(delta);
                }
                Operand::Literal(_) => {}
            }
        }
    }

    for id in &mut function.result_ids {
        id.0 = id.0.saturating_add(delta);
    }
    out
}

fn constant_program(value: f32) -> Program {
    let spec = tensor_spec_static(DType::F32, &[1]);
    let literal = gpt_rs::backend::spec::TensorLiteral::new(
        spec.clone(),
        Arc::from(value.to_le_bytes().to_vec()),
    );
    let mut builder = ProgramBuilder::new();
    let constant = builder.emit_single(
        Operation::Constant(literal),
        Vec::new(),
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![constant]);
    Program::new("main").with_functions(vec![function])
}

struct DummyTarget {
    name: String,
    hits: Arc<AtomicUsize>,
}

impl ConversionTarget for DummyTarget {
    fn name(&self) -> &str {
        &self.name
    }

    fn file_extension(&self) -> &str {
        "dummy"
    }

    fn convert(
        &self,
        program: &Program,
        _options: &ConversionOptions,
    ) -> gpt_rs::backend::conversion::ConversionResult<ConvertedIr> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        Ok(ConvertedIr {
            module: format!("module @{}", program.entry),
            entrypoints: vec![ConvertedEntrypoint {
                ptir: program.entry.clone(),
                symbol: program.entry.clone(),
            }],
        })
    }
}

#[test]
fn default_entrypoint_name_is_stable() {
    let program = sample_program();
    let name = default_entrypoint_name(&program).expect("entrypoint name");
    let base = sanitize_symbol(&program.entry);
    let hash = hash_program(&program).expect("program hash");
    assert_eq!(name, format!("{base}__{hash:016x}"));
}

#[test]
fn registry_returns_registered_target() {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let name = format!("dummy_target_{id}");
    let target = Arc::new(DummyTarget {
        name: name.clone(),
        hits: Arc::new(AtomicUsize::new(0)),
    });
    register_conversion_target(target);
    let fetched = get_conversion_target(&name).expect("target exists");
    assert_eq!(fetched.name(), name);
}

#[test]
fn conversion_cache_dedups_builds() {
    let program = sample_program();
    let hits = Arc::new(AtomicUsize::new(0));
    let target = DummyTarget {
        name: "dummy_cache".to_string(),
        hits: Arc::clone(&hits),
    };
    let options = ConversionOptions::default();
    let key = ConversionCacheKey::new(&program, &target, &options, None).expect("cache key");
    let cache = ConversionCache::new();

    let _first = cache
        .get_or_convert(key.clone(), || target.convert(&program, &options))
        .expect("first conversion");
    let _second = cache
        .get_or_convert(key, || target.convert(&program, &options))
        .expect("second conversion");

    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[test]
fn conversion_cache_key_ignores_value_id_renumbering() {
    let program = sample_program();
    let renumbered = renumber_function_values(&program, 37);
    let target = DummyTarget {
        name: "dummy_cache_id_renumber".to_string(),
        hits: Arc::new(AtomicUsize::new(0)),
    };
    let options = ConversionOptions::default();
    let lhs = ConversionCacheKey::new(&program, &target, &options, None).expect("lhs key");
    let rhs = ConversionCacheKey::new(&renumbered, &target, &options, None).expect("rhs key");

    assert_eq!(
        lhs, rhs,
        "conversion cache key should be stable under pure ValueId renumbering"
    );
}

#[test]
fn conversion_cache_key_changes_when_literal_tensor_changes() {
    let first = constant_program(1.0);
    let second = constant_program(2.0);
    let target = DummyTarget {
        name: "dummy_cache_literal_change".to_string(),
        hits: Arc::new(AtomicUsize::new(0)),
    };
    let options = ConversionOptions::default();
    let lhs = ConversionCacheKey::new(&first, &target, &options, None).expect("lhs key");
    let rhs = ConversionCacheKey::new(&second, &target, &options, None).expect("rhs key");

    assert_ne!(
        lhs, rhs,
        "literal value change must produce a different conversion cache key"
    );
}

#[test]
fn walk_program_visits_functions_and_regions() {
    let spec = tensor_spec_static(DType::F32, &[1, 1]);
    let mut builder = ProgramBuilder::new();
    let param = builder.add_parameter(value_type_tensor(spec.clone()));
    let out = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(param)],
        value_type_tensor(spec.clone()),
    );
    let function = builder.finish("main", vec![out]);

    let region = Region {
        id: RegionId(0),
        parameters: vec![value_type_tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(1),
            op: Operation::StopGradient,
            operands: vec![Operand::Value(ValueId(0))],
            output: value_type_tensor(spec.clone()),
        }],
        results: vec![value_type_tensor(spec)],
    };

    let program = Program::new("main")
        .with_functions(vec![function])
        .with_regions(vec![region]);

    struct RecordingVisitor {
        events: Vec<String>,
    }

    impl ProgramVisitor for RecordingVisitor {
        fn on_program(&mut self, program: &Program) {
            self.events.push(format!("program:{}", program.entry));
        }

        fn on_function(&mut self, function: &gpt_rs::backend::spec::Function) {
            self.events.push(format!("func:{}", function.name));
        }

        fn on_instruction(
            &mut self,
            function: &gpt_rs::backend::spec::Function,
            index: usize,
            _inst: &Instruction,
        ) {
            self.events
                .push(format!("inst:{}:{}", function.name, index));
        }

        fn on_region(&mut self, region: &Region) {
            self.events.push(format!("region:{}", region.id.0));
        }

        fn on_region_instruction(&mut self, region: &Region, index: usize, _inst: &Instruction) {
            self.events
                .push(format!("region_inst:{}:{}", region.id.0, index));
        }
    }

    let mut visitor = RecordingVisitor { events: Vec::new() };
    walk_program(&program, &mut visitor);

    assert_eq!(
        visitor.events,
        vec![
            "program:main",
            "func:main",
            "inst:main:0",
            "region:0",
            "region_inst:0:0"
        ]
    );
}

#[test]
fn legality_checks_reject_disallowed_ops() {
    let program = sample_program();
    let spec = LegalitySpec::default().allow_ops([OperationKind::ElementwiseUnary]);

    let report = check_program_legality(&program, &spec).expect_err("expected failure");
    assert!(!report.diagnostics.is_empty());
    assert!(report
        .diagnostics
        .iter()
        .any(|diag| diag.message.contains("not allowed")));
}

#[test]
fn legality_checks_reject_dynamic_dims() {
    let spec = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::Dynamic(DimSymbol::new("B"))]),
    );
    let mut builder = ProgramBuilder::new();
    let param = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let out = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(param)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![out]);
    let program = Program::new("main").with_functions(vec![function]);

    let legality = LegalitySpec::default().with_dynamic_dims(false);
    let report = check_program_legality(&program, &legality).expect_err("expected failure");
    assert!(report
        .diagnostics
        .iter()
        .any(|diag| diag.message.contains("dynamic dimensions")));
}

#[test]
fn buffer_plan_tracks_static_buffers() {
    let (program, param, out) = sample_program_builder();
    let plan = plan_buffers(&program).expect("buffer plan");
    let function = plan.function("main").expect("function plan");

    let param_buf = function.buffer_for(param).expect("param buffer");
    assert_eq!(param_buf.byte_len, Some(2 * 3 * 4));
    assert!(param_buf.usage.contains_parameter());

    let out_buf = function.buffer_for(out).expect("output buffer");
    assert!(out_buf.usage.contains_temporary());
    assert!(out_buf.usage.contains_result());
}

#[test]
fn buffer_plan_rejects_dynamic_shapes_when_required() {
    let spec = TensorSpec::new(
        DType::F32,
        Shape::new(vec![Dimension::Dynamic(DimSymbol::new("N"))]),
    );
    let mut builder = ProgramBuilder::new();
    let param = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let out = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(param)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![out]);
    let program = Program::new("main").with_functions(vec![function]);

    let options = BufferizeOptions {
        require_static_shapes: true,
        require_known_dtypes: false,
    };
    let err = plan_buffers_with(&program, &options).expect_err("expected failure");
    let err_text = format!("{err}");
    assert!(err_text.contains("dynamic shape"));
}

#[test]
fn buffer_plan_tracks_tuple_outputs() {
    let input_spec = tensor_spec_static(DType::F32, &[4]);
    let values_spec = tensor_spec_static(DType::F32, &[2]);
    let indices_spec = tensor_spec_static(DType::Si32, &[2]);

    let mut builder = ProgramBuilder::new();
    let input_id = builder.add_parameter(ValueType::Tensor(input_spec));
    let tuple_id = builder.emit_single(
        Operation::TopK(TopKSpec {
            k: 2,
            axis: 0,
            largest: true,
            indices_dtype: DType::Si32,
        }),
        vec![Operand::Value(input_id)],
        ValueType::Tuple(vec![
            ValueType::Tensor(values_spec.clone()),
            ValueType::Tensor(indices_spec.clone()),
        ]),
    );
    let function = builder.finish("main", vec![tuple_id]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_buffers(&program).expect("buffer plan");
    let func_plan = plan.function("main").expect("function plan");
    let buffers = func_plan.buffers_for_value(tuple_id);

    assert_eq!(buffers.len(), 2);
    assert_eq!(buffers[0].dtype, values_spec.dtype);
    assert_eq!(buffers[1].dtype, indices_spec.dtype);
    assert!(buffers[0].usage.contains_result());
    assert!(buffers[1].usage.contains_result());
}
