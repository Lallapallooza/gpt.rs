use std::sync::Arc;

use gpt_rs::backend::ptir_utils::{tensor_spec_static, value_type_tensor};
use gpt_rs::backend::spec::{
    BackendResult, DType, Instruction, Operation, PortableBackend, Program, TensorInit,
    TensorLiteral, ValueId,
};
use gpt_rs::backend::text_ir::SnippetBindings;
use gpt_rs::ops::graph::GraphArena;
use gpt_rs::ops::ptir::{tensor, PtirGraph};
use gpt_rs::ptir_snippet;
use gpt_rs::tensor::Shape as DeviceShape;

#[derive(Default)]
struct DummyBackend;

impl PortableBackend for DummyBackend {
    type TensorHandle = ();

    fn backend_name(&self) -> &str {
        "dummy"
    }

    fn materialize(&self, _init: TensorInit) -> BackendResult<Self::TensorHandle> {
        Ok(())
    }

    fn to_literal(&self, _tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        unreachable!("dummy backend does not materialize tensors")
    }

    fn execute_instruction(
        &self,
        _instruction: &Instruction,
        _inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        unreachable!("dummy backend does not execute instructions")
    }

    fn run_program(
        &self,
        _program: &Program,
        _entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        Ok(Vec::new())
    }
}

#[test]
fn snippet_instantiation_populates_program() {
    let snippet = ptir_snippet!(
        r#"
func @unit(%x: tensor<{{dtype}}, {{shape}}>) -> tensor<{{dtype}}, {{shape}}> {
  %y = add %x, %x -> tensor<{{dtype}}, {{shape}}>
  return %y
}
"#
    );

    let bindings = SnippetBindings::new()
        .dtype("dtype", DType::F32)
        .shape("shape", [2, 3]);

    let parsed = snippet
        .instantiate(&bindings)
        .expect("snippet should parse");
    let function = parsed
        .program
        .functions
        .first()
        .expect("snippet defines function");

    assert_eq!(
        function.parameters[0],
        value_type_tensor(tensor_spec_static(DType::F32, &[2, 3]))
    );
    assert_eq!(function.body.len(), 1);

    // Ensure value naming persisted.
    let x_id = parsed.value_names.get("x").copied().expect("value tracked");
    let y_id = parsed
        .value_names
        .get("y")
        .copied()
        .expect("result tracked");
    assert_ne!(x_id, y_id);
}

#[test]
fn emit_snippet_records_value_ids() {
    let backend = Arc::new(DummyBackend);
    let arena = GraphArena::new(backend);

    let snippet = ptir_snippet!(
        r#"
func @softmax(%x: tensor<{{dtype}}, {{shape}}>) -> tensor<{{dtype}}, {{shape}}> {
  %max = reduce_max(%x) axes[{{axis}}] keepdims[true] -> tensor<{{dtype}}, {{reduce_shape}}>
  %max_bcast = broadcast_to(%max) shape[{{shape_list}}] -> tensor<{{dtype}}, {{shape}}>
  %shift = sub %x, %max_bcast -> tensor<{{dtype}}, {{shape}}>
  %exp = exp %shift -> tensor<{{dtype}}, {{shape}}>
  return %exp
}
"#
    );

    let bindings = SnippetBindings::new()
        .value("x", ValueId(42))
        .dtype("dtype", DType::F32)
        .shape("shape", [2, 4])
        .shape_list("shape_list", [2, 4])
        .shape("reduce_shape", [2, 1])
        .int("axis", 1usize);

    let result = arena
        .capture(|ctx| ctx.emit_snippet(snippet, &bindings))
        .expect("capture should succeed");

    let (ids, specs) = result.into_parts();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], ValueId(3), "expected final SSA value id");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0], tensor_spec_static(DType::F32, &[2, 4]));
}

#[test]
fn ptir_graph_emitter_wraps_builder() {
    let backend = Arc::new(DummyBackend);
    let arena = GraphArena::new(backend);

    let snippet = ptir_snippet!(
        r#"
func @softmax(%x: tensor<{{dtype}}, {{shape}}>) -> tensor<{{dtype}}, {{shape}}> {
  %max = reduce_max(%x) axes[{{axis}}] keepdims[true] -> tensor<{{dtype}}, {{reduce_shape}}>
  %max_bcast = broadcast_to(%max) shape[{{shape_list}}] -> tensor<{{dtype}}, {{shape}}>
  %shift = sub %x, %max_bcast -> tensor<{{dtype}}, {{shape}}>
  %exp = exp %shift -> tensor<{{dtype}}, {{shape}}>
  return %exp
}
"#
    );

    let value_id = arena
        .capture(|ctx| {
            let snippet_copy = snippet;
            let captured_value = {
                let mut graph = PtirGraph::new(ctx);
                let placeholder = tensor::<f32>(DeviceShape::new(vec![2, 4]));
                let x = graph.import("x", ValueId(12), &placeholder);

                let emitter = graph
                    .emit_snippet(snippet_copy)
                    .value("x", &x)
                    .dtype("dtype", DType::F32)
                    .shape("reduce_shape", [2usize, 1usize])
                    .dims("shape_list", [2usize, 4usize])
                    .int("axis", 1usize);

                let emitter = emitter.shape_like("shape", &x)?;
                emitter.finish_value()
            }?;
            Ok(captured_value.id())
        })
        .expect("capture should succeed");

    assert_eq!(value_id, ValueId(3));
}

#[test]
fn snippet_parses_dot_general() {
    let snippet = ptir_snippet!(
        r#"
func @dot(%lhs: tensor<{{dtype}}, {{lhs_shape}}>, %rhs: tensor<{{dtype}}, {{rhs_shape}}>) -> tensor<{{dtype}}, {{out_shape}}> {
  %out = dot_general(%lhs, %rhs) contract_lhs[{{contract_lhs}}] contract_rhs[{{contract_rhs}}] batch[{{batch_axes}}] -> tensor<{{dtype}}, {{out_shape}}>
  return %out
}
"#
    );

    let bindings = SnippetBindings::new()
        .dtype("dtype", DType::F32)
        .shape("lhs_shape", [2, 3])
        .shape("rhs_shape", [3, 4])
        .shape("out_shape", [2, 4])
        .dims("contract_lhs", [1])
        .dims("contract_rhs", [0])
        .dims("batch_axes", []);

    let parsed = snippet
        .instantiate(&bindings)
        .expect("dot_general snippet should parse");
    let function = parsed
        .program
        .functions
        .first()
        .expect("snippet defines function");
    assert_eq!(
        function.body.len(),
        1,
        "expected single dot_general instruction"
    );
    let Operation::DotGeneral(spec) = &function.body[0].op else {
        panic!("expected dot_general operation");
    };
    assert_eq!(spec.contract_lhs, vec![1]);
    assert_eq!(spec.contract_rhs, vec![0]);
    assert!(spec.batch_lhs.is_empty());
    assert!(spec.batch_rhs.is_empty());
    assert_eq!(spec.accum_dtype, None);
    assert_eq!(spec.out_dtype, None);
}

#[test]
fn snippet_finish_supports_tuple_results() {
    let backend = Arc::new(DummyBackend);
    let arena = GraphArena::new(backend);

    let snippet = ptir_snippet!(
        r#"
func @pair(%x: tensor<f32, 2x2>) -> (tensor<f32, 2x2>, tensor<f32, 2x2>) {
  %sum = add %x, %x -> tensor<f32, 2x2>
  %prod = mul %x, %x -> tensor<f32, 2x2>
  return %sum, %prod
}
"#
    );

    let (first_id, second_id) = arena
        .capture(|ctx| {
            let mut graph = PtirGraph::new(ctx);
            let placeholder = tensor::<f32>(DeviceShape::new(vec![2, 2]));
            let x = graph.import("x", ValueId(10), &placeholder);

            let results = graph.emit_snippet(snippet).value("x", &x).finish()?;

            assert_eq!(results.len(), 2, "tuple results should report length");
            let first = results.tuple_element(0)?;
            let second = results.tuple_element(1)?;
            assert_ne!(
                first.id(),
                second.id(),
                "tuple elements should map to distinct values"
            );

            assert!(
                results.clone().into_value().is_err(),
                "multi-result tuple cannot collapse into a single value"
            );

            let taken_second = results.into_tuple_element(1)?;
            Ok((first.id(), taken_second.id()))
        })
        .expect("capture should succeed");

    assert_ne!(first_id, second_id, "tuple elements should remain distinct");
}
