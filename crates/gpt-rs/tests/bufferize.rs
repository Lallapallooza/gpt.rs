use std::collections::{HashMap, HashSet};

use gpt_rs::backend::conversion::{
    plan_buffers_with, AliasKind, BufferKey, BufferSpec, BufferizeError, BufferizeOptions,
    FunctionBufferPlan, LiveRange,
};
use gpt_rs::backend::ptir_utils::{
    tensor_literal_f32_zeros, tensor_spec_mixed, tensor_spec_static,
};
use gpt_rs::backend::spec::{
    DType, Dimension, ElementwiseBinaryOp, Instruction, Operand, Operation, Program,
    ProgramBuilder, Region, RegionId, ReshapeDim, ReshapeSpec, SliceSpec, TransposeSpec, ValueId,
    ValueType,
};

fn strict_opts() -> BufferizeOptions {
    BufferizeOptions {
        require_static_shapes: true,
        require_known_dtypes: true,
    }
}

fn relaxed_opts() -> BufferizeOptions {
    BufferizeOptions {
        require_static_shapes: false,
        require_known_dtypes: false,
    }
}

fn plan_main(program: &Program, options: &BufferizeOptions) -> FunctionBufferPlan {
    let plan = plan_buffers_with(program, options).expect("bufferize plan");
    plan.function("main").expect("function plan").clone()
}

fn buffer_key(value: ValueId, path: &[usize]) -> BufferKey {
    BufferKey {
        value,
        path: path.to_vec(),
    }
}

fn buffer_index(plan: &FunctionBufferPlan, value: ValueId, path: &[usize]) -> usize {
    let key = buffer_key(value, path);
    *plan
        .values
        .get(&key)
        .unwrap_or_else(|| panic!("buffer index missing for {key:?}"))
}

fn buffer_for<'a>(plan: &'a FunctionBufferPlan, value: ValueId, path: &[usize]) -> &'a BufferSpec {
    plan.buffer_for_path(value, path)
        .unwrap_or_else(|| panic!("buffer missing for {value:?} path {path:?}"))
}

fn live(plan: &FunctionBufferPlan, value: ValueId, path: &[usize]) -> LiveRange {
    buffer_for(plan, value, path).live_range
}

fn assert_usage(buffer: &BufferSpec, parameter: bool, result: bool, temporary: bool) {
    assert_eq!(buffer.usage.contains_parameter(), parameter);
    assert_eq!(buffer.usage.contains_result(), result);
    assert_eq!(buffer.usage.contains_temporary(), temporary);
}

fn assert_function_plan_eq(left: &FunctionBufferPlan, right: &FunctionBufferPlan) {
    assert_eq!(left.values, right.values);
    assert_eq!(left.slots, right.slots);
    assert_eq!(left.buffers.len(), right.buffers.len());
    for (left_buf, right_buf) in left.buffers.iter().zip(right.buffers.iter()) {
        let mut left_norm = left_buf.clone();
        let mut right_norm = right_buf.clone();
        left_norm.alias_group = 0;
        right_norm.alias_group = 0;
        assert_eq!(left_norm, right_norm);
    }
    for (i, left_buf) in left.buffers.iter().enumerate() {
        for (j, other_left) in left.buffers.iter().enumerate() {
            let same_left = left_buf.alias_group == other_left.alias_group;
            let same_right = right.buffers[i].alias_group == right.buffers[j].alias_group;
            assert_eq!(same_left, same_right);
        }
    }
}

#[test]
fn deterministic_plan_is_stable_strict() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan_a = plan_buffers_with(&program, &strict_opts()).expect("plan a");
    let plan_b = plan_buffers_with(&program, &strict_opts()).expect("plan b");

    let func_a = plan_a.function("main").expect("plan a func");
    let func_b = plan_b.function("main").expect("plan b func");
    assert_function_plan_eq(func_a, func_b);
}

#[test]
fn deterministic_plan_is_stable_relaxed() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_mixed(DType::F32, &[None, Some(4)]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan_a = plan_buffers_with(&program, &relaxed_opts()).expect("plan a");
    let plan_b = plan_buffers_with(&program, &relaxed_opts()).expect("plan b");

    let func_a = plan_a.function("main").expect("plan a func");
    let func_b = plan_b.function("main").expect("plan b func");
    assert_function_plan_eq(func_a, func_b);
}

#[test]
fn enumeration_order_params_then_instr_outputs_then_results() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p1)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let idx_p0 = buffer_index(&plan, p0, &[]);
    let idx_p1 = buffer_index(&plan, p1, &[]);
    let idx_v0 = buffer_index(&plan, v0, &[]);
    assert!(idx_p0 < idx_p1 && idx_p1 < idx_v0);
}

#[test]
fn reinsertion_merges_usage_instead_of_duplication() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p1)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffers = plan.buffers_for_value(v0);
    assert_eq!(buffers.len(), 1);
    let buffer = buffers[0];
    assert_usage(buffer, false, true, true);
}

#[test]
fn param_returned_merges_parameter_and_result_usage() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, p0, &[]);
    assert_usage(buffer, true, true, false);
}

#[test]
fn tuple_param_expands_to_paths_only_no_root_tuple_buffer() {
    let mut builder = ProgramBuilder::new();
    let elem = ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2]));
    let tuple = ValueType::Tuple(vec![elem.clone(), elem.clone()]);
    let t = builder.add_parameter(tuple);
    let v0 = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::TupleElement { tuple: t, index: 0 }],
        elem,
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(plan.buffer_for(t).is_none());
    assert!(plan.buffer_for_path(t, &[0]).is_some());
    assert!(plan.buffer_for_path(t, &[1]).is_some());
}

#[test]
fn nested_tuple_paths_are_recursive_and_deterministic() {
    let mut builder = ProgramBuilder::new();
    let elem = ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2]));
    let nested = ValueType::Tuple(vec![
        ValueType::Tuple(vec![elem.clone(), elem.clone()]),
        elem,
    ]);
    let t = builder.add_parameter(nested);
    let function = builder.finish("main", vec![t]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffers = plan.buffers_for_value(t);
    let paths: Vec<Vec<usize>> = buffers.iter().map(|buf| buf.path.clone()).collect();
    assert_eq!(paths, vec![vec![0, 0], vec![0, 1], vec![1]]);
}

#[test]
fn type_mismatch_same_key_tensor_dtype() {
    let param_id = ValueId(0);
    let f32_spec = tensor_spec_static(DType::F32, &[2, 2]);
    let i32_spec = tensor_spec_static(DType::Si32, &[2, 2]);
    let function = gpt_rs::backend::spec::Function {
        name: "main".to_string(),
        parameters: vec![ValueType::Tensor(f32_spec)],
        parameter_ids: vec![param_id],
        results: Vec::new(),
        body: vec![Instruction {
            id: param_id,
            op: Operation::StopGradient,
            operands: vec![Operand::Value(param_id)],
            output: ValueType::Tensor(i32_spec),
        }],
        result_ids: Vec::new(),
    };
    let program = Program::new("main").with_functions(vec![function]);
    let err = plan_buffers_with(&program, &strict_opts()).expect_err("type mismatch");
    assert!(matches!(err, BufferizeError::TypeMismatch { value } if value == param_id));
}

#[test]
fn type_mismatch_same_key_tensor_shape() {
    let param_id = ValueId(0);
    let spec_a = tensor_spec_static(DType::F32, &[2, 2]);
    let spec_b = tensor_spec_static(DType::F32, &[2, 3]);
    let function = gpt_rs::backend::spec::Function {
        name: "main".to_string(),
        parameters: vec![ValueType::Tensor(spec_a)],
        parameter_ids: vec![param_id],
        results: Vec::new(),
        body: vec![Instruction {
            id: param_id,
            op: Operation::StopGradient,
            operands: vec![Operand::Value(param_id)],
            output: ValueType::Tensor(spec_b),
        }],
        result_ids: Vec::new(),
    };
    let program = Program::new("main").with_functions(vec![function]);
    let err = plan_buffers_with(&program, &strict_opts()).expect_err("type mismatch");
    assert!(matches!(err, BufferizeError::TypeMismatch { value } if value == param_id));
}

#[test]
fn type_mismatch_tuple_element_path_specific() {
    let param_id = ValueId(0);
    let elem_a = tensor_spec_static(DType::F32, &[2, 2]);
    let elem_b = tensor_spec_static(DType::Si32, &[2, 2]);
    let param_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem_a.clone()),
        ValueType::Tensor(elem_a),
    ]);
    let output_ty = ValueType::Tuple(vec![
        ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2])),
        ValueType::Tensor(elem_b),
    ]);
    let function = gpt_rs::backend::spec::Function {
        name: "main".to_string(),
        parameters: vec![param_ty],
        parameter_ids: vec![param_id],
        results: Vec::new(),
        body: vec![Instruction {
            id: param_id,
            op: Operation::StopGradient,
            operands: vec![Operand::Value(param_id)],
            output: output_ty,
        }],
        result_ids: Vec::new(),
    };
    let program = Program::new("main").with_functions(vec![function]);
    let err = plan_buffers_with(&program, &strict_opts()).expect_err("type mismatch");
    assert!(matches!(err, BufferizeError::TypeMismatch { value } if value == param_id));
}

#[test]
fn byte_len_scalar_tensor_rank0() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, p0, &[]);
    assert_eq!(buffer.byte_len, Some(4));
}

#[test]
fn byte_len_zero_dim_is_zero() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[0, 10]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, p0, &[]);
    assert_eq!(buffer.byte_len, Some(0));
}

#[test]
fn byte_len_known_shape_known_dtype() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F16, &[2, 3, 4]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, p0, &[]);
    assert_eq!(buffer.byte_len, Some(48));
}

#[test]
fn dynamic_shape_errors_when_require_static_shapes() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_mixed(DType::F32, &[None, Some(4)]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("dynamic shape");
    assert!(matches!(err, BufferizeError::DynamicShape { .. }));
}

#[test]
fn dynamic_shape_allowed_byte_len_none_when_relaxed() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_mixed(DType::F32, &[None, Some(4)]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let options = BufferizeOptions {
        require_static_shapes: false,
        require_known_dtypes: true,
    };
    let plan = plan_main(&program, &options);
    let buffer = buffer_for(&plan, p0, &[]);
    assert_eq!(buffer.byte_len, None);
}

#[test]
fn unknown_dtype_errors_when_require_known_dtypes() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::Si4, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("unknown dtype");
    assert!(matches!(err, BufferizeError::UnknownDType { .. }));
}

#[test]
fn unknown_dtype_allowed_byte_len_none_when_relaxed() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::Si4, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let options = BufferizeOptions {
        require_static_shapes: true,
        require_known_dtypes: false,
    };
    let plan = plan_main(&program, &options);
    let buffer = buffer_for(&plan, p0, &[]);
    assert_eq!(buffer.byte_len, None);
}

#[test]
fn dynamic_shape_takes_precedence_over_unknown_dtype() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_mixed(DType::Si4, &[None, Some(2)]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("dynamic shape");
    assert!(matches!(err, BufferizeError::DynamicShape { .. }));
}

#[test]
fn byte_len_overflow_errors() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[usize::MAX, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("overflow");
    assert!(matches!(err, BufferizeError::ByteLenOverflow { .. }));
}

#[test]
fn byte_len_overflow_even_when_relaxed() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[usize::MAX, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let options = BufferizeOptions {
        require_static_shapes: false,
        require_known_dtypes: false,
    };
    let err = plan_buffers_with(&program, &options).expect_err("overflow");
    assert!(matches!(err, BufferizeError::ByteLenOverflow { .. }));
}

#[test]
fn byte_len_none_if_any_input_unknown_and_relaxed() {
    let mut builder = ProgramBuilder::new();
    let spec_dynamic = tensor_spec_mixed(DType::F32, &[None, Some(4)]);
    let spec_unknown = tensor_spec_static(DType::Si4, &[2, 2]);
    let spec_both = tensor_spec_mixed(DType::Fp8E4M3, &[None, Some(2)]);

    let p0 = builder.add_parameter(ValueType::Tensor(spec_dynamic));
    let p1 = builder.add_parameter(ValueType::Tensor(spec_unknown));
    let p2 = builder.add_parameter(ValueType::Tensor(spec_both));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &relaxed_opts());

    assert_eq!(buffer_for(&plan, p0, &[]).byte_len, None);
    assert_eq!(buffer_for(&plan, p1, &[]).byte_len, None);
    assert_eq!(buffer_for(&plan, p2, &[]).byte_len, None);
}

#[test]
fn live_range_unused_param_stays_0_0() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let literal = tensor_literal_f32_zeros(&[2, 2]);
    let c0 = builder.emit_single(
        Operation::Constant(literal),
        Vec::new(),
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![c0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 0));
}

#[test]
fn live_range_param_used_in_first_instruction() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 1));
    assert_eq!(live(&plan, v0, &[]), LiveRange::new(1, 2));
}

#[test]
fn live_range_param_used_late_extends_to_late_pos() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p1), Operand::Value(p1)],
        ValueType::Tensor(spec.clone()),
    );
    let v1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(v0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v1]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 2));
    assert_eq!(live(&plan, p1, &[]), LiveRange::new(0, 1));
    assert_eq!(live(&plan, v0, &[]), LiveRange::new(1, 2));
    assert_eq!(live(&plan, v1, &[]), LiveRange::new(2, 3));
}

#[test]
fn live_range_multiple_uses_extend_end_to_max_pos() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let v1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let v2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v0), Operand::Value(v1)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v2]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, v0, &[]), LiveRange::new(1, 3));
    assert_eq!(live(&plan, v2, &[]), LiveRange::new(3, 4));
}

#[test]
fn live_range_param_returned_extends_to_end() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let _v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 2));
}

#[test]
fn tuple_element_operand_extends_tuple_value_id_not_path() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let tuple = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.add_parameter(tuple);
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![
            Operand::TupleElement { tuple: t, index: 0 },
            Operand::TupleElement { tuple: t, index: 0 },
        ],
        ValueType::Tensor(elem),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, t, &[0]), LiveRange::new(0, 1));
    assert_eq!(live(&plan, t, &[1]), LiveRange::new(0, 1));
}

#[test]
fn instruction_tuple_output_extends_all_paths_on_use() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(elem.clone()));
    let tuple_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.emit_single(Operation::StopGradient, vec![Operand::Value(p0)], tuple_ty);
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![
            Operand::TupleElement { tuple: t, index: 0 },
            Operand::Value(p0),
        ],
        ValueType::Tensor(elem),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, t, &[0]), LiveRange::new(1, 2));
    assert_eq!(live(&plan, t, &[1]), LiveRange::new(1, 2));
}

#[test]
fn reshape_is_identity_alias_root_path() {
    let mut builder = ProgramBuilder::new();
    let in_spec = tensor_spec_static(DType::F32, &[2, 2]);
    let out_spec = tensor_spec_static(DType::F32, &[4]);
    let input = builder.add_parameter(ValueType::Tensor(in_spec.clone()));
    let reshape = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(input)],
        ValueType::Tensor(out_spec),
    );
    let function = builder.finish("main", vec![reshape]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let input_spec = buffer_for(&plan, input, &[]);
    let reshape_spec = buffer_for(&plan, reshape, &[]);

    assert_eq!(input_spec.alias_group, reshape_spec.alias_group);
    assert_eq!(reshape_spec.alias_kind, AliasKind::Identity);
    assert_eq!(reshape_spec.alias_of, Some(buffer_key(input, &[])));
    assert!(reshape_spec.slot.is_none());
}

#[test]
fn stop_gradient_is_identity_alias() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let input = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let sg = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(input)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![sg]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let sg_spec = buffer_for(&plan, sg, &[]);
    assert_eq!(sg_spec.alias_kind, AliasKind::Identity);
    assert_eq!(sg_spec.alias_of, Some(buffer_key(input, &[])));
}

#[test]
fn transpose_is_view_alias() {
    let mut builder = ProgramBuilder::new();
    let in_spec = tensor_spec_static(DType::F32, &[2, 3]);
    let out_spec = tensor_spec_static(DType::F32, &[3, 2]);
    let input = builder.add_parameter(ValueType::Tensor(in_spec.clone()));
    let transpose = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![1, 0] }),
        vec![Operand::Value(input)],
        ValueType::Tensor(out_spec.clone()),
    );
    let add = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(transpose), Operand::Value(transpose)],
        ValueType::Tensor(out_spec),
    );
    let function = builder.finish("main", vec![add]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let transpose_spec = buffer_for(&plan, transpose, &[]);

    assert_eq!(transpose_spec.alias_kind, AliasKind::View);
    assert_eq!(transpose_spec.alias_of, Some(buffer_key(input, &[])));
    assert!(transpose_spec.slot.is_some());
}

#[test]
fn slice_alias_uses_first_operand() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let slice = builder.emit_single(
        Operation::Slice(SliceSpec {
            starts: vec![0, 0],
            sizes: vec![2, 2],
        }),
        vec![Operand::Value(p0), Operand::Value(p1)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![slice]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let slice_spec = buffer_for(&plan, slice, &[]);
    assert_eq!(slice_spec.alias_kind, AliasKind::View);
    assert_eq!(slice_spec.alias_of, Some(buffer_key(p0, &[])));
}

#[test]
fn non_alias_ops_have_alias_none_and_alias_of_cleared() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let v0_spec = buffer_for(&plan, v0, &[]);
    assert_eq!(v0_spec.alias_kind, AliasKind::None);
    assert!(v0_spec.alias_of.is_none());
}

#[test]
fn alias_groups_union_transitively_across_identity_and_view_edges() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v1 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v2 = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![0] }),
        vec![Operand::Value(v1)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v3 = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(v2)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v3]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let group = buffer_for(&plan, p0, &[]).alias_group;
    assert_eq!(buffer_for(&plan, v1, &[]).alias_group, group);
    assert_eq!(buffer_for(&plan, v2, &[]).alias_group, group);
    assert_eq!(buffer_for(&plan, v3, &[]).alias_group, group);
}

#[test]
fn alias_groups_separate_components() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v1 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p1)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let group_p0 = buffer_for(&plan, p0, &[]).alias_group;
    let group_p1 = buffer_for(&plan, p1, &[]).alias_group;
    assert_ne!(group_p0, group_p1);
    assert_eq!(buffer_for(&plan, v0, &[]).alias_group, group_p0);
    assert_eq!(buffer_for(&plan, v1, &[]).alias_group, group_p1);
}

#[test]
fn alias_analysis_root_path_only_breaks_tuple_element_aliasing() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let tuple = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.add_parameter(tuple);
    let v0 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::TupleElement { tuple: t, index: 0 }],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let v0_spec = buffer_for(&plan, v0, &[]);
    assert_eq!(v0_spec.alias_kind, AliasKind::None);
    assert!(v0_spec.alias_of.is_none());
    assert_ne!(v0_spec.alias_group, buffer_for(&plan, t, &[0]).alias_group);
}

#[test]
fn identity_chain_merges_live_ranges_to_union() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v1 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v1), Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v2]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 2));
    assert_eq!(live(&plan, v1, &[]), LiveRange::new(0, 2));
}

#[test]
fn identity_branching_unions_across_component() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v1 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v2 = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let v3 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v1), Operand::Value(v2)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v3]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 3));
    assert_eq!(live(&plan, v1, &[]), LiveRange::new(0, 3));
    assert_eq!(live(&plan, v2, &[]), LiveRange::new(0, 3));
}

#[test]
fn view_edges_do_not_participate_in_identity_live_range_union() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v1 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v2 = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![0] }),
        vec![Operand::Value(v1)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v3 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v2), Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v3]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, p0, &[]), LiveRange::new(0, 3));
    assert_eq!(live(&plan, v1, &[]), LiveRange::new(0, 3));
    assert_eq!(live(&plan, v2, &[]), LiveRange::new(2, 3));
}

#[test]
fn identity_live_range_union_affects_slot_reuse() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(a)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let c = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let d = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(b), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![d]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let a_spec = buffer_for(&plan, a, &[]);
    let b_spec = buffer_for(&plan, b, &[]);
    let c_spec = buffer_for(&plan, c, &[]);
    assert_eq!(a_spec.slot, b_spec.slot);
    assert_ne!(a_spec.slot, c_spec.slot);
}

#[test]
fn temporary_gets_slot_when_eligible() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let t0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, t0, &[]);
    assert!(buffer.usage.contains_temporary());
    assert!(buffer.slot.is_some());
    assert_eq!(plan.slots.len(), 1);
}

#[test]
fn parameter_never_gets_slot() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let buffer = buffer_for(&plan, p0, &[]);
    assert!(buffer.slot.is_none());
}

#[test]
fn result_non_identity_excluded_from_slotting() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(buffer_for(&plan, v0, &[]).slot.is_none());
}

#[test]
fn constant_excluded_from_slotting() {
    let mut builder = ProgramBuilder::new();
    let literal = tensor_literal_f32_zeros(&[2, 2]);
    let spec = literal.spec.clone();
    let c0 = builder.emit_single(
        Operation::Constant(literal),
        Vec::new(),
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![c0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(buffer_for(&plan, c0, &[]).slot.is_none());
}

#[test]
fn identity_alias_output_inherits_slot_from_source() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(a)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(b), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let a_spec = buffer_for(&plan, a, &[]);
    let b_spec = buffer_for(&plan, b, &[]);
    assert_eq!(a_spec.slot, b_spec.slot);
}

#[test]
fn identity_alias_of_parameter_does_not_get_slot() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let b = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(buffer_for(&plan, b, &[]).slot.is_none());
}

#[test]
fn identity_alias_of_constant_does_not_get_slot() {
    let mut builder = ProgramBuilder::new();
    let literal = tensor_literal_f32_zeros(&[2, 2]);
    let spec = literal.spec.clone();
    let c0 = builder.emit_single(
        Operation::Constant(literal),
        Vec::new(),
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(c0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![c0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(buffer_for(&plan, b, &[]).slot.is_none());
}

#[test]
fn view_alias_is_slot_eligible_and_can_get_slot() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![0, 1] }),
        vec![Operand::Value(a)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(buffer_for(&plan, b, &[]).slot.is_some());
}

#[test]
fn slot_reuse_when_non_overlapping_end_lt_start() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[4]);
    let input = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let t1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(input), Operand::Value(input)],
        ValueType::Tensor(spec.clone()),
    );
    let t2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(input), Operand::Value(input)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![input]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let t1_spec = buffer_for(&plan, t1, &[]);
    let t2_spec = buffer_for(&plan, t2, &[]);

    assert_eq!(t1_spec.slot, t2_spec.slot);
}

#[test]
fn slot_not_reused_when_overlapping() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[4]);
    let input = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let t1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(input), Operand::Value(input)],
        ValueType::Tensor(spec.clone()),
    );
    let t2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(t1), Operand::Value(input)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![input]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let t1_spec = buffer_for(&plan, t1, &[]);
    let t2_spec = buffer_for(&plan, t2, &[]);

    assert_ne!(t1_spec.slot, t2_spec.slot);
    assert_eq!(t1_spec.live_range.end, t2_spec.live_range.start);
}

#[test]
fn slot_grouping_by_dtype_not_size() {
    let mut builder = ProgramBuilder::new();
    let spec_f32 = tensor_spec_static(DType::F32, &[2, 2]);
    let spec_i32 = tensor_spec_static(DType::Si32, &[2, 2]);
    let p_f32 = builder.add_parameter(ValueType::Tensor(spec_f32.clone()));
    let p_i32 = builder.add_parameter(ValueType::Tensor(spec_i32.clone()));
    let t_f32 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p_f32), Operand::Value(p_f32)],
        ValueType::Tensor(spec_f32),
    );
    let t_i32 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p_i32), Operand::Value(p_i32)],
        ValueType::Tensor(spec_i32),
    );
    let function = builder.finish("main", vec![p_f32]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let slot_f32 = buffer_for(&plan, t_f32, &[]).slot;
    let slot_i32 = buffer_for(&plan, t_i32, &[]).slot;
    assert!(slot_f32.is_some() && slot_i32.is_some());
    assert_ne!(slot_f32, slot_i32);
}

#[test]
fn slot_grouping_by_byte_len_exact_match() {
    let mut builder = ProgramBuilder::new();
    let spec_a = tensor_spec_static(DType::F32, &[2, 2]);
    let spec_b = tensor_spec_static(DType::F32, &[2, 3]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec_a.clone()));
    let p1 = builder.add_parameter(ValueType::Tensor(spec_b.clone()));
    let t0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec_a),
    );
    let t1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p1), Operand::Value(p1)],
        ValueType::Tensor(spec_b),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let plan = plan_main(&program, &strict_opts());
    let slot_a = buffer_for(&plan, t0, &[]).slot;
    let slot_b = buffer_for(&plan, t1, &[]).slot;
    assert!(slot_a.is_some() && slot_b.is_some());
    assert_ne!(slot_a, slot_b);
}

#[test]
fn byte_len_none_buffers_group_and_reuse_under_relaxed_options() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_mixed(DType::F32, &[None, Some(4)]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let t0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let t1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);

    let options = BufferizeOptions {
        require_static_shapes: false,
        require_known_dtypes: true,
    };
    let plan = plan_main(&program, &options);
    let slot_a = buffer_for(&plan, t0, &[]).slot;
    let slot_b = buffer_for(&plan, t1, &[]).slot;
    assert_eq!(slot_a, slot_b);
    assert_eq!(buffer_for(&plan, t0, &[]).byte_len, None);
}

#[test]
fn slot_usage_flags_merge_including_identity_result_inheritance() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let r = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(a)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![r]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let slot_id = buffer_for(&plan, a, &[]).slot.expect("slot");
    let slot = plan.slots.get(slot_id).expect("slot spec");
    assert!(slot.usage.contains_temporary());
    assert!(slot.usage.contains_result());
}

#[test]
fn no_slots_when_no_eligible_temporaries() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec));
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(plan.slots.is_empty());
}

#[test]
fn two_buffers_same_start_cannot_reuse_need_two_slots() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(elem.clone()));
    let tuple_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.emit_single(Operation::StopGradient, vec![Operand::Value(p0)], tuple_ty);
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let slot0 = buffer_for(&plan, t, &[0]).slot;
    let slot1 = buffer_for(&plan, t, &[1]).slot;
    assert!(slot0.is_some() && slot1.is_some());
    assert_ne!(slot0, slot1);
}

#[test]
fn tuple_result_marks_each_element_as_result_and_excludes_from_slotting() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(elem.clone()));
    let tuple_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.emit_single(Operation::StopGradient, vec![Operand::Value(p0)], tuple_ty);
    let function = builder.finish("main", vec![t]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    for path in [&[0][..], &[1][..]] {
        let buf = buffer_for(&plan, t, path);
        assert_usage(buf, false, true, true);
        assert!(buf.slot.is_none());
    }
}

#[test]
fn tuple_element_use_extends_all_elements_live_ranges() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let tuple = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t = builder.add_parameter(tuple);
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![
            Operand::TupleElement { tuple: t, index: 0 },
            Operand::TupleElement { tuple: t, index: 0 },
        ],
        ValueType::Tensor(elem.clone()),
    );
    let v1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![
            Operand::Value(v0),
            Operand::TupleElement { tuple: t, index: 0 },
        ],
        ValueType::Tensor(elem),
    );
    let function = builder.finish("main", vec![v1]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert_eq!(live(&plan, t, &[0]), LiveRange::new(0, 2));
    assert_eq!(live(&plan, t, &[1]), LiveRange::new(0, 2));
}

#[test]
fn tuple_paths_preserved_in_values_map() {
    let mut builder = ProgramBuilder::new();
    let elem = ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2]));
    let nested = ValueType::Tuple(vec![
        ValueType::Tuple(vec![elem.clone(), elem.clone()]),
        elem,
    ]);
    let t = builder.add_parameter(nested);
    let function = builder.finish("main", vec![t]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let expected: HashSet<BufferKey> = HashSet::from([
        buffer_key(t, &[0, 0]),
        buffer_key(t, &[0, 1]),
        buffer_key(t, &[1]),
    ]);
    let actual: HashSet<BufferKey> = plan.values.keys().cloned().collect();
    assert_eq!(expected, actual);
}

#[test]
fn region_param_ids_are_implicit_0_to_n_minus_1() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![
            ValueType::Tensor(spec.clone()),
            ValueType::Tensor(spec.clone()),
        ],
        body: vec![Instruction {
            id: ValueId(5),
            op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
            operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(1))],
            output: ValueType::Tensor(spec.clone()),
        }],
        results: vec![ValueType::Tensor(spec)],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    assert!(region_plan.buffer_for(ValueId(0)).is_some());
    assert!(region_plan.buffer_for(ValueId(1)).is_some());
}

#[test]
fn region_results_inferred_from_tail_when_region_results_non_empty() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![
            Instruction {
                id: ValueId(10),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
            Instruction {
                id: ValueId(11),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(10)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
            Instruction {
                id: ValueId(12),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(11)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
        ],
        results: vec![ValueType::Tensor(spec)],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    let buf = region_plan.buffer_for(ValueId(12)).expect("buffer");
    assert_eq!(buf.live_range.end, 4);
    assert!(!buf.usage.contains_result());
}

#[test]
fn region_results_empty_means_no_inference() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(10),
            op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
            operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(0))],
            output: ValueType::Tensor(spec),
        }],
        results: Vec::new(),
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    let buf = region_plan.buffer_for(ValueId(10)).expect("buffer");
    assert!(!buf.usage.contains_result());
}

#[test]
fn region_tuple_result_inference_marks_all_tuple_paths_as_result() {
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let tuple_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(elem.clone())],
        body: vec![Instruction {
            id: ValueId(10),
            op: Operation::StopGradient,
            operands: vec![Operand::Value(ValueId(0))],
            output: tuple_ty.clone(),
        }],
        results: vec![tuple_ty],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    for path in [&[0][..], &[1][..]] {
        let buf = region_plan
            .buffer_for_path(ValueId(10), path)
            .expect("buffer");
        assert_eq!(buf.live_range.end, 2);
        assert!(!buf.usage.contains_result());
    }
}

#[test]
fn region_param_liveness_is_not_initialized_regression() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![
            Instruction {
                id: ValueId(10),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
            Instruction {
                id: ValueId(11),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(10)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec.clone()),
            },
            Instruction {
                id: ValueId(12),
                op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
                operands: vec![Operand::Value(ValueId(11)), Operand::Value(ValueId(0))],
                output: ValueType::Tensor(spec),
            },
        ],
        results: vec![ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2]))],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    assert_eq!(
        region_plan
            .buffer_for(ValueId(0))
            .expect("param buffer")
            .live_range,
        LiveRange::new(0, 0)
    );
}

#[test]
fn region_results_inference_out_of_bounds_is_handled() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(10),
            op: Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
            operands: vec![Operand::Value(ValueId(0)), Operand::Value(ValueId(0))],
            output: ValueType::Tensor(spec.clone()),
        }],
        results: vec![ValueType::Tensor(spec.clone()), ValueType::Tensor(spec)],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    assert!(!region_plan
        .buffer_for(ValueId(10))
        .expect("buffer")
        .usage
        .contains_result());
}

#[test]
fn multiple_regions_get_separate_plans_and_no_cross_contamination() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region0 = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(10),
            op: Operation::StopGradient,
            operands: vec![Operand::Value(ValueId(0))],
            output: ValueType::Tensor(spec.clone()),
        }],
        results: vec![ValueType::Tensor(spec.clone())],
    };
    let region1 = Region {
        id: RegionId(1),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(20),
            op: Operation::StopGradient,
            operands: vec![Operand::Value(ValueId(0))],
            output: ValueType::Tensor(spec),
        }],
        results: vec![ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2]))],
    };
    let program = Program::new("main").with_regions(vec![region0, region1]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");

    let region_plan0 = plan.region(RegionId(0)).expect("region0 plan");
    let region_plan1 = plan.region(RegionId(1)).expect("region1 plan");
    assert!(region_plan0.buffer_for(ValueId(0)).is_some());
    assert!(region_plan1.buffer_for(ValueId(0)).is_some());
    assert_ne!(region_plan0.buffers.len(), 0);
    assert_ne!(region_plan1.buffers.len(), 0);
}

#[test]
fn alias_ops_in_regions_work_like_functions() {
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let region = Region {
        id: RegionId(0),
        parameters: vec![ValueType::Tensor(spec.clone())],
        body: vec![Instruction {
            id: ValueId(10),
            op: Operation::Reshape(ReshapeSpec {
                new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
            }),
            operands: vec![Operand::Value(ValueId(0))],
            output: ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
        }],
        results: vec![ValueType::Tensor(tensor_spec_static(DType::F32, &[4]))],
    };
    let program = Program::new("main").with_regions(vec![region]);
    let plan = plan_buffers_with(&program, &strict_opts()).expect("plan");
    let region_plan = plan.region(RegionId(0)).expect("region plan");

    let buf = region_plan.buffer_for(ValueId(10)).expect("buffer");
    assert_eq!(buf.alias_kind, AliasKind::Identity);
    assert_eq!(buf.alias_of, Some(buffer_key(ValueId(0), &[])));
}

#[test]
fn view_alias_does_not_merge_live_ranges_can_enable_unsafe_reuse_regression() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![0, 1] }),
        vec![Operand::Value(a)],
        ValueType::Tensor(spec.clone()),
    );
    let c = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let d = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(b), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![d]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let a_spec = buffer_for(&plan, a, &[]);
    let c_spec = buffer_for(&plan, c, &[]);
    assert_eq!(a_spec.slot, c_spec.slot);
    assert_eq!(buffer_for(&plan, b, &[]).alias_kind, AliasKind::View);
}

#[test]
fn identity_alias_prevents_reuse_due_to_live_range_union() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let a = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let b = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(a)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let c = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let d = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(b), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![d]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let a_spec = buffer_for(&plan, a, &[]);
    let c_spec = buffer_for(&plan, c, &[]);
    assert_ne!(a_spec.slot, c_spec.slot);
}

#[test]
fn all_params_and_instr_outputs_have_buffer_specs_in_strict_mode() {
    let mut builder = ProgramBuilder::new();
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(elem.clone()));
    let tuple_param = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let t0 = builder.add_parameter(tuple_param);
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(elem.clone()),
    );
    let tuple_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem.clone()),
    ]);
    let v1 = builder.emit_single(Operation::StopGradient, vec![Operand::Value(p0)], tuple_ty);
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    assert!(plan.buffer_for(p0).is_some());
    assert!(plan.buffer_for_path(t0, &[0]).is_some());
    assert!(plan.buffer_for_path(t0, &[1]).is_some());
    assert!(plan.buffer_for(v0).is_some());
    assert!(plan.buffer_for_path(v1, &[0]).is_some());
    assert!(plan.buffer_for_path(v1, &[1]).is_some());
}

#[test]
fn strict_mode_produces_no_byte_len_none() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    for buffer in &plan.buffers {
        assert!(buffer.byte_len.is_some());
    }
}

#[test]
fn slot_indices_are_in_range_and_slots_match_buffers_dtype_len() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let _v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let _v1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![p0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    for buffer in &plan.buffers {
        if let Some(slot_id) = buffer.slot {
            let slot = plan.slots.get(slot_id).expect("slot in range");
            assert_eq!(slot.dtype, buffer.dtype);
            assert_eq!(slot.byte_len, buffer.byte_len);
        }
    }
}

#[test]
fn alias_kind_none_implies_alias_of_none() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![v0]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    for buffer in &plan.buffers {
        if buffer.alias_kind == AliasKind::None {
            assert!(buffer.alias_of.is_none());
        }
    }
}

#[test]
fn alias_kind_identity_or_view_implies_same_alias_group_as_source() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v1 = builder.emit_single(
        Operation::Transpose(TransposeSpec { perm: vec![0] }),
        vec![Operand::Value(v0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v1]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    for buffer in &plan.buffers {
        if buffer.alias_kind == AliasKind::None {
            continue;
        }
        let alias_of = buffer.alias_of.as_ref().expect("alias_of");
        let src = plan
            .values
            .get(alias_of)
            .and_then(|idx| plan.buffers.get(*idx))
            .expect("source buffer");
        assert_eq!(buffer.alias_group, src.alias_group);
    }
}

#[test]
fn identity_connected_component_has_uniform_live_range() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[2, 2]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let v0 = builder.emit_single(
        Operation::Reshape(ReshapeSpec {
            new_shape: vec![ReshapeDim::Explicit(Dimension::Static(4))],
        }),
        vec![Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v1 = builder.emit_single(
        Operation::StopGradient,
        vec![Operand::Value(v0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let v2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(v1), Operand::Value(p0)],
        ValueType::Tensor(tensor_spec_static(DType::F32, &[4])),
    );
    let function = builder.finish("main", vec![v2]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let mut parent: HashMap<BufferKey, BufferKey> = HashMap::new();
    for key in plan.values.keys() {
        parent.insert(key.clone(), key.clone());
    }
    fn find(parent: &mut HashMap<BufferKey, BufferKey>, key: &BufferKey) -> BufferKey {
        let mut current = key.clone();
        loop {
            let next = parent
                .get(&current)
                .cloned()
                .unwrap_or_else(|| current.clone());
            if next == current {
                return current;
            }
            current = next;
        }
    }
    for buffer in &plan.buffers {
        if buffer.alias_kind == AliasKind::Identity {
            if let Some(alias_of) = buffer.alias_of.clone() {
                let out_key = buffer_key(buffer.value, &buffer.path);
                let root_out = find(&mut parent, &out_key);
                let root_in = find(&mut parent, &alias_of);
                if root_out != root_in {
                    parent.insert(root_out, root_in);
                }
            }
        }
    }

    let mut ranges: HashMap<BufferKey, LiveRange> = HashMap::new();
    for buffer in &plan.buffers {
        let key = buffer_key(buffer.value, &buffer.path);
        let root = find(&mut parent, &key);
        ranges
            .entry(root)
            .and_modify(|range| {
                range.start = range.start.min(buffer.live_range.start);
                range.end = range.end.max(buffer.live_range.end);
            })
            .or_insert(buffer.live_range);
    }
    for buffer in &plan.buffers {
        let key = buffer_key(buffer.value, &buffer.path);
        let root = find(&mut parent, &key);
        let expected = ranges.get(&root).expect("range");
        assert_eq!(&buffer.live_range, expected);
    }
}

#[test]
fn slot_reuse_only_when_strictly_non_overlapping_for_non_identity_reuse() {
    let mut builder = ProgramBuilder::new();
    let spec = tensor_spec_static(DType::F32, &[4]);
    let p0 = builder.add_parameter(ValueType::Tensor(spec.clone()));
    let t0 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let t1 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(p0), Operand::Value(p0)],
        ValueType::Tensor(spec.clone()),
    );
    let t2 = builder.emit_single(
        Operation::ElementwiseBinary(ElementwiseBinaryOp::Add),
        vec![Operand::Value(t0), Operand::Value(t1)],
        ValueType::Tensor(spec),
    );
    let function = builder.finish("main", vec![t2]);
    let program = Program::new("main").with_functions(vec![function]);
    let plan = plan_main(&program, &strict_opts());

    let mut by_slot: HashMap<usize, Vec<&BufferSpec>> = HashMap::new();
    for buffer in &plan.buffers {
        if buffer.alias_kind == AliasKind::Identity {
            continue;
        }
        if let Some(slot) = buffer.slot {
            by_slot.entry(slot).or_default().push(buffer);
        }
    }
    for buffers in by_slot.values_mut() {
        buffers.sort_by_key(|buf| buf.live_range.start);
        for pair in buffers.windows(2) {
            assert!(pair[0].live_range.end < pair[1].live_range.start);
        }
    }
}

#[test]
fn type_mismatch_in_tuple_element_on_reinsertion() {
    let param_id = ValueId(0);
    let elem = tensor_spec_static(DType::F32, &[2, 2]);
    let param_ty = ValueType::Tuple(vec![
        ValueType::Tensor(elem.clone()),
        ValueType::Tensor(elem),
    ]);
    let output_ty = ValueType::Tuple(vec![
        ValueType::Tensor(tensor_spec_static(DType::F32, &[2, 2])),
        ValueType::Tensor(tensor_spec_static(DType::Si32, &[2, 2])),
    ]);
    let function = gpt_rs::backend::spec::Function {
        name: "main".to_string(),
        parameters: vec![param_ty],
        parameter_ids: vec![param_id],
        results: Vec::new(),
        body: vec![Instruction {
            id: param_id,
            op: Operation::StopGradient,
            operands: vec![Operand::Value(param_id)],
            output: output_ty,
        }],
        result_ids: Vec::new(),
    };
    let program = Program::new("main").with_functions(vec![function]);
    let err = plan_buffers_with(&program, &strict_opts()).expect_err("type mismatch");
    assert!(matches!(err, BufferizeError::TypeMismatch { value } if value == param_id));
}

#[test]
fn byte_len_overflow_in_one_buffer_fails_entire_plan() {
    let mut builder = ProgramBuilder::new();
    let spec_ok = tensor_spec_static(DType::F32, &[2, 2]);
    let spec_overflow = tensor_spec_static(DType::F32, &[usize::MAX, 2]);
    let _p0 = builder.add_parameter(ValueType::Tensor(spec_ok));
    let _p1 = builder.add_parameter(ValueType::Tensor(spec_overflow));
    let function = builder.finish("main", Vec::new());
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("overflow");
    assert!(matches!(err, BufferizeError::ByteLenOverflow { .. }));
}

#[test]
fn mixed_dynamic_and_static_buffers_strict_mode_fails() {
    let mut builder = ProgramBuilder::new();
    let spec_static = tensor_spec_static(DType::F32, &[2, 2]);
    let spec_dynamic = tensor_spec_mixed(DType::F32, &[None, Some(2)]);
    let _p0 = builder.add_parameter(ValueType::Tensor(spec_static));
    let _p1 = builder.add_parameter(ValueType::Tensor(spec_dynamic));
    let function = builder.finish("main", Vec::new());
    let program = Program::new("main").with_functions(vec![function]);

    let err = plan_buffers_with(&program, &strict_opts()).expect_err("dynamic shape");
    assert!(matches!(err, BufferizeError::DynamicShape { .. }));
}
