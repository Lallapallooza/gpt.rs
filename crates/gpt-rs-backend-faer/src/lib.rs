use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use gpt_rs::backend::param_resolver::ParamResolver;
use gpt_rs::backend::shape_helpers::static_dims_or_error;
use gpt_rs::backend::spec::{
    BackendError, BackendResult, CustomCallAttr, CustomCallSpec, DType, DotGeneralSpec,
    ExtractPatchesSpec, Function, Instruction, Literal, Operand, Operation, PortableBackend,
    Program, ReduceKind, ReduceWindowSpec, SpecErrorCode, TensorInit, TensorLiteral, TensorSpec,
    TransposeSpec, ValueId, ValueType,
};
use gpt_rs_backend_ref_cpu::{CpuKernelInterceptor, CpuTensor, GenericCpuBackend, TensorData};

mod optimizer;

thread_local! {
    static CONV2D_IM2COL_SCRATCH_F32: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

fn faer_parallelism() -> Par {
    let par = faer::get_global_parallelism();
    if par.degree() == 1 {
        Par::Seq
    } else {
        par
    }
}

#[derive(Default, Clone)]
pub struct FaerCpuInterceptor;

impl FaerCpuInterceptor {
    pub fn new() -> Self {
        Self
    }
}

impl CpuKernelInterceptor for FaerCpuInterceptor {
    fn try_execute(
        &self,
        op: &Operation,
        inputs: &[CpuTensor],
        outputs: &[TensorSpec],
    ) -> Option<BackendResult<Vec<CpuTensor>>> {
        match op {
            Operation::CustomCall(spec) => try_custom_call(inputs, outputs, spec),
            Operation::DotGeneral(spec) => try_dot_general(inputs, outputs, spec),
            Operation::ElementwiseBinary(kind) => try_elementwise_binary(inputs, outputs, *kind),
            Operation::Transpose(spec) => try_transpose(inputs, outputs, spec),
            Operation::ReduceWindow(spec) => try_reduce_window(inputs, outputs, spec),
            _ => None,
        }
    }
}

type FaerInnerBackend = GenericCpuBackend<FaerCpuInterceptor>;

/// Fused backend that accelerates `dot_general` via faer and recognizes common conv patterns.
///
/// The implementation keeps the reference CPU backend semantics for everything it does not
/// explicitly fuse.
#[derive(Clone)]
pub struct FaerPortableBackend {
    inner: FaerInnerBackend,
    params: Arc<FaerDerivedParamResolver>,
}

impl Default for FaerPortableBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl FaerPortableBackend {
    pub fn new() -> Self {
        FaerPortableBackend {
            inner: FaerInnerBackend::with_interceptor(FaerCpuInterceptor),
            params: Arc::new(FaerDerivedParamResolver::new()),
        }
    }

    fn materialize_literal(&self, lit: &TensorLiteral) -> BackendResult<CpuTensor> {
        self.inner.materialize(TensorInit::Literal(lit.clone()))
    }

    fn try_fuse_depthwise_conv2d_nhwc(
        &self,
        function: &Function,
        inst_index: usize,
        use_counts: &HashMap<ValueId, usize>,
        values: &HashMap<ValueId, CpuTensor>,
    ) -> BackendResult<Option<(usize, ValueId, CpuTensor)>> {
        let body = &function.body;
        let extract = match body.get(inst_index) {
            Some(inst) => inst,
            None => return Ok(None),
        };
        let extract_spec = match &extract.op {
            Operation::ExtractPatches(spec) => spec,
            _ => return Ok(None),
        };
        if !matches!(extract_spec.pad_value, Literal::Float(v) if v == 0.0) {
            return Ok(None);
        }

        let extract_input = match extract.operands.as_slice() {
            [Operand::Value(id)] => *id,
            _ => return Ok(None),
        };
        if !matches!(use_counts.get(&extract.id), Some(1)) {
            return Ok(None);
        }

        // patches.reshape([flattened, KH*KW, G, C_in/G])
        let reshape_patches = match body.get(inst_index + 1) {
            Some(inst) => inst,
            None => return Ok(None),
        };
        if !matches!(reshape_patches.op, Operation::Reshape(_)) {
            return Ok(None);
        }
        let reshape_patches_input = match reshape_patches.operands.as_slice() {
            [Operand::Value(id)] if *id == extract.id => *id,
            _ => return Ok(None),
        };
        let _ = reshape_patches_input;
        if !matches!(use_counts.get(&reshape_patches.id), Some(1)) {
            return Ok(None);
        }

        // Optional weight reshape before dot_general.
        let mut cursor = inst_index + 2;
        let (weight_source, dot_inst) = match body.get(cursor) {
            Some(inst) => match &inst.op {
                Operation::Reshape(_) => {
                    let weight_source = match inst.operands.as_slice() {
                        [Operand::Value(id)] => *id,
                        _ => return Ok(None),
                    };
                    if !matches!(use_counts.get(&inst.id), Some(1)) {
                        return Ok(None);
                    }
                    cursor += 1;
                    let dot = match body.get(cursor) {
                        Some(dot) => dot,
                        None => return Ok(None),
                    };
                    (Some(weight_source), dot)
                }
                Operation::DotGeneral(_) => (None, inst),
                _ => return Ok(None),
            },
            None => return Ok(None),
        };

        let dot_spec = match &dot_inst.op {
            Operation::DotGeneral(spec) => spec,
            _ => return Ok(None),
        };
        if dot_spec.batch_lhs.as_slice() != [2]
            || dot_spec.batch_rhs.as_slice() != [0]
            || dot_spec.contract_lhs.as_slice() != [1, 3]
            || dot_spec.contract_rhs.as_slice() != [1, 2]
        {
            return Ok(None);
        }

        let (lhs_id, rhs_id) = match dot_inst.operands.as_slice() {
            [Operand::Value(lhs), Operand::Value(rhs)] if *lhs == reshape_patches.id => {
                (*lhs, *rhs)
            }
            _ => return Ok(None),
        };
        let _ = lhs_id;

        // If we had a weight reshape op, the rhs is that reshape value; use the original source.
        let weight_id = match weight_source {
            Some(src) => {
                let weight_reshape = body
                    .get(inst_index + 2)
                    .ok_or_else(|| BackendError::execution("missing weight reshape instruction"))?;
                if weight_reshape.id != rhs_id {
                    return Ok(None);
                }
                src
            }
            None => rhs_id,
        };

        if !matches!(use_counts.get(&dot_inst.id), Some(1)) {
            return Ok(None);
        }
        cursor += 1;

        // out.transpose([1,0,2])
        let transpose = match body.get(cursor) {
            Some(inst) => inst,
            None => return Ok(None),
        };
        let transpose_spec = match &transpose.op {
            Operation::Transpose(spec) => spec,
            _ => return Ok(None),
        };
        if transpose_spec.perm.as_slice() != [1, 0, 2] {
            return Ok(None);
        }
        match transpose.operands.as_slice() {
            [Operand::Value(id)] if *id == dot_inst.id => {}
            _ => return Ok(None),
        }
        if !matches!(use_counts.get(&transpose.id), Some(1)) {
            return Ok(None);
        }
        cursor += 1;

        // Either reshape(transpose) -> output, or reshape(transpose) -> [flattened, c_out] -> reshape -> output.
        let reshape_a = match body.get(cursor) {
            Some(inst) => inst,
            None => return Ok(None),
        };
        if !matches!(reshape_a.op, Operation::Reshape(_)) {
            return Ok(None);
        }
        match reshape_a.operands.as_slice() {
            [Operand::Value(id)] if *id == transpose.id => {}
            _ => return Ok(None),
        }

        let reshape_a_spec = match &reshape_a.output {
            ValueType::Tensor(spec) => spec,
            _ => return Ok(None),
        };
        if reshape_a_spec.dtype != DType::F32 {
            return Ok(None);
        }

        let (final_reshape_id, final_output_spec, cursor) =
            match static_dims_or_error(&reshape_a_spec.shape, |sym| {
                BackendError::execution(format!(
                    "dynamic dimension {} not supported at runtime",
                    sym.as_str()
                ))
            })?
            .len()
            {
                4 => (reshape_a.id, reshape_a_spec.clone(), cursor + 1),
                2 => {
                    if !matches!(use_counts.get(&reshape_a.id), Some(1)) {
                        return Ok(None);
                    }
                    let reshape_b = match body.get(cursor + 1) {
                        Some(inst) => inst,
                        None => return Ok(None),
                    };
                    if !matches!(reshape_b.op, Operation::Reshape(_)) {
                        return Ok(None);
                    }
                    match reshape_b.operands.as_slice() {
                        [Operand::Value(id)] if *id == reshape_a.id => {}
                        _ => return Ok(None),
                    }
                    let spec_b = match &reshape_b.output {
                        ValueType::Tensor(spec) => spec,
                        _ => return Ok(None),
                    };
                    if spec_b.dtype != DType::F32 {
                        return Ok(None);
                    }
                    if static_dims_or_error(&spec_b.shape, |sym| {
                        BackendError::execution(format!(
                            "dynamic dimension {} not supported at runtime",
                            sym.as_str()
                        ))
                    })?
                    .len()
                        != 4
                    {
                        return Ok(None);
                    }
                    cursor += 2;
                    (reshape_b.id, spec_b.clone(), cursor)
                }
                _ => return Ok(None),
            };

        // Optional bias broadcast + add.
        let bias_and_add = body.get(cursor).zip(body.get(cursor + 1));
        let (consumed, output_id, bias_id) = match bias_and_add {
            Some((broadcast, add)) => {
                let bias_id = match broadcast.operands.as_slice() {
                    [Operand::Value(id)] => *id,
                    _ => return Ok(None),
                };
                let broadcast_ok = match (&broadcast.op, &broadcast.output) {
                    (Operation::BroadcastTo(spec), ValueType::Tensor(out)) => {
                        out.dtype == DType::F32
                            && out.shape == spec.result_shape
                            && out.shape == final_output_spec.shape
                    }
                    _ => false,
                };
                if !broadcast_ok {
                    return Ok(None);
                }
                if !matches!(use_counts.get(&broadcast.id), Some(1)) {
                    return Ok(None);
                }
                let add_ok = match (&add.op, add.operands.as_slice(), &add.output) {
                    (
                        Operation::ElementwiseBinary(op),
                        [Operand::Value(a), Operand::Value(b)],
                        ValueType::Tensor(out),
                    ) => {
                        *op == gpt_rs::backend::spec::ElementwiseBinaryOp::Add
                            && *a == final_reshape_id
                            && *b == broadcast.id
                            && out.dtype == final_output_spec.dtype
                            && out.shape == final_output_spec.shape
                    }
                    _ => false,
                };
                if !add_ok {
                    return Ok(None);
                }
                if !matches!(use_counts.get(&final_reshape_id), Some(1)) {
                    return Ok(None);
                }
                (
                    (cursor + 2).saturating_sub(inst_index),
                    add.id,
                    Some(bias_id),
                )
            }
            None => (cursor.saturating_sub(inst_index), final_reshape_id, None),
        };

        let input = values.get(&extract_input).cloned().ok_or_else(|| {
            BackendError::execution("depthwise conv fusion missing extract_patches input value")
        })?;
        let weight = values.get(&weight_id).cloned().ok_or_else(|| {
            BackendError::execution("depthwise conv fusion missing dot_general weight value")
        })?;
        let bias = match bias_id {
            Some(id) => Some(values.get(&id).cloned().ok_or_else(|| {
                BackendError::execution("depthwise conv fusion missing bias value")
            })?),
            None => None,
        };

        // Only fuse depthwise case for now: weight is [C, KH*KW] (or an equivalent packed 4D view)
        // and produces [N, out_h, out_w, C].
        let weight_dims = static_dims_or_error(&weight.spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        let out_dims = static_dims_or_error(&final_output_spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        if out_dims.len() != 4 {
            return Ok(None);
        }
        let c_out = out_dims[3];
        let input_dims = static_dims_or_error(&input.spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        if input_dims.len() != 4 {
            return Ok(None);
        }
        let c_in = input_dims[3];
        if c_out != c_in {
            return Ok(None);
        }
        let k_h = extract_spec.window[0];
        let k_w = extract_spec.window[1];
        let khkw = k_h.saturating_mul(k_w);
        let is_depthwise_packed = match weight_dims.as_slice() {
            [c, k] => *c == c_in && *k == khkw,
            [c, k, one_in, one_out] => *c == c_in && *k == khkw && *one_in == 1 && *one_out == 1,
            _ => false,
        };
        if !is_depthwise_packed {
            return Ok(None);
        }

        let _prof_guard = gpt_rs::profiling::backend_scope("backend.fused_depthwise_conv2d_nhwc");
        let fused = depthwise_conv2d_nhwc_direct_f32(
            &input,
            &weight,
            bias.as_ref(),
            extract_spec,
            &final_output_spec,
        )?;
        Ok(Some((consumed, output_id, fused)))
    }

    fn try_fuse_relu6(
        &self,
        function: &Function,
        inst_index: usize,
        use_counts: &HashMap<ValueId, usize>,
        values: &HashMap<ValueId, CpuTensor>,
    ) -> BackendResult<Option<(usize, ValueId, CpuTensor)>> {
        fn literal_f32(lit: &TensorLiteral) -> Option<f32> {
            if lit.spec.dtype != DType::F32 {
                return None;
            }
            if !lit.spec.shape.dims().is_empty() || lit.bytes.len() != 4 {
                return None;
            }
            let bytes: [u8; 4] = lit.bytes.as_ref().try_into().ok()?;
            Some(f32::from_le_bytes(bytes))
        }

        let body = &function.body;
        // Current PTIR lowering for relu6 typically looks like:
        //   s0 = broadcast_to([] , literal 0.0)
        //   z  = broadcast_to([..], value s0)
        //   s6 = broadcast_to([] , literal 6.0)
        //   s  = broadcast_to([..], value s6)
        //   m  = maximum(x, z)
        //   y  = minimum(m, s)
        //
        // Older variants may broadcast literal scalars directly to the full shape; handle both.
        if let (Some(a0), Some(a1), Some(b0), Some(b1), Some(max_inst), Some(min_inst)) = (
            body.get(inst_index),
            body.get(inst_index + 1),
            body.get(inst_index + 2),
            body.get(inst_index + 3),
            body.get(inst_index + 4),
            body.get(inst_index + 5),
        ) {
            let parse_scalar = |inst: &Instruction| -> Option<f32> {
                match (&inst.op, &inst.output, inst.operands.as_slice()) {
                    (
                        Operation::BroadcastTo(spec),
                        ValueType::Tensor(out),
                        [Operand::Literal(lit)],
                    ) if out.dtype == DType::F32
                        && out.shape == spec.result_shape
                        && out.shape.dims().is_empty() =>
                    {
                        literal_f32(lit)
                    }
                    _ => None,
                }
            };

            let parse_broadcast_like =
                |inst: &Instruction, scalar_id: ValueId| -> Option<TensorSpec> {
                    match (&inst.op, &inst.output, inst.operands.as_slice()) {
                        (
                            Operation::BroadcastTo(spec),
                            ValueType::Tensor(out),
                            [Operand::Value(id)],
                        ) if *id == scalar_id
                            && out.dtype == DType::F32
                            && out.shape == spec.result_shape =>
                        {
                            Some(out.clone())
                        }
                        _ => None,
                    }
                };

            if let (Some(a_val), Some(b_val)) = (parse_scalar(a0), parse_scalar(b0)) {
                let a_out = match parse_broadcast_like(a1, a0.id) {
                    Some(s) => s,
                    None => return Ok(None),
                };
                let b_out = match parse_broadcast_like(b1, b0.id) {
                    Some(s) => s,
                    None => return Ok(None),
                };
                if a_out.shape != b_out.shape {
                    return Ok(None);
                }

                let (zero_id, six_id) = match (a_val, b_val) {
                    (v0, v6) if v0 == 0.0 && v6 == 6.0 => (a1.id, b1.id),
                    (v6, v0) if v0 == 0.0 && v6 == 6.0 => (b1.id, a1.id),
                    _ => return Ok(None),
                };

                if !matches!(use_counts.get(&a0.id), Some(1))
                    || !matches!(use_counts.get(&a1.id), Some(1))
                    || !matches!(use_counts.get(&b0.id), Some(1))
                    || !matches!(use_counts.get(&b1.id), Some(1))
                    || !matches!(use_counts.get(&max_inst.id), Some(1))
                {
                    return Ok(None);
                }

                let (x_id, max_out_spec) =
                    match (&max_inst.op, max_inst.operands.as_slice(), &max_inst.output) {
                        (
                            Operation::ElementwiseBinary(
                                gpt_rs::backend::spec::ElementwiseBinaryOp::Maximum,
                            ),
                            [Operand::Value(lhs), Operand::Value(rhs)],
                            ValueType::Tensor(out),
                        ) => {
                            let has_zero = (*lhs == zero_id) || (*rhs == zero_id);
                            if !has_zero {
                                return Ok(None);
                            }
                            let x_id = if *lhs == zero_id { *rhs } else { *lhs };
                            (x_id, out)
                        }
                        _ => return Ok(None),
                    };

                if max_out_spec.dtype != DType::F32 || max_out_spec.shape != a_out.shape {
                    return Ok(None);
                }

                let min_out_spec =
                    match (&min_inst.op, min_inst.operands.as_slice(), &min_inst.output) {
                        (
                            Operation::ElementwiseBinary(
                                gpt_rs::backend::spec::ElementwiseBinaryOp::Minimum,
                            ),
                            [Operand::Value(lhs), Operand::Value(rhs)],
                            ValueType::Tensor(out),
                        ) => {
                            let has_six = (*lhs == six_id) || (*rhs == six_id);
                            let has_max = (*lhs == max_inst.id) || (*rhs == max_inst.id);
                            if !has_six || !has_max {
                                return Ok(None);
                            }
                            out
                        }
                        _ => return Ok(None),
                    };
                if min_out_spec.dtype != DType::F32 || min_out_spec.shape != a_out.shape {
                    return Ok(None);
                }

                let x = values.get(&x_id).cloned().ok_or_else(|| {
                    BackendError::execution("relu6 fusion missing maximum input tensor value")
                })?;
                let x_values = match &x.data {
                    TensorData::F32(values) => values.as_ref(),
                    _ => return Ok(None),
                };
                let out_dims = static_dims_or_error(&min_out_spec.shape, |sym| {
                    BackendError::execution(format!(
                        "dynamic dimension {} not supported at runtime",
                        sym.as_str()
                    ))
                })?;
                let out_len: usize = out_dims.iter().product();
                if out_len != x_values.len() {
                    return Ok(None);
                }

                let _prof_guard = gpt_rs::profiling::backend_scope("backend.fused_relu6");
                let mut out = vec![0.0f32; out_len];
                for (dst, &v) in out.iter_mut().zip(x_values.iter()) {
                    *dst = v.clamp(0.0, 6.0);
                }
                return Ok(Some((
                    6,
                    min_inst.id,
                    CpuTensor {
                        spec: min_out_spec.clone(),
                        data: TensorData::F32(Arc::from(out.into_boxed_slice())),
                    },
                )));
            }
        }

        let a = body.get(inst_index);
        let b = body.get(inst_index + 1);
        let max_inst = body.get(inst_index + 2);
        let min_inst = body.get(inst_index + 3);
        let (a, b, max_inst, min_inst) = match (a, b, max_inst, min_inst) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Ok(None),
        };

        let operand_f32 = |operand: &Operand| -> Option<f32> {
            match operand {
                Operand::Literal(lit) => literal_f32(lit),
                Operand::Value(id) => {
                    let scalar = values.get(id)?;
                    if scalar.spec.dtype != DType::F32 || !scalar.spec.shape.dims().is_empty() {
                        return None;
                    }
                    match &scalar.data {
                        TensorData::F32(values) if values.len() == 1 => Some(values[0]),
                        _ => None,
                    }
                }
                _ => None,
            }
        };

        let (a_val, a_out) = match (&a.op, &a.output, a.operands.as_slice()) {
            (Operation::BroadcastTo(spec), ValueType::Tensor(out), [operand]) => {
                if out.dtype != DType::F32 || out.shape != spec.result_shape {
                    return Ok(None);
                }
                let val = match operand_f32(operand) {
                    Some(v) => v,
                    None => return Ok(None),
                };
                (val, out)
            }
            _ => return Ok(None),
        };
        let (b_val, b_out) = match (&b.op, &b.output, b.operands.as_slice()) {
            (Operation::BroadcastTo(spec), ValueType::Tensor(out), [operand]) => {
                if out.dtype != DType::F32 || out.shape != spec.result_shape {
                    return Ok(None);
                }
                let val = match operand_f32(operand) {
                    Some(v) => v,
                    None => return Ok(None),
                };
                (val, out)
            }
            _ => return Ok(None),
        };

        // Identify which broadcast corresponds to 0.0 and 6.0.
        let (zero_id, six_id) = match (a_val, b_val) {
            (v0, v6) if v0 == 0.0 && v6 == 6.0 => (a.id, b.id),
            (v6, v0) if v0 == 0.0 && v6 == 6.0 => (b.id, a.id),
            _ => return Ok(None),
        };

        if a_out.shape != b_out.shape {
            return Ok(None);
        }
        if !matches!(use_counts.get(&a.id), Some(1)) || !matches!(use_counts.get(&b.id), Some(1)) {
            return Ok(None);
        }

        let (x_id, max_out_spec) =
            match (&max_inst.op, max_inst.operands.as_slice(), &max_inst.output) {
                (
                    Operation::ElementwiseBinary(
                        gpt_rs::backend::spec::ElementwiseBinaryOp::Maximum,
                    ),
                    [Operand::Value(lhs), Operand::Value(rhs)],
                    ValueType::Tensor(out),
                ) => {
                    let has_zero = (*lhs == zero_id) || (*rhs == zero_id);
                    if !has_zero {
                        return Ok(None);
                    }
                    let x_id = if *lhs == zero_id { *rhs } else { *lhs };
                    (x_id, out)
                }
                _ => return Ok(None),
            };

        if max_out_spec.dtype != DType::F32 || max_out_spec.shape != a_out.shape {
            return Ok(None);
        }
        if !matches!(use_counts.get(&max_inst.id), Some(1)) {
            return Ok(None);
        }

        let min_out_spec = match (&min_inst.op, min_inst.operands.as_slice(), &min_inst.output) {
            (
                Operation::ElementwiseBinary(gpt_rs::backend::spec::ElementwiseBinaryOp::Minimum),
                [Operand::Value(lhs), Operand::Value(rhs)],
                ValueType::Tensor(out),
            ) => {
                let has_six = (*lhs == six_id) || (*rhs == six_id);
                let has_max = (*lhs == max_inst.id) || (*rhs == max_inst.id);
                if !has_six || !has_max {
                    return Ok(None);
                }
                out
            }
            _ => return Ok(None),
        };
        if min_out_spec.dtype != DType::F32 || min_out_spec.shape != a_out.shape {
            return Ok(None);
        }

        let x = values.get(&x_id).cloned().ok_or_else(|| {
            BackendError::execution("relu6 fusion missing maximum input tensor value")
        })?;
        let x_values = match &x.data {
            TensorData::F32(values) => values.as_ref(),
            _ => return Ok(None),
        };
        let out_dims = static_dims_or_error(&min_out_spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        let out_len: usize = out_dims.iter().product();
        if out_len != x_values.len() {
            return Ok(None);
        }

        let _prof_guard = gpt_rs::profiling::backend_scope("backend.fused_relu6");
        let mut out = vec![0.0f32; out_len];
        for (dst, &v) in out.iter_mut().zip(x_values.iter()) {
            *dst = v.clamp(0.0, 6.0);
        }
        Ok(Some((
            4,
            min_inst.id,
            CpuTensor {
                spec: min_out_spec.clone(),
                data: TensorData::F32(Arc::from(out.into_boxed_slice())),
            },
        )))
    }

    fn try_fuse_conv2d_nhwc(
        &self,
        function: &Function,
        inst_index: usize,
        use_counts: &HashMap<ValueId, usize>,
        values: &HashMap<ValueId, CpuTensor>,
    ) -> BackendResult<Option<(usize, ValueId, CpuTensor)>> {
        let body = &function.body;
        let extract = body.get(inst_index);
        let reshape1 = body.get(inst_index + 1);
        let dot = body.get(inst_index + 2);
        let reshape2 = body.get(inst_index + 3);

        let (extract, reshape1, dot, reshape2) = match (extract, reshape1, dot, reshape2) {
            (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
            _ => return Ok(None),
        };

        let extract_spec = match &extract.op {
            Operation::ExtractPatches(spec) => spec,
            _ => return Ok(None),
        };
        if !matches!(extract_spec.pad_value, Literal::Float(v) if v == 0.0) {
            return Ok(None);
        }

        let extract_input = match extract.operands.as_slice() {
            [Operand::Value(id)] => *id,
            _ => return Ok(None),
        };

        if !matches!(use_counts.get(&extract.id), Some(1)) {
            return Ok(None);
        }

        let reshape1_input = match reshape1.operands.as_slice() {
            [Operand::Value(id)] if *id == extract.id => *id,
            _ => return Ok(None),
        };
        let _ = reshape1_input;
        if !matches!(reshape1.op, Operation::Reshape(_)) {
            return Ok(None);
        }
        if !matches!(use_counts.get(&reshape1.id), Some(1)) {
            return Ok(None);
        }

        let (lhs_id, rhs_id) = match dot.operands.as_slice() {
            [Operand::Value(lhs), Operand::Value(rhs)] if *lhs == reshape1.id => (*lhs, *rhs),
            _ => return Ok(None),
        };
        let _ = lhs_id;
        let dot_spec = match &dot.op {
            Operation::DotGeneral(spec) => spec,
            _ => return Ok(None),
        };
        if !dot_spec.batch_lhs.is_empty()
            || !dot_spec.batch_rhs.is_empty()
            || dot_spec.contract_lhs.as_slice() != [1]
            || dot_spec.contract_rhs.as_slice() != [0]
        {
            return Ok(None);
        }
        if !matches!(use_counts.get(&dot.id), Some(1)) {
            return Ok(None);
        }

        let reshape2_input = match reshape2.operands.as_slice() {
            [Operand::Value(id)] if *id == dot.id => *id,
            _ => return Ok(None),
        };
        let _ = reshape2_input;
        if !matches!(reshape2.op, Operation::Reshape(_)) {
            return Ok(None);
        }

        let output_spec = match &reshape2.output {
            ValueType::Tensor(spec) => spec,
            _ => return Ok(None),
        };
        if output_spec.dtype != DType::F32 {
            return Ok(None);
        }

        let bias_and_add = body.get(inst_index + 4).zip(body.get(inst_index + 5));
        let (consumed, output_id, bias_id) = match bias_and_add {
            Some((broadcast, add)) => {
                let bias_id = match broadcast.operands.as_slice() {
                    [Operand::Value(id)] => *id,
                    _ => return Ok(None),
                };
                let broadcast_ok = match (&broadcast.op, &broadcast.output) {
                    (Operation::BroadcastTo(spec), ValueType::Tensor(out)) => {
                        out.dtype == DType::F32
                            && out.shape == spec.result_shape
                            && out.shape == output_spec.shape
                    }
                    _ => false,
                };
                if !broadcast_ok {
                    return Ok(None);
                }
                if !matches!(use_counts.get(&broadcast.id), Some(1)) {
                    return Ok(None);
                }
                let add_ok = match (&add.op, add.operands.as_slice(), &add.output) {
                    (
                        Operation::ElementwiseBinary(op),
                        [Operand::Value(a), Operand::Value(b)],
                        ValueType::Tensor(out),
                    ) => {
                        *op == gpt_rs::backend::spec::ElementwiseBinaryOp::Add
                            && *a == reshape2.id
                            && *b == broadcast.id
                            && out.dtype == output_spec.dtype
                            && out.shape == output_spec.shape
                    }
                    _ => false,
                };
                if !add_ok {
                    return Ok(None);
                }
                if !matches!(use_counts.get(&reshape2.id), Some(1)) {
                    return Ok(None);
                }
                (6usize, add.id, Some(bias_id))
            }
            None => (4usize, reshape2.id, None),
        };

        let input = values.get(&extract_input).cloned().ok_or_else(|| {
            BackendError::execution("conv fusion missing extract_patches input value")
        })?;
        let weight = values.get(&rhs_id).cloned().ok_or_else(|| {
            BackendError::execution("conv fusion missing dot_general weight value")
        })?;
        let bias = match bias_id {
            Some(id) => Some(values.get(&id).cloned().ok_or_else(|| {
                BackendError::execution("conv fusion missing dot_general bias value")
            })?),
            None => None,
        };

        let _prof_guard = gpt_rs::profiling::backend_scope("backend.fused_conv2d_nhwc");
        let fused = conv2d_nhwc_im2col_tiled_f32(
            &input,
            &weight,
            bias.as_ref(),
            extract_spec,
            output_spec,
        )?;
        Ok(Some((consumed, output_id, fused)))
    }
}

#[derive(Default)]
struct FaerDerivedParamResolver {
    entries: Mutex<HashMap<u128, CpuTensor>>,
}

impl FaerDerivedParamResolver {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl ParamResolver for FaerDerivedParamResolver {
    type Handle = CpuTensor;

    fn get(&self, key: u128) -> Option<Self::Handle> {
        self.entries
            .lock()
            .expect("const cache poisoned")
            .get(&key)
            .cloned()
    }

    fn set(&self, key: u128, handle: Self::Handle) {
        self.entries
            .lock()
            .expect("const cache poisoned")
            .insert(key, handle);
    }
}

impl PortableBackend for FaerPortableBackend {
    type TensorHandle = CpuTensor;

    fn backend_name(&self) -> &str {
        "faer"
    }

    fn pipeline(&self) -> Option<Arc<dyn gpt_rs::backend::pipeline::BackendPipeline<Self>>> {
        Some(Arc::new(optimizer::FaerPipeline))
    }

    fn param_resolver(
        &self,
    ) -> Option<Arc<dyn gpt_rs::backend::param_resolver::ParamResolver<Handle = Self::TensorHandle>>>
    {
        Some(self.params.clone())
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle> {
        self.inner.materialize(init)
    }

    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        self.inner.to_literal(tensor)
    }

    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        self.inner.execute_instruction(instruction, inputs)
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let function = program
            .functions
            .iter()
            .find(|f| f.name == program.entry)
            .ok_or_else(|| BackendError::execution("entry function not found"))?;

        if function.parameter_ids.len() != entry_inputs.len() {
            return Err(BackendError::execution("entry input arity mismatch"));
        }

        let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
        for inst in &function.body {
            for operand in &inst.operands {
                match operand {
                    Operand::Value(id) => *use_counts.entry(*id).or_default() += 1,
                    Operand::TupleElement { tuple, .. } => {
                        *use_counts.entry(*tuple).or_default() += 1
                    }
                    Operand::Literal(_) => {}
                }
            }
        }
        for id in &function.result_ids {
            *use_counts.entry(*id).or_default() += 1;
        }

        let mut values: HashMap<ValueId, CpuTensor> = HashMap::new();
        for (param_id, handle) in function.parameter_ids.iter().zip(entry_inputs.iter()) {
            values.insert(*param_id, handle.clone());
        }

        let mut idx = 0usize;
        while idx < function.body.len() {
            if let Some((consumed, out_id, out_tensor)) =
                self.try_fuse_depthwise_conv2d_nhwc(function, idx, &use_counts, &values)?
            {
                values.insert(out_id, out_tensor);
                idx = idx
                    .checked_add(consumed)
                    .ok_or_else(|| BackendError::execution("instruction index overflow"))?;
                continue;
            }
            if let Some((consumed, out_id, out_tensor)) =
                self.try_fuse_conv2d_nhwc(function, idx, &use_counts, &values)?
            {
                values.insert(out_id, out_tensor);
                idx = idx
                    .checked_add(consumed)
                    .ok_or_else(|| BackendError::execution("instruction index overflow"))?;
                continue;
            }
            if let Some((consumed, out_id, out_tensor)) =
                self.try_fuse_relu6(function, idx, &use_counts, &values)?
            {
                values.insert(out_id, out_tensor);
                idx = idx
                    .checked_add(consumed)
                    .ok_or_else(|| BackendError::execution("instruction index overflow"))?;
                continue;
            }

            let instruction = &function.body[idx];
            let mut inputs = Vec::with_capacity(instruction.operands.len());
            for (operand_idx, operand) in instruction.operands.iter().enumerate() {
                let tensor = match operand {
                    Operand::Value(id) => {
                        if matches!(&instruction.op, Operation::DynamicUpdateSlice(_))
                            && operand_idx == 0
                            && matches!(use_counts.get(id), Some(&1))
                        {
                            values
                                .remove(id)
                                .ok_or_else(|| BackendError::execution("operand value missing"))?
                        } else {
                            values
                                .get(id)
                                .cloned()
                                .ok_or_else(|| BackendError::execution("operand value missing"))?
                        }
                    }
                    Operand::TupleElement { .. } => {
                        return Err(BackendError::execution("tuple operands not supported"))
                    }
                    Operand::Literal(lit) => self.materialize_literal(lit)?,
                };
                inputs.push(tensor);
            }

            let output = match &instruction.op {
                Operation::DynamicUpdateSlice(spec) => {
                    let output_spec = match &instruction.output {
                        ValueType::Tensor(spec) => spec,
                        ValueType::Tuple(_) => {
                            return Err(BackendError::execution(
                                "tuple outputs are not supported in faer backend",
                            ))
                        }
                    };

                    if inputs.len() != 3 {
                        return Err(BackendError::execution(
                            "dynamic_update_slice expects (base, update, starts)",
                        ));
                    }

                    let mut inputs = inputs;
                    let mut base = inputs.remove(0);
                    let update = &inputs[0];
                    let starts = &inputs[1];

                    if dynamic_update_slice_inplace_fast(
                        &mut base,
                        update,
                        starts,
                        output_spec,
                        spec,
                    )? {
                        base
                    } else {
                        let mut full_inputs = Vec::with_capacity(3);
                        full_inputs.push(base);
                        full_inputs.extend(inputs);
                        let mut outputs =
                            self.inner.execute_instruction(instruction, &full_inputs)?;
                        if outputs.len() != 1 {
                            return Err(BackendError::execution(
                                "instructions must produce exactly one result",
                            ));
                        }
                        outputs
                            .pop()
                            .expect("single output guaranteed by length check")
                    }
                }
                _ => {
                    let mut outputs = self.inner.execute_instruction(instruction, &inputs)?;
                    if outputs.len() != 1 {
                        return Err(BackendError::execution(
                            "instructions must produce exactly one result",
                        ));
                    }
                    outputs
                        .pop()
                        .expect("single output guaranteed by length check")
                }
            };

            values.insert(instruction.id, output);
            idx += 1;
        }

        let mut results = Vec::with_capacity(function.result_ids.len());
        for id in &function.result_ids {
            let value = values
                .get(id)
                .cloned()
                .ok_or_else(|| BackendError::execution("missing function result value"))?;
            results.push(value);
        }
        Ok(results)
    }
}

fn dynamic_update_slice_inplace_fast(
    base: &mut CpuTensor,
    update: &CpuTensor,
    starts_tensor: &CpuTensor,
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::DynamicUpdateSliceSpec,
) -> BackendResult<bool> {
    let base_dims = static_dims_or_error(&base.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    let update_dims = static_dims_or_error(&update.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;

    if base_dims.len() != spec.sizes.len() {
        return Ok(false);
    }
    if update_dims != spec.sizes {
        return Ok(false);
    }
    if base.spec.dtype != update.spec.dtype {
        return Ok(false);
    }

    let starts_vals = match &starts_tensor.data {
        TensorData::Si32(values) => values.as_ref(),
        _ => return Ok(false),
    };
    let starts_dims = static_dims_or_error(&starts_tensor.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    if starts_dims.len() != 1 || starts_dims[0] != base_dims.len() {
        return Ok(false);
    }

    let mut clamped_starts = Vec::with_capacity(base_dims.len());
    for axis in 0..base_dims.len() {
        let dim = base_dims[axis];
        let size = spec.sizes[axis];
        if size > dim {
            return Ok(false);
        }
        let max_start = dim - size;
        let mut start = starts_vals[axis] as isize;
        if start < 0 {
            start = 0;
        }
        let start = start as usize;
        clamped_starts.push(if start > max_start { max_start } else { start });
    }

    if &base.spec != output {
        return Ok(false);
    }

    if base_dims.len() != 3
        || update_dims.len() != 3
        || update_dims[0] != base_dims[0]
        || update_dims[2] != base_dims[2]
        || clamped_starts[0] != 0
        || clamped_starts[2] != 0
    {
        return Ok(false);
    }

    let heads = base_dims[0];
    let base_seq = base_dims[1];
    let head_dim = base_dims[2];
    let update_seq = update_dims[1];
    let start_seq = clamped_starts[1];

    match (&mut base.data, &update.data) {
        (TensorData::F32(base_vals), TensorData::F32(update_vals)) => {
            let base_slice = Arc::make_mut(base_vals);
            let update_slice = update_vals.as_ref();
            for head in 0..heads {
                let dst_head_base = head * base_seq * head_dim;
                let src_head_base = head * update_seq * head_dim;
                for t in 0..update_seq {
                    let dst_offset = dst_head_base + (start_seq + t) * head_dim;
                    let src_offset = src_head_base + t * head_dim;
                    base_slice[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&update_slice[src_offset..src_offset + head_dim]);
                }
            }
            Ok(true)
        }
        (TensorData::Si32(base_vals), TensorData::Si32(update_vals)) => {
            let base_slice = Arc::make_mut(base_vals);
            let update_slice = update_vals.as_ref();
            for head in 0..heads {
                let dst_head_base = head * base_seq * head_dim;
                let src_head_base = head * update_seq * head_dim;
                for t in 0..update_seq {
                    let dst_offset = dst_head_base + (start_seq + t) * head_dim;
                    let src_offset = src_head_base + t * head_dim;
                    base_slice[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&update_slice[src_offset..src_offset + head_dim]);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub struct FaerCpuBackend;

impl FaerCpuBackend {
    pub fn create() -> FaerPortableBackend {
        FaerPortableBackend::new()
    }
}

fn conv2d_nhwc_im2col_tiled_f32(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    spec: &ExtractPatchesSpec,
    output_spec: &TensorSpec,
) -> BackendResult<CpuTensor> {
    let input_dims = static_dims_or_error(&input.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    let weight_dims = static_dims_or_error(&weight.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    let out_dims = static_dims_or_error(&output_spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;

    if input.spec.dtype != DType::F32
        || weight.spec.dtype != DType::F32
        || output_spec.dtype != DType::F32
    {
        return Err(BackendError::execution("conv2d fusion only supports f32"));
    }
    if input_dims.len() != 4 || out_dims.len() != 4 || weight_dims.len() != 2 {
        return Err(BackendError::execution("conv2d fusion shape rank mismatch"));
    }
    let (n, h, w, c_in) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, c_out) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n {
        return Err(BackendError::execution("conv2d fusion batch mismatch"));
    }
    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return Err(BackendError::execution(
            "conv2d fusion expects 2D window/stride/dilation/padding",
        ));
    }

    let (k_h, k_w) = (spec.window[0], spec.window[1]);
    let (s_h, s_w) = (spec.strides[0], spec.strides[1]);
    let (d_h, d_w) = (spec.dilation[0], spec.dilation[1]);
    let pad_top = spec.padding[0].0;
    let pad_left = spec.padding[1].0;

    let k = k_h
        .checked_mul(k_w)
        .and_then(|v| v.checked_mul(c_in))
        .ok_or_else(|| BackendError::execution("conv2d fusion kernel size overflow"))?;

    if weight_dims[0] != k || weight_dims[1] != c_out {
        return Err(BackendError::execution(
            "conv2d fusion weight shape does not match kernel",
        ));
    }

    if let Some(bias) = bias {
        let bias_dims = static_dims_or_error(&bias.spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        if bias.spec.dtype != DType::F32 || bias_dims.as_slice() != [c_out] {
            return Err(BackendError::execution("conv2d fusion bias shape mismatch"));
        }
    }

    let input_values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(BackendError::execution("conv2d fusion expects f32 input")),
    };
    let weight_values = match &weight.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return Err(BackendError::execution("conv2d fusion expects f32 weight")),
    };
    let bias_values = match bias {
        Some(bias) => match &bias.data {
            TensorData::F32(values) => Some(values.as_ref()),
            _ => return Err(BackendError::execution("conv2d fusion expects f32 bias")),
        },
        None => None,
    };

    let m = out_n
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .ok_or_else(|| BackendError::execution("conv2d fusion output size overflow"))?;

    if m == 0 || c_out == 0 {
        return Ok(CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(Vec::<f32>::new().into_boxed_slice())),
        });
    }

    let output_len = m
        .checked_mul(c_out)
        .ok_or_else(|| BackendError::execution("conv2d fusion output overflow"))?;
    let mut output_uninit = vec![std::mem::MaybeUninit::<f32>::uninit(); output_len];
    // Safety: `MaybeUninit<f32>` has the same memory layout as `f32`, and `process_parallel`
    // (via `matmul` with `Accum::Replace`) fully initializes all `output_len` elements on the
    // success path.
    let output_slice = unsafe {
        std::slice::from_raw_parts_mut(output_uninit.as_mut_ptr().cast::<f32>(), output_len)
    };

    let rhs_view = MatRef::from_row_major_slice(weight_values, k, c_out);
    let rhs_t = rhs_view.transpose();

    let max_scratch_floats = 1_048_576usize; // 4 MiB
    let mut block_rows_cap = m.min(8192);
    while block_rows_cap > 1 && block_rows_cap.saturating_mul(k) > max_scratch_floats {
        block_rows_cap /= 2;
    }
    block_rows_cap = block_rows_cap.max(1);

    let out_hw = out_h
        .checked_mul(out_w)
        .ok_or_else(|| BackendError::execution("conv2d fusion out_hw overflow"))?;

    let in_hw = h
        .checked_mul(w)
        .ok_or_else(|| BackendError::execution("conv2d fusion in_hw overflow"))?;
    let in_stride_n = in_hw
        .checked_mul(c_in)
        .ok_or_else(|| BackendError::execution("conv2d fusion in_stride overflow"))?;
    let in_stride_h = w
        .checked_mul(c_in)
        .ok_or_else(|| BackendError::execution("conv2d fusion in_stride overflow"))?;

    let expected_input_len = n
        .checked_mul(in_stride_n)
        .ok_or_else(|| BackendError::execution("conv2d fusion input size overflow"))?;
    if input_values.len() != expected_input_len {
        return Err(BackendError::execution(
            "conv2d fusion input length mismatch",
        ));
    }

    let expected_weight_len = k
        .checked_mul(c_out)
        .ok_or_else(|| BackendError::execution("conv2d fusion weight size overflow"))?;
    if weight_values.len() != expected_weight_len {
        return Err(BackendError::execution(
            "conv2d fusion weight length mismatch",
        ));
    }

    if let Some(bias) = bias_values {
        if bias.len() != c_out {
            return Err(BackendError::execution(
                "conv2d fusion bias length mismatch",
            ));
        }
    }

    let s_h_isize = s_h as isize;
    let s_w_isize = s_w as isize;
    let d_h_isize = d_h as isize;
    let d_w_isize = d_w as isize;
    let pad_top_isize = pad_top as isize;
    let pad_left_isize = pad_left as isize;
    let h_isize = h as isize;
    let w_isize = w as isize;
    let par = faer_parallelism();

    #[derive(Clone, Copy)]
    struct Conv2dIm2colContext<'a> {
        out_w: usize,
        out_hw: usize,
        in_stride_n: usize,
        in_stride_h: usize,
        k_h: usize,
        k_w: usize,
        c_in: usize,
        k: usize,
        c_out: usize,
        block_rows_cap: usize,
        s_h_isize: isize,
        s_w_isize: isize,
        d_h_isize: isize,
        d_w_isize: isize,
        pad_top_isize: isize,
        pad_left_isize: isize,
        h_isize: isize,
        w_isize: isize,
        input_values: &'a [f32],
        rhs_t: MatRef<'a, f32>,
        bias_values: Option<&'a [f32]>,
    }

    fn process_range(
        ctx: Conv2dIm2colContext<'_>,
        row_start: usize,
        row_end: usize,
        out_slice: &mut [f32],
        matmul_par: Par,
    ) -> BackendResult<()> {
        let range_rows = row_end - row_start;
        if range_rows == 0 {
            return Ok(());
        }

        let out_h = ctx.out_hw / ctx.out_w;
        let step_h = (ctx.d_h_isize as usize)
            .checked_mul(ctx.in_stride_h)
            .ok_or_else(|| BackendError::execution("conv2d fusion h offset overflow"))?;
        let step_w = (ctx.d_w_isize as usize)
            .checked_mul(ctx.c_in)
            .ok_or_else(|| BackendError::execution("conv2d fusion w offset overflow"))?;
        let k_h_span = (ctx.k_h.saturating_sub(1) as isize)
            .checked_mul(ctx.d_h_isize)
            .ok_or_else(|| BackendError::execution("conv2d fusion h offset overflow"))?;
        let k_w_span = (ctx.k_w.saturating_sub(1) as isize)
            .checked_mul(ctx.d_w_isize)
            .ok_or_else(|| BackendError::execution("conv2d fusion w offset overflow"))?;
        let input_ptr = ctx.input_values.as_ptr();

        let scratch_rows = range_rows.min(ctx.block_rows_cap);
        let scratch_capacity = scratch_rows
            .checked_mul(ctx.k)
            .ok_or_else(|| BackendError::execution("conv2d fusion scratch overflow"))?;
        CONV2D_IM2COL_SCRATCH_F32.with(|scratch_cell| {
            {
                let mut scratch = scratch_cell.borrow_mut();
                if scratch.len() < scratch_capacity {
                    scratch.resize(scratch_capacity, 0.0);
                }
            }

            for block_offset in (0..range_rows).step_by(ctx.block_rows_cap) {
                let cur_rows = (range_rows - block_offset).min(ctx.block_rows_cap);
                let scratch_len = cur_rows
                    .checked_mul(ctx.k)
                    .ok_or_else(|| BackendError::execution("conv2d fusion scratch overflow"))?;
                let out_offset = block_offset
                    .checked_mul(ctx.c_out)
                    .ok_or_else(|| BackendError::execution("conv2d fusion output overflow"))?;
                let out_chunk = &mut out_slice[out_offset..out_offset + cur_rows * ctx.c_out];

                let flat0 = row_start + block_offset;
                let n_idx = flat0 / ctx.out_hw;
                let rem = flat0 - n_idx * ctx.out_hw;
                let mut oh = rem / ctx.out_w;
                let mut ow = rem - oh * ctx.out_w;
                let mut src_n_base = n_idx * ctx.in_stride_n;

                let mut fill_scratch = |scratch_buf: &mut [f32]| -> BackendResult<()> {
                    for row in 0..cur_rows {
                        let base_h = oh as isize * ctx.s_h_isize - ctx.pad_top_isize;
                        let base_w = ow as isize * ctx.s_w_isize - ctx.pad_left_isize;

                        let dst_row = &mut scratch_buf[row * ctx.k..(row + 1) * ctx.k];
                        let dst_ptr = dst_row.as_mut_ptr();

                        let max_h = base_h + k_h_span;
                        let max_w = base_w + k_w_span;
                        let interior = base_h >= 0
                            && base_w >= 0
                            && max_h < ctx.h_isize
                            && max_w < ctx.w_isize;

                        if interior {
                            let mut dst_off = 0usize;
                            let mut src_row_base = src_n_base
                                + (base_h as usize) * ctx.in_stride_h
                                + (base_w as usize) * ctx.c_in;
                            for _kh in 0..ctx.k_h {
                                let mut src = src_row_base;
                                for _kw in 0..ctx.k_w {
                                    // Safety: `input_values`/`dst_row` lengths are validated by the
                                    // caller, and the `interior` bounds guarantee each patch element
                                    // points at a valid `[C_in]` slice.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            input_ptr.add(src),
                                            dst_ptr.add(dst_off),
                                            ctx.c_in,
                                        );
                                    }
                                    dst_off += ctx.c_in;
                                    src += step_w;
                                }
                                src_row_base += step_h;
                            }
                        } else {
                            dst_row.fill(0.0);
                            for kh in 0..ctx.k_h {
                                let ih = base_h + kh as isize * ctx.d_h_isize;
                                if ih < 0 || ih >= ctx.h_isize {
                                    continue;
                                }
                                let src_h_base = src_n_base + (ih as usize) * ctx.in_stride_h;
                                for kw in 0..ctx.k_w {
                                    let iw = base_w + kw as isize * ctx.d_w_isize;
                                    if iw < 0 || iw >= ctx.w_isize {
                                        continue;
                                    }
                                    let dst_off = (kh * ctx.k_w + kw) * ctx.c_in;
                                    let src = src_h_base + (iw as usize) * ctx.c_in;
                                    // Safety: bounds checks above guarantee `ih/iw` are in range.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            input_ptr.add(src),
                                            dst_ptr.add(dst_off),
                                            ctx.c_in,
                                        );
                                    }
                                }
                            }
                        }

                        ow += 1;
                        if ow == ctx.out_w {
                            ow = 0;
                            oh += 1;
                            if oh == out_h {
                                oh = 0;
                                src_n_base += ctx.in_stride_n;
                            }
                        }
                    }

                    Ok(())
                };

                if matmul_par.degree() > 1 {
                    let scratch_ptr = {
                        let mut scratch = scratch_cell.borrow_mut();
                        let scratch_buf = &mut scratch[..scratch_len];
                        let _fill_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.custom_call.conv2d.im2col_fill",
                            || {
                                let bytes = (scratch_len as u64).saturating_mul(4);
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: scratch_len as u64,
                                        bytes_read: bytes,
                                        bytes_written: bytes,
                                        flops: 0,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        fill_scratch(scratch_buf)?;
                        scratch_buf.as_ptr()
                    };

                    // Safety: `scratch_ptr` points into the thread-local scratch buffer for the
                    // current thread, which stays alive for the duration of this call. We do not
                    // borrow the scratch buffer again until after the parallel matmul completes.
                    let scratch_view =
                        unsafe { std::slice::from_raw_parts(scratch_ptr, scratch_len) };
                    let lhs_view = MatRef::from_row_major_slice(scratch_view, cur_rows, ctx.k);
                    let lhs_t = lhs_view.transpose();

                    let out_view =
                        MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, cur_rows);
                    let _matmul_scope = gpt_rs::profiling::backend_scope_with_meta(
                        "backend.custom_call.conv2d.matmul",
                        || {
                            let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                            let k = ctx.k as u64;
                            let bytes_per_elem = 4u64;
                            let rhs_bytes = (ctx.k as u64)
                                .saturating_mul(ctx.c_out as u64)
                                .saturating_mul(bytes_per_elem);
                            let lhs_bytes = (cur_rows as u64)
                                .saturating_mul(ctx.k as u64)
                                .saturating_mul(bytes_per_elem);
                            let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                            let flops = (out_elems as u128)
                                .saturating_mul(k as u128)
                                .saturating_mul(2)
                                .min(u64::MAX as u128)
                                as u64;
                            gpt_rs::profiling::ScopeMeta::default().with_work(
                                gpt_rs::profiling::WorkStats {
                                    elements: out_elems,
                                    bytes_read: rhs_bytes.saturating_add(lhs_bytes),
                                    bytes_written: out_bytes,
                                    flops,
                                    alloc_bytes: 0,
                                    alloc_count: 0,
                                },
                            )
                        },
                    );
                    matmul_tiled_parallel(matmul_par, out_view, ctx.rhs_t, lhs_t)?;
                } else {
                    let mut scratch = scratch_cell.borrow_mut();
                    let scratch_buf = &mut scratch[..scratch_len];
                    {
                        let _fill_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.custom_call.conv2d.im2col_fill",
                            || {
                                let bytes = (scratch_len as u64).saturating_mul(4);
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: scratch_len as u64,
                                        bytes_read: bytes,
                                        bytes_written: bytes,
                                        flops: 0,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        fill_scratch(scratch_buf)?;
                    }
                    let lhs_view = MatRef::from_row_major_slice(scratch_buf, cur_rows, ctx.k);
                    let lhs_t = lhs_view.transpose();

                    let mut out_view =
                        MatMut::from_column_major_slice_mut(out_chunk, ctx.c_out, cur_rows);
                    {
                        let _matmul_scope = gpt_rs::profiling::backend_scope_with_meta(
                            "backend.custom_call.conv2d.matmul",
                            || {
                                let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                                let k = ctx.k as u64;
                                let bytes_per_elem = 4u64;
                                let rhs_bytes = (ctx.k as u64)
                                    .saturating_mul(ctx.c_out as u64)
                                    .saturating_mul(bytes_per_elem);
                                let lhs_bytes = (cur_rows as u64)
                                    .saturating_mul(ctx.k as u64)
                                    .saturating_mul(bytes_per_elem);
                                let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                                let flops = (out_elems as u128)
                                    .saturating_mul(k as u128)
                                    .saturating_mul(2)
                                    .min(u64::MAX as u128)
                                    as u64;
                                gpt_rs::profiling::ScopeMeta::default().with_work(
                                    gpt_rs::profiling::WorkStats {
                                        elements: out_elems,
                                        bytes_read: rhs_bytes.saturating_add(lhs_bytes),
                                        bytes_written: out_bytes,
                                        flops,
                                        alloc_bytes: 0,
                                        alloc_count: 0,
                                    },
                                )
                            },
                        );
                        matmul(
                            &mut out_view,
                            Accum::Replace,
                            ctx.rhs_t,
                            lhs_t,
                            1.0f32,
                            Par::Seq,
                        );
                    }
                }

                if let Some(bias) = ctx.bias_values {
                    let _bias_scope = gpt_rs::profiling::backend_scope_with_meta(
                        "backend.custom_call.conv2d.bias_add",
                        || {
                            let out_elems = (cur_rows as u64).saturating_mul(ctx.c_out as u64);
                            let bytes_per_elem = 4u64;
                            let out_bytes = out_elems.saturating_mul(bytes_per_elem);
                            let bias_bytes = (ctx.c_out as u64).saturating_mul(bytes_per_elem);
                            gpt_rs::profiling::ScopeMeta::default().with_work(
                                gpt_rs::profiling::WorkStats {
                                    elements: out_elems,
                                    bytes_read: out_bytes.saturating_add(bias_bytes),
                                    bytes_written: out_bytes,
                                    flops: out_elems,
                                    alloc_bytes: 0,
                                    alloc_count: 0,
                                },
                            )
                        },
                    );
                    for row in out_chunk.chunks_exact_mut(ctx.c_out) {
                        for (slot, b) in row.iter_mut().zip(bias.iter()) {
                            *slot += *b;
                        }
                    }
                }
            }

            Ok(())
        })
    }

    fn matmul_tiled_parallel(
        par: Par,
        mut out: MatMut<'_, f32>,
        rhs_t: MatRef<'_, f32>,
        lhs_t: MatRef<'_, f32>,
    ) -> BackendResult<()> {
        const MIN_ROWS: usize = 64;
        const MIN_COLS: usize = 16;

        let nrows = out.nrows();
        let ncols = out.ncols();
        if par.degree() <= 1 || nrows == 0 || ncols == 0 {
            matmul(&mut out, Accum::Replace, rhs_t, lhs_t, 1.0f32, Par::Seq);
            return Ok(());
        }

        let row_splits = nrows / MIN_ROWS;
        let col_splits = ncols / MIN_COLS;
        let split_rows = nrows >= 2 * MIN_ROWS && row_splits >= col_splits;

        let mut left_res: BackendResult<()> = Ok(());
        let mut right_res: BackendResult<()> = Ok(());

        if split_rows {
            let mid = nrows / 2;
            let (out_left, out_right) = out.split_at_row_mut(mid);
            let (rhs_left, rhs_right) = rhs_t.split_at_row(mid);

            faer::utils::thread::join_raw(
                |par_left| {
                    left_res = matmul_tiled_parallel(par_left, out_left, rhs_left, lhs_t);
                },
                |par_right| {
                    right_res = matmul_tiled_parallel(par_right, out_right, rhs_right, lhs_t);
                },
                par,
            );
        } else if ncols >= 2 * MIN_COLS {
            let mid = ncols / 2;
            let (out_left, out_right) = out.split_at_col_mut(mid);
            let (lhs_left, lhs_right) = lhs_t.split_at_col(mid);

            faer::utils::thread::join_raw(
                |par_left| {
                    left_res = matmul_tiled_parallel(par_left, out_left, rhs_t, lhs_left);
                },
                |par_right| {
                    right_res = matmul_tiled_parallel(par_right, out_right, rhs_t, lhs_right);
                },
                par,
            );
        } else {
            matmul(&mut out, Accum::Replace, rhs_t, lhs_t, 1.0f32, Par::Seq);
            return Ok(());
        }

        left_res?;
        right_res?;
        Ok(())
    }

    fn process_parallel<F>(
        par: Par,
        row_start: usize,
        row_end: usize,
        c_out: usize,
        out_slice: &mut [f32],
        process: &F,
    ) -> BackendResult<()>
    where
        F: Fn(usize, usize, &mut [f32]) -> BackendResult<()> + Send + Sync,
    {
        let rows = row_end - row_start;
        if rows == 0 {
            return Ok(());
        }

        if par.degree() <= 1 || rows == 1 {
            return process(row_start, row_end, out_slice);
        }

        let mid = row_start + rows / 2;
        let left_rows = mid - row_start;
        let (out_left, out_right) = out_slice.split_at_mut(left_rows * c_out);
        let mut left_res: BackendResult<()> = Ok(());
        let mut right_res: BackendResult<()> = Ok(());

        faer::utils::thread::join_raw(
            |par_left| {
                left_res = process_parallel(par_left, row_start, mid, c_out, out_left, process)
            },
            |par_right| {
                right_res = process_parallel(par_right, mid, row_end, c_out, out_right, process)
            },
            par,
        );

        left_res?;
        right_res?;
        Ok(())
    }

    let ctx = Conv2dIm2colContext {
        out_w,
        out_hw,
        in_stride_n,
        in_stride_h,
        k_h,
        k_w,
        c_in,
        k,
        c_out,
        block_rows_cap,
        s_h_isize,
        s_w_isize,
        d_h_isize,
        d_w_isize,
        pad_top_isize,
        pad_left_isize,
        h_isize,
        w_isize,
        input_values,
        rhs_t,
        bias_values,
    };

    let use_channel_parallel = par.degree() > 1 && m <= 64 && c_out >= 256;
    if use_channel_parallel {
        process_range(ctx, 0, m, output_slice, par)?;
    } else {
        let process =
            |row_start: usize, row_end: usize, out_slice: &mut [f32]| -> BackendResult<()> {
                process_range(ctx, row_start, row_end, out_slice, Par::Seq)
            };
        process_parallel(par, 0, m, c_out, output_slice, &process)?;
    }

    let output = unsafe {
        let ptr = output_uninit.as_mut_ptr().cast::<f32>();
        let len = output_uninit.len();
        let cap = output_uninit.capacity();
        std::mem::forget(output_uninit);
        Vec::from_raw_parts(ptr, len, cap)
    };

    Ok(CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(output.into_boxed_slice())),
    })
}

fn depthwise_conv2d_nhwc_direct_f32(
    input: &CpuTensor,
    weight: &CpuTensor,
    bias: Option<&CpuTensor>,
    spec: &ExtractPatchesSpec,
    output_spec: &TensorSpec,
) -> BackendResult<CpuTensor> {
    let input_dims = static_dims_or_error(&input.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    let weight_dims = static_dims_or_error(&weight.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;
    let out_dims = static_dims_or_error(&output_spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    })?;

    if input.spec.dtype != DType::F32
        || weight.spec.dtype != DType::F32
        || output_spec.dtype != DType::F32
    {
        return Err(BackendError::execution(
            "depthwise conv fusion only supports f32",
        ));
    }
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return Err(BackendError::execution(
            "depthwise conv fusion expects rank-4 NHWC",
        ));
    }

    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return Err(BackendError::execution(
            "depthwise conv fusion expects 2D window/stride/dilation/padding",
        ));
    }

    let (n, h, w, c) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, out_c) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n || out_c != c {
        return Err(BackendError::execution(
            "depthwise conv fusion output shape mismatch",
        ));
    }

    let (k_h, k_w) = (spec.window[0], spec.window[1]);
    let (s_h, s_w) = (spec.strides[0], spec.strides[1]);
    let (d_h, d_w) = (spec.dilation[0], spec.dilation[1]);
    let pad_top = spec.padding[0].0;
    let pad_left = spec.padding[1].0;

    let khkw = k_h
        .checked_mul(k_w)
        .ok_or_else(|| BackendError::execution("depthwise conv fusion kernel overflow"))?;

    match weight_dims.as_slice() {
        [wc, wk] => {
            if *wc != c || *wk != khkw {
                return Err(BackendError::execution(
                    "depthwise conv fusion weight shape mismatch",
                ));
            }
        }
        [wc, wk, one_in, one_out] => {
            if *wc != c || *wk != khkw || *one_in != 1 || *one_out != 1 {
                return Err(BackendError::execution(
                    "depthwise conv fusion weight shape mismatch",
                ));
            }
        }
        _ => {
            return Err(BackendError::execution(
                "depthwise conv fusion expects weight [C, KH*KW] or [C, KH*KW, 1, 1]",
            ))
        }
    }

    if let Some(bias) = bias {
        let bias_dims = static_dims_or_error(&bias.spec.shape, |sym| {
            BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))
        })?;
        if bias.spec.dtype != DType::F32 || bias_dims.as_slice() != [c] {
            return Err(BackendError::execution(
                "depthwise conv fusion bias shape mismatch",
            ));
        }
    }

    let input_values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => {
            return Err(BackendError::execution(
                "depthwise conv fusion expects f32 input",
            ))
        }
    };
    let weight_values = match &weight.data {
        TensorData::F32(values) => values.as_ref(),
        _ => {
            return Err(BackendError::execution(
                "depthwise conv fusion expects f32 weight",
            ))
        }
    };
    let bias_values = match bias {
        Some(bias) => match &bias.data {
            TensorData::F32(values) => Some(values.as_ref()),
            _ => {
                return Err(BackendError::execution(
                    "depthwise conv fusion expects f32 bias",
                ))
            }
        },
        None => None,
    };

    let out_len = out_dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| BackendError::execution("depthwise conv fusion output size overflow"))?;
    let mut out = vec![0.0f32; out_len];

    let in_hw = h
        .checked_mul(w)
        .ok_or_else(|| BackendError::execution("depthwise conv fusion in_hw overflow"))?;
    let in_stride_n = in_hw
        .checked_mul(c)
        .ok_or_else(|| BackendError::execution("depthwise conv fusion in_stride overflow"))?;
    let in_stride_h = w
        .checked_mul(c)
        .ok_or_else(|| BackendError::execution("depthwise conv fusion in_stride overflow"))?;

    let out_hw = out_h
        .checked_mul(out_w)
        .ok_or_else(|| BackendError::execution("depthwise conv fusion out_hw overflow"))?;

    let s_h_isize = s_h as isize;
    let s_w_isize = s_w as isize;
    let d_h_isize = d_h as isize;
    let d_w_isize = d_w as isize;
    let pad_top_isize = pad_top as isize;
    let pad_left_isize = pad_left as isize;
    let h_isize = h as isize;
    let w_isize = w as isize;

    for n_idx in 0..n {
        let in_n_base = n_idx * in_stride_n;
        let out_n_base = n_idx * out_hw * c;
        for oh in 0..out_h {
            let base_h = oh as isize * s_h_isize - pad_top_isize;
            let out_h_base = out_n_base + oh * out_w * c;
            for ow in 0..out_w {
                let base_w = ow as isize * s_w_isize - pad_left_isize;
                let out_base = out_h_base + ow * c;

                // Initialize with bias.
                if let Some(bias) = bias_values {
                    out[out_base..out_base + c].copy_from_slice(bias);
                }

                for kh in 0..k_h {
                    let ih = base_h + kh as isize * d_h_isize;
                    if ih < 0 || ih >= h_isize {
                        continue;
                    }
                    let in_h_base = in_n_base
                        + (ih as usize).checked_mul(in_stride_h).ok_or_else(|| {
                            BackendError::execution("depthwise conv fusion offset overflow")
                        })?;
                    for kw in 0..k_w {
                        let iw = base_w + kw as isize * d_w_isize;
                        if iw < 0 || iw >= w_isize {
                            continue;
                        }
                        let in_base = in_h_base
                            + (iw as usize).checked_mul(c).ok_or_else(|| {
                                BackendError::execution("depthwise conv fusion offset overflow")
                            })?;

                        let khkw_idx = kh * k_w + kw;
                        // For each channel, accumulate input * weight[channel, khkw].
                        for ch in 0..c {
                            let w_idx = ch * khkw + khkw_idx;
                            out[out_base + ch] += input_values[in_base + ch] * weight_values[w_idx];
                        }
                    }
                }
            }
        }
    }

    Ok(CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(out.into_boxed_slice())),
    })
}

fn try_elementwise_binary(
    inputs: &[CpuTensor],
    outputs: &[TensorSpec],
    op: gpt_rs::backend::spec::ElementwiseBinaryOp,
) -> Option<BackendResult<Vec<CpuTensor>>> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return None;
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let output_spec = &outputs[0];

    let (lhs_values, rhs_values) = match (&lhs.data, &rhs.data) {
        (TensorData::F32(a), TensorData::F32(b)) => (a.as_ref(), b.as_ref()),
        _ => return None,
    };

    if lhs.spec.dtype != DType::F32
        || rhs.spec.dtype != DType::F32
        || output_spec.dtype != DType::F32
    {
        return None;
    }
    if lhs_values.len() != rhs_values.len() {
        return Some(Err(BackendError::execution("elementwise size mismatch")));
    }

    match op {
        gpt_rs::backend::spec::ElementwiseBinaryOp::Add
        | gpt_rs::backend::spec::ElementwiseBinaryOp::Maximum
        | gpt_rs::backend::spec::ElementwiseBinaryOp::Minimum
        | gpt_rs::backend::spec::ElementwiseBinaryOp::Mul => {}
        _ => return None,
    }

    let len = lhs_values.len();
    let mut out = vec![0.0f32; len];
    match op {
        gpt_rs::backend::spec::ElementwiseBinaryOp::Add => {
            for i in 0..len {
                out[i] = lhs_values[i] + rhs_values[i];
            }
        }
        gpt_rs::backend::spec::ElementwiseBinaryOp::Maximum => {
            for i in 0..len {
                out[i] = lhs_values[i].max(rhs_values[i]);
            }
        }
        gpt_rs::backend::spec::ElementwiseBinaryOp::Minimum => {
            for i in 0..len {
                out[i] = lhs_values[i].min(rhs_values[i]);
            }
        }
        gpt_rs::backend::spec::ElementwiseBinaryOp::Mul => {
            for i in 0..len {
                out[i] = lhs_values[i] * rhs_values[i];
            }
        }
        _ => unreachable!("filtered above"),
    }

    Some(Ok(vec![CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(out.into_boxed_slice())),
    }]))
}

fn try_transpose(
    inputs: &[CpuTensor],
    outputs: &[TensorSpec],
    spec: &TransposeSpec,
) -> Option<BackendResult<Vec<CpuTensor>>> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return None;
    }
    let input = &inputs[0];
    let output_spec = &outputs[0];
    let input_dims = match static_dims_or_error(&input.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    let out_dims = match static_dims_or_error(&output_spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    if input.spec.dtype != DType::F32 || output_spec.dtype != DType::F32 {
        return None;
    }
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return None;
    }
    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return None,
    };

    // Fast paths for the vision entrypoint layout conversions.
    // NCHW -> NHWC
    if spec.perm.as_slice() == [0, 2, 3, 1]
        && out_dims[0] == input_dims[0]
        && out_dims[1] == input_dims[2]
        && out_dims[2] == input_dims[3]
        && out_dims[3] == input_dims[1]
    {
        let (n, c, h, w) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
        let out_len = match out_dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        {
            Some(v) => v,
            None => return Some(Err(BackendError::execution("transpose size overflow"))),
        };
        let mut out = vec![0.0f32; out_len];
        let hw = h * w;
        let out_hw = h * w;
        for n_idx in 0..n {
            let in_n_base = n_idx * c * hw;
            let out_n_base = n_idx * out_hw * c;
            for h_idx in 0..h {
                let in_h_base = in_n_base + h_idx * w;
                let out_h_base = out_n_base + h_idx * w * c;
                for w_idx in 0..w {
                    let in_hw_index = in_h_base + w_idx;
                    let out_base = out_h_base + w_idx * c;
                    for c_idx in 0..c {
                        out[out_base + c_idx] = values[in_hw_index + c_idx * hw];
                    }
                }
            }
        }
        return Some(Ok(vec![CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(out.into_boxed_slice())),
        }]));
    }

    // NHWC -> NCHW
    if spec.perm.as_slice() == [0, 3, 1, 2]
        && out_dims[0] == input_dims[0]
        && out_dims[1] == input_dims[3]
        && out_dims[2] == input_dims[1]
        && out_dims[3] == input_dims[2]
    {
        let (n, h, w, c) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
        let out_len = match out_dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        {
            Some(v) => v,
            None => return Some(Err(BackendError::execution("transpose size overflow"))),
        };
        let mut out = vec![0.0f32; out_len];
        let hw = h * w;
        for n_idx in 0..n {
            let in_n_base = n_idx * hw * c;
            let out_n_base = n_idx * c * hw;
            for h_idx in 0..h {
                let in_h_base = in_n_base + h_idx * w * c;
                let out_h_base = out_n_base + h_idx * w;
                for w_idx in 0..w {
                    let in_base = in_h_base + w_idx * c;
                    let out_hw_index = out_h_base + w_idx;
                    for c_idx in 0..c {
                        out[out_hw_index + c_idx * hw] = values[in_base + c_idx];
                    }
                }
            }
        }
        return Some(Ok(vec![CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(out.into_boxed_slice())),
        }]));
    }

    None
}

fn try_reduce_window(
    inputs: &[CpuTensor],
    outputs: &[TensorSpec],
    spec: &ReduceWindowSpec,
) -> Option<BackendResult<Vec<CpuTensor>>> {
    if inputs.len() != 1 || outputs.len() != 1 {
        return None;
    }
    if spec.reduce != ReduceKind::Max {
        return None;
    }
    let input = &inputs[0];
    let output_spec = &outputs[0];
    if input.spec.dtype != DType::F32 || output_spec.dtype != DType::F32 {
        return None;
    }
    if spec.window_dims.as_slice() != [1, 3, 3, 1]
        || spec.strides.as_slice() != [1, 2, 2, 1]
        || spec.base_dilation.as_slice() != [1, 1, 1, 1]
        || spec.window_dilation.as_slice() != [1, 1, 1, 1]
    {
        return None;
    }
    if spec.padding.as_slice() != [(0, 0), (1, 1), (1, 1), (0, 0)] {
        return None;
    }

    let input_dims = match static_dims_or_error(&input.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    let out_dims = match static_dims_or_error(&output_spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    if input_dims.len() != 4 || out_dims.len() != 4 {
        return None;
    }

    let (n, h, w, c) = (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (out_n, out_h, out_w, out_c) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    if out_n != n || out_c != c {
        return Some(Err(BackendError::execution(
            "reduce_window expects batch/channels preserved",
        )));
    }

    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => return None,
    };

    let out_len = match out_n
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_c))
    {
        Some(v) => v,
        None => return Some(Err(BackendError::execution("reduce_window size overflow"))),
    };
    let mut out = vec![f32::NEG_INFINITY; out_len];

    let in_hw = match h.checked_mul(w) {
        Some(v) => v,
        None => return Some(Err(BackendError::execution("reduce_window in_hw overflow"))),
    };
    let in_stride_n = match in_hw.checked_mul(c) {
        Some(v) => v,
        None => {
            return Some(Err(BackendError::execution(
                "reduce_window stride overflow",
            )))
        }
    };
    let in_stride_h = match w.checked_mul(c) {
        Some(v) => v,
        None => {
            return Some(Err(BackendError::execution(
                "reduce_window stride overflow",
            )))
        }
    };

    for n_idx in 0..n {
        let in_n_base = n_idx * in_stride_n;
        for oh in 0..out_h {
            let ih0 = oh as isize * 2 - 1;
            for ow in 0..out_w {
                let iw0 = ow as isize * 2 - 1;
                let dst_offset = ((n_idx * out_h + oh) * out_w + ow) * c;
                let dst = &mut out[dst_offset..dst_offset + c];

                for kh in 0..3 {
                    let ih = ih0 + kh;
                    if ih < 0 || ih >= h as isize {
                        continue;
                    }
                    let ih = ih as usize;
                    let in_h_base = in_n_base + ih * in_stride_h;
                    for kw in 0..3 {
                        let iw = iw0 + kw;
                        if iw < 0 || iw >= w as isize {
                            continue;
                        }
                        let iw = iw as usize;
                        let src_offset = in_h_base + iw * c;
                        let src = &values[src_offset..src_offset + c];
                        for i in 0..c {
                            dst[i] = dst[i].max(src[i]);
                        }
                    }
                }
            }
        }
    }

    Some(Ok(vec![CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(out.into_boxed_slice())),
    }]))
}

fn custom_call_attr_i64_array(
    spec: &CustomCallSpec,
    key: &str,
    expected_len: usize,
) -> BackendResult<Vec<usize>> {
    let attr = spec.attrs.get(key).ok_or_else(|| {
        BackendError::spec(
            SpecErrorCode::InvalidAttributeValue,
            format!("custom_call `{}` missing attribute `{}`", spec.target, key),
        )
    })?;
    let CustomCallAttr::I64Array(values) = attr else {
        return Err(BackendError::spec(
            SpecErrorCode::InvalidAttributeValue,
            format!(
                "custom_call `{}` attribute `{}` must be an i64 array",
                spec.target, key
            ),
        ));
    };
    if values.len() != expected_len {
        return Err(BackendError::spec(
            SpecErrorCode::InvalidAttributeValue,
            format!(
                "custom_call `{}` attribute `{}` must have length {}, got {}",
                spec.target,
                key,
                expected_len,
                values.len()
            ),
        ));
    }
    values
        .iter()
        .map(|value| {
            if *value < 0 {
                return Err(BackendError::spec(
                    SpecErrorCode::InvalidAttributeValue,
                    format!(
                        "custom_call `{}` attribute `{}` contains negative value {}",
                        spec.target, key, value
                    ),
                ));
            }
            Ok(*value as usize)
        })
        .collect()
}

fn custom_call_conv2d_extract_spec(spec: &CustomCallSpec) -> BackendResult<ExtractPatchesSpec> {
    let window = custom_call_attr_i64_array(spec, "window", 2)?;
    let strides = custom_call_attr_i64_array(spec, "strides", 2)?;
    let dilation = custom_call_attr_i64_array(spec, "dilation", 2)?;
    let padding = custom_call_attr_i64_array(spec, "padding", 4)?;

    Ok(ExtractPatchesSpec {
        window,
        strides,
        dilation,
        padding: vec![(padding[0], padding[1]), (padding[2], padding[3])],
        pad_value: Literal::Float(0.0),
    })
}

fn try_custom_call(
    inputs: &[CpuTensor],
    outputs: &[TensorSpec],
    spec: &CustomCallSpec,
) -> Option<BackendResult<Vec<CpuTensor>>> {
    let [output_spec] = outputs else {
        return Some(Err(BackendError::execution(
            "custom_call expects exactly one output",
        )));
    };

    match spec.target.as_str() {
        optimizer::TARGET_CONV2D_NHWC_F32_V1 => {
            let extract_spec = match custom_call_conv2d_extract_spec(spec) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };

            let (input, weight, bias) = match inputs {
                [input, weight] => (input, weight, None),
                [input, weight, bias] => (input, weight, Some(bias)),
                _ => {
                    return Some(Err(BackendError::execution(
                        "custom_call conv2d expects 2 or 3 operands",
                    )))
                }
            };

            let _prof_guard = gpt_rs::profiling::backend_scope_with_meta(
                "backend.custom_call.conv2d.nhwc.f32.v1",
                || {
                    let signature = {
                        let mut sig = String::new();
                        sig.push_str("in=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(&input.spec));
                        sig.push_str(" weight=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(&weight.spec));
                        sig.push_str(" out=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(output_spec));
                        sig.push_str(" window=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.window[0], extract_spec.window[1]
                        ));
                        sig.push_str(" strides=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.strides[0], extract_spec.strides[1]
                        ));
                        sig.push_str(" dilation=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.dilation[0], extract_spec.dilation[1]
                        ));
                        sig.push_str(" padding=");
                        sig.push_str(&format!(
                            "[{},{}],[{},{}]",
                            extract_spec.padding[0].0,
                            extract_spec.padding[0].1,
                            extract_spec.padding[1].0,
                            extract_spec.padding[1].1
                        ));
                        sig
                    };

                    let work = {
                        let input_dims = static_dims_or_error(&input.spec.shape, |sym| {
                            BackendError::execution(format!(
                                "dynamic dimension {} not supported at runtime",
                                sym.as_str()
                            ))
                        })
                        .ok();
                        let out_dims = static_dims_or_error(&output_spec.shape, |sym| {
                            BackendError::execution(format!(
                                "dynamic dimension {} not supported at runtime",
                                sym.as_str()
                            ))
                        })
                        .ok();
                        let bytes_per_elem = 4u64;

                        let input_bytes = input_dims
                            .as_ref()
                            .map(|dims| {
                                dims.iter()
                                    .fold(1u64, |acc, &v| acc.saturating_mul(v as u64))
                                    .saturating_mul(bytes_per_elem)
                            })
                            .unwrap_or(0);
                        let weight_bytes = {
                            let dims = static_dims_or_error(&weight.spec.shape, |sym| {
                                BackendError::execution(format!(
                                    "dynamic dimension {} not supported at runtime",
                                    sym.as_str()
                                ))
                            })
                            .ok();
                            dims.as_ref()
                                .map(|dims| {
                                    dims.iter()
                                        .fold(1u64, |acc, &v| acc.saturating_mul(v as u64))
                                        .saturating_mul(bytes_per_elem)
                                })
                                .unwrap_or(0)
                        };
                        let bias_bytes = bias
                            .and_then(|b| {
                                static_dims_or_error(&b.spec.shape, |sym| {
                                    BackendError::execution(format!(
                                        "dynamic dimension {} not supported at runtime",
                                        sym.as_str()
                                    ))
                                })
                                .ok()
                            })
                            .map(|dims| {
                                dims.iter()
                                    .fold(1u64, |acc, &v| acc.saturating_mul(v as u64))
                                    .saturating_mul(bytes_per_elem)
                            })
                            .unwrap_or(0);
                        let out_bytes = out_dims
                            .as_ref()
                            .map(|dims| {
                                dims.iter()
                                    .fold(1u64, |acc, &v| acc.saturating_mul(v as u64))
                                    .saturating_mul(bytes_per_elem)
                            })
                            .unwrap_or(0);

                        let flops = match (input_dims.as_ref(), out_dims.as_ref()) {
                            (Some(in_dims), Some(out_dims))
                                if in_dims.len() == 4 && out_dims.len() == 4 =>
                            {
                                let c_in = in_dims[3] as u128;
                                let c_out = out_dims[3] as u128;
                                let out_spatial = (out_dims[0] as u128)
                                    .saturating_mul(out_dims[1] as u128)
                                    .saturating_mul(out_dims[2] as u128);
                                let k = (extract_spec.window[0] as u128)
                                    .saturating_mul(extract_spec.window[1] as u128)
                                    .saturating_mul(c_in);
                                let flops = out_spatial
                                    .saturating_mul(c_out)
                                    .saturating_mul(k)
                                    .saturating_mul(2);
                                flops.min(u64::MAX as u128) as u64
                            }
                            _ => 0,
                        };

                        gpt_rs::profiling::WorkStats {
                            elements: out_bytes / bytes_per_elem,
                            bytes_read: input_bytes
                                .saturating_add(weight_bytes)
                                .saturating_add(bias_bytes),
                            bytes_written: out_bytes,
                            flops,
                            alloc_bytes: out_bytes,
                            alloc_count: 1,
                        }
                    };

                    let signature = gpt_rs::profiling::signature_id(&signature);
                    let meta = signature
                        .map(gpt_rs::profiling::ScopeMeta::signature)
                        .unwrap_or_default();
                    meta.with_work(work)
                },
            );
            Some(
                conv2d_nhwc_im2col_tiled_f32(input, weight, bias, &extract_spec, output_spec)
                    .map(|tensor| vec![tensor]),
            )
        }
        optimizer::TARGET_DEPTHWISE_CONV2D_NHWC_F32_V1 => {
            let extract_spec = match custom_call_conv2d_extract_spec(spec) {
                Ok(value) => value,
                Err(err) => return Some(Err(err)),
            };

            let (input, weight, bias) = match inputs {
                [input, weight] => (input, weight, None),
                [input, weight, bias] => (input, weight, Some(bias)),
                _ => {
                    return Some(Err(BackendError::execution(
                        "custom_call depthwise conv2d expects 2 or 3 operands",
                    )))
                }
            };

            let _prof_guard = gpt_rs::profiling::backend_scope_with_meta(
                "backend.custom_call.depthwise_conv2d.nhwc.f32.v1",
                || {
                    let signature = {
                        let mut sig = String::new();
                        sig.push_str("in=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(&input.spec));
                        sig.push_str(" weight=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(&weight.spec));
                        sig.push_str(" out=");
                        sig.push_str(&gpt_rs::profiling::tensor_spec_signature(output_spec));
                        sig.push_str(" window=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.window[0], extract_spec.window[1]
                        ));
                        sig.push_str(" strides=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.strides[0], extract_spec.strides[1]
                        ));
                        sig.push_str(" dilation=");
                        sig.push_str(&format!(
                            "{}x{}",
                            extract_spec.dilation[0], extract_spec.dilation[1]
                        ));
                        sig.push_str(" padding=");
                        sig.push_str(&format!(
                            "[{},{}],[{},{}]",
                            extract_spec.padding[0].0,
                            extract_spec.padding[0].1,
                            extract_spec.padding[1].0,
                            extract_spec.padding[1].1
                        ));
                        sig
                    };

                    let work = {
                        let out_dims = static_dims_or_error(&output_spec.shape, |sym| {
                            BackendError::execution(format!(
                                "dynamic dimension {} not supported at runtime",
                                sym.as_str()
                            ))
                        })
                        .ok();
                        let bytes_per_elem = 4u64;
                        let out_bytes = out_dims
                            .as_ref()
                            .map(|dims| {
                                dims.iter()
                                    .fold(1u64, |acc, &v| acc.saturating_mul(v as u64))
                                    .saturating_mul(bytes_per_elem)
                            })
                            .unwrap_or(0);

                        gpt_rs::profiling::WorkStats {
                            elements: out_bytes / bytes_per_elem,
                            bytes_read: 0,
                            bytes_written: out_bytes,
                            flops: 0,
                            alloc_bytes: out_bytes,
                            alloc_count: 1,
                        }
                    };

                    let signature = gpt_rs::profiling::signature_id(&signature);
                    let meta = signature
                        .map(gpt_rs::profiling::ScopeMeta::signature)
                        .unwrap_or_default();
                    meta.with_work(work)
                },
            );
            Some(
                depthwise_conv2d_nhwc_direct_f32(input, weight, bias, &extract_spec, output_spec)
                    .map(|tensor| vec![tensor]),
            )
        }
        _ => Some(Err(BackendError::unimplemented(
            "custom_call",
            format!("unknown target `{}`", spec.target),
        ))),
    }
}

fn try_dot_general(
    inputs: &[CpuTensor],
    outputs: &[TensorSpec],
    spec: &DotGeneralSpec,
) -> Option<BackendResult<Vec<CpuTensor>>> {
    if inputs.len() != 2 || outputs.len() != 1 {
        return None;
    }

    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let output_spec = &outputs[0];

    let (lhs_values, rhs_values) = match (&lhs.data, &rhs.data) {
        (TensorData::F32(lhs_values), TensorData::F32(rhs_values)) => {
            (lhs_values.as_ref(), rhs_values.as_ref())
        }
        _ => return None,
    };

    if lhs.spec.dtype != DType::F32
        || rhs.spec.dtype != DType::F32
        || output_spec.dtype != DType::F32
    {
        return None;
    }

    if let Some(accum) = spec.accum_dtype {
        if accum != DType::F32 {
            return None;
        }
    }

    let lhs_dims = match static_dims_or_error(&lhs.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    let rhs_dims = match static_dims_or_error(&rhs.spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };
    let out_dims = match static_dims_or_error(&output_spec.shape, |sym| {
        BackendError::execution(format!(
            "dynamic dimension {} not supported at runtime",
            sym.as_str()
        ))
    }) {
        Ok(dims) => dims,
        Err(err) => return Some(Err(err)),
    };

    if lhs_dims.len() == 2
        && rhs_dims.len() == 2
        && spec.batch_lhs.is_empty()
        && spec.batch_rhs.is_empty()
        && spec.contract_lhs.as_slice() == [1]
        && spec.contract_rhs.as_slice() == [0]
    {
        return Some(dot_general_2d(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    if lhs_dims.len() == 3
        && rhs_dims.len() == 3
        && spec.batch_lhs.as_slice() == [0]
        && spec.batch_rhs.as_slice() == [0]
        && spec.contract_lhs.as_slice() == [2]
        && spec.contract_rhs.as_slice() == [1]
    {
        return Some(dot_general_batched(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    // Batched matmul where rhs is laid out as [B, N, K] (contracting rhs axis 2).
    if lhs_dims.len() == 3
        && rhs_dims.len() == 3
        && spec.batch_lhs.as_slice() == [0]
        && spec.batch_rhs.as_slice() == [0]
        && spec.contract_lhs.as_slice() == [2]
        && spec.contract_rhs.as_slice() == [2]
    {
        return Some(dot_general_batched_rhs_transposed(
            lhs_values,
            rhs_values,
            &lhs_dims,
            &rhs_dims,
            &out_dims,
            output_spec,
        ));
    }

    None
}

fn dot_general_2d(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorSpec,
) -> BackendResult<Vec<CpuTensor>> {
    let (m, k_lhs) = (lhs_dims[0], lhs_dims[1]);
    let (k_rhs, n) = (rhs_dims[0], rhs_dims[1]);

    if k_lhs != k_rhs {
        return Err(BackendError::execution(
            "dot_general lhs/rhs contract mismatch",
        ));
    }

    if out_dims.len() != 2 || out_dims[0] != m || out_dims[1] != n {
        return Err(BackendError::execution("dot_general output shape mismatch"));
    }

    if lhs.len() != m * k_lhs || rhs.len() != k_rhs * n {
        return Err(BackendError::execution("dot_general operand size mismatch"));
    }

    if m == 0 || n == 0 || k_lhs == 0 {
        return Ok(vec![CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(Vec::<f32>::new())),
        }]);
    }

    let lhs_view = MatRef::from_row_major_slice(lhs, m, k_lhs);
    let rhs_view = MatRef::from_row_major_slice(rhs, k_rhs, n);

    // faer prefers column-major output. To avoid an explicit output transpose/copy, compute
    // C^T = B^T * A^T into a column-major (n x m) matrix. The underlying buffer layout matches
    // row-major (m x n) for C, so we can return it directly.
    let a_t = lhs_view.transpose();
    let b_t = rhs_view.transpose();
    let mut row_major = vec![0.0f32; m * n];
    let mut out_view = MatMut::from_column_major_slice_mut(row_major.as_mut_slice(), n, m);
    let par = faer_parallelism();
    matmul(&mut out_view, Accum::Replace, b_t, a_t, 1.0f32, par);

    Ok(vec![CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(row_major.into_boxed_slice())),
    }])
}

fn dot_general_batched(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorSpec,
) -> BackendResult<Vec<CpuTensor>> {
    let (batch, m, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, k_rhs, n) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(BackendError::execution("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(BackendError::execution(
            "dot_general lhs/rhs contract mismatch",
        ));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != m || out_dims[2] != n {
        return Err(BackendError::execution("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * m * k_lhs || rhs.len() != batch * k_rhs * n {
        return Err(BackendError::execution("dot_general operand size mismatch"));
    }

    if batch == 0 || m == 0 || n == 0 || k_lhs == 0 {
        return Ok(vec![CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(Vec::<f32>::new())),
        }]);
    }

    let lhs_batch_stride = m * k_lhs;
    let rhs_batch_stride = k_rhs * n;
    let out_batch_stride = m * n;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for b in 0..batch {
        let lhs_slice = &lhs[b * lhs_batch_stride..(b + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[b * rhs_batch_stride..(b + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, m, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, k_rhs, n);

        let a_t = lhs_view.transpose();
        let b_t = rhs_view.transpose();
        let dst = &mut row_output[b * out_batch_stride..(b + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, n, m);
        matmul(&mut out_view, Accum::Replace, b_t, a_t, 1.0f32, par);
    }

    Ok(vec![CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(row_output.into_boxed_slice())),
    }])
}

fn dot_general_batched_rhs_transposed(
    lhs: &[f32],
    rhs: &[f32],
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    out_dims: &[usize],
    output_spec: &TensorSpec,
) -> BackendResult<Vec<CpuTensor>> {
    let (batch, m, k_lhs) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let (batch_rhs, n, k_rhs) = (rhs_dims[0], rhs_dims[1], rhs_dims[2]);

    if batch != batch_rhs {
        return Err(BackendError::execution("dot_general batch mismatch"));
    }

    if k_lhs != k_rhs {
        return Err(BackendError::execution(
            "dot_general lhs/rhs contract mismatch",
        ));
    }

    if out_dims.len() != 3 || out_dims[0] != batch || out_dims[1] != m || out_dims[2] != n {
        return Err(BackendError::execution("dot_general output shape mismatch"));
    }

    if lhs.len() != batch * m * k_lhs || rhs.len() != batch * n * k_rhs {
        return Err(BackendError::execution("dot_general operand size mismatch"));
    }

    if batch == 0 || m == 0 || n == 0 || k_lhs == 0 {
        return Ok(vec![CpuTensor {
            spec: output_spec.clone(),
            data: TensorData::F32(Arc::from(Vec::<f32>::new())),
        }]);
    }

    let lhs_batch_stride = m * k_lhs;
    let rhs_batch_stride = n * k_rhs;
    let out_batch_stride = m * n;

    let mut row_output = vec![0.0f32; batch * out_batch_stride];
    let par = faer_parallelism();

    for b in 0..batch {
        let lhs_slice = &lhs[b * lhs_batch_stride..(b + 1) * lhs_batch_stride];
        let rhs_slice = &rhs[b * rhs_batch_stride..(b + 1) * rhs_batch_stride];

        let lhs_view = MatRef::from_row_major_slice(lhs_slice, m, k_lhs);
        let rhs_view = MatRef::from_row_major_slice(rhs_slice, n, k_rhs);

        // Compute C^T = B * A^T into a column-major (n x m) matrix.
        // The underlying buffer layout matches row-major (m x n) for C.
        let a_t = lhs_view.transpose();
        let dst = &mut row_output[b * out_batch_stride..(b + 1) * out_batch_stride];
        let mut out_view = MatMut::from_column_major_slice_mut(dst, n, m);
        matmul(&mut out_view, Accum::Replace, rhs_view, a_t, 1.0f32, par);
    }

    Ok(vec![CpuTensor {
        spec: output_spec.clone(),
        data: TensorData::F32(Arc::from(row_output.into_boxed_slice())),
    }])
}

/// Register the Faer backend with the global backend registry.
///
/// This function is called automatically via a static initializer, but can also
/// be called manually to ensure the backend is registered.
pub fn register_faer_backend() {
    gpt_rs::backend::registry::register_portable_backend("faer", FaerCpuBackend::create);
}

// Auto-register on library load
#[gpt_rs::linkme::distributed_slice(gpt_rs::backend::registry::BACKEND_REGISTRARS)]
static REGISTER_FAER_BACKEND: fn() = register_faer_backend;
