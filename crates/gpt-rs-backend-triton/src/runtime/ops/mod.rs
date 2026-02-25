use super::*;

pub(super) fn literal_to_tensor(
    driver: &Arc<CudaDriver>,
    literal: &TensorLiteral,
    output: Option<TritonTensor>,
) -> BackendResult<TritonTensor> {
    let expected = byte_len(&literal.spec)?;
    if expected != literal.bytes.len() {
        return Err(BackendError::execution(format!(
            "literal byte length mismatch for dtype {:?}: expected {}, got {}",
            literal.spec.dtype,
            expected,
            literal.bytes.len()
        )));
    }

    let out = allocate_output_tensor(driver, &literal.spec, output)?;
    if expected != out.buffer.bytes() {
        return Err(BackendError::execution(format!(
            "literal destination byte length mismatch: expected {}, got {}",
            expected,
            out.buffer.bytes()
        )));
    }
    driver.upload_to_device(out.buffer.device_ptr(), literal.bytes.as_ref())?;
    Ok(out)
}

pub(super) fn allocate_output_tensor(
    driver: &Arc<CudaDriver>,
    spec: &TensorSpec,
    output: Option<TritonTensor>,
) -> BackendResult<TritonTensor> {
    match output {
        Some(tensor) => {
            if tensor.spec != *spec {
                return Err(BackendError::execution(format!(
                    "slot output spec mismatch: expected {:?}, got {:?}",
                    spec, tensor.spec
                )));
            }
            Ok(tensor)
        }
        None => Ok(TritonTensor::new(
            spec.clone(),
            driver.alloc_zeroed(byte_len(spec)?)?,
        )),
    }
}

pub(super) fn output_tensor_spec(output: &ValueType) -> BackendResult<TensorSpec> {
    match output {
        ValueType::Tensor(spec) => Ok(spec.clone()),
        ValueType::Tuple(_) => Err(BackendError::execution(
            "tuple outputs are not supported by triton runtime",
        )),
    }
}

pub(super) fn byte_len(spec: &TensorSpec) -> BackendResult<usize> {
    spec.byte_len().ok_or_else(|| {
        BackendError::execution(format!(
            "cannot compute byte length for dtype {:?} and shape {:?}",
            spec.dtype,
            spec.shape.dims()
        ))
    })
}

pub(super) fn static_element_count(shape: &gpt_rs::backend::spec::Shape) -> BackendResult<usize> {
    let dims = static_dims_or_error(shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    checked_element_count_or_error(dims.as_slice(), || {
        BackendError::execution("element count overflow")
    })
}

pub(super) fn ensure_static_broadcastable(
    out_shape: &gpt_rs::backend::spec::Shape,
    in_shape: &gpt_rs::backend::spec::Shape,
    context: &str,
) -> BackendResult<()> {
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > out_dims.len() {
        return Err(BackendError::execution(format!(
            "{context} rank mismatch: input rank {} exceeds output rank {}",
            in_dims.len(),
            out_dims.len()
        )));
    }
    let offset = out_dims.len() - in_dims.len();
    for (idx, in_dim) in in_dims.iter().enumerate() {
        let out_dim = out_dims[offset + idx];
        if *in_dim != 1 && *in_dim != out_dim {
            return Err(BackendError::execution(format!(
                "{context} shape mismatch: input dim {} (axis {}) is incompatible with output dim {}",
                in_dim,
                idx,
                out_dim
            )));
        }
    }
    Ok(())
}

pub(super) fn align_dims4(dims: &[usize]) -> BackendResult<(i32, i32, i32, i32)> {
    if dims.len() > 4 {
        return Err(BackendError::execution(format!(
            "rank {} exceeds rank-4 support",
            dims.len()
        )));
    }
    let mut aligned = [1usize; 4];
    let offset = 4 - dims.len();
    for (idx, value) in dims.iter().enumerate() {
        aligned[offset + idx] = *value;
    }
    Ok((
        i32::try_from(aligned[0]).map_err(|_| BackendError::execution("dim0 exceeds i32 range"))?,
        i32::try_from(aligned[1]).map_err(|_| BackendError::execution("dim1 exceeds i32 range"))?,
        i32::try_from(aligned[2]).map_err(|_| BackendError::execution("dim2 exceeds i32 range"))?,
        i32::try_from(aligned[3]).map_err(|_| BackendError::execution("dim3 exceeds i32 range"))?,
    ))
}

pub(super) fn broadcast_rank4_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
) -> BackendResult<([i32; 4], [i32; 4])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if out_dims.len() > 4 || in_dims.len() > 4 {
        return Err(BackendError::execution(
            "broadcast runtime supports rank <= 4 only",
        ));
    }
    if out_dims.len() < in_dims.len() {
        return Err(BackendError::execution(
            "broadcast result rank must be >= input rank",
        ));
    }

    let mut out4 = [1usize; 4];
    let mut in4 = [1usize; 4];
    for (idx, dim) in out_dims.iter().enumerate() {
        out4[4 - out_dims.len() + idx] = *dim;
    }
    for (idx, dim) in in_dims.iter().enumerate() {
        in4[4 - in_dims.len() + idx] = *dim;
    }

    let base_strides =
        contiguous_strides_or_error(&in4, || BackendError::execution("stride overflow"))?;
    let mut in_strides = [0i32; 4];
    let mut out_i32 = [0i32; 4];
    for axis in 0..4 {
        let in_dim = in4[axis];
        let out_dim = out4[axis];
        if !(in_dim == out_dim || in_dim == 1) {
            return Err(BackendError::execution(format!(
                "broadcast dim mismatch at axis {axis}: input={in_dim}, output={out_dim}"
            )));
        }
        let stride = if in_dim == 1 && out_dim > 1 {
            0usize
        } else {
            base_strides[axis]
        };
        in_strides[axis] = i32::try_from(stride)
            .map_err(|_| BackendError::execution("broadcast input stride exceeds i32 range"))?;
        out_i32[axis] = i32::try_from(out_dim)
            .map_err(|_| BackendError::execution("broadcast output dim exceeds i32 range"))?;
    }

    Ok((out_i32, in_strides))
}

pub(super) fn slice_rank4_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    starts: &[usize],
) -> BackendResult<([i32; 4], [i32; 4], [i32; 4])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "slice runtime supports rank <= 4 only",
        ));
    }
    if in_dims.len() != out_dims.len() || starts.len() != in_dims.len() {
        return Err(BackendError::execution("slice starts/sizes rank mismatch"));
    }
    for axis in 0..in_dims.len() {
        if match starts[axis].checked_add(out_dims[axis]) {
            Some(end) => end > in_dims[axis],
            None => true,
        } {
            return Err(BackendError::execution(format!(
                "slice out of bounds at axis {axis}"
            )));
        }
    }

    let mut in4 = [1usize; 4];
    let mut out4 = [1usize; 4];
    let mut starts4 = [0usize; 4];
    let offset = 4 - in_dims.len();
    in4[offset..(offset + in_dims.len())].copy_from_slice(&in_dims);
    out4[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    starts4[offset..(offset + starts.len())].copy_from_slice(starts);
    let strides = contiguous_strides_or_error(&in4, || BackendError::execution("stride overflow"))?;
    let mut out_i32 = [0i32; 4];
    let mut strides_i32 = [0i32; 4];
    let mut starts_i32 = [0i32; 4];
    for axis in 0..4 {
        out_i32[axis] = i32::try_from(out4[axis])
            .map_err(|_| BackendError::execution("slice output dim exceeds i32 range"))?;
        strides_i32[axis] = i32::try_from(strides[axis])
            .map_err(|_| BackendError::execution("slice input stride exceeds i32 range"))?;
        starts_i32[axis] = i32::try_from(starts4[axis])
            .map_err(|_| BackendError::execution("slice start exceeds i32 range"))?;
    }
    Ok((out_i32, strides_i32, starts_i32))
}

pub(super) struct ContiguousSliceCopyPlan {
    pub(super) byte_offset: u64,
    pub(super) byte_len: usize,
}

pub(super) fn contiguous_slice_copy_plan(
    input_spec: &TensorSpec,
    out_spec: &TensorSpec,
    starts: &[usize],
) -> BackendResult<Option<ContiguousSliceCopyPlan>> {
    let in_dims = static_dims_or_error(&input_spec.shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(&out_spec.shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() != out_dims.len() || starts.len() != in_dims.len() {
        return Ok(None);
    }

    for axis in 0..in_dims.len() {
        if match starts[axis].checked_add(out_dims[axis]) {
            Some(end) => end > in_dims[axis],
            None => true,
        } {
            return Err(BackendError::execution(format!(
                "slice out of bounds at axis {axis}"
            )));
        }
    }

    let first_non_full =
        (0..in_dims.len()).find(|&axis| starts[axis] != 0 || out_dims[axis] != in_dims[axis]);
    if let Some(axis) = first_non_full {
        // A contiguous subrange in row-major memory can only have one truncated axis,
        // with all leading axes selecting exactly one index and all trailing axes full.
        if (0..axis).any(|idx| out_dims[idx] != 1) {
            return Ok(None);
        }
        if ((axis + 1)..in_dims.len()).any(|idx| starts[idx] != 0 || out_dims[idx] != in_dims[idx])
        {
            return Ok(None);
        }
    }

    let strides =
        contiguous_strides_or_error(&in_dims, || BackendError::execution("stride overflow"))?;
    let mut offset_elems = 0usize;
    for axis in 0..in_dims.len() {
        offset_elems = offset_elems
            .checked_add(
                starts[axis]
                    .checked_mul(strides[axis])
                    .ok_or_else(|| BackendError::execution("slice contiguous offset overflow"))?,
            )
            .ok_or_else(|| BackendError::execution("slice contiguous offset overflow"))?;
    }

    let elem_size = out_spec
        .dtype
        .size_in_bytes()
        .ok_or_else(|| BackendError::execution("slice contiguous unsupported dtype size"))?;
    let byte_offset = u64::try_from(
        offset_elems
            .checked_mul(elem_size)
            .ok_or_else(|| BackendError::execution("slice contiguous byte offset overflow"))?,
    )
    .map_err(|_| BackendError::execution("slice contiguous byte offset exceeds u64"))?;
    let byte_len = out_spec
        .byte_len()
        .ok_or_else(|| BackendError::execution("slice contiguous output byte length unknown"))?;

    Ok(Some(ContiguousSliceCopyPlan {
        byte_offset,
        byte_len,
    }))
}

pub(super) fn transpose_rank5_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    perm: &[usize],
) -> BackendResult<([i32; 5], [i32; 5])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > 5 || out_dims.len() > 5 {
        return Err(BackendError::execution(
            "transpose runtime supports rank <= 5 only",
        ));
    }
    if in_dims.len() != out_dims.len() || perm.len() != in_dims.len() {
        return Err(BackendError::execution(
            "transpose rank/permutation mismatch",
        ));
    }
    let input_strides =
        contiguous_strides_or_error(&in_dims, || BackendError::execution("stride overflow"))?;
    let mut mapped_strides = vec![0usize; perm.len()];
    let mut seen = vec![false; perm.len()];
    for (axis, src_axis) in perm.iter().enumerate() {
        if *src_axis >= perm.len() || seen[*src_axis] {
            return Err(BackendError::execution(
                "transpose permutation must be a valid unique permutation",
            ));
        }
        seen[*src_axis] = true;
        mapped_strides[axis] = input_strides[*src_axis];
    }

    let mut out5 = [1usize; 5];
    let mut strides5 = [0usize; 5];
    let offset = 5 - out_dims.len();
    out5[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    strides5[offset..(offset + out_dims.len())].copy_from_slice(&mapped_strides);

    let mut out_i32 = [0i32; 5];
    let mut strides_i32 = [0i32; 5];
    for axis in 0..5 {
        out_i32[axis] = i32::try_from(out5[axis])
            .map_err(|_| BackendError::execution("transpose output dim exceeds i32 range"))?;
        strides_i32[axis] = i32::try_from(strides5[axis])
            .map_err(|_| BackendError::execution("transpose input stride exceeds i32 range"))?;
    }
    Ok((out_i32, strides_i32))
}

pub(super) fn normalize_axis(axis: isize, rank: usize) -> BackendResult<usize> {
    let rank_isize =
        isize::try_from(rank).map_err(|_| BackendError::execution("rank exceeds isize"))?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(BackendError::execution(format!(
            "axis {axis} out of bounds for rank {rank}"
        )));
    }
    usize::try_from(normalized).map_err(|_| BackendError::execution("axis conversion overflow"))
}

pub(super) type ConcatRank4Layout = ([i32; 4], [i32; 4], [i32; 4], i32, i32);

pub(super) fn concat_rank4_layout(
    lhs_shape: &gpt_rs::backend::spec::Shape,
    rhs_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    axis: isize,
) -> BackendResult<ConcatRank4Layout> {
    let lhs_dims = static_dims_or_error(lhs_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let rhs_dims = static_dims_or_error(rhs_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if lhs_dims.len() > 4 || rhs_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "concat runtime supports rank <= 4 only",
        ));
    }
    if lhs_dims.len() != rhs_dims.len() || lhs_dims.len() != out_dims.len() {
        return Err(BackendError::execution("concat rank mismatch"));
    }
    let axis = normalize_axis(axis, lhs_dims.len())?;
    for idx in 0..lhs_dims.len() {
        if idx == axis {
            if match lhs_dims[idx].checked_add(rhs_dims[idx]) {
                Some(sum) => sum != out_dims[idx],
                None => true,
            } {
                return Err(BackendError::execution(
                    "concat output axis dimension mismatch",
                ));
            }
        } else if lhs_dims[idx] != rhs_dims[idx] || lhs_dims[idx] != out_dims[idx] {
            return Err(BackendError::execution(format!(
                "concat non-axis dimension mismatch at axis {idx}"
            )));
        }
    }

    let lhs_strides =
        contiguous_strides_or_error(&lhs_dims, || BackendError::execution("stride overflow"))?;
    let rhs_strides =
        contiguous_strides_or_error(&rhs_dims, || BackendError::execution("stride overflow"))?;

    let mut out4 = [1usize; 4];
    let mut lhs4 = [0usize; 4];
    let mut rhs4 = [0usize; 4];
    let offset = 4 - out_dims.len();
    out4[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    lhs4[offset..(offset + out_dims.len())].copy_from_slice(&lhs_strides);
    rhs4[offset..(offset + out_dims.len())].copy_from_slice(&rhs_strides);

    let mut out_i32 = [0i32; 4];
    let mut lhs_i32 = [0i32; 4];
    let mut rhs_i32 = [0i32; 4];
    for idx in 0..4 {
        out_i32[idx] = i32::try_from(out4[idx])
            .map_err(|_| BackendError::execution("concat output dim exceeds i32 range"))?;
        lhs_i32[idx] = i32::try_from(lhs4[idx])
            .map_err(|_| BackendError::execution("concat lhs stride exceeds i32 range"))?;
        rhs_i32[idx] = i32::try_from(rhs4[idx])
            .map_err(|_| BackendError::execution("concat rhs stride exceeds i32 range"))?;
    }

    let axis_i32 = i32::try_from(offset + axis)
        .map_err(|_| BackendError::execution("concat axis exceeds i32 range"))?;
    let split_i32 = i32::try_from(lhs_dims[axis])
        .map_err(|_| BackendError::execution("concat split exceeds i32 range"))?;
    Ok((out_i32, lhs_i32, rhs_i32, axis_i32, split_i32))
}

pub(super) fn dynamic_update_rank4_layout(
    update_dims: &[usize],
    out_dims: &[usize],
    starts: &[usize],
) -> BackendResult<([i32; 4], [i32; 4], [i32; 4])> {
    if update_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "dynamic_update_slice runtime supports rank <= 4 only",
        ));
    }
    if update_dims.len() != out_dims.len() || starts.len() != out_dims.len() {
        return Err(BackendError::execution(
            "dynamic_update_slice rank mismatch",
        ));
    }
    let out_strides =
        contiguous_strides_or_error(out_dims, || BackendError::execution("stride overflow"))?;

    let mut update4 = [1usize; 4];
    let mut out_strides4 = [0usize; 4];
    let mut starts4 = [0usize; 4];
    let offset = 4 - out_dims.len();
    update4[offset..(offset + out_dims.len())].copy_from_slice(update_dims);
    out_strides4[offset..(offset + out_dims.len())].copy_from_slice(&out_strides);
    starts4[offset..(offset + out_dims.len())].copy_from_slice(starts);

    let mut update_i32 = [0i32; 4];
    let mut out_strides_i32 = [0i32; 4];
    let mut starts_i32 = [0i32; 4];
    for idx in 0..4 {
        update_i32[idx] = i32::try_from(update4[idx])
            .map_err(|_| BackendError::execution("dynamic_update update dim exceeds i32 range"))?;
        out_strides_i32[idx] = i32::try_from(out_strides4[idx]).map_err(|_| {
            BackendError::execution("dynamic_update output stride exceeds i32 range")
        })?;
        starts_i32[idx] = i32::try_from(starts4[idx])
            .map_err(|_| BackendError::execution("dynamic_update start exceeds i32 range"))?;
    }
    Ok((update_i32, out_strides_i32, starts_i32))
}

pub(super) fn launch_1d(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    n: u32,
    block_x: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if n == 0 {
        return Ok(());
    }
    let grid_x = n.div_ceil(block_x);
    launch_program_grid(driver, kernel, grid_x, block_x, n, params)
}

pub(super) fn launch_program_grid(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    grid_x: u32,
    block_x: u32,
    work_elements: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if grid_x == 0 {
        return Ok(());
    }
    launch_program_grid_2d(driver, kernel, grid_x, 1, block_x, work_elements, params)
}

pub(super) fn launch_program_grid_2d(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    work_elements: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if grid_x == 0 || grid_y == 0 {
        return Ok(());
    }
    KERNEL_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
    let _scope = profiling::backend_scope_with_meta("backend.triton.kernel", || {
        let meta = kernel
            .profile_signature
            .map(ScopeMeta::signature)
            .unwrap_or_default();
        meta.with_work(WorkStats {
            elements: u64::from(work_elements),
            ..WorkStats::default()
        })
    });
    if GPU_EVENT_TIMING_ENABLED.load(Ordering::Relaxed) {
        let host_start = std::time::Instant::now();
        let elapsed_ms = driver.time_with_events("backend.triton.kernel", || {
            driver.launch_kernel(
                &kernel.function,
                (grid_x, grid_y, 1),
                (block_x, 1, 1),
                0,
                params,
            )
        })?;
        let host_duration = host_start.elapsed();
        let gpu_duration = Duration::from_secs_f64(f64::from(elapsed_ms) * 1e-3);
        let work = WorkStats {
            elements: u64::from(work_elements),
            ..WorkStats::default()
        };
        profiling::record_backend_aggregate_with_signature(
            "backend.triton.kernel_gpu",
            kernel.profile_signature,
            1,
            gpu_duration,
            work,
        );
        profiling::record_backend_aggregate_with_signature(
            "backend.triton.kernel_host",
            kernel.profile_signature,
            1,
            host_duration,
            work,
        );
        Ok(())
    } else {
        driver.launch_kernel(
            &kernel.function,
            (grid_x, grid_y, 1),
            (block_x, 1, 1),
            0,
            params,
        )
    }
}

pub(super) fn read_i32_tensor(tensor: &TritonTensor) -> BackendResult<Vec<i32>> {
    if tensor.spec.dtype != DType::Si32 {
        return Err(BackendError::execution(
            "read_i32_tensor requires Si32 tensor",
        ));
    }
    let bytes = tensor.buffer.read_to_vec()?;
    if bytes.len() % 4 != 0 {
        return Err(BackendError::execution(
            "Si32 tensor byte length is not divisible by 4",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

pub(super) fn binary_opcode(op: ElementwiseBinaryOp) -> u32 {
    match op {
        ElementwiseBinaryOp::Add => 0,
        ElementwiseBinaryOp::Sub => 1,
        ElementwiseBinaryOp::Mul => 2,
        ElementwiseBinaryOp::Div => 3,
        ElementwiseBinaryOp::Maximum => 4,
        ElementwiseBinaryOp::Minimum => 5,
    }
}

pub(super) fn unary_opcode(op: ElementwiseUnaryOp) -> BackendResult<u32> {
    match op {
        ElementwiseUnaryOp::Neg => Ok(0),
        ElementwiseUnaryOp::Abs => Ok(1),
        ElementwiseUnaryOp::Exp => Ok(2),
        ElementwiseUnaryOp::Log => Ok(3),
        ElementwiseUnaryOp::Tanh => Ok(4),
        ElementwiseUnaryOp::Erf => Ok(5),
        ElementwiseUnaryOp::Rsqrt => Ok(6),
        ElementwiseUnaryOp::Reciprocal => Ok(7),
    }
}

pub(super) fn prefer_custom_dot_bias_kernel(
    _lhs: &TensorSpec,
    _rhs: &TensorSpec,
    _out: &TensorSpec,
) -> bool {
    true
}

pub(super) fn compare_opcode(op: ComparisonOp) -> u32 {
    match op {
        ComparisonOp::Less => 0,
        ComparisonOp::LessEqual => 1,
        ComparisonOp::Equal => 2,
        ComparisonOp::GreaterEqual => 3,
        ComparisonOp::Greater => 4,
        ComparisonOp::NotEqual => 5,
    }
}
