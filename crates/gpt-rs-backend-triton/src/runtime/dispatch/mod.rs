use super::*;

mod custom_call;
mod dot;
mod elementwise;
mod reduction;
mod shape;
mod vision;

impl TritonExecutor {
    pub(super) fn execute_instruction(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &DenseValueStore,
        slot_allocator: &mut SlotAllocator,
        instruction: &gpt_rs::backend::spec::Instruction,
        instruction_pos: usize,
    ) -> BackendResult<TritonTensor> {
        match &instruction.op {
            Operation::Constant(literal) => {
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.materialize_literal(driver, literal, output)
            }
            Operation::StopGradient | Operation::Reshape(_) => {
                let source_value = match instruction.operands.first() {
                    Some(Operand::Value(id)) => Some(*id),
                    Some(Operand::TupleElement { tuple, .. }) => Some(*tuple),
                    _ => None,
                };
                let source =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let src_bytes = byte_len(&source.spec)?;
                let out_bytes = byte_len(&out_spec)?;
                if src_bytes != out_bytes {
                    return Err(BackendError::execution(format!(
                        "alias byte-size mismatch: source={} bytes, output={} bytes",
                        src_bytes, out_bytes
                    )));
                }
                if let Some(source_value) = source_value {
                    slot_allocator.propagate_alias(source_value, instruction.id);
                }
                Ok(TritonTensor::new(out_spec, source.buffer))
            }
            Operation::ElementwiseBinary(op) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(EWISE_BINARY_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing elementwise binary kernel in triton artifact")
                })?;
                self.execute_elementwise_binary(
                    driver,
                    kernel,
                    *op,
                    &lhs,
                    &rhs,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::ElementwiseUnary(op) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(EWISE_UNARY_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing elementwise unary kernel in triton artifact")
                })?;
                self.execute_elementwise_unary(driver, kernel, *op, &input, &out_spec, output)
            }
            Operation::BroadcastTo(_) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_broadcast(driver, kernels, &input, &out_spec, output)
            }
            Operation::Slice(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_slice(driver, kernels, &input, spec, &out_spec, output)
            }
            Operation::DynamicSlice(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let starts =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_dynamic_slice(
                    driver,
                    kernels,
                    &input,
                    &starts,
                    spec,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::Transpose(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_transpose(driver, kernels, &input, spec, &out_spec, output)
            }
            Operation::Concat(spec) => {
                if instruction.operands.len() != 2 {
                    return Err(BackendError::execution(
                        "triton concat runtime currently supports exactly two operands",
                    ));
                }
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_concat(
                    driver,
                    kernels,
                    &lhs,
                    &rhs,
                    spec,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::Iota(spec) => {
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(IOTA_SI32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing iota kernel in triton artifact")
                })?;
                self.execute_iota(driver, kernel, spec, &out_spec, output)
            }
            Operation::Compare(spec) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(COMPARE_SI32_I1_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing compare kernel in triton artifact")
                })?;
                self.execute_compare(
                    driver,
                    kernel,
                    spec.op,
                    &lhs,
                    &rhs,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::Select => {
                let pred =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let when_true =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let when_false =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(2))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(SELECT_I1_F32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing select kernel in triton artifact")
                })?;
                self.execute_select(
                    driver,
                    kernel,
                    &pred,
                    &when_true,
                    &when_false,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::Take => {
                let params =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let indices =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(TAKE_F32_I32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing take kernel in triton artifact")
                })?;
                self.execute_take(driver, kernel, &params, &indices, &out_spec, output)
            }
            Operation::DynamicUpdateSlice(_spec) => {
                let base =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let update =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let starts =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(2))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels
                    .get(DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID)
                    .ok_or_else(|| {
                        BackendError::execution(
                            "missing dynamic_update_slice kernel in triton artifact",
                        )
                    })?;
                self.execute_dynamic_update_slice(
                    driver,
                    kernel,
                    &base,
                    &update,
                    &starts,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::DotGeneral(spec) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let args = DotGeneralArgs {
                    spec,
                    lhs_spec: &lhs.spec,
                    rhs_spec: &rhs.spec,
                    out_spec: &out_spec,
                };
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_dot_general(driver, args, &lhs, &rhs, output)
            }
            Operation::Reduce(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_reduce(
                    driver,
                    kernels,
                    &input.spec,
                    spec,
                    &input,
                    OutputBinding::new(&out_spec, output),
                )
            }
            Operation::ExtractPatches(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels.get(EXTRACT_PATCHES_NHWC_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing extract_patches kernel in triton artifact")
                })?;
                self.execute_extract_patches(driver, kernel, &input, spec, &out_spec, output)
            }
            Operation::ReduceWindow(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                let kernel = kernels
                    .get(REDUCE_WINDOW_MAX_NHWC_KERNEL_ID)
                    .ok_or_else(|| {
                        BackendError::execution("missing reduce_window kernel in triton artifact")
                    })?;
                self.execute_reduce_window(driver, kernel, &input, spec, &out_spec, output)
            }
            Operation::CustomCall(spec) => {
                let out_spec = output_tensor_spec(&instruction.output)?;
                let output = slot_allocator.output_for_value(
                    driver,
                    instruction.id,
                    &out_spec,
                    instruction_pos,
                )?;
                self.execute_custom_call(
                    driver,
                    kernels,
                    values,
                    instruction,
                    spec,
                    OutputBinding::new(&out_spec, output),
                )
            }
            other => Err(BackendError::execution(format!(
                "triton runtime does not support instruction op: {:?}",
                other
            ))),
        }
    }

    pub(super) fn resolve_operand_tensor(
        &self,
        driver: &Arc<CudaDriver>,
        values: &DenseValueStore,
        operand: Option<&Operand>,
    ) -> BackendResult<TritonTensor> {
        match operand {
            Some(Operand::Value(id)) => values.get_cloned(*id).ok_or_else(|| {
                BackendError::execution(format!(
                    "operand value {} missing from triton runtime state",
                    id.0
                ))
            }),
            Some(Operand::Literal(literal)) => self.materialize_literal(driver, literal, None),
            Some(Operand::TupleElement { .. }) => Err(BackendError::execution(
                "tuple element operands are not supported by triton runtime",
            )),
            None => Err(BackendError::execution("missing instruction operand")),
        }
    }

    fn execute_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &DenseValueStore,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out: OutputBinding<'_>,
    ) -> BackendResult<TritonTensor> {
        match spec.target.as_str() {
            TARGET_ELEMENTWISE_FUSED_F32_V1 => self.execute_fused_elementwise_custom_call(
                driver,
                values,
                instruction,
                spec,
                out.spec,
                out.tensor,
            ),
            TARGET_DOT_BIAS_FUSED_F32_V1 => self.execute_fused_dot_bias_custom_call(
                driver,
                kernels,
                values,
                instruction,
                spec,
                out,
            ),
            TARGET_LAYER_NORM_FUSED_F32_V1 => {
                self.execute_layer_norm_custom_call(driver, kernels, values, instruction, spec, out)
            }
            TARGET_SOFTMAX_LAST_AXIS_FUSED_F32_V1 => self.execute_softmax_last_axis_custom_call(
                driver,
                kernels,
                values,
                instruction,
                spec,
                out,
            ),
            _ => Err(BackendError::execution(format!(
                "unsupported triton custom_call target '{}'",
                spec.target
            ))),
        }
    }
}
