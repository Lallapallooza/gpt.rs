use std::os::raw::{c_char, c_int, c_void};
use std::sync::Arc;

use gpt_rs::backend::spec::{
    BackendError, DType, Dimension, Program, Shape, TensorInit, TensorLiteral, TensorSpec,
};
use gpt_rs::PortableBackend;
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

#[repr(C)]
pub struct PtirTensor {
    pub dtype: u32,
    pub rank: u32,
    pub dims: *const i64,
    pub data: *mut c_void,
}

#[repr(C)]
pub struct PtirProgramJson {
    pub json: *const c_char,
    pub json_len: usize,
}

#[no_mangle]
/// # Safety
/// Caller must provide valid pointers for `program`, `inputs`, and `outputs` and ensure the
/// buffers they reference are readable/writable for the sizes implied by the tensor metadata.
pub unsafe extern "C" fn ptir_execute_json(
    program: *const PtirProgramJson,
    inputs: *const PtirTensor,
    input_count: usize,
    outputs: *mut PtirTensor,
    output_count: usize,
) -> c_int {
    if program.is_null() {
        return -1;
    }
    let program = unsafe { &*program };
    if program.json.is_null() {
        return -1;
    }

    let json = unsafe { std::slice::from_raw_parts(program.json as *const u8, program.json_len) };
    let json_str = match std::str::from_utf8(json) {
        Ok(value) => value,
        Err(_) => return -1,
    };

    let program = match Program::from_json_str(json_str) {
        Ok(value) => value,
        Err(_) => return -1,
    };

    let backend = CpuPortableBackend::new();
    let input_slice = unsafe { std::slice::from_raw_parts(inputs, input_count) };
    let mut handles = Vec::with_capacity(input_slice.len());
    for tensor in input_slice {
        let literal = match tensor_to_literal(tensor) {
            Ok(literal) => literal,
            Err(_) => return -1,
        };
        let handle = match backend.materialize(TensorInit::Literal(literal)) {
            Ok(handle) => handle,
            Err(_) => return -1,
        };
        handles.push(handle);
    }

    let outputs_vec = match backend.run_program(&program, &handles) {
        Ok(values) => values,
        Err(_) => return -1,
    };

    if outputs_vec.len() != output_count {
        return -1;
    }

    let output_slice = unsafe { std::slice::from_raw_parts_mut(outputs, output_count) };
    for (tensor, handle) in output_slice.iter_mut().zip(outputs_vec.iter()) {
        let literal = match backend.to_literal(handle) {
            Ok(literal) => literal,
            Err(_) => return -1,
        };
        if write_literal(tensor, &literal).is_err() {
            return -1;
        }
    }

    0
}

fn tensor_to_literal(tensor: &PtirTensor) -> Result<TensorLiteral, BackendError> {
    if tensor.dims.is_null() || tensor.data.is_null() {
        return Err(BackendError::execution("null tensor pointers"));
    }
    let rank = tensor.rank as usize;
    let dims = unsafe { std::slice::from_raw_parts(tensor.dims, rank) };
    let dtype = tag_to_dtype(tensor.dtype)
        .ok_or_else(|| BackendError::execution("unsupported dtype tag"))?;
    let elem_size = dtype_size(dtype)?;
    let elem_count = dims.iter().try_fold(1usize, |acc, dim| {
        if *dim <= 0 {
            return None;
        }
        acc.checked_mul(*dim as usize)
    });
    let elem_count = elem_count.ok_or_else(|| BackendError::execution("invalid dims"))?;
    let byte_len = elem_count
        .checked_mul(elem_size)
        .ok_or_else(|| BackendError::execution("tensor byte size overflow"))?;
    let bytes = unsafe { std::slice::from_raw_parts(tensor.data as *const u8, byte_len).to_vec() };
    let shape = Shape::new(
        dims.iter()
            .map(|d| Dimension::from_usize(*d as usize))
            .collect::<Vec<_>>(),
    );
    let spec = TensorSpec::new(dtype, shape);
    Ok(TensorLiteral::new(spec, Arc::<[u8]>::from(bytes)))
}

fn write_literal(target: &mut PtirTensor, literal: &TensorLiteral) -> Result<(), BackendError> {
    if target.dims.is_null() || target.data.is_null() {
        return Err(BackendError::execution("null output tensor pointers"));
    }
    let rank = target.rank as usize;
    let dims = unsafe { std::slice::from_raw_parts(target.dims, rank) };
    if dims.len() != literal.spec.shape.rank() {
        return Err(BackendError::execution("output rank mismatch"));
    }
    let out_dtype = tag_to_dtype(target.dtype)
        .ok_or_else(|| BackendError::execution("unsupported output dtype tag"))?;
    if out_dtype != literal.spec.dtype {
        return Err(BackendError::execution("output dtype mismatch"));
    }
    for (dim, spec_dim) in dims.iter().zip(literal.spec.shape.dims().iter()) {
        let expected = match spec_dim {
            gpt_rs::backend::spec::Dimension::Static(value) => *value as i64,
            gpt_rs::backend::spec::Dimension::Dynamic(_) => {
                return Err(BackendError::execution("dynamic dims not supported"));
            }
        };
        if *dim != expected {
            return Err(BackendError::execution("output shape mismatch"));
        }
    }
    let byte_len = literal.byte_len();
    unsafe {
        std::ptr::copy_nonoverlapping(literal.bytes.as_ptr(), target.data as *mut u8, byte_len);
    }
    Ok(())
}

fn tag_to_dtype(tag: u32) -> Option<DType> {
    match tag {
        0 => Some(DType::F32),
        1 => Some(DType::F16),
        2 => Some(DType::Bf16),
        3 => Some(DType::Si32),
        _ => None,
    }
}

fn dtype_size(dtype: DType) -> Result<usize, BackendError> {
    Ok(match dtype {
        DType::F32 => 4,
        DType::F16 | DType::Bf16 => 2,
        DType::Si32 => 4,
        _ => {
            return Err(BackendError::execution(
                "unsupported dtype for C runtime execution",
            ))
        }
    })
}
