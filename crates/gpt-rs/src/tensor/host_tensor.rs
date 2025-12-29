//! Host-backed tensor used for literals, debugging, and interoperability.

use super::{dtype::DType, shape::Shape, spec_utils, storage::StorageElement};
use anyhow::{bail, ensure, Result};
use rand::Rng;
use std::mem::{size_of, ManuallyDrop};
use std::sync::Arc;

use crate::backend::spec::{Dimension, TensorLiteral, TensorSpec};

/// Simple host-backed tensor used for literals, debugging, and tests.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Shape,
    dtype: DType,
    data: Vec<u8>,
    grad: Option<Vec<u8>>,
    requires_grad: bool,
}

impl Tensor {
    /// Constructs an `F32` tensor from raw values, validating the length against the shape.
    pub fn from_vec(shape: Shape, data: Vec<f32>) -> Result<Self> {
        if data.len() != shape.num_elements() {
            bail!(
                "tensor data length ({}) does not match shape {:?}",
                data.len(),
                shape.dims()
            );
        }
        Ok(Tensor {
            shape,
            dtype: DType::F32,
            data: vec_into_bytes(data),
            grad: None,
            requires_grad: false,
        })
    }

    /// Constructs an `I32` tensor, ensuring the payload matches the expected element count.
    pub fn from_i32(shape: Shape, data: Vec<i32>) -> Result<Self> {
        if data.len() != shape.num_elements() {
            bail!(
                "tensor data length ({}) does not match shape {:?}",
                data.len(),
                shape.dims()
            );
        }
        Ok(Tensor {
            shape,
            dtype: DType::I32,
            data: vec_into_bytes(data),
            grad: None,
            requires_grad: false,
        })
    }

    /// Returns a zero-initialized `F32` tensor of the requested shape.
    pub fn zeros(shape: Shape) -> Self {
        let len = shape.num_elements();
        Tensor {
            shape,
            dtype: DType::F32,
            data: vec_into_bytes(vec![0.0; len]),
            grad: None,
            requires_grad: false,
        }
    }

    /// Returns a one-initialized `F32` tensor of the requested shape.
    pub fn ones(shape: Shape) -> Self {
        let len = shape.num_elements();
        Tensor {
            shape,
            dtype: DType::F32,
            data: vec_into_bytes(vec![1.0; len]),
            grad: None,
            requires_grad: false,
        }
    }

    /// Samples from a normal distribution (`N(0, std^2)`) using the Box-Muller transform.
    pub fn randn(shape: Shape, std: f32, rng: &mut impl Rng) -> Self {
        let len = shape.num_elements();
        let mut values = Vec::with_capacity(len);
        while values.len() < len {
            let u1: f32 = rng.gen::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.gen::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z0 = r * theta.cos() * std;
            let z1 = r * theta.sin() * std;
            values.push(z0);
            if values.len() < len {
                values.push(z1);
            }
        }
        Tensor {
            shape,
            dtype: DType::F32,
            data: vec_into_bytes(values),
            grad: None,
            requires_grad: false,
        }
    }

    /// Returns the total number of elements stored in the tensor.
    pub fn len(&self) -> usize {
        self.shape.num_elements()
    }

    /// Reports whether the tensor contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Provides access to the tensor shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the scalar dtype of the tensor payload.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrows the underlying `f32` data slice, panicking if the dtype differs.
    pub fn data(&self) -> &[f32] {
        match self.dtype {
            DType::F32 => bytes_as_slice::<f32>(&self.data),
            _ => panic!("tensor data is not stored as f32"),
        }
    }

    /// Mutably borrows the `f32` data slice, panicking if the dtype differs.
    pub fn data_mut(&mut self) -> &mut [f32] {
        match self.dtype {
            DType::F32 => bytes_as_slice_mut::<f32>(&mut self.data),
            _ => panic!("tensor data is not stored as mutable f32"),
        }
    }

    /// Borrows the underlying `i32` data slice, panicking if the dtype differs.
    pub fn data_i32(&self) -> &[i32] {
        match self.dtype {
            DType::I32 => bytes_as_slice::<i32>(&self.data),
            _ => panic!("tensor data is not stored as i32"),
        }
    }

    /// Toggles gradient tracking and allocates a gradient buffer when necessary.
    pub fn requires_grad(mut self, flag: bool) -> Self {
        self.requires_grad = flag;
        if flag {
            match self.dtype {
                DType::F32 => {
                    if self.grad.is_none() {
                        self.grad = Some(vec_into_bytes(vec![0.0; self.len()]));
                    }
                }
                _ => {
                    self.grad = None;
                }
            }
        } else {
            self.grad = None;
        }
        self
    }

    /// Returns the current gradient tracking flag.
    pub fn requires_grad_flag(&self) -> bool {
        self.requires_grad
    }

    /// Borrows the gradient buffer as `f32` values when gradients are available.
    pub fn grad(&self) -> Option<&[f32]> {
        match (self.dtype, self.grad.as_ref()) {
            (DType::F32, Some(bytes)) => Some(bytes_as_slice::<f32>(bytes)),
            _ => None,
        }
    }

    /// Mutably borrows the gradient buffer as `f32` values.
    pub fn grad_mut(&mut self) -> Option<&mut [f32]> {
        match (self.dtype, self.grad.as_mut()) {
            (DType::F32, Some(bytes)) => Some(bytes_as_slice_mut::<f32>(bytes)),
            _ => None,
        }
    }

    /// Applies a unary function in place over every scalar element.
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(f32) -> f32,
    {
        let data = self.data_mut();
        for v in data {
            *v = f(*v);
        }
    }

    /// Applies a binary function over two tensors, storing the results in `self`.
    pub fn apply_binary_inplace<F>(&mut self, rhs: &Tensor, mut f: F)
    where
        F: FnMut(f32, f32) -> f32,
    {
        self.shape.assert_same(&rhs.shape);
        let lhs = self.data_mut();
        let rhs_slice = rhs.data();
        for (a, b) in lhs.iter_mut().zip(rhs_slice.iter()) {
            *a = f(*a, *b);
        }
    }

    /// Fills the tensor with a constant value.
    pub fn fill(&mut self, value: f32) {
        self.data_mut().fill(value);
    }

    /// Converts the tensor contents into another storage element type.
    pub fn astype<E: StorageElement>(&self) -> Vec<E> {
        match self.dtype {
            DType::F32 => self.data().iter().map(|&x| E::from_f32(x)).collect(),
            DType::I32 => self
                .data_i32()
                .iter()
                .map(|&x| E::from_f32(x as f32))
                .collect(),
            DType::F16 | DType::BF16 => {
                panic!("astype is not supported for dtype {:?}", self.dtype)
            }
        }
    }

    /// Wraps the tensor in a backend-neutral literal for graph initialization.
    pub fn to_literal(&self) -> TensorLiteral {
        let spec = TensorSpec::new(
            spec_utils::backend_dtype(self.dtype),
            spec_utils::backend_shape_from_shape(&self.shape),
        );
        TensorLiteral::new(spec, Arc::from(self.data.clone().into_boxed_slice()))
    }

    /// Reconstructs a host tensor from a backend literal.
    pub fn from_literal(literal: &TensorLiteral) -> Result<Self> {
        let dtype = match literal.spec.dtype {
            crate::backend::spec::DType::F32 => DType::F32,
            crate::backend::spec::DType::Si32 => DType::I32,
            other => {
                bail!("portable backend produced unsupported dtype {:?}", other)
            }
        };
        let dims: Vec<usize> = literal
            .spec
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                Dimension::Static(value) => Ok(*value),
                Dimension::Dynamic(symbol) => {
                    bail!("portable backend produced dynamic dimension {:?}", symbol)
                }
            })
            .collect::<Result<_>>()?;
        let expected_bytes = Shape::new(dims.clone()).num_elements() * dtype.size_in_bytes();
        ensure!(
            literal.bytes.len() == expected_bytes,
            "literal byte length {} does not match expected {}",
            literal.bytes.len(),
            expected_bytes
        );
        Ok(Tensor {
            shape: Shape::new(dims),
            dtype,
            data: literal.bytes.as_ref().to_vec(),
            grad: None,
            requires_grad: false,
        })
    }
}

/// Converts an owned vector into a raw byte buffer without copying.
fn vec_into_bytes<T>(data: Vec<T>) -> Vec<u8> {
    let mut data = ManuallyDrop::new(data);
    let ptr = data.as_mut_ptr() as *mut u8;
    let len = data.len() * size_of::<T>();
    let cap = data.capacity() * size_of::<T>();
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Views a byte slice as a typed slice, asserting that the layout matches.
fn bytes_as_slice<T>(bytes: &[u8]) -> &[T] {
    assert_eq!(
        bytes.len() % size_of::<T>(),
        0,
        "byte length {} is not a multiple of element size {}",
        bytes.len(),
        size_of::<T>()
    );
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / size_of::<T>()) }
}

/// Views a mutable byte slice as a typed mutable slice, asserting the layout.
fn bytes_as_slice_mut<T>(bytes: &mut [u8]) -> &mut [T] {
    assert_eq!(
        bytes.len() % size_of::<T>(),
        0,
        "byte length {} is not a multiple of element size {}",
        bytes.len(),
        size_of::<T>()
    );
    unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, bytes.len() / size_of::<T>())
    }
}
