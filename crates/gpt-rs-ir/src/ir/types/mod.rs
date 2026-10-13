//! Value types shared by every dialect: tensors, buffers, index and scalar values, and the dtypes,
//! extents and float formats they are made of.

mod dtype;
mod float;
mod shape;

pub use self::dtype::*;
pub use self::float::*;
pub use self::shape::*;

/// Dense row-major tensor type, of rank at most [`MAX_RANK`]: `tensor<4x{T}xf32>`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`tensor` `<` shape($dims, $dtype) `>`")]
pub struct TensorType {
    pub dtype: DType,
    pub dims: Ranked<Dim>,
}

impl TensorType {
    /// The type of `dims`. Panics, at the caller, when there are more than [`MAX_RANK`].
    #[track_caller]
    pub fn new(dtype: DType, dims: impl IntoIterator<Item = impl Into<Dim>>) -> Self {
        let dims = match ranked(dims.into_iter().map(Into::into)) {
            Ok(dims) => dims,
            Err(err) => panic!("tensor type: {err}"),
        };
        Self { dtype, dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The extents, when every dimension is static.
    pub fn static_dims(&self) -> Option<Ranked<usize>> {
        let mut dims = Ranked::new();
        for dim in &self.dims {
            dims.push(dim.as_static()?);
        }
        Some(dims)
    }

    /// The extent of `axis`, when it is static.
    pub fn static_dim(&self, axis: usize) -> Option<usize> {
        self.dims[axis].as_static()
    }

    /// The element count, when every dimension is static and the count fits in `usize`.
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim.as_static()?))
    }

    pub fn with_dims(&self, dims: Ranked<Dim>) -> Self {
        Self {
            dtype: self.dtype,
            dims,
        }
    }

    pub fn with_dtype(&self, dtype: DType) -> Self {
        Self {
            dtype,
            dims: self.dims.clone(),
        }
    }

    /// The type with axes reordered by `perm`, a permutation of `0..rank`.
    pub fn permuted(&self, perm: &Perm) -> Self {
        self.with_dims(perm.apply(&self.dims))
    }
}

/// The memory a buffer of the `loop` dialect lives in: who owns it, and for how long.
#[derive(crate::Attr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Space {
    /// Memory that the caller of a launch owns: the parameters of a `loop.func`.
    Global,
    /// Scratch memory of one unit of work, which lives as long as the region that allocates it.
    Private,
}

/// A buffer of the `loop` dialect: elements of `dtype` at the coordinates of `dims`, row-major, in
/// memory of `space`. Its text form is `!loop.buf<4x{T}xf32, global>`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`!loop.buf` `<` shape($dims, $dtype) `,` $space `>`")]
pub struct BufferType {
    pub dtype: DType,
    pub dims: Ranked<Dim>,
    pub space: Space,
}

impl BufferType {
    /// The buffer of `space` that holds the elements of a tensor of type `tensor`.
    pub fn of(tensor: &TensorType, space: Space) -> Self {
        Self {
            dtype: tensor.dtype,
            dims: tensor.dims.clone(),
            space,
        }
    }

    /// The type of the tensor its elements make.
    pub fn tensor_type(&self) -> TensorType {
        TensorType {
            dtype: self.dtype,
            dims: self.dims.clone(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// The type of an SSA value.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    #[attr(format = "`index`")]
    Index,
    #[attr(format = "$0")]
    Scalar(DType),
    #[attr(format = "$0")]
    Tensor(TensorType),
    #[attr(format = "$0")]
    Buffer(BufferType),
}

impl Type {
    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn as_buffer(&self) -> Option<&BufferType> {
        match self {
            Type::Buffer(buffer) => Some(buffer),
            _ => None,
        }
    }

    /// The element dtype of a tensor or a buffer, or the dtype of a scalar.
    pub fn dtype(&self) -> Option<DType> {
        match self {
            Type::Tensor(tensor) => Some(tensor.dtype),
            Type::Buffer(buffer) => Some(buffer.dtype),
            Type::Scalar(dtype) => Some(*dtype),
            Type::Index => None,
        }
    }

    pub fn as_scalar(&self) -> Option<DType> {
        match self {
            Type::Scalar(dtype) => Some(*dtype),
            _ => None,
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, Type::Index)
    }
}

impl From<TensorType> for Type {
    fn from(tensor: TensorType) -> Self {
        Type::Tensor(tensor)
    }
}

/// `types` in parentheses, as the text form and error messages show them.
pub(crate) fn type_list(types: &[&Type]) -> String {
    let items: Vec<String> = types.iter().map(|t| t.to_string()).collect();
    format!("({})", items.join(", "))
}
