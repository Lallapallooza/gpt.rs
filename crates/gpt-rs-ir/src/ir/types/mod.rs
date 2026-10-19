//! Value types shared by every dialect: tensors, buffers, vectors, index and scalar values, and the
//! dtypes, extents and float formats they are made of.

mod dtype;
mod float;
mod shape;

use tinyvec::ArrayVec;

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
        let extents = match ranked(dims.into_iter().map(Into::into)) {
            Ok(extents) => extents,
            Err(err) => panic!("tensor type: {err}"),
        };
        Self {
            dtype,
            dims: extents,
        }
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

/// The memory a buffer of the `loop` dialect lives in: who owns it, and for how long. A target
/// file names a space without parameters as its text form does: `private`.
#[derive(
    crate::Attr,
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Space {
    /// Memory that the caller of a launch owns: the parameters of a `loop.func`.
    Global,
    /// Memory that the caller owns and that nothing writes: the elements of a tensor that a
    /// function only reads, such as a parameter mapped from a checkpoint. A launch reads it
    /// through an `in` parameter, which is global.
    ReadOnly,
    /// Scratch memory of one unit of work, which lives as long as the region that allocates it.
    #[default]
    Private,
    /// Vector registers of one unit of work: scratch, as `private` is, that the code keeps in
    /// registers, such as the accumulators of a register tile.
    Reg,
    /// The bytes of a `host.arena`, which start at a multiple of `alignment` bytes and which the
    /// buffers of a function are slices of. Only `host.arena` gives a buffer of this space.
    Arena { alignment: u64 },
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

/// Vector registers of the `isa` dialect: `!isa.vec<6x4x16xf32>` is a group of 6x4 registers that
/// each hold 16 lanes of f32. The last extent is the lanes of one register, and the extents before
/// it, at most [`MAX_GROUP_RANK`], count registers. A vector of `i1` is a mask: one flag per lane.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = VectorText)]
pub struct VectorType {
    pub dtype: DType,
    pub dims: Ranked<usize>,
}

/// The most axes a group of vector registers has, its lanes left out.
pub const MAX_GROUP_RANK: usize = 3;

/// A list indexed by the axes of a group of vector registers, its lanes left out, stored inline:
/// the group's extents, a register's coordinates in it, or a memory op's steps.
pub type PerGroupAxis<T> = ArrayVec<[T; MAX_GROUP_RANK]>;

impl VectorType {
    /// The vectors of `dims`, the lanes last. Panics, at the caller, when
    /// [`VectorType::try_new`] fails.
    #[track_caller]
    pub fn new(dtype: DType, dims: &[usize]) -> Self {
        match Self::try_new(dtype, dims) {
            Ok(vector) => vector,
            Err(err) => panic!("vector type: {err}"),
        }
    }

    /// The vectors of `dims`, the lanes last. Fails unless there are lanes, at most
    /// [`MAX_GROUP_RANK`] axes before them, and no axis of extent 0.
    pub fn try_new(dtype: DType, dims: &[usize]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !dims.is_empty() && dims.len() <= MAX_GROUP_RANK + 1,
            "a vector has lanes and a group rank of at most {MAX_GROUP_RANK}, not the extents \
             {dims:?}"
        );
        anyhow::ensure!(!dims.contains(&0), "a vector has no empty axis");
        Ok(Self {
            dtype,
            dims: dims.iter().copied().collect(),
        })
    }

    /// The lanes of one register.
    pub fn lanes(&self) -> usize {
        *self.dims.last().expect("a vector has lanes")
    }

    /// The extents of the group of registers, the lanes left out.
    pub fn group(&self) -> &[usize] {
        &self.dims[..self.dims.len() - 1]
    }

    /// The registers in the group.
    pub fn registers(&self) -> usize {
        self.group().iter().product()
    }

    pub fn with_dtype(&self, dtype: DType) -> Self {
        Self {
            dtype,
            dims: self.dims,
        }
    }

    /// The tensor type of its elements, the lanes the last axis.
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype, self.dims.iter().copied())
    }
}

/// The text form of [`VectorType`], whose extents are static and which has lanes.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`!isa.vec` `<` shape($dims, $dtype) `>`")]
pub struct VectorText {
    pub dtype: DType,
    pub dims: Ranked<Dim>,
}

impl From<&VectorType> for VectorText {
    fn from(vector: &VectorType) -> Self {
        Self {
            dtype: vector.dtype,
            dims: vector.dims.iter().map(|&dim| Dim::Static(dim)).collect(),
        }
    }
}

impl TryFrom<VectorText> for VectorType {
    type Error = anyhow::Error;

    fn try_from(text: VectorText) -> anyhow::Result<Self> {
        let mut dims = Ranked::new();
        for dim in &text.dims {
            let Some(extent) = dim.as_static() else {
                anyhow::bail!("a vector's extents are static, not {dim}");
            };
            dims.push(extent);
        }
        Self::try_new(text.dtype, &dims)
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
    #[attr(format = "$0")]
    Vector(VectorType),
}

impl Type {
    /// The bytes of an index, a 64-bit integer, and of a buffer, which a call passes as the 64-bit
    /// address of its elements.
    pub const INDEX_BYTES: u64 = 8;

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

    pub fn as_vector(&self) -> Option<&VectorType> {
        match self {
            Type::Vector(vector) => Some(vector),
            _ => None,
        }
    }

    /// The element dtype of a tensor or a buffer, or the dtype of a scalar.
    pub fn dtype(&self) -> Option<DType> {
        match self {
            Type::Tensor(tensor) => Some(tensor.dtype),
            Type::Buffer(buffer) => Some(buffer.dtype),
            Type::Vector(vector) => Some(vector.dtype),
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
    let items: Vec<String> = types.iter().map(|ty| ty.to_string()).collect();
    format!("({})", items.join(", "))
}
