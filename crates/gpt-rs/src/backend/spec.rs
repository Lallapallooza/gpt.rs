use std::{
    collections::{BTreeMap, HashMap},
    fmt, fs, io,
    path::Path,
    sync::Arc,
};

use serde::{ser::SerializeStruct, Deserialize, Serialize};
use thiserror::Error;

/// Frozen PTIR specification version enforced by this interface.
pub const SPEC_VERSION: &str = "ptir.v0.4";

fn default_spec_version() -> String {
    SPEC_VERSION.to_string()
}

/// Enumerates scalar element types supported by the PTIR backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum DType {
    I1,
    Si4,
    Ui4,
    Si8,
    Ui8,
    Si16,
    Ui16,
    Si32,
    Ui32,
    Si64,
    Ui64,
    Fp8E4M3,
    Fp8E5M2,
    Bf16,
    F16,
    F32,
    F64,
    Cf32,
    Cf64,
}

impl DType {
    /// Returns `true` when the dtype is any signed or unsigned integer, including quantized 4-bit types.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            DType::Si4
                | DType::Ui4
                | DType::Si8
                | DType::Ui8
                | DType::Si16
                | DType::Ui16
                | DType::Si32
                | DType::Ui32
                | DType::Si64
                | DType::Ui64
        )
    }

    /// Returns `true` when the dtype is a signed integer (including 4-bit quantized values).
    pub fn is_signed_integer(self) -> bool {
        matches!(
            self,
            DType::Si4 | DType::Si8 | DType::Si16 | DType::Si32 | DType::Si64
        )
    }

    /// Returns `true` when the dtype is an unsigned integer (including 4-bit quantized values).
    pub fn is_unsigned_integer(self) -> bool {
        matches!(
            self,
            DType::Ui4 | DType::Ui8 | DType::Ui16 | DType::Ui32 | DType::Ui64
        )
    }

    /// Returns `true` when the dtype is a floating-point representation.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DType::Fp8E4M3 | DType::Fp8E5M2 | DType::Bf16 | DType::F16 | DType::F32 | DType::F64
        )
    }

    /// Returns `true` when the dtype is complex.
    pub fn is_complex(self) -> bool {
        matches!(self, DType::Cf32 | DType::Cf64)
    }

    /// Returns the storage bit-width when well-defined for the logical scalar.
    pub fn bitwidth(self) -> Option<usize> {
        match self {
            DType::I1 => Some(1),
            DType::Si4 | DType::Ui4 => Some(4),
            DType::Si8 | DType::Ui8 | DType::Fp8E4M3 | DType::Fp8E5M2 => Some(8),
            DType::Si16 | DType::Ui16 | DType::Bf16 | DType::F16 => Some(16),
            DType::Si32 | DType::Ui32 | DType::F32 => Some(32),
            DType::Si64 | DType::Ui64 | DType::F64 => Some(64),
            DType::Cf32 => Some(64),
            DType::Cf64 => Some(128),
        }
    }

    /// Returns the size in bytes when storage size is well-defined.
    pub fn size_in_bytes(self) -> Option<usize> {
        match self {
            DType::I1 => Some(1),
            DType::Si8 | DType::Ui8 => Some(1),
            DType::Si16 | DType::Ui16 | DType::Bf16 | DType::F16 => Some(2),
            DType::Si32 | DType::Ui32 | DType::F32 => Some(4),
            DType::Si64 | DType::Ui64 | DType::F64 => Some(8),
            DType::Cf32 => Some(8),
            DType::Cf64 => Some(16),
            DType::Si4 | DType::Ui4 | DType::Fp8E4M3 | DType::Fp8E5M2 => None,
        }
    }
}

/// Names a symbolic dynamic dimension (e.g. `?B`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DimSymbol(Arc<str>);

impl DimSymbol {
    pub fn new(name: impl Into<String>) -> Self {
        Self(Arc::<str>::from(name.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for DimSymbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DimSymbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Ok(DimSymbol::new(name))
    }
}

/// Represents a single axis extent in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dimension {
    Static(usize),
    Dynamic(DimSymbol),
}

impl Dimension {
    /// Convenience constructor for static extents.
    pub fn from_usize(value: usize) -> Self {
        Self::Static(value)
    }
}

/// Logical tensor shape as an ordered list of dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<Dimension>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<Dimension>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[Dimension] {
        &self.dims
    }

    pub fn into_dims(self) -> Vec<Dimension> {
        self.dims
    }

    /// Returns static dimensions when all dims are static.
    pub fn static_dims(&self) -> Option<Vec<usize>> {
        let mut dims = Vec::with_capacity(self.dims.len());
        for dim in &self.dims {
            match dim {
                Dimension::Static(value) => dims.push(*value),
                Dimension::Dynamic(_) => return None,
            }
        }
        Some(dims)
    }

    /// Returns element count when all dims are static and non-zero.
    pub fn element_count(&self) -> Option<usize> {
        let dims = self.static_dims()?;
        let mut count = 1usize;
        for dim in dims {
            count = count.checked_mul(dim)?;
        }
        Some(count)
    }
}

/// Tensor metadata coupling dtype and shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorSpec {
    pub dtype: DType,
    pub shape: Shape,
}

impl TensorSpec {
    pub fn new(dtype: DType, shape: Shape) -> Self {
        Self { dtype, shape }
    }

    /// Returns total element count when shape is fully static.
    pub fn element_count(&self) -> Option<usize> {
        self.shape.element_count()
    }

    /// Returns total byte length when shape is static and dtype size is known.
    pub fn byte_len(&self) -> Option<usize> {
        let elem_count = self.element_count()?;
        let elem_size = self.dtype.size_in_bytes()?;
        elem_count.checked_mul(elem_size)
    }
}

/// Scalar literal used for attributes (e.g., padding values).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    I1(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Complex { re: f64, im: f64 },
}

/// Dense literal tensor payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLiteral {
    pub spec: TensorSpec,
    pub bytes: Arc<[u8]>,
}

impl TensorLiteral {
    pub fn new(spec: TensorSpec, bytes: Arc<[u8]>) -> Self {
        Self { spec, bytes }
    }

    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

impl Serialize for TensorLiteral {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TensorLiteral", 2)?;
        state.serialize_field("spec", &self.spec)?;
        state.serialize_field("bytes", &self.bytes.as_ref())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TensorLiteral {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TensorLiteralHelper {
            spec: TensorSpec,
            bytes: Vec<u8>,
        }

        let helper = TensorLiteralHelper::deserialize(deserializer)?;
        Ok(TensorLiteral {
            spec: helper.spec,
            bytes: Arc::<[u8]>::from(helper.bytes),
        })
    }
}

/// Initialization payload when materialising tensors on a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorInit {
    Literal(TensorLiteral),
    Zeroed(TensorSpec),
}

/// Comparator used by the `compare` op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Less,
    LessEqual,
    Equal,
    GreaterEqual,
    Greater,
    NotEqual,
}

/// Elementwise unary ops defined by the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementwiseUnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Tanh,
    Erf,
    Rsqrt,
    Reciprocal,
}

/// Elementwise binary ops defined by the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementwiseBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
}

/// Reduction families with optional accumulator dtype promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
}

/// Segment reduction variants extending the reduction family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentReduceKind {
    Sum,
    Max,
}

/// Fully describes a `dot_general` contraction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DotGeneralSpec {
    pub batch_lhs: Vec<usize>,
    pub batch_rhs: Vec<usize>,
    pub contract_lhs: Vec<usize>,
    pub contract_rhs: Vec<usize>,
    pub accum_dtype: Option<DType>,
    pub out_dtype: Option<DType>,
}

/// Configuration shared by `reduce_sum`, `reduce_max`, and `reduce_min`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceSpec {
    pub kind: ReduceKind,
    pub axes: Vec<usize>,
    pub keepdims: bool,
    pub accum_dtype: Option<DType>,
    pub out_dtype: Option<DType>,
}

/// Describes the `argmax` op.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgMaxSpec {
    pub axis: isize,
    pub keepdims: bool,
    pub output_dtype: DType,
}

/// Attribute payload for `compare`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompareSpec {
    pub op: ComparisonOp,
}

/// Attribute payload for `cast`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CastSpec {
    pub dtype: DType,
}

/// Entry in the requested output shape for `reshape`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReshapeDim {
    Explicit(Dimension),
    Infer,
}

/// Attribute payload for `reshape`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReshapeSpec {
    pub new_shape: Vec<ReshapeDim>,
}

/// Permutation payload for `transpose`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransposeSpec {
    pub perm: Vec<usize>,
}

/// Attribute payload for `broadcast_to`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BroadcastToSpec {
    pub result_shape: Shape,
}

/// Attribute payload for `slice`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceSpec {
    pub starts: Vec<usize>,
    pub sizes: Vec<usize>,
}

/// Attribute payload for `concat`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcatSpec {
    pub axis: isize,
}

/// Attribute payload for `pad`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PadSpec {
    pub low: Vec<usize>,
    pub high: Vec<usize>,
    pub interior: Vec<usize>,
    pub pad_value: Literal,
}

/// Attribute payload for `tile`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileSpec {
    pub repeats: Vec<usize>,
}

/// Attribute payload for `iota`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IotaSpec {
    pub shape: Shape,
    pub dtype: DType,
    pub axis: usize,
}

/// Attribute payload for `gather`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatherSpec {
    pub axis: isize,
}

/// Dimension numbers defining a `scatter_add`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScatterSpec {
    pub update_window_dims: Vec<usize>,
    pub inserted_window_dims: Vec<usize>,
    pub scatter_dims_to_operand_dims: Vec<usize>,
    pub index_vector_dim: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScatterReduceKind {
    Add,
    Max,
    Min,
    Replace,
}

/// Attribute payload for `scatter_reduce`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScatterReduceSpec {
    pub axis: isize,
    pub reduce: ScatterReduceKind,
}

/// Attribute payload for `dynamic_slice`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicSliceSpec {
    pub sizes: Vec<usize>,
}

/// Attribute payload for `dynamic_update_slice`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicUpdateSliceSpec {
    pub sizes: Vec<usize>,
}

/// Attribute payload for `extract_patches`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractPatchesSpec {
    pub window: Vec<usize>,
    pub strides: Vec<usize>,
    pub dilation: Vec<usize>,
    pub padding: Vec<(usize, usize)>,
    pub pad_value: Literal,
}

/// Identifies a region referenced by control-flow ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegionId(pub usize);

/// Control-flow payload for `cond`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CondSpec {
    pub true_region: RegionId,
    pub false_region: RegionId,
}

/// Control-flow payload for `while`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhileSpec {
    pub cond_region: RegionId,
    pub body_region: RegionId,
}

/// Control-flow payload for `scan`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanSpec {
    pub body_region: RegionId,
    pub carry_count: usize,
    pub scan_output_count: usize,
}

/// Attribute payload for `reduce_window`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReduceWindowSpec {
    pub window_dims: Vec<usize>,
    pub strides: Vec<usize>,
    pub padding: Vec<(usize, usize)>,
    pub base_dilation: Vec<usize>,
    pub window_dilation: Vec<usize>,
    pub reduce: ReduceKind,
    pub accum_dtype: Option<DType>,
}

/// Attribute payload for `rng_uniform`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngUniformSpec {
    pub shape: Shape,
    pub dtype: DType,
}

/// Attribute payload for `rng_normal`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RngNormalSpec {
    pub shape: Shape,
    pub dtype: DType,
}

/// Attribute payload for `top_k`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopKSpec {
    pub k: usize,
    pub axis: isize,
    pub largest: bool,
    pub indices_dtype: DType,
}

/// Attribute payload for segment reductions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentReduceSpec {
    pub kind: SegmentReduceKind,
    pub num_segments: usize,
    pub accum_dtype: Option<DType>,
    pub indices_are_sorted: bool,
    pub unique: bool,
}

/// Attribute payload for `quantize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizeSpec {
    pub output_dtype: DType,
    pub axis: Option<usize>,
}

/// Attribute payload for `dequantize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DequantizeSpec {
    pub axis: Option<usize>,
    pub output_dtype: Option<DType>,
}

/// Attribute payload for `requantize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequantizeSpec {
    pub output_dtype: DType,
    pub axis: Option<usize>,
}

/// Attribute payload for `custom_call`.
///
/// Custom-call attributes are intentionally limited to simple primitives and arrays so they are
/// easy to serialize, hash, and validate across backends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum CustomCallAttr {
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    I64Array(Vec<i64>),
    F64Array(Vec<f64>),
    BoolArray(Vec<bool>),
    StringArray(Vec<String>),
}

/// Attribute payload for `custom_call`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomCallSpec {
    pub target: String,
    #[serde(default)]
    pub attrs: BTreeMap<String, CustomCallAttr>,
}

/// Unique identifier for SSA values in a PTIR program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

/// Typing information for SSA values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Tensor(TensorSpec),
    Tuple(Vec<ValueType>),
}

/// Operand reference in an instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operand {
    Value(ValueId),
    TupleElement { tuple: ValueId, index: usize },
    Literal(TensorLiteral),
}

/// Declarative form of PTIR operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    Constant(TensorLiteral),
    ElementwiseUnary(ElementwiseUnaryOp),
    ElementwiseBinary(ElementwiseBinaryOp),
    DotGeneral(DotGeneralSpec),
    Reduce(ReduceSpec),
    ArgMax(ArgMaxSpec),
    Compare(CompareSpec),
    Select,
    Cast(CastSpec),
    StopGradient,
    Reshape(ReshapeSpec),
    Transpose(TransposeSpec),
    BroadcastTo(BroadcastToSpec),
    Slice(SliceSpec),
    Concat(ConcatSpec),
    Pad(PadSpec),
    Tile(TileSpec),
    Iota(IotaSpec),
    Take,
    Gather(GatherSpec),
    ScatterAdd(ScatterSpec),
    ScatterReduce(ScatterReduceSpec),
    DynamicSlice(DynamicSliceSpec),
    DynamicUpdateSlice(DynamicUpdateSliceSpec),
    Cond(CondSpec),
    While(WhileSpec),
    Scan(ScanSpec),
    ExtractPatches(ExtractPatchesSpec),
    ReduceWindow(ReduceWindowSpec),
    RngUniform(RngUniformSpec),
    RngNormal(RngNormalSpec),
    TopK(TopKSpec),
    SegmentReduce(SegmentReduceSpec),
    Quantize(QuantizeSpec),
    Dequantize(DequantizeSpec),
    Requantize(RequantizeSpec),
    CustomCall(CustomCallSpec),
}

/// Single SSA instruction in the declarative PTIR program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub id: ValueId,
    pub op: Operation,
    pub operands: Vec<Operand>,
    pub output: ValueType,
}

/// Region used by control-flow constructs. Regions have their own parameter list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: RegionId,
    pub parameters: Vec<ValueType>,
    pub body: Vec<Instruction>,
    pub results: Vec<ValueType>,
}

/// High-level fusion hint kinds discovered by optimizer analyses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HintKind {
    ElementwiseDag,
    DotEpilogue,
    ReductionChain,
}

/// Execution preference for a fusion hint region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HintPolicy {
    Optional,
    Preferred,
    Required,
}

/// Function-level fusion hint that groups a PTIR instruction region.
///
/// Hints are advisory and live above normal PTIR operations. Backends can decide
/// whether to materialize them as fused artifacts or inline their `body` as plain
/// instructions depending on support and policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HintRegion {
    pub id: u32,
    pub kind: HintKind,
    pub policy: HintPolicy,
    pub inputs: Vec<ValueId>,
    pub exports: Vec<ValueId>,
    pub body: Vec<Instruction>,
    #[serde(default)]
    pub attrs: BTreeMap<String, CustomCallAttr>,
}

/// PTIR function describing a reusable computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<ValueType>,
    pub parameter_ids: Vec<ValueId>,
    pub results: Vec<ValueType>,
    pub body: Vec<Instruction>,
    #[serde(default)]
    pub hints: Vec<HintRegion>,
    pub result_ids: Vec<ValueId>,
}

/// Complete PTIR module with optional helper regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    #[serde(default = "default_spec_version")]
    pub spec_version: String,
    pub entry: String,
    pub functions: Vec<Function>,
    pub regions: Vec<Region>,
}

#[derive(Debug, Error)]
pub enum ProgramSerdeError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bincode error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("program spec version '{found}' does not match expected '{expected}'")]
    SpecVersionMismatch {
        found: String,
        expected: &'static str,
    },
}

#[derive(Debug, Error)]
pub enum ProgramIoError {
    #[error(transparent)]
    Serialization(#[from] ProgramSerdeError),
    #[error("i/o error: {0}")]
    Io(#[from] io::Error),
}

impl Program {
    pub fn new(entry: impl Into<String>) -> Self {
        Self {
            spec_version: SPEC_VERSION.to_string(),
            entry: entry.into(),
            functions: Vec::new(),
            regions: Vec::new(),
        }
    }

    pub fn with_functions(mut self, functions: Vec<Function>) -> Self {
        self.functions = functions;
        self
    }

    pub fn with_regions(mut self, regions: Vec<Region>) -> Self {
        self.regions = regions;
        self
    }

    pub fn to_json_string(&self) -> Result<String, ProgramSerdeError> {
        serde_json::to_string_pretty(self).map_err(ProgramSerdeError::from)
    }

    pub fn from_json_str(src: &str) -> Result<Self, ProgramSerdeError> {
        let mut program: Program = serde_json::from_str(src).map_err(ProgramSerdeError::from)?;
        program.spec_version = normalize_spec_version(program.spec_version)?;
        Ok(program)
    }

    pub fn to_bincode_bytes(&self) -> Result<Vec<u8>, ProgramSerdeError> {
        bincode::serialize(self).map_err(ProgramSerdeError::from)
    }

    pub fn from_bincode_slice(bytes: &[u8]) -> Result<Self, ProgramSerdeError> {
        let mut program: Program = bincode::deserialize(bytes).map_err(ProgramSerdeError::from)?;
        program.spec_version = normalize_spec_version(program.spec_version)?;
        Ok(program)
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), ProgramIoError> {
        let contents = self.to_json_string()?;
        fs::write(path, contents).map_err(ProgramIoError::from)
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, ProgramIoError> {
        let contents = fs::read_to_string(path).map_err(ProgramIoError::from)?;
        Program::from_json_str(&contents).map_err(ProgramIoError::from)
    }

    pub fn save_bincode<P: AsRef<Path>>(&self, path: P) -> Result<(), ProgramIoError> {
        let bytes = self.to_bincode_bytes()?;
        fs::write(path, bytes).map_err(ProgramIoError::from)
    }

    pub fn load_bincode<P: AsRef<Path>>(path: P) -> Result<Self, ProgramIoError> {
        let bytes = fs::read(path).map_err(ProgramIoError::from)?;
        Program::from_bincode_slice(&bytes).map_err(ProgramIoError::from)
    }

    pub fn to_text(&self) -> String {
        format!("{self}")
    }
}

fn normalize_spec_version(version: String) -> Result<String, ProgramSerdeError> {
    if version.is_empty() {
        return Ok(SPEC_VERSION.to_string());
    }
    if version == SPEC_VERSION {
        Ok(version)
    } else {
        Err(ProgramSerdeError::SpecVersionMismatch {
            found: version,
            expected: SPEC_VERSION,
        })
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_line(
            f,
            0,
            &format!(
                "program @{} (spec_version = {}) {{",
                self.entry, self.spec_version
            ),
        )?;
        for function in &self.functions {
            fmt_function(function, 1, f)?;
        }
        for region in &self.regions {
            fmt_region(region, 1, f)?;
        }
        write_line(f, 0, "}")
    }
}

fn fmt_function(function: &Function, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write_line(f, indent, &format!("func @{} {{", function.name))?;
    if !function.parameter_ids.is_empty() {
        write_line(f, indent + 1, "params:")?;
        for (value_id, value_type) in function
            .parameter_ids
            .iter()
            .zip(function.parameters.iter())
        {
            write_line(
                f,
                indent + 2,
                &format!("%{} : {}", value_id.0, format_value_type(value_type)),
            )?;
        }
    }
    if !function.body.is_empty() {
        write_line(f, indent + 1, "body:")?;
        for instruction in &function.body {
            fmt_instruction(instruction, indent + 2, f)?;
        }
    }
    if !function.hints.is_empty() {
        write_line(f, indent + 1, "hints:")?;
        for hint in &function.hints {
            fmt_hint_region(hint, indent + 2, f)?;
        }
    }
    if !function.result_ids.is_empty() {
        write_line(f, indent + 1, "results:")?;
        for (value_id, value_type) in function.result_ids.iter().zip(function.results.iter()) {
            write_line(
                f,
                indent + 2,
                &format!("%{} : {}", value_id.0, format_value_type(value_type)),
            )?;
        }
    }
    write_line(f, indent, "}")
}

fn fmt_region(region: &Region, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write_line(f, indent, &format!("region ^r{} {{", region.id.0))?;
    if !region.parameters.is_empty() {
        write_line(f, indent + 1, "params:")?;
        for (index, value_type) in region.parameters.iter().enumerate() {
            write_line(
                f,
                indent + 2,
                &format!("%arg{} : {}", index, format_value_type(value_type)),
            )?;
        }
    }
    if !region.body.is_empty() {
        write_line(f, indent + 1, "body:")?;
        for instruction in &region.body {
            fmt_instruction(instruction, indent + 2, f)?;
        }
    }
    if !region.results.is_empty() {
        write_line(f, indent + 1, "results:")?;
        for (index, value_type) in region.results.iter().enumerate() {
            write_line(
                f,
                indent + 2,
                &format!("%res{} : {}", index, format_value_type(value_type)),
            )?;
        }
    }
    write_line(f, indent, "}")
}

fn fmt_hint_region(hint: &HintRegion, indent: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write_line(
        f,
        indent,
        &format!(
            "fused_hint id={} kind={:?} policy={:?} {{",
            hint.id, hint.kind, hint.policy
        ),
    )?;
    if !hint.inputs.is_empty() {
        let inputs = hint
            .inputs
            .iter()
            .map(|id| format!("%{}", id.0))
            .collect::<Vec<_>>()
            .join(", ");
        write_line(f, indent + 1, &format!("inputs: [{inputs}]"))?;
    }
    if !hint.exports.is_empty() {
        let exports = hint
            .exports
            .iter()
            .map(|id| format!("%{}", id.0))
            .collect::<Vec<_>>()
            .join(", ");
        write_line(f, indent + 1, &format!("exports: [{exports}]"))?;
    }
    if !hint.attrs.is_empty() {
        write_line(f, indent + 1, "attrs:")?;
        for (key, value) in &hint.attrs {
            write_line(f, indent + 2, &format!("{key} = {:?}", value))?;
        }
    }
    if !hint.body.is_empty() {
        write_line(f, indent + 1, "body:")?;
        for instruction in &hint.body {
            fmt_instruction(instruction, indent + 2, f)?;
        }
    }
    write_line(f, indent, "}")
}

fn fmt_instruction(
    instruction: &Instruction,
    indent: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let operands = instruction
        .operands
        .iter()
        .map(format_operand)
        .collect::<Vec<_>>();
    let operands_str = if operands.is_empty() {
        String::new()
    } else {
        operands.join(", ")
    };
    let op_repr = format!("{:?}", instruction.op);
    let line = if operands_str.is_empty() {
        format!(
            "%{} = {} -> {}",
            instruction.id.0,
            op_repr,
            format_value_type(&instruction.output)
        )
    } else {
        format!(
            "%{} = {}({}) -> {}",
            instruction.id.0,
            op_repr,
            operands_str,
            format_value_type(&instruction.output)
        )
    };
    write_line(f, indent, &line)
}

fn format_value_type(value_type: &ValueType) -> String {
    match value_type {
        ValueType::Tensor(spec) => {
            format!("tensor<{:?} x {}>", spec.dtype, format_shape(&spec.shape))
        }
        ValueType::Tuple(elements) => {
            let inner = elements
                .iter()
                .map(format_value_type)
                .collect::<Vec<_>>()
                .join(", ");
            format!("tuple<{}>", inner)
        }
    }
}

fn format_shape(shape: &Shape) -> String {
    let dims = shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => v.to_string(),
            Dimension::Dynamic(sym) => format!("?{}", sym.as_str()),
        })
        .collect::<Vec<_>>();
    if dims.is_empty() {
        "[]".to_string()
    } else {
        dims.join("x")
    }
}

fn format_operand(operand: &Operand) -> String {
    match operand {
        Operand::Value(id) => format!("%{}", id.0),
        Operand::TupleElement { tuple, index } => format!("%{}[{}]", tuple.0, index),
        Operand::Literal(lit) => format!(
            "literal(dtype={:?}, shape={})",
            lit.spec.dtype,
            format_shape(&lit.spec.shape)
        ),
    }
}

fn write_line(f: &mut fmt::Formatter<'_>, indent: usize, line: &str) -> fmt::Result {
    for _ in 0..indent {
        f.write_str("  ")?;
    }
    writeln!(f, "{line}")
}

/// Lightweight builder for constructing PTIR functions programmatically.
#[derive(Default, Serialize, Deserialize)]
pub struct ProgramBuilder {
    next_value_id: u32,
    parameters: Vec<(ValueId, ValueType)>,
    instructions: Vec<Instruction>,
    value_types: HashMap<ValueId, ValueType>,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_parameter(&mut self, ty: ValueType) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        self.value_types.insert(id, ty.clone());
        self.parameters.push((id, ty));
        id
    }

    pub fn emit_single(
        &mut self,
        op: Operation,
        operands: Vec<Operand>,
        output: ValueType,
    ) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        self.value_types.insert(id, output.clone());
        self.instructions.push(Instruction {
            id,
            op,
            operands,
            output,
        });
        id
    }

    pub fn value_type(&self, id: ValueId) -> Option<&ValueType> {
        self.value_types.get(&id)
    }

    pub fn finish(self, name: impl Into<String>, result_ids: Vec<ValueId>) -> Function {
        let mut results = Vec::with_capacity(result_ids.len());
        for id in &result_ids {
            let ty = self
                .value_types
                .get(id)
                .expect("result value id must have a recorded type")
                .clone();
            results.push(ty);
        }
        let (parameter_ids, parameters): (Vec<_>, Vec<_>) = self.parameters.into_iter().unzip();
        Function {
            name: name.into(),
            parameters,
            parameter_ids,
            results,
            body: self.instructions,
            hints: Vec::new(),
            result_ids,
        }
    }
}

/// Stable set of specification error identifiers referenced throughout the doc.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpecErrorCode {
    NegativeDimension,
    DynamicDimensionMismatch,
    DTypeNotSupported,
    InvalidAttributeValue,
    InvalidAccumDTypeOverride,
    IntegerPromotionExceedsSi64,
    IntegerDivideByZero,
    CompareOperandsMustMatchShape,
    CompareDTypePromotionInvalid,
    BroadcastRankMismatch,
    BroadcastDimsInvalid,
    ScatterIndexOutOfBounds,
    GatherIndexOutOfBounds,
    RegionSignatureMismatch,
    Unspecified(&'static str),
}

impl SpecErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecErrorCode::NegativeDimension => "SpecError: negative dimension",
            SpecErrorCode::DynamicDimensionMismatch => "SpecError: dynamic dimension mismatch",
            SpecErrorCode::DTypeNotSupported => "SpecError: dtype not supported for op",
            SpecErrorCode::InvalidAttributeValue => "SpecError: invalid attribute value",
            SpecErrorCode::InvalidAccumDTypeOverride => "SpecError: invalid accum_dtype override",
            SpecErrorCode::IntegerPromotionExceedsSi64 => {
                "SpecError: integer promotion exceeds si64 range"
            }
            SpecErrorCode::IntegerDivideByZero => "SpecError: integer divide by zero",
            SpecErrorCode::CompareOperandsMustMatchShape => {
                "SpecError: compare operands must match shape"
            }
            SpecErrorCode::CompareDTypePromotionInvalid => {
                "SpecError: compare dtype promotion invalid"
            }
            SpecErrorCode::BroadcastRankMismatch => "SpecError: broadcast rank mismatch",
            SpecErrorCode::BroadcastDimsInvalid => {
                "SpecError: broadcast dims must be sorted and unique"
            }
            SpecErrorCode::ScatterIndexOutOfBounds => "SpecError: scatter index out of bounds",
            SpecErrorCode::GatherIndexOutOfBounds => "SpecError: gather index out of bounds",
            SpecErrorCode::RegionSignatureMismatch => "SpecError: region signature mismatch",
            SpecErrorCode::Unspecified(code) => code,
        }
    }
}

/// Validation failure captured before execution.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecError {
    pub code: SpecErrorCode,
    pub detail: Option<String>,
}

impl SpecError {
    pub fn new(code: SpecErrorCode, detail: impl Into<Option<String>>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.detail {
            Some(detail) => write!(f, "{} ({detail})", self.code.as_str()),
            None => write!(f, "{}", self.code.as_str()),
        }
    }
}

impl std::error::Error for SpecError {}

/// Backend error surfaced to higher layers.
#[derive(Debug)]
pub enum BackendError {
    SpecViolation(SpecError),
    Unimplemented { op: &'static str, reason: String },
    Execution { message: String },
}

impl BackendError {
    pub fn spec(code: SpecErrorCode, detail: impl Into<Option<String>>) -> Self {
        BackendError::SpecViolation(SpecError::new(code, detail))
    }

    pub fn unimplemented(op: &'static str, reason: impl Into<String>) -> Self {
        BackendError::Unimplemented {
            op,
            reason: reason.into(),
        }
    }

    pub fn execution(message: impl Into<String>) -> Self {
        BackendError::Execution {
            message: message.into(),
        }
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::SpecViolation(err) => write!(f, "{err}"),
            BackendError::Unimplemented { op, reason } => {
                write!(f, "{op} is not implemented: {reason}")
            }
            BackendError::Execution { message } => {
                write!(f, "backend execution failure: {message}")
            }
        }
    }
}

impl std::error::Error for BackendError {}

/// Convenience alias for results returned by backend routines.
pub type BackendResult<T> = Result<T, BackendError>;

/// Request metadata for backend-side decode token sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeSampleRequest {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub random_u: Option<f32>,
}

impl DecodeSampleRequest {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            random_u: None,
        }
    }
}

/// Portable backend trait that evaluates PTIR programs.
pub trait PortableBackend: Send + Sync {
    type TensorHandle: Clone + Send + Sync + 'static;

    /// Returns a human-readable backend identifier (e.g., `"cpu"`, `"cuda"`).
    fn backend_name(&self) -> &str;

    /// Returns optional backend hooks that extend the default optimizer pipeline.
    ///
    /// Backends should install legalization/fusion passes here without leaking backend-specific
    /// APIs into model/functionals.
    fn pipeline(&self) -> Option<Arc<dyn crate::backend::pipeline::BackendPipeline<Self>>> {
        None
    }

    /// Returns a backend-provided resolver for parameter handles keyed by stable ids.
    ///
    /// Optimizers may use this to memoize derived parameter representations (for example, packed
    /// weights) and the runtime uses it to feed parameter inputs when executing cached plans.
    fn param_resolver(
        &self,
    ) -> Option<Arc<dyn crate::backend::param_resolver::ParamResolver<Handle = Self::TensorHandle>>>
    {
        None
    }

    /// Materialises a tensor handle from host initialisation data.
    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle>;

    /// Reads back a tensor handle into a dense literal (debug/development only).
    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral>;

    /// Executes a single instruction given already materialised operand handles.
    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>>;

    /// Executes an entire PTIR program starting from the entry function.
    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>>;

    /// Returns true when this backend can serve `sample_decode_token` for the request.
    fn supports_decode_sampling(&self, _request: DecodeSampleRequest) -> bool {
        false
    }

    /// Optional backend-side decode sampling hook.
    ///
    /// Backends can return `Some(token_id)` to bypass host logits materialization for decode.
    /// Returning `Ok(None)` preserves existing CPU sampler behavior.
    fn sample_decode_token(
        &self,
        _logits: &Self::TensorHandle,
        _logits_spec: &TensorSpec,
        _request: DecodeSampleRequest,
    ) -> BackendResult<Option<usize>> {
        Ok(None)
    }
}
