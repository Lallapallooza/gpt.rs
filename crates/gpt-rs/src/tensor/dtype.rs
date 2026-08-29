//! Enumerates the scalar element types supported by portable tensor backends.

/// Logical dtype identifier shared between host tensors and backend handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// 32-bit floating point following IEEE-754 semantics.
    F32,
    /// 16-bit floating point with full mantissa (fp16).
    F16,
    /// 16-bit bfloat16 precision as used by many accelerators.
    BF16,
    /// 32-bit signed integer, primarily for index buffers and token ids.
    I32,
}

impl DType {
    /// Returns the number of bytes required per scalar element.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I32 => 4,
        }
    }

    /// Produces a stable tag used when serializing or crossing FFI boundaries.
    pub fn tag(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            DType::I32 => 3,
        }
    }

    /// Reconstructs a `DType` from its serialized tag representation.
    pub fn from_tag(tag: u32) -> Option<Self> {
        match tag {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::BF16),
            3 => Some(DType::I32),
            _ => None,
        }
    }
}

impl std::str::FromStr for DType {
    type Err = anyhow::Error;

    /// Parses a dtype name as written in configs (`bf16`), ignoring case and surrounding whitespace.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(DType::F32),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "i32" => Ok(DType::I32),
            other => anyhow::bail!("unknown dtype '{other}', expected f32, f16, bf16, or i32"),
        }
    }
}
