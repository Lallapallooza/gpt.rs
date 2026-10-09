//! Dense constant payloads: the value of a constant op.

use std::fmt;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use half::{bf16, f16};

use crate::attr::{print_float, Attr};
use crate::text::Cursor;
use crate::types::{DType, Element, TensorType};

/// Little-endian elements of a static-shape tensor. `i1` stores one byte per element.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dense {
    pub dtype: DType,
    pub dims: Vec<usize>,
    pub bytes: Arc<[u8]>,
}

impl Dense {
    pub fn tensor_type(&self) -> TensorType {
        TensorType::new(self.dtype, self.dims.iter().copied())
    }

    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// A tensor of `dims` whose every element is `element`, little-endian.
    pub fn splat(dtype: DType, dims: impl AsRef<[usize]>, element: &[u8]) -> Self {
        let dims = dims.as_ref().to_vec();
        let count: usize = dims.iter().product();
        Self {
            dtype,
            dims,
            bytes: element.repeat(count).into(),
        }
    }

    /// The bytes of one element, when the tensor is non-empty and all its elements are equal.
    pub fn splat_bytes(&self) -> Option<&[u8]> {
        let size = self.dtype.size_in_bytes()?;
        let first = self.bytes.get(..size)?;
        self.bytes
            .chunks_exact(size)
            .all(|chunk| chunk == first)
            .then_some(first)
    }

    /// Whether every element is `value`, compared by bits.
    pub fn is_splat<T: Element>(&self, value: T) -> bool {
        self.dtype == T::DTYPE && self.splat_bytes() == Some(value.to_le().as_ref())
    }

    /// The same elements in row-major order with a new shape of the same element count.
    pub fn reshaped(&self, dims: Vec<usize>) -> Self {
        debug_assert_eq!(dims.iter().product::<usize>(), self.element_count());
        Self {
            dtype: self.dtype,
            dims,
            bytes: self.bytes.clone(),
        }
    }

    /// The elements as text, or `None` when they must be printed as hex.
    fn element_texts(&self) -> Option<Vec<String>> {
        let size = self.dtype.size_in_bytes()?;
        let chunks = self.bytes.chunks_exact(size);
        let mut texts = Vec::with_capacity(self.element_count());
        for chunk in chunks {
            let text = match self.dtype {
                DType::F32 => float_text(f32::from_le_bytes(chunk.try_into().ok()?))?,
                DType::Bf16 => float_text(bf16::from_le_bytes(chunk.try_into().ok()?).to_f32())?,
                DType::F16 => float_text(f16::from_le_bytes(chunk.try_into().ok()?).to_f32())?,
                DType::F64 => float_text(f64::from_le_bytes(chunk.try_into().ok()?))?,
                DType::I1 => match chunk[0] {
                    0 => "false".to_string(),
                    1 => "true".to_string(),
                    _ => return None,
                },
                DType::Si8 => i8::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Ui8 => chunk[0].to_string(),
                DType::Si16 => i16::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Ui16 => u16::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Si32 => i32::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Ui32 => u32::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Si64 => i64::from_le_bytes(chunk.try_into().ok()?).to_string(),
                DType::Ui64 => u64::from_le_bytes(chunk.try_into().ok()?).to_string(),
                _ => return None,
            };
            texts.push(text);
        }
        Some(texts)
    }

    fn parse_element(dtype: DType, c: &mut Cursor<'_>, out: &mut Vec<u8>) -> Result<()> {
        match dtype {
            DType::F32 => out.extend(c.float::<f32>()?.to_le_bytes()),
            DType::Bf16 => out.extend(bf16::from_f32(c.float()?).to_le_bytes()),
            DType::F16 => out.extend(f16::from_f32(c.float()?).to_le_bytes()),
            DType::F64 => out.extend(c.float::<f64>()?.to_le_bytes()),
            DType::I1 => out.push(u8::from(bool::parse(c)?)),
            DType::Si8 => out.extend(i8::parse(c)?.to_le_bytes()),
            DType::Ui8 => out.extend(u8::parse(c)?.to_le_bytes()),
            DType::Si16 => out.extend(i16::parse(c)?.to_le_bytes()),
            DType::Ui16 => out.extend(u16::parse(c)?.to_le_bytes()),
            DType::Si32 => out.extend(i32::parse(c)?.to_le_bytes()),
            DType::Ui32 => out.extend(u32::parse(c)?.to_le_bytes()),
            DType::Si64 => out.extend(i64::parse(c)?.to_le_bytes()),
            DType::Ui64 => out.extend(u64::parse(c)?.to_le_bytes()),
            other => bail!(c.error(format!("{other} constants are written in hex"))),
        }
        Ok(())
    }
}

/// The shortest decimal that parses back to `value`, or `None` for a NaN, whose bits only hex
/// keeps.
fn float_text<F: fmt::Debug + Into<f64> + Copy>(value: F) -> Option<String> {
    if value.into().is_nan() {
        return None;
    }
    let mut text = String::new();
    print_float(value, &mut text);
    Some(text)
}

/// Printed as `dense<[2, 3] x f32: [1.0, 2.0, ...]>`, as `dense<[2, 3] x f32: 0.0>` when every
/// element is equal, and as hex bytes (`dense<[2] x f8e4m3: 0x3840>`) for dtypes or values
/// without an exact decimal form.
impl Attr for Dense {
    fn print(&self, out: &mut String) {
        out.push_str("dense<");
        self.dims.print(out);
        out.push_str(" x ");
        self.dtype.print(out);
        out.push_str(": ");
        match self.element_texts() {
            Some(texts) if !texts.is_empty() && texts.iter().all(|t| *t == texts[0]) => {
                out.push_str(&texts[0]);
            }
            Some(texts) => {
                out.push('[');
                out.push_str(&texts.join(", "));
                out.push(']');
            }
            None => {
                out.push_str("0x");
                for byte in self.bytes.iter() {
                    out.push_str(&format!("{byte:02x}"));
                }
            }
        }
        out.push('>');
    }

    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.expect("dense")?;
        c.expect("<")?;
        let dims = Vec::<usize>::parse(c)?;
        c.expect("x")?;
        let dtype = DType::parse(c)?;
        c.expect(":")?;
        // Elements narrower than a byte pack as tensors store them, and only hex writes them.
        let count = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        let len = count
            .and_then(|count| dtype.bytes_for(count))
            .ok_or_else(|| c.error(format!("dense<{dims:?} x {dtype}> takes too many bytes")))?;
        let count = count.expect("the byte count fits");
        let mut bytes = Vec::new();
        if c.eat("[") {
            c.list("]", |c| Self::parse_element(dtype, c, &mut bytes))?;
        } else if let Some(hex) = c.hex_bytes()? {
            bytes = hex;
        } else {
            Self::parse_element(dtype, c, &mut bytes)?;
            bytes = bytes.repeat(count);
        }
        c.expect(">")?;
        ensure!(
            bytes.len() == len,
            c.error(format!(
                "dense<{dims:?} x {dtype}> needs {len} bytes for {count} elements, got {}",
                bytes.len()
            ))
        );
        ensure!(
            dtype != DType::I1 || bytes.iter().all(|&b| b <= 1),
            c.error("i1 elements are 0 or 1")
        );
        Ok(Dense {
            dtype,
            dims,
            bytes: bytes.into(),
        })
    }
}
