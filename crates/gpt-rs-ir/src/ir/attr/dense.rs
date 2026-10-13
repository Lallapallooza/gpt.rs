//! Dense constant payloads: the value of a constant op.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Result};
use half::{bf16, f16};

use crate::ir::{print_float, print_shape, DType, Dim, Element, Number, Ranked, TensorType};

/// Little-endian elements of a static-shape tensor. `i1` stores one byte per element.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = DenseText)]
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

    /// The elements as their text forms write them, or `None` when only hex keeps their bits.
    fn element_texts(&self) -> Option<Vec<DenseElement>> {
        let size = self.dtype.size_in_bytes()?;
        let chunks = self.bytes.chunks_exact(size);
        let mut texts = Vec::with_capacity(self.element_count());
        for chunk in chunks {
            let number = match self.dtype {
                DType::F32 => float_text(f32::from_le_bytes(chunk.try_into().ok()?))?,
                DType::Bf16 => float_text(bf16::from_le_bytes(chunk.try_into().ok()?).to_f32())?,
                DType::F16 => float_text(f16::from_le_bytes(chunk.try_into().ok()?).to_f32())?,
                DType::F64 => float_text(f64::from_le_bytes(chunk.try_into().ok()?))?,
                DType::I1 => {
                    texts.push(DenseElement::Bool(match chunk[0] {
                        0 => false,
                        1 => true,
                        _ => return None,
                    }));
                    continue;
                }
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
            texts.push(DenseElement::Number(Number(number)));
        }
        Some(texts)
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

/// The text form of a dense constant: `dense<2x3xf32: [1.0, 2.0, ...]>`, `dense<2x3xf32: 0.0>`
/// when every element is equal, and its bytes in hex, `dense<2xf8e4m3: "0x3840">`, for dtypes or
/// values without an exact decimal form.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`dense` `<` shape($dims, $dtype) `` `:` $elements `>`")]
struct DenseText {
    dims: Ranked<Dim>,
    dtype: DType,
    elements: DenseElements,
}

/// The elements of a dense constant.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
enum DenseElements {
    /// Each element, in row-major order.
    #[attr(format = "$0")]
    List(Vec<DenseElement>),
    /// The little-endian bytes in hex, `"0x3840"`.
    #[attr(format = "$0")]
    Hex(String),
    /// One element, which every position holds.
    #[attr(format = "$0")]
    Splat(DenseElement),
}

/// One element of a dense constant: `true` or `false` for `i1`, a number for any other dtype.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
enum DenseElement {
    #[attr(format = "$0")]
    Bool(bool),
    #[attr(format = "$0")]
    Number(Number),
}

impl From<&Dense> for DenseText {
    fn from(dense: &Dense) -> Self {
        let mut dims = Ranked::new();
        for &extent in &dense.dims {
            dims.push(Dim::Static(extent));
        }
        let elements = match dense.element_texts() {
            Some(texts) if !texts.is_empty() && texts.iter().all(|t| *t == texts[0]) => {
                DenseElements::Splat(texts[0].clone())
            }
            Some(texts) => DenseElements::List(texts),
            None => {
                let mut hex = String::from("0x");
                for byte in dense.bytes.iter() {
                    hex.push_str(&format!("{byte:02x}"));
                }
                DenseElements::Hex(hex)
            }
        };
        DenseText {
            dims,
            dtype: dense.dtype,
            elements,
        }
    }
}

impl TryFrom<DenseText> for Dense {
    type Error = anyhow::Error;

    fn try_from(text: DenseText) -> Result<Self> {
        let dtype = text.dtype;
        let mut dims = Vec::with_capacity(text.dims.len());
        for dim in &text.dims {
            let Some(extent) = dim.as_static() else {
                bail!("a constant has static extents, not {dim}");
            };
            dims.push(extent);
        }
        let mut shape = String::from("dense<");
        print_shape(&text.dims, &dtype, &mut shape);
        shape.push('>');
        // Elements narrower than a byte pack as tensors store them, and only hex writes them.
        let sizes = dims
            .iter()
            .try_fold(1usize, |product, &extent| product.checked_mul(extent))
            .and_then(|count| Some((count, dtype.bytes_for(count)?)));
        let (count, len) = sizes.ok_or_else(|| anyhow!("{shape} takes too many bytes"))?;
        let mut bytes = Vec::new();
        match &text.elements {
            DenseElements::List(elements) => {
                for element in elements {
                    encode(dtype, element, &mut bytes)?;
                }
            }
            DenseElements::Splat(element) => {
                encode(dtype, element, &mut bytes)?;
                bytes = bytes.repeat(count);
            }
            DenseElements::Hex(hex) => bytes = hex_bytes(hex)?,
        }
        ensure!(
            bytes.len() == len,
            "{shape} needs {len} bytes for {count} elements, got {}",
            bytes.len()
        );
        ensure!(
            dtype != DType::I1 || bytes.iter().all(|&b| b <= 1),
            "i1 elements are 0 or 1"
        );
        Ok(Dense {
            dtype,
            dims,
            bytes: bytes.into(),
        })
    }
}

/// The bytes that `0x0a1b...` writes, two hex digits each.
fn hex_bytes(hex: &str) -> Result<Vec<u8>> {
    let Some(digits) = hex.strip_prefix("0x") else {
        bail!("hex bytes start with `0x`");
    };
    ensure!(
        digits.len() % 2 == 0 && digits.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "hex bytes are pairs of hex digits"
    );
    let mut bytes = Vec::with_capacity(digits.len() / 2);
    for pair in (0..digits.len()).step_by(2) {
        bytes.push(u8::from_str_radix(&digits[pair..pair + 2], 16).expect("hex digits"));
    }
    Ok(bytes)
}

/// The value of `text`, an element of `dtype`.
fn value_of<T: FromStr>(text: &Number, dtype: DType) -> Result<T> {
    text.0
        .parse()
        .map_err(|_| anyhow!("`{}` is not a {dtype} value", text.0))
}

/// Appends the little-endian bytes of `element`, an element of `dtype`.
fn encode(dtype: DType, element: &DenseElement, out: &mut Vec<u8>) -> Result<()> {
    let text = match element {
        DenseElement::Bool(value) if dtype == DType::I1 => {
            out.push(u8::from(*value));
            return Ok(());
        }
        DenseElement::Number(text) if dtype != DType::I1 => text,
        DenseElement::Bool(_) => bail!("a {dtype} element is a number"),
        DenseElement::Number(_) => bail!("an i1 element is `true` or `false`"),
    };
    if dtype.is_integer() && text.is_float() {
        bail!("`{}` is not a {dtype} value", text.0);
    }
    match dtype {
        DType::F32 => out.extend(value_of::<f32>(text, dtype)?.to_le_bytes()),
        DType::Bf16 => out.extend(bf16::from_f32(value_of(text, dtype)?).to_le_bytes()),
        DType::F16 => out.extend(f16::from_f32(value_of(text, dtype)?).to_le_bytes()),
        DType::F64 => out.extend(value_of::<f64>(text, dtype)?.to_le_bytes()),
        DType::Si8 => out.extend(value_of::<i8>(text, dtype)?.to_le_bytes()),
        DType::Ui8 => out.extend(value_of::<u8>(text, dtype)?.to_le_bytes()),
        DType::Si16 => out.extend(value_of::<i16>(text, dtype)?.to_le_bytes()),
        DType::Ui16 => out.extend(value_of::<u16>(text, dtype)?.to_le_bytes()),
        DType::Si32 => out.extend(value_of::<i32>(text, dtype)?.to_le_bytes()),
        DType::Ui32 => out.extend(value_of::<u32>(text, dtype)?.to_le_bytes()),
        DType::Si64 => out.extend(value_of::<i64>(text, dtype)?.to_le_bytes()),
        DType::Ui64 => out.extend(value_of::<u64>(text, dtype)?.to_le_bytes()),
        other => bail!("{other} constants are written in hex"),
    }
    Ok(())
}
