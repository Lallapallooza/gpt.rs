//! The projection layer built on [`functional::linear`]: PyTorch `nn.Linear`, and `transformers`'
//! `Conv1D` with its transposed weight.

use anyhow::{ensure, Result};

use crate::backend::PortableBackend;
use crate::nn::{self, Params};
use crate::ops::functional::{self, WeightLayout};
use crate::tensor::{DType, DeviceTensor};

/// Fully connected layer `y = x W^T + b`. `weight` is stored as `layout` says: `[out_features,
/// in_features]` like `nn.Linear`, or `[in_features, out_features]` like the `Conv1D` of GPT-2
/// checkpoints. An FP8 checkpoint stores `W` as `weight * weight_scale_inv`, in the dtype of the
/// scale.
///
/// A 4-bit checkpoint (compressed-tensors' `pack-quantized`) stores no `weight`. It stores codes
/// in `weight_packed`, eight to an int32 word along `in_features`, from the least significant bits
/// up. The layer reads them in place as `[out_features, in_features]` codes. `weight_scale` holds
/// the scale of each group of codes along `in_features`, and `W` is `(code - 8) * scale`. Loading
/// fails on codes with zero points (`weight_zero_point`) and on a `weight` with a `weight_scale`.
/// The layer does not check the dtype or packing of `weight_packed`, so it misreads other packed
/// codes, such as 8-bit or NVFP4 ones.
#[nn::module]
pub struct Linear {
    pub weight: Option<Tensor>,
    pub weight_scale_inv: Option<Tensor>,
    pub weight_packed: Option<Tensor>,
    pub weight_scale: Option<Tensor>,
    pub bias: Option<Tensor>,
    #[module(config)]
    pub layout: WeightLayout,
    #[module(config)]
    pub input_dtype: Option<DType>,
}

#[nn::module]
impl Linear {
    /// A projection with an `[out_features, in_features]` weight, like `nn.Linear`.
    pub fn new(p: &mut Params<B>) -> Result<Self> {
        Self::with_layout(p, WeightLayout::OutIn)
    }

    /// A projection whose weight the checkpoint stores as `layout` says, or packed in 4-bit codes,
    /// which are stored `[out_features, in_features]`.
    pub fn with_layout(p: &mut Params<B>, layout: WeightLayout) -> Result<Self> {
        let weight_packed = p.scoped("weight_packed", |packed| {
            packed.optional_param_as(DType::Ui4)
        })?;
        let weight = match weight_packed {
            Some(_) => None,
            None => Some(p.scoped("weight", Params::param)?),
        };
        let path = p.path();
        let zero_point = p.scoped("weight_zero_point", Params::optional_param)?;
        ensure!(
            zero_point.is_none(),
            "{path}: 4-bit codes with zero points (`weight_zero_point`) are not supported"
        );
        let linear = Self {
            weight,
            weight_scale_inv: p.optional_param()?,
            weight_packed,
            weight_scale: p.optional_param()?,
            bias: p.optional_param()?,
            layout,
            input_dtype: p.linear_input_dtype(),
        };
        let packed = linear.weight_packed.is_some();
        ensure!(
            packed == linear.weight_scale.is_some(),
            "{path}: a `weight_scale` comes with packed 4-bit codes (`weight_packed`) and only \
             with them"
        );
        ensure!(
            !packed || layout == WeightLayout::OutIn,
            "{path}: packed 4-bit codes are stored [out_features, in_features], not {layout:?}"
        );
        Ok(linear)
    }

    /// The projection of a 1x1 convolution of an image of one pixel, whose weight the checkpoint
    /// stores as `[out_features, in_features, 1, 1]`. Where neither extent is 1, the weight reads
    /// as `[out_features, in_features]` ([`Params::param_without_unit_axes`]), a view of the same
    /// elements.
    pub fn pointwise(p: &mut Params<B>) -> Result<Self> {
        Ok(Self {
            weight: Some(p.param_without_unit_axes()?),
            weight_scale_inv: p.optional_param()?,
            weight_packed: None,
            weight_scale: None,
            bias: p.optional_param()?,
            layout: WeightLayout::OutIn,
            input_dtype: p.linear_input_dtype(),
        })
    }

    /// A bias-free projection that shares `weight` with another layer, like a language-model head
    /// tied to the token embedding.
    pub fn tied(weight: &Tensor, input_dtype: Option<DType>) -> Self {
        Self {
            weight: Some(weight.clone()),
            weight_scale_inv: None,
            weight_packed: None,
            weight_scale: None,
            bias: None,
            layout: WeightLayout::OutIn,
            input_dtype,
        }
    }

    /// Projects `input` (`[N, in_features]`) to `[N, out_features]` in f32.
    fn forward(&self, input: &Tensor) -> Tensor {
        let cast_input = match self.input_dtype {
            Some(dtype) => input.cast(dtype),
            None => input.clone(),
        };
        let output = functional::linear(&cast_input, &self.dequantized_weight(), self.layout);
        match &self.bias {
            Some(bias) => output + bias,
            None => output,
        }
    }

    /// `W`, in the dtype of its scale when the checkpoint stores it quantized. The dequantization
    /// stays in the graph, so a backend can read the stored weight in place.
    fn dequantized_weight(&self) -> Tensor {
        let stored = (&self.weight, &self.weight_scale_inv);
        let packed = (&self.weight_packed, &self.weight_scale);
        match (stored, packed) {
            ((Some(weight), Some(scale)), _) => weight.cast(scale.dtype()) * scale,
            ((Some(weight), None), _) => weight.clone(),
            ((None, _), (Some(codes), Some(scale))) => unpacked_weight(codes, scale),
            _ => panic!("a linear layer's packed codes come with their scales"),
        }
    }
}

/// The weight of the 4-bit `codes`, in the dtype of `scale`, which holds one scale for each group
/// of codes along the last axis. Each element is `(code - 8) * scale`, because the codes store
/// signed values plus 8.
fn unpacked_weight<B: PortableBackend + 'static>(
    codes: &DeviceTensor<B>,
    scale: &DeviceTensor<B>,
) -> DeviceTensor<B> {
    let (&[_, in_features], &[_, groups]) = (codes.shape().dims(), scale.shape().dims()) else {
        panic!(
            "packed codes {:?} and scales {:?} are matrices",
            codes.shape().dims(),
            scale.shape().dims()
        );
    };
    let offset = scale * -8.0;
    codes.dequantize(
        scale,
        Some(&offset),
        [1, in_features / groups],
        scale.dtype(),
    )
}
