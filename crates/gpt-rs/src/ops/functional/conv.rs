//! 2D convolution as `extract_patches` and `dot_general`, which backends run on their own
//! convolution kernels.

use gpt_rs_ir::dialects::tensor::DotGeneral;

use crate::{ensure_rank, ensure_same_dtype, functional};

/// 2D convolution of NHWC activations `x` (`[N, H, W, C_in]`) with an OIHW `weight`
/// (`[C_out, C_in / groups, KH, KW]`) and an optional `bias` (`[C_out]`). It computes what Torch's
/// `conv2d` computes on NCHW activations, and returns `[N, OH, OW, C_out]`.
#[functional]
pub fn conv2d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, params: Conv2dParams) -> Tensor {
    ensure_rank!(x, 4);
    ensure_rank!(weight, 4);
    ensure_same_dtype!(x, weight);
    let Conv2dParams {
        stride,
        dilation,
        padding,
        groups,
    } = params;
    let [c_out, c_in_per_group, kernel_h, kernel_w] = [0, 1, 2, 3].map(|axis| weight.size(axis));
    let c_in = x.size(-1);
    assert!(
        groups > 0 && c_in == groups * c_in_per_group && c_out.is_multiple_of(groups),
        "{FUNCTIONAL}: {groups} groups do not fit {c_in} input channels and the weight {:?}",
        weight.shape().dims()
    );
    if let Some(bias_tensor) = bias {
        let bias_dims = bias_tensor.shape().dims();
        assert!(
            bias_dims == [c_out],
            "{FUNCTIONAL}: bias must be [{c_out}], got {bias_dims:?}"
        );
    }
    let patches = x.extract_patches(
        [kernel_h, kernel_w],
        stride,
        dilation,
        padding.as_hw_pairs(),
        0.0,
    );
    let [batch, out_h, out_w] = [0, 1, 2].map(|axis| patches.size(axis));
    let out = if groups == 1 {
        // Patches are [N, OH, OW, KH, KW, C_in]. The weight [C_out, C_in, KH, KW] contracts with
        // them.
        patches
            .reshape([batch, out_h, out_w, kernel_h, kernel_w, c_in])
            .dot_general(weight, DotGeneral::contract([3, 4, 5], [2, 3, 1]))
    } else {
        // The patches [N, OH, OW, KH, KW, G, C_in/G] contract with the weight viewed as
        // [G, C_out/G, C_in/G, KH, KW], batched over the groups. The result is
        // [G, N, OH, OW, C_out/G], as batch axes come first.
        let grouped_patches = patches.reshape([
            batch,
            out_h,
            out_w,
            kernel_h,
            kernel_w,
            groups,
            c_in_per_group,
        ]);
        let grouped_weight =
            weight.reshape([groups, c_out / groups, c_in_per_group, kernel_h, kernel_w]);
        grouped_patches
            .dot_general(
                &grouped_weight,
                DotGeneral::contract([3, 4, 6], [3, 4, 2]).batch([5], [0]),
            )
            .transpose([1, 2, 3, 0, 4])
            .reshape([batch, out_h, out_w, c_out])
    };
    match bias {
        Some(bias_tensor) => out + bias_tensor,
        None => out,
    }
}

/// Zero padding before and after the spatial axes of NHWC activations.
#[derive(Debug, Clone, Copy, Default)]
pub struct Padding2d {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

impl Padding2d {
    /// `padding` on every side.
    pub fn uniform(padding: usize) -> Self {
        Self {
            top: padding,
            bottom: padding,
            left: padding,
            right: padding,
        }
    }

    /// `(before, after)` of the height and of the width.
    pub fn as_hw_pairs(self) -> [(usize, usize); 2] {
        [(self.top, self.bottom), (self.left, self.right)]
    }
}

/// How [`conv2d`] slides its kernel, as the arguments of Torch's `conv2d` besides the tensors.
/// The kernel size comes from the weight.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dParams {
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: Padding2d,
    pub groups: usize,
}

impl Default for Conv2dParams {
    /// Stride 1, no dilation, no padding, one group.
    fn default() -> Self {
        Self {
            stride: [1, 1],
            dilation: [1, 1],
            padding: Padding2d::default(),
            groups: 1,
        }
    }
}
