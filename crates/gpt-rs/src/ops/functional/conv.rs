//! Convolution kernels implemented via portable graph capture.
//!
//! The implementations here intentionally stay within the PTIR operator set so that backends can
//! override or fuse these kernels (e.g., `extract_patches + dot_general` -> conv).

use anyhow::{bail, ensure, Result};
use gpt_rs_macros::{capture_ptir, ptir_pattern, support_runtime_overload};

use crate::backend::spec::{ExtractPatchesSpec, Literal, PortableBackend};
use crate::ops::functional::common::CaptureIntoDeviceTensor;
use crate::tensor::DeviceTensor;

/// Semantic 2D convolution.
///
/// - Activations are expected in NHWC ([N, H, W, C]) layout (the layout supported by
///   `extract_patches`).
/// - Weights are expected in canonical OIHW ([C_out, C_in/groups, KH, KW]) layout.
#[support_runtime_overload]
#[ptir_pattern(target = "gpt_rs.conv2d_nhwc_f32")]
pub fn conv2d<B: PortableBackend + 'static>(
    _backend: &B,
    x: &DeviceTensor<B>,
    weight: &DeviceTensor<B>,
    bias: Option<&DeviceTensor<B>>,
    params: Conv2dParams2d,
) -> Result<DeviceTensor<B>> {
    let _scope = crate::profiling::functional_scope(
        "gpt_rs::ops::functional::conv::conv2d",
        "extract_patches_dot_general",
    );

    let validated = validate_conv2d_inputs(x, weight, bias, params)?;
    let n = validated.n;
    let out_h = validated.out_h;
    let out_w = validated.out_w;
    let c_in = validated.c_in;
    let c_out = validated.c_out;
    let kernel_h = validated.kernel_h;
    let kernel_w = validated.kernel_w;
    let groups = validated.groups;
    let c_in_per_group = validated.c_in_per_group;
    let c_out_per_group = validated.c_out_per_group;

    let requires_grad = x.requires_grad_flag()
        || weight.requires_grad_flag()
        || bias.map(|b| b.requires_grad_flag()).unwrap_or(false);

    let spec = ExtractPatchesSpec {
        window: vec![kernel_h, kernel_w],
        strides: vec![params.stride[0], params.stride[1]],
        dilation: vec![params.dilation[0], params.dilation[1]],
        padding: vec![
            (params.padding.top, params.padding.bottom),
            (params.padding.left, params.padding.right),
        ],
        pad_value: Literal::Float(0.0),
    };

    if groups == 1 {
        let result = match bias {
            Some(bias) => capture_ptir!({ x, weight, bias }, |_session| {
                let patches = x.extract_patches(
                    spec.window.clone(),
                    spec.strides.clone(),
                    spec.dilation.clone(),
                    spec.padding.clone(),
                    spec.pad_value.clone(),
                );
                let patches =
                    patches.reshape(vec![n, out_h, out_w, kernel_h, kernel_w, c_in]);
                let out = patches.dot_general(
                    &weight,
                    &crate::ops::ptir::DotDims::new(
                        crate::ops::ptir::axes_iter([]),
                        crate::axes!(3, 4, 5),
                        crate::axes!(2, 3, 1),
                    ),
                    &crate::ops::ptir::DotAttrs::default(),
                );
                let out = out + bias.broadcast_to(vec![n, out_h, out_w, c_out]);
                Ok(out.id())
            })?,
            None => capture_ptir!({ x, weight }, |_session| {
                let patches = x.extract_patches(
                    spec.window.clone(),
                    spec.strides.clone(),
                    spec.dilation.clone(),
                    spec.padding.clone(),
                    spec.pad_value.clone(),
                );
                let patches =
                    patches.reshape(vec![n, out_h, out_w, kernel_h, kernel_w, c_in]);
                let out = patches.dot_general(
                    &weight,
                    &crate::ops::ptir::DotDims::new(
                        crate::ops::ptir::axes_iter([]),
                        crate::axes!(3, 4, 5),
                        crate::axes!(2, 3, 1),
                    ),
                    &crate::ops::ptir::DotAttrs::default(),
                );
                Ok(out.id())
            })?,
        }
        .into_device_tensor(requires_grad)?;

        return Ok(result);
    }

    let result = match bias {
        Some(bias) => capture_ptir!({ x, weight, bias }, |_session| {
            let patches = x.extract_patches(
                spec.window.clone(),
                spec.strides.clone(),
                spec.dilation.clone(),
                spec.padding.clone(),
                spec.pad_value.clone(),
            );
            let patches = patches.reshape(vec![
                n,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                groups,
                c_in_per_group,
            ]);
            let weight = weight.reshape(vec![
                groups,
                c_out_per_group,
                c_in_per_group,
                kernel_h,
                kernel_w,
            ]);
            let out = patches.dot_general(
                &weight,
                &crate::ops::ptir::DotDims::new(
                    crate::axes!(5),
                    crate::axes!(3, 4, 6),
                    crate::axes!(3, 4, 2),
                )
                .with_rhs_batch(crate::axes!(0)),
                &crate::ops::ptir::DotAttrs::default(),
            );
            // dot_general yields shape [G, N, OH, OW, C_out/G] because batch dims come first.
            let out = out.transpose(vec![1, 2, 3, 0, 4]);
            let out = out.reshape(vec![n, out_h, out_w, c_out]);
            let out = out + bias.broadcast_to(vec![n, out_h, out_w, c_out]);
            Ok(out.id())
        })?,
        None => capture_ptir!({ x, weight }, |_session| {
            let patches = x.extract_patches(
                spec.window.clone(),
                spec.strides.clone(),
                spec.dilation.clone(),
                spec.padding.clone(),
                spec.pad_value.clone(),
            );
            let patches = patches.reshape(vec![
                n,
                out_h,
                out_w,
                kernel_h,
                kernel_w,
                groups,
                c_in_per_group,
            ]);
            let weight = weight.reshape(vec![
                groups,
                c_out_per_group,
                c_in_per_group,
                kernel_h,
                kernel_w,
            ]);
            let out = patches.dot_general(
                &weight,
                &crate::ops::ptir::DotDims::new(
                    crate::axes!(5),
                    crate::axes!(3, 4, 6),
                    crate::axes!(3, 4, 2),
                )
                .with_rhs_batch(crate::axes!(0)),
                &crate::ops::ptir::DotAttrs::default(),
            );
            let out = out.transpose(vec![1, 2, 3, 0, 4]);
            let out = out.reshape(vec![n, out_h, out_w, c_out]);
            Ok(out.id())
        })?,
    }
    .into_device_tensor(requires_grad)?;

    Ok(result)
}

#[derive(Debug, Clone, Copy)]
pub struct Padding2d {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

impl Padding2d {
    pub fn zero() -> Self {
        Self {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        }
    }

    pub fn as_hw_pairs(self) -> [(usize, usize); 2] {
        [(self.top, self.bottom), (self.left, self.right)]
    }
}

impl crate::ops::functional::runtime::CacheKeyArg for Padding2d {
    fn add_to_cache_key(&self, builder: &mut crate::ops::functional::runtime::CacheKeyBuilder) {
        builder.combine_hash(&self.top);
        builder.combine_hash(&self.bottom);
        builder.combine_hash(&self.left);
        builder.combine_hash(&self.right);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dParams2d {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: Padding2d,
    pub groups: usize,
}

impl crate::ops::functional::runtime::CacheKeyArg for Conv2dParams2d {
    fn add_to_cache_key(&self, builder: &mut crate::ops::functional::runtime::CacheKeyBuilder) {
        builder.combine_hash(&self.kernel);
        builder.combine_hash(&self.stride);
        builder.combine_hash(&self.dilation);
        builder.combine_hash(&self.padding.top);
        builder.combine_hash(&self.padding.bottom);
        builder.combine_hash(&self.padding.left);
        builder.combine_hash(&self.padding.right);
        builder.combine_hash(&self.groups);
    }
}

fn conv2d_out_dim(
    input: usize,
    window: usize,
    stride: usize,
    dilation: usize,
    pad_before: usize,
    pad_after: usize,
) -> Result<usize> {
    ensure!(window > 0, "conv2d window must be > 0");
    ensure!(stride > 0, "conv2d stride must be > 0");
    ensure!(dilation > 0, "conv2d dilation must be > 0");
    let effective = (window - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("conv2d effective window overflow"))?;
    let padded = input
        .checked_add(pad_before)
        .and_then(|v| v.checked_add(pad_after))
        .ok_or_else(|| anyhow::anyhow!("conv2d padded dimension overflow"))?;
    ensure!(
        padded >= effective,
        "conv2d window ({}) exceeds padded input ({})",
        effective,
        padded
    );
    Ok((padded - effective) / stride + 1)
}

fn validate_conv2d_inputs<B: PortableBackend + 'static>(
    x: &DeviceTensor<B>,
    weight: &DeviceTensor<B>,
    bias: Option<&DeviceTensor<B>>,
    params: Conv2dParams2d,
) -> Result<ValidatedConv2d> {
    ensure!(
        x.shape().rank() == 4,
        "conv2d expects rank-4 NHWC input, got {:?}",
        x.shape().dims()
    );
    if let Some(bias) = bias {
        ensure!(
            bias.shape().rank() == 1,
            "conv2d expects rank-1 bias [C_out], got {:?}",
            bias.shape().dims()
        );
    }

    let dims = x.shape().dims();
    let (n, h, w, c_in) = (dims[0], dims[1], dims[2], dims[3]);

    let kernel_h = params.kernel[0];
    let kernel_w = params.kernel[1];
    let groups = params.groups;
    ensure!(groups > 0, "conv2d groups must be > 0");
    ensure!(
        c_in.is_multiple_of(groups),
        "conv2d input channels {} must be divisible by groups {}",
        c_in,
        groups
    );
    let c_in_per_group = c_in / groups;

    let weight_dims = weight.shape().dims();
    if weight.shape().rank() != 4 {
        bail!(
            "conv2d weight must be canonical OIHW [C_out, C_in/groups, KH, KW], got rank {} ({:?})",
            weight.shape().rank(),
            weight_dims
        );
    }

    // Canonical (Torch): [C_out, C_in/groups, KH, KW]
    ensure!(
        weight_dims[1] == c_in_per_group,
        "conv2d weight expects C_in/groups={}, got {:?}",
        c_in_per_group,
        weight_dims
    );
    ensure!(
        weight_dims[2] == kernel_h && weight_dims[3] == kernel_w,
        "conv2d weight kernel [{}, {}] must match params [{}, {}]",
        weight_dims[2],
        weight_dims[3],
        kernel_h,
        kernel_w
    );
    let c_out = weight_dims[0];
    ensure!(
        c_out.is_multiple_of(groups),
        "conv2d weight expects C_out divisible by groups={}, got {:?}",
        groups,
        weight_dims
    );
    let c_out_per_group = c_out / groups;

    if let Some(bias) = bias {
        ensure!(
            bias.shape().dims()[0] == c_out,
            "conv2d bias length {} must match weight output channels {}",
            bias.shape().dims()[0],
            c_out
        );
    }

    let out_h = conv2d_out_dim(
        h,
        params.kernel[0],
        params.stride[0],
        params.dilation[0],
        params.padding.top,
        params.padding.bottom,
    )?;
    let out_w = conv2d_out_dim(
        w,
        params.kernel[1],
        params.stride[1],
        params.dilation[1],
        params.padding.left,
        params.padding.right,
    )?;

    Ok(ValidatedConv2d {
        n,
        out_h,
        out_w,
        c_in,
        c_out,
        c_out_per_group,
        kernel_h,
        kernel_w,
        groups,
        c_in_per_group,
    })
}

#[derive(Debug, Clone, Copy)]
struct ValidatedConv2d {
    n: usize,
    out_h: usize,
    out_w: usize,
    c_in: usize,
    c_out: usize,
    c_out_per_group: usize,
    kernel_h: usize,
    kernel_w: usize,
    groups: usize,
    c_in_per_group: usize,
}
