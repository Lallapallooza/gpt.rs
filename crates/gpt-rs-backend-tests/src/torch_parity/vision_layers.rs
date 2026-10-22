use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::module::Layer;
use gpt_rs::nn::layers::{BatchNorm2d, Conv2d, ConvPadding};
use gpt_rs::ops::functional;
use tch::Tensor as TchTensor;

use super::common::*;

impl HostBatchNorm {
    /// Torch eval-mode batch norm of an NCHW tensor.
    fn torch(&self, x: &TchTensor, eps: f64) -> TchTensor {
        let channels = self.weight.len();
        let param = |data: &[f32]| tch_tensor_from_vec(&[channels], data);
        x.batch_norm(
            Some(&param(&self.weight)),
            Some(&param(&self.bias)),
            Some(&param(&self.running_mean)),
            Some(&param(&self.running_var)),
            false,
            0.1,
            eps,
            false,
        )
    }
}

fn nhwc_to_nchw(x: &TchTensor) -> TchTensor {
    x.permute([0, 3, 1, 2])
}

fn nchw_to_nhwc(x: &TchTensor) -> TchTensor {
    x.permute([0, 2, 3, 1])
}

pub fn batch_norm2d_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xB4_7C);
    let shape = [2, 5, 7, 6];
    let eps = 1e-3;
    let x_data = random_vec(&mut rng, shape.iter().product());
    let norm = HostBatchNorm::random(&mut rng, shape[3]);

    let expected = timed_torch(|| {
        let x = nhwc_to_nchw(&tch_tensor_from_vec(&shape, &x_data));
        tensor_to_vec(&nchw_to_nhwc(&norm.torch(&x, eps)))
    });

    let actual = timed_gpt(|| {
        let layer = load_layer(backend, norm.tensors("bn"), |params| {
            params.scoped("bn", |p| BatchNorm2d::new(p, eps as f32))
        })
        .unwrap();
        let x = device_tensor_from_data(backend, &shape, &x_data);
        to_host_vec(&layer.call(&x))
    });

    assert_close(&expected, &actual);
}

/// Bias-free convolution, batch norm, and ReLU, as in ResNet's `ResNetConvLayer`.
pub fn conv2d_batch_norm_relu_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(0xC0_B7);
    let (batch, height, width, c_in, c_out) = (1, 9, 8, 4, 6);
    let eps = 1e-5;
    let x_data = random_vec(&mut rng, batch * height * width * c_in);
    let weight = random_vec(&mut rng, c_out * c_in * 3 * 3);
    let norm = HostBatchNorm::random(&mut rng, c_out);

    let expected = timed_torch(|| {
        let x = nhwc_to_nchw(&tch_tensor_from_vec(&[batch, height, width, c_in], &x_data));
        let weight_tch = tch_tensor_from_vec(&[c_out, c_in, 3, 3], &weight);
        let output = x.conv2d(&weight_tch, None::<&TchTensor>, [2, 2], [1, 1], [1, 1], 1);
        tensor_to_vec(&nchw_to_nhwc(&norm.torch(&output, eps).relu()))
    });

    let actual = timed_gpt(|| {
        let weight_host = tensor_from_vec(&[c_out, c_in, 3, 3], weight.clone());
        let mut tensors = norm.tensors("bn");
        tensors.push(("conv.weight".into(), weight_host));
        let (conv, bn) = load_layer(backend, tensors, |params| {
            let conv = params.scoped("conv", |p| Conv2d::new(p, 2, 1, ConvPadding::Centered))?;
            Ok((
                conv,
                params.scoped("bn", |p| BatchNorm2d::new(p, eps as f32))?,
            ))
        })
        .unwrap();
        let x = device_tensor_from_data(backend, &[batch, height, width, c_in], &x_data);
        let output = bn.call(&conv.call(&x));
        to_host_vec(&functional::relu(&output))
    });

    assert_close(&expected, &actual);
}

/// Compares a bias-free depthwise 3x3 `Conv2d` with TensorFlow `SAME` padding to a Torch
/// convolution of the input padded by `[top, bottom, left, right]`.
fn run_tf_padding_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    (height, width): (usize, usize),
    (stride, dilation): (usize, usize),
    [top, bottom, left, right]: [i64; 4],
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let (batch, channels) = (2, 4);
    let x_data = random_vec(&mut rng, batch * height * width * channels);
    let weight = random_vec(&mut rng, channels * 3 * 3);

    let expected = timed_torch(|| {
        let x = nhwc_to_nchw(&tch_tensor_from_vec(
            &[batch, height, width, channels],
            &x_data,
        ));
        let padded = x.constant_pad_nd([left, right, top, bottom]);
        let weight_tch = tch_tensor_from_vec(&[channels, 1, 3, 3], &weight);
        let (torch_stride, torch_dilation) = (stride as i64, dilation as i64);
        let output = padded.conv2d(
            &weight_tch,
            None::<&TchTensor>,
            [torch_stride, torch_stride],
            [0, 0],
            [torch_dilation, torch_dilation],
            channels as i64,
        );
        tensor_to_vec(&nchw_to_nhwc(&output))
    });

    let actual = timed_gpt(|| {
        let tensors = [(
            "conv.weight".to_string(),
            tensor_from_vec(&[channels, 1, 3, 3], weight.clone()),
        )];
        let conv = load_layer(backend, tensors, |params| {
            params.scoped("conv", |p| {
                Conv2d::new(p, stride, dilation, ConvPadding::TfSame)
            })
        })
        .unwrap();
        let x = device_tensor_from_data(backend, &[batch, height, width, channels], &x_data);
        to_host_vec(&conv.call(&x))
    });

    assert_close(&expected, &actual);
}

/// With stride 2, `SAME` pads an even input size by one after the input, and an odd size by one
/// on each side.
pub fn conv2d_tf_padding_stride2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_tf_padding_case(backend, (8, 9), (2, 1), [0, 1, 1, 1], 0x7F_01);
}

/// Dilation 2 makes the 3x3 kernel span 5, so `SAME` pads 2 on every side.
pub fn conv2d_tf_padding_dilated_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_tf_padding_case(backend, (7, 6), (1, 2), [2, 2, 2, 2], 0x7F_02);
}
