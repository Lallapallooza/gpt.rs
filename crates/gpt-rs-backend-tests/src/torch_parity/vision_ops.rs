use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::ops::functional::{self, Conv2dParams, Padding2d};
use tch::Tensor as TchTensor;

use super::common::*;

fn tch_permute(tensor: &TchTensor, perm: &[i64]) -> TchTensor {
    tensor.permute(perm)
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    height: usize,
    width: usize,
    c_in: usize,
    c_out: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
    dilation: [usize; 2],
    groups: usize,
    bias: bool,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x_len = batch * height * width * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * (c_in / groups) * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in / groups, kernel[0], kernel[1]], &weight_data);
        let bias_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_nhwc = tch_tensor_from_vec(&[batch, height, width, c_in], &x_data);
        let x_nchw = tch_permute(&x_nhwc, &[0, 3, 1, 2]);
        let y_nchw = x_nchw.conv2d(
            &w_t,
            if bias { Some(&bias_t) } else { None },
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [dilation[0] as i64, dilation[1] as i64],
            groups as i64,
        );
        let y_nhwc = tch_permute(&y_nchw, &[0, 2, 3, 1]);
        tensor_to_vec(&y_nhwc)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[batch, height, width, c_in], &x_data);
        let weight = device_tensor_from_data(
            backend,
            &[c_out, c_in / groups, kernel[0], kernel[1]],
            &weight_data,
        );
        let bias_dev = device_tensor_from_data(backend, &[c_out], &bias_data);
        let params = Conv2dParams {
            stride,
            dilation,
            padding,
            groups,
        };
        let output = if bias {
            functional::conv2d(&x, &weight, Some(&bias_dev), params)
        } else {
            functional::conv2d(&x, &weight, None, params)
        };
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

#[allow(clippy::too_many_arguments)]
fn run_max_pool_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    batch: usize,
    height: usize,
    width: usize,
    channels: usize,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x_data = random_vec(&mut rng, batch * height * width * channels);
    check_max_pool(
        backend,
        [batch, height, width, channels],
        &x_data,
        window,
        stride,
        padding,
    );
}

fn check_max_pool<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: [usize; 4],
    x_data: &[f32],
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
) {
    let expected = timed_torch(|| {
        let x_nhwc = tch_tensor_from_vec(&shape, x_data);
        let x_nchw = tch_permute(&x_nhwc, &[0, 3, 1, 2]);
        let y_nchw = x_nchw.max_pool2d(
            [window[0] as i64, window[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            false,
        );
        let y_nhwc = tch_permute(&y_nchw, &[0, 2, 3, 1]);
        tensor_to_vec(&y_nhwc)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, x_data);
        let output = functional::max_pool2d(&x, window, stride, padding);
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        1,
        7,
        8,
        3,
        4,
        [3, 3],
        [2, 2],
        Padding2d::uniform(1),
        [1, 1],
        1,
        true,
        101,
    );
}

pub fn conv2d_nhwc_kernel3_stride1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    // Covers the most common ResNet block conv: 3x3, stride 1, padding 1.
    run_conv2d_case(
        backend,
        2,
        6,
        7,
        5,
        4,
        [3, 3],
        [1, 1],
        Padding2d::uniform(1),
        [1, 1],
        1,
        true,
        106,
    );
}

pub fn conv2d_nhwc_kernel7_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        1,
        9,
        10,
        3,
        5,
        [7, 7],
        [2, 2],
        Padding2d::uniform(3),
        [1, 1],
        1,
        true,
        102,
    );
}

pub fn conv2d_nhwc_kernel1_stride2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        1,
        5,
        6,
        8,
        4,
        [1, 1],
        [2, 2],
        Padding2d::default(),
        [1, 1],
        1,
        true,
        104,
    );
}

pub fn conv2d_nhwc_kernel1_stride1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        2,
        5,
        4,
        6,
        7,
        [1, 1],
        [1, 1],
        Padding2d::default(),
        [1, 1],
        1,
        true,
        105,
    );
}

pub fn conv2d_nhwc_kernel1_stride2_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        28,
        28,
        32,
        64,
        [1, 1],
        [2, 2],
        Padding2d::default(),
        [1, 1],
        1,
        true,
        201,
    );
}

pub fn conv2d_nhwc_kernel3_stride1_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        28,
        28,
        32,
        32,
        [3, 3],
        [1, 1],
        Padding2d::uniform(1),
        [1, 1],
        1,
        true,
        202,
    );
}

pub fn conv2d_nhwc_kernel7_stride2_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        64,
        64,
        3,
        32,
        [7, 7],
        [2, 2],
        Padding2d::uniform(3),
        [1, 1],
        1,
        true,
        203,
    );
}

pub fn depthwise_conv2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        1,
        6,
        5,
        4,
        4,
        [3, 3],
        [1, 1],
        Padding2d::uniform(1),
        [1, 1],
        4,
        true,
        103,
    );
}

pub fn depthwise_conv2d_nhwc_stride2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        1,
        7,
        6,
        4,
        4,
        [3, 3],
        [2, 2],
        Padding2d::uniform(1),
        [1, 1],
        4,
        true,
        111,
    );
}

pub fn max_pool2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_max_pool_case(
        backend,
        1,
        7,
        8,
        3,
        [3, 3],
        [2, 2],
        Padding2d::uniform(1),
        107,
    );
}

pub fn relu6_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(109);
    let shape = [2usize, 3usize, 4usize];
    let len: usize = shape.iter().product();
    let x_data = random_vec(&mut rng, len);
    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&shape, &x_data);
        let y_t = x_t.clamp(0.0, 6.0);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &x_data);
        let output = functional::relu6(&x);
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

/// Inputs across the kinks of the hard sigmoid at -3 and 3: random values in [-5, 5), the kinks
/// themselves, 0, and -6.
fn hard_activation_inputs(seed: u64) -> ([usize; 2], Vec<f32>) {
    let mut rng = seeded_rng(seed);
    let mut data = Vec::new();
    for value in random_vec(&mut rng, 60) {
        data.push(value * 5.0);
    }
    data.extend([-3.0, 3.0, 0.0, -6.0]);
    ([4, 16], data)
}

pub fn hard_sigmoid_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let (shape, x_data) = hard_activation_inputs(111);
    let expected =
        timed_torch(|| tensor_to_vec(&tch_tensor_from_vec(&shape, &x_data).hardsigmoid()));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &x_data);
        to_host_vec(&functional::hard_sigmoid(&x))
    });
    assert_close(&expected, &actual);
}

pub fn hard_swish_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let (shape, x_data) = hard_activation_inputs(112);
    let expected = timed_torch(|| tensor_to_vec(&tch_tensor_from_vec(&shape, &x_data).hardswish()));
    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &x_data);
        to_host_vec(&functional::hard_swish(&x))
    });
    assert_close(&expected, &actual);
}

pub fn global_avg_pool2d_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(110);
    let shape = [2usize, 5, 7, 6];
    let x_data = random_vec(&mut rng, shape.iter().product());
    let expected = timed_torch(|| {
        let x_t = tch_permute(&tch_tensor_from_vec(&shape, &x_data), &[0, 3, 1, 2]);
        tensor_to_vec(&x_t.adaptive_avg_pool2d([1, 1]).flatten(1, -1).contiguous())
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &x_data);
        let output = functional::global_avg_pool2d(&x);
        assert_eq!(output.shape().dims(), [2, 6]);
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_k3_s1_p1_bias_n1_h11_w13_c3_cout8<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        11,
        13,
        3,
        8,
        [3, 3],
        [1, 1],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
        [1, 1],
        1,
        true,
        301,
    );
}

pub fn conv2d_nhwc_k3_s2_p1_nobias_n2_h9_w10_c5_cout7<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        2,
        9,
        10,
        5,
        7,
        [3, 3],
        [2, 2],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
        [1, 1],
        1,
        false,
        302,
    );
}

pub fn conv2d_nhwc_k5_s1_p2_bias_n1_h15_w17_c4_cout6<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        15,
        17,
        4,
        6,
        [5, 5],
        [1, 1],
        Padding2d {
            top: 2,
            bottom: 2,
            left: 2,
            right: 2,
        },
        [1, 1],
        1,
        true,
        303,
    );
}

pub fn conv2d_nhwc_k1_s1_p0_bias_n4_h7_w7_c8_cout8<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_conv2d_case(
        backend,
        4,
        7,
        7,
        8,
        8,
        [1, 1],
        [1, 1],
        Padding2d::default(),
        [1, 1],
        1,
        true,
        304,
    );
}

pub fn conv2d_nhwc_k3x5_s2x1_p1x2_bias_n1_h11_w12_c4_cout6<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        11,
        12,
        4,
        6,
        [3, 5],
        [2, 1],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 2,
            right: 2,
        },
        [1, 1],
        1,
        true,
        305,
    );
}

pub fn conv2d_nhwc_k3_s1_p2_d2_bias_n1_h13_w13_c4_cout8<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        13,
        13,
        4,
        8,
        [3, 3],
        [1, 1],
        Padding2d {
            top: 2,
            bottom: 2,
            left: 2,
            right: 2,
        },
        [2, 2],
        1,
        true,
        306,
    );
}

pub fn group_conv2d_nhwc_g2_k3_s1_p1_bias_n1_h11_w11_c8_cout12<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        11,
        11,
        8,
        12,
        [3, 3],
        [1, 1],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
        [1, 1],
        2,
        true,
        307,
    );
}

pub fn depthwise_conv2d_nhwc_k5_s2_p2_n1_h15_w17_c8<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        15,
        17,
        8,
        8,
        [5, 5],
        [2, 2],
        Padding2d {
            top: 2,
            bottom: 2,
            left: 2,
            right: 2,
        },
        [1, 1],
        8,
        true,
        308,
    );
}

pub fn depthwise_conv2d_nhwc_k3_s1_p1_n2_h28_w28_c32<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        2,
        28,
        28,
        32,
        32,
        [3, 3],
        [1, 1],
        Padding2d::uniform(1),
        [1, 1],
        32,
        true,
        309,
    );
}

pub fn group_conv2d_nhwc_g4_k3_s1_p1_bias_n1_h9_w9_c8_cout4<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    run_conv2d_case(
        backend,
        1,
        9,
        9,
        8,
        4,
        [3, 3],
        [1, 1],
        Padding2d::uniform(1),
        [1, 1],
        4,
        true,
        310,
    );
}

pub fn max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_max_pool_case(
        backend,
        1,
        8,
        8,
        3,
        [2, 2],
        [2, 2],
        Padding2d::default(),
        309,
    );
}

pub fn max_pool2d_nhwc_special_values_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(311);
    let shape = [2, 9, 9, 5];
    let x_data = special_vec(&mut rng, shape.iter().product());
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };
    check_max_pool(backend, shape, &x_data, [3, 3], [2, 2], padding);
    check_max_pool(
        backend,
        shape,
        &x_data,
        [2, 2],
        [2, 2],
        Padding2d::default(),
    );
}

pub fn relu6_edge_values_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let shape = [2usize, 3usize, 4usize];
    let mut values = vec![-10.0, -1.0, 0.0, 1.0, 5.5, 6.0, 7.0, 10.0];
    while values.len() < shape.iter().product() {
        values.extend_from_within(..8);
    }
    values.truncate(shape.iter().product());

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&shape, &values);
        let y_t = x_t.clamp(0.0, 6.0);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &shape, &values);
        let output = functional::relu6(&x);
        to_host_vec(&output)
    });

    assert_close(&expected, &actual);
}
