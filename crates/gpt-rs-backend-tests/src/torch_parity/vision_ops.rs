use std::sync::Arc;

use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::ops::functional::{self, Conv2dParams2d, Padding2d};
use tch::Tensor as TchTensor;

use super::common::*;

fn conv_params(kernel: [usize; 2], stride: [usize; 2], padding: Padding2d) -> Conv2dParams2d {
    Conv2dParams2d {
        kernel,
        stride,
        dilation: [1, 1],
        padding,
        groups: 1,
    }
}

fn tch_permute(tensor: &TchTensor, perm: &[i64]) -> TchTensor {
    tensor.permute(perm)
}

#[allow(clippy::too_many_arguments)]
fn run_conv2d_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    n: usize,
    h: usize,
    w: usize,
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
    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * (c_in / groups) * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in / groups, kernel[0], kernel[1]], &weight_data);
        let bias_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            if bias { Some(&bias_t) } else { None },
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [dilation[0] as i64, dilation[1] as i64],
            groups as i64,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w = device_tensor_from_data(
            backend,
            &[c_out, c_in / groups, kernel[0], kernel[1]],
            &weight_data,
        );
        let bias_dev = device_tensor_from_data(backend, &[c_out], &bias_data);
        let params = Conv2dParams2d {
            kernel,
            stride,
            dilation,
            padding,
            groups,
        };
        let y = if bias {
            functional::conv2d(backend.as_ref(), &x, &w, Some(&bias_dev), params).unwrap()
        } else {
            functional::conv2d(backend.as_ref(), &x, &w, None, params).unwrap()
        };
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

#[allow(clippy::too_many_arguments)]
fn run_max_pool_case<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    n: usize,
    h: usize,
    w: usize,
    c: usize,
    window: [usize; 2],
    stride: [usize; 2],
    padding: Padding2d,
    seed: u64,
) {
    let mut rng = seeded_rng(seed);
    let x_len = n * h * w * c;
    let x_data = random_vec(&mut rng, x_len);

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&[n, h, w, c], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.max_pool2d(
            [window[0] as i64, window[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            false,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c], &x_data);
        let y = functional::max_pool2d_nhwc(backend.as_ref(), &x, window, stride, padding).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(101);
    let (n, h, w, c_in, c_out) = (1usize, 7usize, 8usize, 3usize, 4usize);
    let kernel = [3usize, 3usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        // Torch expects NCHW.
        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel3_stride1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    // Covers the most common ResNet block conv: 3x3, stride 1, padding 1.
    let mut rng = seeded_rng(106);
    let (n, h, w, c_in, c_out) = (2usize, 6usize, 7usize, 5usize, 4usize);
    let kernel = [3usize, 3usize];
    let stride = [1usize, 1usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel7_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(102);
    let (n, h, w, c_in, c_out) = (1usize, 9usize, 10usize, 3usize, 5usize);
    let kernel = [7usize, 7usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d {
        top: 3,
        bottom: 3,
        left: 3,
        right: 3,
    };

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel1_stride2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(104);
    let (n, h, w, c_in, c_out) = (1usize, 5usize, 6usize, 8usize, 4usize);
    let kernel = [1usize, 1usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d::zero();

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [0, 0],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel1_stride1_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(105);
    let (n, h, w, c_in, c_out) = (2usize, 5usize, 4usize, 6usize, 7usize);
    let kernel = [1usize, 1usize];
    let stride = [1usize, 1usize];
    let padding = Padding2d::zero();

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(&w_t, Some(&b_t), [1, 1], [0, 0], [1, 1], 1);
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel1_stride2_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(201);
    let (n, h, w, c_in, c_out) = (1usize, 28usize, 28usize, 32usize, 64usize);
    let kernel = [1usize, 1usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d::zero();

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [0, 0],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel3_stride1_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(202);
    let (n, h, w, c_in, c_out) = (1usize, 28usize, 28usize, 32usize, 32usize);
    let kernel = [3usize, 3usize];
    let stride = [1usize, 1usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn conv2d_nhwc_kernel7_stride2_resnet_matches_torch<B: PortableBackend + 'static>(
    backend: &Arc<B>,
) {
    let mut rng = seeded_rng(203);
    let (n, h, w, c_in, c_out) = (1usize, 64usize, 64usize, 3usize, 32usize);
    let kernel = [7usize, 7usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d {
        top: 3,
        bottom: 3,
        left: 3,
        right: 3,
    };

    let x_len = n * h * w * c_in;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c_out * c_in * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c_out);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c_out], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c_in], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            1,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c_in], &x_data);
        let w =
            device_tensor_from_data(backend, &[c_out, c_in, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c_out], &bias_data);

        let params = conv_params(kernel, stride, padding);
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn depthwise_conv2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(103);
    let (n, h, w, c) = (1usize, 6usize, 5usize, 4usize);
    let kernel = [3usize, 3usize];
    let stride = [1usize, 1usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let x_len = n * h * w * c;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c, 1, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            c as i64,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c], &x_data);
        let w = device_tensor_from_data(backend, &[c, 1, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c], &bias_data);

        let params = Conv2dParams2d {
            groups: c,
            ..conv_params(kernel, stride, padding)
        };
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn depthwise_conv2d_nhwc_stride2_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(111);
    let (n, h, w, c) = (1usize, 7usize, 6usize, 4usize);
    let kernel = [3usize, 3usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let x_len = n * h * w * c;
    let x_data = random_vec(&mut rng, x_len);
    let weight_len = c * kernel[0] * kernel[1];
    let weight_data = random_vec(&mut rng, weight_len);
    let bias_data = random_vec(&mut rng, c);

    let expected = timed_torch(|| {
        let w_t = tch_tensor_from_vec(&[c, 1, kernel[0], kernel[1]], &weight_data);
        let b_t = tch_tensor_from_vec(&[c], &bias_data);

        let x_t = tch_tensor_from_vec(&[n, h, w, c], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.conv2d(
            &w_t,
            Some(&b_t),
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            c as i64,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c], &x_data);
        let w = device_tensor_from_data(backend, &[c, 1, kernel[0], kernel[1]], &weight_data);
        let bias = device_tensor_from_data(backend, &[c], &bias_data);

        let params = Conv2dParams2d {
            groups: c,
            ..conv_params(kernel, stride, padding)
        };
        let y = functional::conv2d(backend.as_ref(), &x, &w, Some(&bias), params).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}

pub fn max_pool2d_nhwc_matches_torch<B: PortableBackend + 'static>(backend: &Arc<B>) {
    let mut rng = seeded_rng(107);
    let (n, h, w, c) = (1usize, 7usize, 8usize, 3usize);
    let x_len = n * h * w * c;
    let x_data = random_vec(&mut rng, x_len);

    let window = [3usize, 3usize];
    let stride = [2usize, 2usize];
    let padding = Padding2d {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    let expected = timed_torch(|| {
        let x_t = tch_tensor_from_vec(&[n, h, w, c], &x_data);
        let x_t = tch_permute(&x_t, &[0, 3, 1, 2]);
        let y_t = x_t.max_pool2d(
            [window[0] as i64, window[1] as i64],
            [stride[0] as i64, stride[1] as i64],
            [padding.top as i64, padding.left as i64],
            [1, 1],
            false,
        );
        let y_t = tch_permute(&y_t, &[0, 2, 3, 1]);
        tensor_to_vec(&y_t)
    });

    let actual = timed_gpt(|| {
        let x = device_tensor_from_data(backend, &[n, h, w, c], &x_data);
        let y = functional::max_pool2d_nhwc(backend.as_ref(), &x, window, stride, padding).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
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
        let y = functional::relu6(backend.as_ref(), &x).unwrap();
        to_host_vec(&y)
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
        Padding2d::zero(),
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

pub fn max_pool2d_nhwc_w2_s2_p0_n1_h8_w8_c3<B: PortableBackend + 'static>(backend: &Arc<B>) {
    run_max_pool_case(backend, 1, 8, 8, 3, [2, 2], [2, 2], Padding2d::zero(), 309);
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
        let y = functional::relu6(backend.as_ref(), &x).unwrap();
        to_host_vec(&y)
    });

    assert_close(&expected, &actual);
}
