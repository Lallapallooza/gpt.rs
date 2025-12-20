use std::sync::Arc;

use gpt_rs::backend::registry;
use gpt_rs::backend::spec::PortableBackend;
use gpt_rs::{DType, DeviceTensor, Shape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::tensor::PyTensor;

type CpuBackend =
    gpt_rs_backend_ref_cpu::GenericCpuBackend<gpt_rs_backend_ref_cpu::NoopInterceptor>;

fn device_tensor_from_py_tensor<B: PortableBackend + 'static>(
    backend: Arc<B>,
    tensor: &PyTensor,
) -> PyResult<DeviceTensor<B>> {
    if tensor.dtype_enum() != DType::F32 {
        return Err(PyValueError::new_err(format!(
            "expected f32 tensor, got {:?}",
            tensor.dtype_enum()
        )));
    }

    let handle = tensor
        .handle()
        .downcast_ref::<B::TensorHandle>()
        .ok_or_else(|| PyValueError::new_err("tensor handle type mismatch for current backend"))?
        .clone();

    Ok(DeviceTensor::from_handle(
        backend,
        Shape::new(tensor.shape_vec().to_vec()),
        tensor.dtype_enum(),
        handle,
    ))
}

fn device_tensor_from_dict<B: PortableBackend + 'static>(
    backend: Arc<B>,
    weights: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<DeviceTensor<B>> {
    let value = weights
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing required weight tensor '{key}'")))?;
    let tensor = value.downcast::<PyTensor>()?.borrow();
    device_tensor_from_py_tensor(backend, &tensor)
}

fn build_resnet34<B: PortableBackend + 'static>(
    backend: Arc<B>,
    weights: &Bound<'_, PyDict>,
) -> PyResult<gpt_rs::vision::ResNet34<B>> {
    use gpt_rs::ops::functional::Padding2d;
    use gpt_rs::vision::layers::Conv2d;
    use gpt_rs::vision::resnet::BasicBlock;

    let conv1_weight = device_tensor_from_dict(Arc::clone(&backend), weights, "conv1.weight")?;
    let conv1_bias = device_tensor_from_dict(Arc::clone(&backend), weights, "conv1.bias")?;
    let conv1 = Conv2d::new(
        Arc::clone(&backend),
        conv1_weight,
        Some(conv1_bias),
        [7, 7],
        [2, 2],
        [1, 1],
        Padding2d {
            top: 3,
            bottom: 3,
            left: 3,
            right: 3,
        },
    )
    .map_err(|e| PyValueError::new_err(format!("failed to create ResNet conv1: {e}")))?;

    const STAGES: [(usize, usize); 4] = [(64, 3), (128, 4), (256, 6), (512, 3)];

    let mut in_channels = 64usize;
    let mut layers: [Vec<BasicBlock<B>>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for (stage_idx, (out_channels, blocks)) in STAGES.iter().copied().enumerate() {
        let stage_num = stage_idx + 1;
        for block_idx in 0..blocks {
            let stride = if stage_idx == 0 {
                1usize
            } else if block_idx == 0 {
                2usize
            } else {
                1usize
            };

            let conv1_key = format!("layer{stage_num}.{block_idx}.conv1");
            let conv2_key = format!("layer{stage_num}.{block_idx}.conv2");

            let block_conv1_weight = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("{conv1_key}.weight"),
            )?;
            let block_conv1_bias = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("{conv1_key}.bias"),
            )?;
            let conv1 = Conv2d::new(
                Arc::clone(&backend),
                block_conv1_weight,
                Some(block_conv1_bias),
                [3, 3],
                [stride, stride],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
            )
            .map_err(|e| {
                PyValueError::new_err(format!("failed to create ResNet {conv1_key}: {e}"))
            })?;

            let block_conv2_weight = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("{conv2_key}.weight"),
            )?;
            let block_conv2_bias = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("{conv2_key}.bias"),
            )?;
            let conv2 = Conv2d::new(
                Arc::clone(&backend),
                block_conv2_weight,
                Some(block_conv2_bias),
                [3, 3],
                [1, 1],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
            )
            .map_err(|e| {
                PyValueError::new_err(format!("failed to create ResNet {conv2_key}: {e}"))
            })?;

            let downsample = if block_idx == 0 && (stride != 1 || in_channels != out_channels) {
                let ds_key = format!("layer{stage_num}.{block_idx}.downsample");
                let ds_weight = device_tensor_from_dict(
                    Arc::clone(&backend),
                    weights,
                    &format!("{ds_key}.weight"),
                )?;
                let ds_bias = device_tensor_from_dict(
                    Arc::clone(&backend),
                    weights,
                    &format!("{ds_key}.bias"),
                )?;
                Some(
                    Conv2d::new(
                        Arc::clone(&backend),
                        ds_weight,
                        Some(ds_bias),
                        [1, 1],
                        [stride, stride],
                        [1, 1],
                        Padding2d::zero(),
                    )
                    .map_err(|e| {
                        PyValueError::new_err(format!("failed to create ResNet {ds_key}: {e}"))
                    })?,
                )
            } else {
                None
            };

            layers[stage_idx].push(BasicBlock::new(
                Arc::clone(&backend),
                conv1,
                conv2,
                downsample,
            ));

            in_channels = out_channels;
        }
    }

    let fc_weight = device_tensor_from_dict(Arc::clone(&backend), weights, "fc.weight")?;
    let fc_bias = device_tensor_from_dict(Arc::clone(&backend), weights, "fc.bias")?;
    let fc_bias_len = fc_bias.shape().dims()[0];
    let fc_weight_dims = fc_weight.shape().dims();
    let fc_weight = if fc_weight_dims.len() == 2 && fc_weight_dims[0] == fc_bias_len {
        gpt_rs::ops::functional::transpose(backend.as_ref(), &fc_weight, &[1, 0])
            .map_err(|e| PyValueError::new_err(format!("failed to transpose fc.weight: {e}")))?
            .freeze()
            .map_err(|e| PyValueError::new_err(format!("failed to materialize fc.weight: {e}")))?
    } else {
        fc_weight
    };
    let fc = gpt_rs::nn::layers::Linear::new(Arc::clone(&backend), fc_weight, Some(fc_bias))
        .map_err(|e| PyValueError::new_err(format!("failed to create fc layer: {e}")))?;

    let [layer1, layer2, layer3, layer4] = layers;
    Ok(gpt_rs::vision::ResNet34::new(
        backend, conv1, layer1, layer2, layer3, layer4, fc,
    ))
}

fn build_mobilenet_v2<B: PortableBackend + 'static>(
    backend: Arc<B>,
    weights: &Bound<'_, PyDict>,
) -> PyResult<gpt_rs::vision::MobileNetV2<B>> {
    use gpt_rs::ops::functional::Padding2d;
    use gpt_rs::vision::layers::Conv2d;
    use gpt_rs::vision::mobilenet_v2::InvertedResidual;

    let stem_weight = device_tensor_from_dict(Arc::clone(&backend), weights, "stem.weight")?;
    let stem_bias = device_tensor_from_dict(Arc::clone(&backend), weights, "stem.bias")?;
    let stem = Conv2d::new(
        Arc::clone(&backend),
        stem_weight,
        Some(stem_bias),
        [3, 3],
        [2, 2],
        [1, 1],
        Padding2d {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
    )
    .map_err(|e| PyValueError::new_err(format!("failed to create MobileNetV2 stem: {e}")))?;

    let mut blocks = Vec::new();
    let mut input_channels = 32usize;
    let mut block_idx = 0usize;

    const SETTINGS: [(usize, usize, usize, usize); 7] = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ];

    for (expand_ratio, out_channels, repeats, stage_stride) in SETTINGS {
        for repeat_idx in 0..repeats {
            let stride = if repeat_idx == 0 { stage_stride } else { 1 };
            let _expanded = input_channels
                .checked_mul(expand_ratio)
                .ok_or_else(|| PyValueError::new_err("mobilenet expanded channel overflow"))?;

            let expand = if expand_ratio != 1 {
                let w = device_tensor_from_dict(
                    Arc::clone(&backend),
                    weights,
                    &format!("blocks.{block_idx}.expand.weight"),
                )?;
                let b = device_tensor_from_dict(
                    Arc::clone(&backend),
                    weights,
                    &format!("blocks.{block_idx}.expand.bias"),
                )?;
                Some(
                    Conv2d::new(
                        Arc::clone(&backend),
                        w,
                        Some(b),
                        [1, 1],
                        [1, 1],
                        [1, 1],
                        Padding2d::zero(),
                    )
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "failed to create MobileNetV2 expand conv: {e}"
                        ))
                    })?,
                )
            } else {
                None
            };

            let dw_weight = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("blocks.{block_idx}.depthwise.weight"),
            )?;
            let dw_bias = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("blocks.{block_idx}.depthwise.bias"),
            )?;
            let groups = dw_weight.shape().dims()[0];
            let depthwise = Conv2d::new_grouped(
                Arc::clone(&backend),
                dw_weight,
                Some(dw_bias),
                [3, 3],
                [stride, stride],
                [1, 1],
                Padding2d {
                    top: 1,
                    bottom: 1,
                    left: 1,
                    right: 1,
                },
                groups,
            )
            .map_err(|e| {
                PyValueError::new_err(format!("failed to create MobileNetV2 depthwise conv: {e}"))
            })?;

            let proj_weight = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("blocks.{block_idx}.project.weight"),
            )?;
            let proj_bias = device_tensor_from_dict(
                Arc::clone(&backend),
                weights,
                &format!("blocks.{block_idx}.project.bias"),
            )?;
            let project = Conv2d::new(
                Arc::clone(&backend),
                proj_weight,
                Some(proj_bias),
                [1, 1],
                [1, 1],
                [1, 1],
                Padding2d::zero(),
            )
            .map_err(|e| {
                PyValueError::new_err(format!("failed to create MobileNetV2 project conv: {e}"))
            })?;

            let use_res_connect = stride == 1 && input_channels == out_channels;
            blocks.push(InvertedResidual::new(
                Arc::clone(&backend),
                expand,
                depthwise,
                project,
                use_res_connect,
            ));

            input_channels = out_channels;
            block_idx += 1;
        }
    }

    let head_weight = device_tensor_from_dict(Arc::clone(&backend), weights, "head.weight")?;
    let head_bias = device_tensor_from_dict(Arc::clone(&backend), weights, "head.bias")?;
    let head = Conv2d::new(
        Arc::clone(&backend),
        head_weight,
        Some(head_bias),
        [1, 1],
        [1, 1],
        [1, 1],
        Padding2d::zero(),
    )
    .map_err(|e| PyValueError::new_err(format!("failed to create MobileNetV2 head: {e}")))?;

    let classifier_weight =
        device_tensor_from_dict(Arc::clone(&backend), weights, "classifier.weight")?;
    let classifier_bias =
        device_tensor_from_dict(Arc::clone(&backend), weights, "classifier.bias")?;
    let bias_len = classifier_bias.shape().dims()[0];
    let weight_dims = classifier_weight.shape().dims();
    let classifier_weight = if weight_dims.len() == 2 && weight_dims[0] == bias_len {
        gpt_rs::ops::functional::transpose(backend.as_ref(), &classifier_weight, &[1, 0])
            .map_err(|e| {
                PyValueError::new_err(format!("failed to transpose classifier.weight: {e}"))
            })?
            .freeze()
            .map_err(|e| {
                PyValueError::new_err(format!("failed to materialize classifier.weight: {e}"))
            })?
    } else {
        classifier_weight
    };
    let classifier = gpt_rs::nn::layers::Linear::new(
        Arc::clone(&backend),
        classifier_weight,
        Some(classifier_bias),
    )
    .map_err(|e| PyValueError::new_err(format!("failed to create classifier: {e}")))?;

    Ok(gpt_rs::vision::MobileNetV2::new(
        backend, stem, blocks, head, classifier,
    ))
}

enum ResNet34Impl {
    Cpu(gpt_rs::vision::ResNet34<CpuBackend>),
    #[cfg(feature = "conversion-c")]
    C(gpt_rs::vision::ResNet34<gpt_rs_backend_c::CBackend>),
    #[cfg(feature = "faer")]
    Faer(gpt_rs::vision::ResNet34<gpt_rs_backend_faer::FaerPortableBackend>),
}

#[pyclass(name = "ResNet34")]
pub struct PyResNet34 {
    inner: ResNet34Impl,
}

#[pymethods]
impl PyResNet34 {
    #[new]
    fn new(weights: &Bound<'_, PyDict>) -> PyResult<Self> {
        let backend = crate::backend::create_current_backend()?;
        let backend_name = crate::backend::get_current_backend_name();

        match backend_name.as_str() {
            "cpu" | "cpu-portable" => {
                let typed = registry::get_typed_backend::<CpuBackend>(backend.as_ref())
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected cpu"))?;
                let model = build_resnet34(typed, weights)?;
                Ok(PyResNet34 {
                    inner: ResNet34Impl::Cpu(model),
                })
            }
            #[cfg(feature = "conversion-c")]
            "c" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_c::CBackend>(backend.as_ref())
                        .ok_or_else(|| {
                            PyValueError::new_err("backend type mismatch: expected c")
                        })?;
                let model = build_resnet34(typed, weights)?;
                Ok(PyResNet34 {
                    inner: ResNet34Impl::C(model),
                })
            }
            #[cfg(feature = "faer")]
            "faer" | "faer-portable" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                        backend.as_ref(),
                    )
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected faer"))?;
                let model = build_resnet34(typed, weights)?;
                Ok(PyResNet34 {
                    inner: ResNet34Impl::Faer(model),
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "backend '{}' is not supported for vision models yet",
                backend_name
            ))),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &self.inner {
            ResNet34Impl::Cpu(model) => forward_logits(model, input),
            #[cfg(feature = "conversion-c")]
            ResNet34Impl::C(model) => forward_logits(model, input),
            #[cfg(feature = "faer")]
            ResNet34Impl::Faer(model) => forward_logits(model, input),
        }
    }

    fn forward_trace<'py>(
        &self,
        py: Python<'py>,
        input: &PyTensor,
    ) -> PyResult<Bound<'py, PyDict>> {
        match &self.inner {
            ResNet34Impl::Cpu(model) => forward_trace(model, py, input),
            #[cfg(feature = "conversion-c")]
            ResNet34Impl::C(model) => forward_trace(model, py, input),
            #[cfg(feature = "faer")]
            ResNet34Impl::Faer(model) => forward_trace(model, py, input),
        }
    }
}

enum MobileNetV2Impl {
    Cpu(gpt_rs::vision::MobileNetV2<CpuBackend>),
    #[cfg(feature = "conversion-c")]
    C(gpt_rs::vision::MobileNetV2<gpt_rs_backend_c::CBackend>),
    #[cfg(feature = "faer")]
    Faer(gpt_rs::vision::MobileNetV2<gpt_rs_backend_faer::FaerPortableBackend>),
}

enum Conv2dImpl {
    Cpu(gpt_rs::vision::layers::Conv2d<CpuBackend>),
    #[cfg(feature = "conversion-c")]
    C(gpt_rs::vision::layers::Conv2d<gpt_rs_backend_c::CBackend>),
    #[cfg(feature = "faer")]
    Faer(gpt_rs::vision::layers::Conv2d<gpt_rs_backend_faer::FaerPortableBackend>),
}

#[pyclass(name = "Conv2d")]
pub struct PyConv2d {
    inner: Conv2dImpl,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (weight, bias=None, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0), groups=1))]
    fn new(
        weight: &PyTensor,
        bias: Option<&PyTensor>,
        kernel: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        padding: (usize, usize, usize, usize),
        groups: usize,
    ) -> PyResult<Self> {
        let backend = crate::backend::create_current_backend()?;
        let backend_name = crate::backend::get_current_backend_name();

        let kernel = [kernel.0, kernel.1];
        let stride = [stride.0, stride.1];
        let dilation = [dilation.0, dilation.1];
        let padding = gpt_rs::ops::functional::Padding2d {
            top: padding.0,
            bottom: padding.1,
            left: padding.2,
            right: padding.3,
        };

        match backend_name.as_str() {
            "cpu" | "cpu-portable" => {
                let typed = registry::get_typed_backend::<CpuBackend>(backend.as_ref())
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected cpu"))?;
                let weight = device_tensor_from_py_tensor(Arc::clone(&typed), weight)?;
                let bias = match bias {
                    Some(bias) => Some(device_tensor_from_py_tensor(Arc::clone(&typed), bias)?),
                    None => None,
                };

                let conv = gpt_rs::vision::layers::Conv2d::new_grouped(
                    Arc::clone(&typed),
                    weight,
                    bias,
                    kernel,
                    stride,
                    dilation,
                    padding,
                    groups,
                )
                .map_err(|e| PyValueError::new_err(format!("failed to create Conv2d: {e}")))?;

                Ok(PyConv2d {
                    inner: Conv2dImpl::Cpu(conv),
                })
            }
            #[cfg(feature = "conversion-c")]
            "c" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_c::CBackend>(backend.as_ref())
                        .ok_or_else(|| {
                            PyValueError::new_err("backend type mismatch: expected c")
                        })?;
                let weight = device_tensor_from_py_tensor(Arc::clone(&typed), weight)?;
                let bias = match bias {
                    Some(bias) => Some(device_tensor_from_py_tensor(Arc::clone(&typed), bias)?),
                    None => None,
                };

                let conv = gpt_rs::vision::layers::Conv2d::new_grouped(
                    Arc::clone(&typed),
                    weight,
                    bias,
                    kernel,
                    stride,
                    dilation,
                    padding,
                    groups,
                )
                .map_err(|e| PyValueError::new_err(format!("failed to create Conv2d: {e}")))?;

                Ok(PyConv2d {
                    inner: Conv2dImpl::C(conv),
                })
            }
            #[cfg(feature = "faer")]
            "faer" | "faer-portable" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                        backend.as_ref(),
                    )
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected faer"))?;
                let weight = device_tensor_from_py_tensor(Arc::clone(&typed), weight)?;
                let bias = match bias {
                    Some(bias) => Some(device_tensor_from_py_tensor(Arc::clone(&typed), bias)?),
                    None => None,
                };

                let conv = gpt_rs::vision::layers::Conv2d::new_grouped(
                    Arc::clone(&typed),
                    weight,
                    bias,
                    kernel,
                    stride,
                    dilation,
                    padding,
                    groups,
                )
                .map_err(|e| PyValueError::new_err(format!("failed to create Conv2d: {e}")))?;

                Ok(PyConv2d {
                    inner: Conv2dImpl::Faer(conv),
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "backend '{}' is not supported for vision layers yet",
                backend_name
            ))),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        if input.dtype_enum() != DType::F32 {
            return Err(PyValueError::new_err("input tensor must be f32"));
        }
        if input.shape_vec().len() != 4 {
            return Err(PyValueError::new_err(format!(
                "input must have shape [N, H, W, C], got {:?}",
                input.shape_vec()
            )));
        }

        match &self.inner {
            Conv2dImpl::Cpu(conv) => forward_conv(conv, input),
            #[cfg(feature = "conversion-c")]
            Conv2dImpl::C(conv) => forward_conv(conv, input),
            #[cfg(feature = "faer")]
            Conv2dImpl::Faer(conv) => forward_conv(conv, input),
        }
    }
}

fn forward_conv<B: PortableBackend + 'static>(
    conv: &gpt_rs::vision::layers::Conv2d<B>,
    input: &PyTensor,
) -> PyResult<PyTensor> {
    let backend = conv.backend();
    let input_device = device_tensor_from_py_tensor(Arc::clone(&backend), input)?;
    let out = conv
        .forward(&input_device)
        .map_err(|e| PyValueError::new_err(format!("conv forward failed: {e}")))?;
    let handle = out
        .materialize()
        .map_err(|e| PyValueError::new_err(format!("failed to materialize output: {e}")))?;
    Ok(PyTensor::from_backend_handle(
        Box::new(handle),
        out.shape().dims().to_vec(),
        out.dtype(),
    ))
}

#[pyclass(name = "MobileNetV2")]
pub struct PyMobileNetV2 {
    inner: MobileNetV2Impl,
}

#[pymethods]
impl PyMobileNetV2 {
    #[new]
    fn new(weights: &Bound<'_, PyDict>) -> PyResult<Self> {
        let backend = crate::backend::create_current_backend()?;
        let backend_name = crate::backend::get_current_backend_name();

        match backend_name.as_str() {
            "cpu" | "cpu-portable" => {
                let typed = registry::get_typed_backend::<CpuBackend>(backend.as_ref())
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected cpu"))?;
                let model = build_mobilenet_v2(typed, weights)?;
                Ok(PyMobileNetV2 {
                    inner: MobileNetV2Impl::Cpu(model),
                })
            }
            #[cfg(feature = "conversion-c")]
            "c" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_c::CBackend>(backend.as_ref())
                        .ok_or_else(|| {
                            PyValueError::new_err("backend type mismatch: expected c")
                        })?;
                let model = build_mobilenet_v2(typed, weights)?;
                Ok(PyMobileNetV2 {
                    inner: MobileNetV2Impl::C(model),
                })
            }
            #[cfg(feature = "faer")]
            "faer" | "faer-portable" => {
                let typed =
                    registry::get_typed_backend::<gpt_rs_backend_faer::FaerPortableBackend>(
                        backend.as_ref(),
                    )
                    .ok_or_else(|| PyValueError::new_err("backend type mismatch: expected faer"))?;
                let model = build_mobilenet_v2(typed, weights)?;
                Ok(PyMobileNetV2 {
                    inner: MobileNetV2Impl::Faer(model),
                })
            }
            _ => Err(PyValueError::new_err(format!(
                "backend '{}' is not supported for vision models yet",
                backend_name
            ))),
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match &self.inner {
            MobileNetV2Impl::Cpu(model) => forward_logits(model, input),
            #[cfg(feature = "conversion-c")]
            MobileNetV2Impl::C(model) => forward_logits(model, input),
            #[cfg(feature = "faer")]
            MobileNetV2Impl::Faer(model) => forward_logits(model, input),
        }
    }

    fn forward_trace<'py>(
        &self,
        py: Python<'py>,
        input: &PyTensor,
    ) -> PyResult<Bound<'py, PyDict>> {
        match &self.inner {
            MobileNetV2Impl::Cpu(model) => forward_trace(model, py, input),
            #[cfg(feature = "conversion-c")]
            MobileNetV2Impl::C(model) => forward_trace(model, py, input),
            #[cfg(feature = "faer")]
            MobileNetV2Impl::Faer(model) => forward_trace(model, py, input),
        }
    }
}

fn forward_logits<B: PortableBackend + 'static, M>(
    model: &M,
    input: &PyTensor,
) -> PyResult<PyTensor>
where
    M: VisionForward<B>,
{
    if input.dtype_enum() != DType::F32 {
        return Err(PyValueError::new_err("input tensor must be f32"));
    }
    if input.shape_vec().len() != 4 {
        return Err(PyValueError::new_err(format!(
            "input must have shape [N, C, H, W], got {:?}",
            input.shape_vec()
        )));
    }

    let backend = model.backend();
    let input_device = device_tensor_from_py_tensor(Arc::clone(&backend), input)?;
    let logits = model
        .forward(&input_device)
        .map_err(|e| PyValueError::new_err(format!("forward failed: {e}")))?;
    let handle = logits
        .materialize()
        .map_err(|e| PyValueError::new_err(format!("failed to materialize output: {e}")))?;
    Ok(PyTensor::from_backend_handle(
        Box::new(handle),
        logits.shape().dims().to_vec(),
        logits.dtype(),
    ))
}

fn forward_trace<'py, B: PortableBackend + 'static, M>(
    model: &M,
    py: Python<'py>,
    input: &PyTensor,
) -> PyResult<Bound<'py, PyDict>>
where
    M: VisionForward<B>,
{
    if input.dtype_enum() != DType::F32 {
        return Err(PyValueError::new_err("input tensor must be f32"));
    }
    if input.shape_vec().len() != 4 {
        return Err(PyValueError::new_err(format!(
            "input must have shape [N, C, H, W], got {:?}",
            input.shape_vec()
        )));
    }

    let backend = model.backend();
    let input_device = device_tensor_from_py_tensor(Arc::clone(&backend), input)?;
    let traced = model
        .forward_trace(&input_device)
        .map_err(|e| PyValueError::new_err(format!("forward_trace failed: {e}")))?;

    let mut names = Vec::with_capacity(traced.len());
    let mut tensors = Vec::with_capacity(traced.len());
    for (name, tensor) in traced {
        names.push(name);
        tensors.push(tensor);
    }

    let refs = tensors.iter().collect::<Vec<_>>();
    let handles = DeviceTensor::materialize_many(&refs)
        .map_err(|e| PyValueError::new_err(format!("failed to materialize trace outputs: {e}")))?;

    let dict = PyDict::new_bound(py);
    for ((name, tensor), handle) in names
        .into_iter()
        .zip(tensors.iter())
        .zip(handles.into_iter())
    {
        let py_tensor = PyTensor::from_backend_handle(
            Box::new(handle),
            tensor.shape().dims().to_vec(),
            tensor.dtype(),
        );
        dict.set_item(name, Py::new(py, py_tensor)?)?;
    }

    Ok(dict)
}

trait VisionForward<B: PortableBackend + 'static> {
    fn backend(&self) -> Arc<B>;
    fn forward(&self, input: &DeviceTensor<B>) -> anyhow::Result<DeviceTensor<B>>;
    fn forward_trace(
        &self,
        input: &DeviceTensor<B>,
    ) -> anyhow::Result<Vec<(String, DeviceTensor<B>)>>;
}

impl<B: PortableBackend + 'static> VisionForward<B> for gpt_rs::vision::ResNet34<B> {
    fn backend(&self) -> Arc<B> {
        self.backend()
    }

    fn forward(&self, input: &DeviceTensor<B>) -> anyhow::Result<DeviceTensor<B>> {
        self.forward(input)
    }

    fn forward_trace(
        &self,
        input: &DeviceTensor<B>,
    ) -> anyhow::Result<Vec<(String, DeviceTensor<B>)>> {
        self.forward_trace(input)
    }
}

impl<B: PortableBackend + 'static> VisionForward<B> for gpt_rs::vision::MobileNetV2<B> {
    fn backend(&self) -> Arc<B> {
        self.backend()
    }

    fn forward(&self, input: &DeviceTensor<B>) -> anyhow::Result<DeviceTensor<B>> {
        self.forward(input)
    }

    fn forward_trace(
        &self,
        input: &DeviceTensor<B>,
    ) -> anyhow::Result<Vec<(String, DeviceTensor<B>)>> {
        self.forward_trace(input)
    }
}
