use serde::{Deserialize, Serialize};

pub const EWISE_BINARY_KERNEL_ID: &str = "gpt_rs.triton.kernel.elementwise_binary_f32.v1";
pub const EWISE_BINARY_SYMBOL: &str = "gpt_rs_triton_ewise_binary_f32";
pub const EWISE_UNARY_KERNEL_ID: &str = "gpt_rs.triton.kernel.elementwise_unary_f32.v1";
pub const EWISE_UNARY_SYMBOL: &str = "gpt_rs_triton_ewise_unary_f32";
pub const BROADCAST_KERNEL_ID: &str = "gpt_rs.triton.kernel.broadcast_f32_rank4.v1";
pub const BROADCAST_SYMBOL: &str = "gpt_rs_triton_broadcast_f32_rank4";
pub const BROADCAST_SI32_KERNEL_ID: &str = "gpt_rs.triton.kernel.broadcast_si32_rank4.v1";
pub const BROADCAST_SI32_SYMBOL: &str = "gpt_rs_triton_broadcast_si32_rank4";
pub const SLICE_KERNEL_ID: &str = "gpt_rs.triton.kernel.slice_f32_rank4.v1";
pub const SLICE_SYMBOL: &str = "gpt_rs_triton_slice_f32_rank4";
pub const DYNAMIC_SLICE_F32_KERNEL_ID: &str = "gpt_rs.triton.kernel.dynamic_slice_f32_rank4.v1";
pub const DYNAMIC_SLICE_F32_SYMBOL: &str = "gpt_rs_triton_dynamic_slice_f32_rank4";
pub const DYNAMIC_SLICE_SI32_RANK1_KERNEL_ID: &str =
    "gpt_rs.triton.kernel.dynamic_slice_si32_rank1.v1";
pub const DYNAMIC_SLICE_SI32_RANK1_SYMBOL: &str = "gpt_rs_triton_dynamic_slice_si32_rank1";
pub const TRANSPOSE_KERNEL_ID: &str = "gpt_rs.triton.kernel.transpose_f32_rank5.v1";
pub const TRANSPOSE_SYMBOL: &str = "gpt_rs_triton_transpose_f32_rank5";
pub const CONCAT_KERNEL_ID: &str = "gpt_rs.triton.kernel.concat_f32_rank4.v1";
pub const CONCAT_SYMBOL: &str = "gpt_rs_triton_concat_f32_rank4";
pub const REDUCE_SUM_LAST_AXIS_KERNEL_ID: &str = "gpt_rs.triton.kernel.reduce_sum_last_axis_f32.v1";
pub const REDUCE_SUM_LAST_AXIS_SYMBOL: &str = "gpt_rs_triton_reduce_sum_last_axis_f32";
pub const REDUCE_MAX_LAST_AXIS_KERNEL_ID: &str = "gpt_rs.triton.kernel.reduce_max_last_axis_f32.v1";
pub const REDUCE_MAX_LAST_AXIS_SYMBOL: &str = "gpt_rs_triton_reduce_max_last_axis_f32";
pub const SOFTMAX_LAST_AXIS_KERNEL_ID: &str = "gpt_rs.triton.kernel.softmax_last_axis_f32.v1";
pub const SOFTMAX_LAST_AXIS_SYMBOL: &str = "gpt_rs_triton_softmax_last_axis_f32";
pub const IOTA_SI32_KERNEL_ID: &str = "gpt_rs.triton.kernel.iota_si32_rank4.v1";
pub const IOTA_SI32_SYMBOL: &str = "gpt_rs_triton_iota_si32_rank4";
pub const COMPARE_SI32_I1_KERNEL_ID: &str = "gpt_rs.triton.kernel.compare_si32_i1.v1";
pub const COMPARE_SI32_I1_SYMBOL: &str = "gpt_rs_triton_compare_si32_i1";
pub const SELECT_I1_F32_KERNEL_ID: &str = "gpt_rs.triton.kernel.select_i1_f32.v1";
pub const SELECT_I1_F32_SYMBOL: &str = "gpt_rs_triton_select_i1_f32";
pub const TAKE_F32_I32_KERNEL_ID: &str = "gpt_rs.triton.kernel.take_f32_i32.v1";
pub const TAKE_F32_I32_SYMBOL: &str = "gpt_rs_triton_take_f32_i32";
pub const DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID: &str =
    "gpt_rs.triton.kernel.dynamic_update_slice_f32_rank4.v1";
pub const DYNAMIC_UPDATE_SLICE_F32_SYMBOL: &str = "gpt_rs_triton_dynamic_update_slice_f32_rank4";
pub const EXTRACT_PATCHES_NHWC_KERNEL_ID: &str = "gpt_rs.triton.kernel.extract_patches_nhwc_f32.v1";
pub const EXTRACT_PATCHES_NHWC_SYMBOL: &str = "gpt_rs_triton_extract_patches_nhwc_f32";
pub const REDUCE_WINDOW_MAX_NHWC_KERNEL_ID: &str =
    "gpt_rs.triton.kernel.reduce_window_max_nhwc_f32.v1";
pub const REDUCE_WINDOW_MAX_NHWC_SYMBOL: &str = "gpt_rs_triton_reduce_window_max_nhwc_f32";
pub const DOT_BIAS_RANK2_KERNEL_ID: &str = "gpt_rs.triton.kernel.dot_bias_rank2_f32.v1";
pub const DOT_BIAS_RANK2_SYMBOL: &str = "gpt_rs_triton_dot_bias_rank2_f32";
pub const LAYER_NORM_F32_KERNEL_ID: &str = "gpt_rs.triton.kernel.layer_norm_f32.v1";
pub const LAYER_NORM_F32_SYMBOL: &str = "gpt_rs_triton_layer_norm_f32";

pub const EWISE_BINARY_KERNEL_SOURCE: &str = include_str!("kernels/elementwise_binary_f32.triton");
pub const EWISE_UNARY_KERNEL_SOURCE: &str = include_str!("kernels/elementwise_unary_f32.triton");
pub const BROADCAST_KERNEL_SOURCE: &str = include_str!("kernels/broadcast_f32_rank4.triton");
pub const BROADCAST_SI32_KERNEL_SOURCE: &str = include_str!("kernels/broadcast_si32_rank4.triton");
pub const SLICE_KERNEL_SOURCE: &str = include_str!("kernels/slice_f32_rank4.triton");
pub const DYNAMIC_SLICE_F32_KERNEL_SOURCE: &str =
    include_str!("kernels/dynamic_slice_f32_rank4.triton");
pub const DYNAMIC_SLICE_SI32_RANK1_KERNEL_SOURCE: &str =
    include_str!("kernels/dynamic_slice_si32_rank1.triton");
pub const TRANSPOSE_KERNEL_SOURCE: &str = include_str!("kernels/transpose_f32_rank5.triton");
pub const CONCAT_KERNEL_SOURCE: &str = include_str!("kernels/concat_f32_rank4.triton");
pub const REDUCE_SUM_LAST_AXIS_KERNEL_SOURCE: &str =
    include_str!("kernels/reduce_sum_last_axis_f32.triton");
pub const REDUCE_MAX_LAST_AXIS_KERNEL_SOURCE: &str =
    include_str!("kernels/reduce_max_last_axis_f32.triton");
pub const SOFTMAX_LAST_AXIS_KERNEL_SOURCE: &str =
    include_str!("kernels/softmax_last_axis_f32.triton");
pub const IOTA_SI32_KERNEL_SOURCE: &str = include_str!("kernels/iota_si32_rank4.triton");
pub const COMPARE_SI32_I1_KERNEL_SOURCE: &str = include_str!("kernels/compare_si32_i1.triton");
pub const SELECT_I1_F32_KERNEL_SOURCE: &str = include_str!("kernels/select_i1_f32.triton");
pub const TAKE_F32_I32_KERNEL_SOURCE: &str = include_str!("kernels/take_f32_i32.triton");
pub const DYNAMIC_UPDATE_SLICE_F32_KERNEL_SOURCE: &str =
    include_str!("kernels/dynamic_update_slice_f32_rank4.triton");
pub const EXTRACT_PATCHES_NHWC_KERNEL_SOURCE: &str =
    include_str!("kernels/extract_patches_nhwc_f32.triton");
pub const REDUCE_WINDOW_MAX_NHWC_KERNEL_SOURCE: &str =
    include_str!("kernels/reduce_window_max_nhwc_f32.triton");
pub const DOT_BIAS_RANK2_KERNEL_SOURCE: &str = include_str!("kernels/dot_bias_rank2_f32.triton");
pub const LAYER_NORM_F32_KERNEL_SOURCE: &str = include_str!("kernels/layer_norm_f32.triton");
pub const MATMUL_F32_KERNEL_SOURCE: &str = include_str!("kernels/matmul_f32.triton");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub kind: KernelKind,
    pub source: String,
    pub symbol: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelKind {
    ElementwiseBinaryF32,
    ElementwiseUnaryF32,
    BroadcastF32Rank4,
    BroadcastSi32Rank4,
    SliceF32Rank4,
    DynamicSliceF32Rank4,
    DynamicSliceSi32Rank1,
    TransposeF32Rank5,
    ConcatF32Rank4,
    ReduceSumLastAxisF32,
    ReduceMaxLastAxisF32,
    SoftmaxLastAxisF32,
    IotaSi32Rank4,
    CompareSi32I1,
    SelectI1F32,
    TakeF32I32,
    DynamicUpdateSliceF32Rank4,
    ExtractPatchesNhwcF32,
    ReduceWindowMaxNhwcF32,
    DotBiasRank2F32,
    LayerNormF32,
    FusedElementwiseF32,
}

#[derive(Clone, Copy)]
struct KernelDescriptor {
    id: &'static str,
    kind: KernelKind,
    symbol: &'static str,
    source: &'static str,
}

const BUILTIN_KERNEL_DESCRIPTORS: &[KernelDescriptor] = &[
    KernelDescriptor {
        id: EWISE_UNARY_KERNEL_ID,
        kind: KernelKind::ElementwiseUnaryF32,
        symbol: EWISE_UNARY_SYMBOL,
        source: EWISE_UNARY_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: EWISE_BINARY_KERNEL_ID,
        kind: KernelKind::ElementwiseBinaryF32,
        symbol: EWISE_BINARY_SYMBOL,
        source: EWISE_BINARY_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: BROADCAST_KERNEL_ID,
        kind: KernelKind::BroadcastF32Rank4,
        symbol: BROADCAST_SYMBOL,
        source: BROADCAST_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: BROADCAST_SI32_KERNEL_ID,
        kind: KernelKind::BroadcastSi32Rank4,
        symbol: BROADCAST_SI32_SYMBOL,
        source: BROADCAST_SI32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: SLICE_KERNEL_ID,
        kind: KernelKind::SliceF32Rank4,
        symbol: SLICE_SYMBOL,
        source: SLICE_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: DYNAMIC_SLICE_F32_KERNEL_ID,
        kind: KernelKind::DynamicSliceF32Rank4,
        symbol: DYNAMIC_SLICE_F32_SYMBOL,
        source: DYNAMIC_SLICE_F32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: DYNAMIC_SLICE_SI32_RANK1_KERNEL_ID,
        kind: KernelKind::DynamicSliceSi32Rank1,
        symbol: DYNAMIC_SLICE_SI32_RANK1_SYMBOL,
        source: DYNAMIC_SLICE_SI32_RANK1_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: TRANSPOSE_KERNEL_ID,
        kind: KernelKind::TransposeF32Rank5,
        symbol: TRANSPOSE_SYMBOL,
        source: TRANSPOSE_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: CONCAT_KERNEL_ID,
        kind: KernelKind::ConcatF32Rank4,
        symbol: CONCAT_SYMBOL,
        source: CONCAT_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: REDUCE_SUM_LAST_AXIS_KERNEL_ID,
        kind: KernelKind::ReduceSumLastAxisF32,
        symbol: REDUCE_SUM_LAST_AXIS_SYMBOL,
        source: REDUCE_SUM_LAST_AXIS_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: REDUCE_MAX_LAST_AXIS_KERNEL_ID,
        kind: KernelKind::ReduceMaxLastAxisF32,
        symbol: REDUCE_MAX_LAST_AXIS_SYMBOL,
        source: REDUCE_MAX_LAST_AXIS_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: IOTA_SI32_KERNEL_ID,
        kind: KernelKind::IotaSi32Rank4,
        symbol: IOTA_SI32_SYMBOL,
        source: IOTA_SI32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: COMPARE_SI32_I1_KERNEL_ID,
        kind: KernelKind::CompareSi32I1,
        symbol: COMPARE_SI32_I1_SYMBOL,
        source: COMPARE_SI32_I1_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: SELECT_I1_F32_KERNEL_ID,
        kind: KernelKind::SelectI1F32,
        symbol: SELECT_I1_F32_SYMBOL,
        source: SELECT_I1_F32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: TAKE_F32_I32_KERNEL_ID,
        kind: KernelKind::TakeF32I32,
        symbol: TAKE_F32_I32_SYMBOL,
        source: TAKE_F32_I32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID,
        kind: KernelKind::DynamicUpdateSliceF32Rank4,
        symbol: DYNAMIC_UPDATE_SLICE_F32_SYMBOL,
        source: DYNAMIC_UPDATE_SLICE_F32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: EXTRACT_PATCHES_NHWC_KERNEL_ID,
        kind: KernelKind::ExtractPatchesNhwcF32,
        symbol: EXTRACT_PATCHES_NHWC_SYMBOL,
        source: EXTRACT_PATCHES_NHWC_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: REDUCE_WINDOW_MAX_NHWC_KERNEL_ID,
        kind: KernelKind::ReduceWindowMaxNhwcF32,
        symbol: REDUCE_WINDOW_MAX_NHWC_SYMBOL,
        source: REDUCE_WINDOW_MAX_NHWC_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: DOT_BIAS_RANK2_KERNEL_ID,
        kind: KernelKind::DotBiasRank2F32,
        symbol: DOT_BIAS_RANK2_SYMBOL,
        source: DOT_BIAS_RANK2_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: LAYER_NORM_F32_KERNEL_ID,
        kind: KernelKind::LayerNormF32,
        symbol: LAYER_NORM_F32_SYMBOL,
        source: LAYER_NORM_F32_KERNEL_SOURCE,
    },
    KernelDescriptor {
        id: SOFTMAX_LAST_AXIS_KERNEL_ID,
        kind: KernelKind::SoftmaxLastAxisF32,
        symbol: SOFTMAX_LAST_AXIS_SYMBOL,
        source: SOFTMAX_LAST_AXIS_KERNEL_SOURCE,
    },
];

pub fn builtin_kernel_specs() -> Vec<KernelSpec> {
    BUILTIN_KERNEL_DESCRIPTORS
        .iter()
        .map(descriptor_to_spec)
        .collect()
}

pub fn kernel_spec_by_id(id: &str) -> Option<KernelSpec> {
    BUILTIN_KERNEL_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.id == id)
        .map(descriptor_to_spec)
}

pub fn builtin_kernel_sources() -> &'static [&'static str] {
    &[
        EWISE_BINARY_KERNEL_SOURCE,
        EWISE_UNARY_KERNEL_SOURCE,
        BROADCAST_KERNEL_SOURCE,
        BROADCAST_SI32_KERNEL_SOURCE,
        SLICE_KERNEL_SOURCE,
        DYNAMIC_SLICE_F32_KERNEL_SOURCE,
        DYNAMIC_SLICE_SI32_RANK1_KERNEL_SOURCE,
        TRANSPOSE_KERNEL_SOURCE,
        CONCAT_KERNEL_SOURCE,
        REDUCE_SUM_LAST_AXIS_KERNEL_SOURCE,
        REDUCE_MAX_LAST_AXIS_KERNEL_SOURCE,
        SOFTMAX_LAST_AXIS_KERNEL_SOURCE,
        IOTA_SI32_KERNEL_SOURCE,
        COMPARE_SI32_I1_KERNEL_SOURCE,
        SELECT_I1_F32_KERNEL_SOURCE,
        TAKE_F32_I32_KERNEL_SOURCE,
        DYNAMIC_UPDATE_SLICE_F32_KERNEL_SOURCE,
        EXTRACT_PATCHES_NHWC_KERNEL_SOURCE,
        REDUCE_WINDOW_MAX_NHWC_KERNEL_SOURCE,
        DOT_BIAS_RANK2_KERNEL_SOURCE,
        LAYER_NORM_F32_KERNEL_SOURCE,
        MATMUL_F32_KERNEL_SOURCE,
    ]
}

fn descriptor_to_spec(descriptor: &KernelDescriptor) -> KernelSpec {
    KernelSpec {
        id: descriptor.id.to_string(),
        kind: descriptor.kind,
        source: descriptor.source.to_string(),
        symbol: descriptor.symbol.to_string(),
    }
}
