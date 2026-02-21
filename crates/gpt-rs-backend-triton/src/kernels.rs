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
pub const TRANSPOSE_KERNEL_ID: &str = "gpt_rs.triton.kernel.transpose_f32_rank5.v1";
pub const TRANSPOSE_SYMBOL: &str = "gpt_rs_triton_transpose_f32_rank5";
pub const CONCAT_KERNEL_ID: &str = "gpt_rs.triton.kernel.concat_f32_rank4.v1";
pub const CONCAT_SYMBOL: &str = "gpt_rs_triton_concat_f32_rank4";
pub const REDUCE_SUM_LAST_AXIS_KERNEL_ID: &str = "gpt_rs.triton.kernel.reduce_sum_last_axis_f32.v1";
pub const REDUCE_SUM_LAST_AXIS_SYMBOL: &str = "gpt_rs_triton_reduce_sum_last_axis_f32";
pub const REDUCE_MAX_LAST_AXIS_KERNEL_ID: &str = "gpt_rs.triton.kernel.reduce_max_last_axis_f32.v1";
pub const REDUCE_MAX_LAST_AXIS_SYMBOL: &str = "gpt_rs_triton_reduce_max_last_axis_f32";
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
pub const PREPACKED_EWISE_BINARY_KERNEL: &str =
    include_str!("kernels/prepacked/elementwise_binary_f32.triton");
pub const PREPACKED_EWISE_UNARY_KERNEL: &str =
    include_str!("kernels/prepacked/elementwise_unary_f32.triton");
pub const PREPACKED_BROADCAST_KERNEL: &str =
    include_str!("kernels/prepacked/broadcast_f32_rank4.triton");
pub const PREPACKED_BROADCAST_SI32_KERNEL: &str =
    include_str!("kernels/prepacked/broadcast_si32_rank4.triton");
pub const PREPACKED_SLICE_KERNEL: &str = include_str!("kernels/prepacked/slice_f32_rank4.triton");
pub const PREPACKED_TRANSPOSE_KERNEL: &str =
    include_str!("kernels/prepacked/transpose_f32_rank5.triton");
pub const PREPACKED_CONCAT_KERNEL: &str = include_str!("kernels/prepacked/concat_f32_rank4.triton");
pub const PREPACKED_REDUCE_MAX_LAST_AXIS_KERNEL: &str =
    include_str!("kernels/prepacked/reduce_max_last_axis_f32.triton");
pub const PREPACKED_IOTA_SI32_KERNEL: &str =
    include_str!("kernels/prepacked/iota_si32_rank4.triton");
pub const PREPACKED_COMPARE_SI32_I1_KERNEL: &str =
    include_str!("kernels/prepacked/compare_si32_i1.triton");
pub const PREPACKED_SELECT_I1_F32_KERNEL: &str =
    include_str!("kernels/prepacked/select_i1_f32.triton");
pub const PREPACKED_TAKE_F32_I32_KERNEL: &str =
    include_str!("kernels/prepacked/take_f32_i32.triton");
pub const PREPACKED_DYNAMIC_UPDATE_SLICE_F32_KERNEL: &str =
    include_str!("kernels/prepacked/dynamic_update_slice_f32_rank4.triton");
pub const PREPACKED_EXTRACT_PATCHES_NHWC_KERNEL: &str =
    include_str!("kernels/prepacked/extract_patches_nhwc_f32.triton");
pub const PREPACKED_REDUCE_WINDOW_MAX_NHWC_KERNEL: &str =
    include_str!("kernels/prepacked/reduce_window_max_nhwc_f32.triton");
pub const PREPACKED_MATMUL_KERNEL: &str = include_str!("kernels/prepacked/matmul_f32.triton");
pub const PREPACKED_REDUCE_SUM_LAST_AXIS_KERNEL: &str =
    include_str!("kernels/prepacked/reduce_sum_last_axis_f32.triton");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSpec {
    pub id: String,
    pub kind: KernelKind,
    pub source: String,
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelKind {
    ElementwiseBinaryF32,
    ElementwiseUnaryF32,
    BroadcastF32Rank4,
    BroadcastSi32Rank4,
    SliceF32Rank4,
    TransposeF32Rank5,
    ConcatF32Rank4,
    ReduceSumLastAxisF32,
    ReduceMaxLastAxisF32,
    IotaSi32Rank4,
    CompareSi32I1,
    SelectI1F32,
    TakeF32I32,
    DynamicUpdateSliceF32Rank4,
    ExtractPatchesNhwcF32,
    ReduceWindowMaxNhwcF32,
}

pub fn elementwise_binary_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EWISE_BINARY_KERNEL_ID.to_string(),
        kind: KernelKind::ElementwiseBinaryF32,
        source: elementwise_binary_triton_source(),
        symbol: EWISE_BINARY_SYMBOL.to_string(),
    }
}

pub fn elementwise_unary_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EWISE_UNARY_KERNEL_ID.to_string(),
        kind: KernelKind::ElementwiseUnaryF32,
        source: elementwise_unary_triton_source(),
        symbol: EWISE_UNARY_SYMBOL.to_string(),
    }
}

pub fn broadcast_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: BROADCAST_KERNEL_ID.to_string(),
        kind: KernelKind::BroadcastF32Rank4,
        source: PREPACKED_BROADCAST_KERNEL.to_string(),
        symbol: BROADCAST_SYMBOL.to_string(),
    }
}

pub fn broadcast_si32_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: BROADCAST_SI32_KERNEL_ID.to_string(),
        kind: KernelKind::BroadcastSi32Rank4,
        source: PREPACKED_BROADCAST_SI32_KERNEL.to_string(),
        symbol: BROADCAST_SI32_SYMBOL.to_string(),
    }
}

pub fn slice_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: SLICE_KERNEL_ID.to_string(),
        kind: KernelKind::SliceF32Rank4,
        source: PREPACKED_SLICE_KERNEL.to_string(),
        symbol: SLICE_SYMBOL.to_string(),
    }
}

pub fn transpose_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: TRANSPOSE_KERNEL_ID.to_string(),
        kind: KernelKind::TransposeF32Rank5,
        source: PREPACKED_TRANSPOSE_KERNEL.to_string(),
        symbol: TRANSPOSE_SYMBOL.to_string(),
    }
}

pub fn concat_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: CONCAT_KERNEL_ID.to_string(),
        kind: KernelKind::ConcatF32Rank4,
        source: PREPACKED_CONCAT_KERNEL.to_string(),
        symbol: CONCAT_SYMBOL.to_string(),
    }
}

pub fn reduce_sum_last_axis_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: REDUCE_SUM_LAST_AXIS_KERNEL_ID.to_string(),
        kind: KernelKind::ReduceSumLastAxisF32,
        source: PREPACKED_REDUCE_SUM_LAST_AXIS_KERNEL.to_string(),
        symbol: REDUCE_SUM_LAST_AXIS_SYMBOL.to_string(),
    }
}

pub fn reduce_max_last_axis_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: REDUCE_MAX_LAST_AXIS_KERNEL_ID.to_string(),
        kind: KernelKind::ReduceMaxLastAxisF32,
        source: PREPACKED_REDUCE_MAX_LAST_AXIS_KERNEL.to_string(),
        symbol: REDUCE_MAX_LAST_AXIS_SYMBOL.to_string(),
    }
}

pub fn iota_si32_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: IOTA_SI32_KERNEL_ID.to_string(),
        kind: KernelKind::IotaSi32Rank4,
        source: PREPACKED_IOTA_SI32_KERNEL.to_string(),
        symbol: IOTA_SI32_SYMBOL.to_string(),
    }
}

pub fn compare_si32_i1_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: COMPARE_SI32_I1_KERNEL_ID.to_string(),
        kind: KernelKind::CompareSi32I1,
        source: PREPACKED_COMPARE_SI32_I1_KERNEL.to_string(),
        symbol: COMPARE_SI32_I1_SYMBOL.to_string(),
    }
}

pub fn select_i1_f32_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: SELECT_I1_F32_KERNEL_ID.to_string(),
        kind: KernelKind::SelectI1F32,
        source: PREPACKED_SELECT_I1_F32_KERNEL.to_string(),
        symbol: SELECT_I1_F32_SYMBOL.to_string(),
    }
}

pub fn take_f32_i32_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: TAKE_F32_I32_KERNEL_ID.to_string(),
        kind: KernelKind::TakeF32I32,
        source: PREPACKED_TAKE_F32_I32_KERNEL.to_string(),
        symbol: TAKE_F32_I32_SYMBOL.to_string(),
    }
}

pub fn dynamic_update_slice_f32_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID.to_string(),
        kind: KernelKind::DynamicUpdateSliceF32Rank4,
        source: PREPACKED_DYNAMIC_UPDATE_SLICE_F32_KERNEL.to_string(),
        symbol: DYNAMIC_UPDATE_SLICE_F32_SYMBOL.to_string(),
    }
}

pub fn extract_patches_nhwc_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: EXTRACT_PATCHES_NHWC_KERNEL_ID.to_string(),
        kind: KernelKind::ExtractPatchesNhwcF32,
        source: PREPACKED_EXTRACT_PATCHES_NHWC_KERNEL.to_string(),
        symbol: EXTRACT_PATCHES_NHWC_SYMBOL.to_string(),
    }
}

pub fn reduce_window_max_nhwc_kernel_spec() -> KernelSpec {
    KernelSpec {
        id: REDUCE_WINDOW_MAX_NHWC_KERNEL_ID.to_string(),
        kind: KernelKind::ReduceWindowMaxNhwcF32,
        source: PREPACKED_REDUCE_WINDOW_MAX_NHWC_KERNEL.to_string(),
        symbol: REDUCE_WINDOW_MAX_NHWC_SYMBOL.to_string(),
    }
}

pub fn elementwise_binary_triton_source() -> String {
    PREPACKED_EWISE_BINARY_KERNEL.to_string()
}

pub fn elementwise_unary_triton_source() -> String {
    PREPACKED_EWISE_UNARY_KERNEL.to_string()
}

pub fn prepacked_kernel_sources() -> &'static [&'static str] {
    &[
        PREPACKED_EWISE_BINARY_KERNEL,
        PREPACKED_EWISE_UNARY_KERNEL,
        PREPACKED_BROADCAST_KERNEL,
        PREPACKED_BROADCAST_SI32_KERNEL,
        PREPACKED_SLICE_KERNEL,
        PREPACKED_TRANSPOSE_KERNEL,
        PREPACKED_CONCAT_KERNEL,
        PREPACKED_REDUCE_MAX_LAST_AXIS_KERNEL,
        PREPACKED_IOTA_SI32_KERNEL,
        PREPACKED_COMPARE_SI32_I1_KERNEL,
        PREPACKED_SELECT_I1_F32_KERNEL,
        PREPACKED_TAKE_F32_I32_KERNEL,
        PREPACKED_DYNAMIC_UPDATE_SLICE_F32_KERNEL,
        PREPACKED_EXTRACT_PATCHES_NHWC_KERNEL,
        PREPACKED_REDUCE_WINDOW_MAX_NHWC_KERNEL,
        PREPACKED_MATMUL_KERNEL,
        PREPACKED_REDUCE_SUM_LAST_AXIS_KERNEL,
    ]
}
