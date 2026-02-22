mod analysis;
mod elementwise_dag;
mod ir;
mod policy;
mod rewrite;
mod select;

pub use analysis::discover_candidates;
pub use elementwise_dag::{
    binary_opcode, build_elementwise_dag, unary_opcode, ElementwiseDag, FusionNode, FusionNodeKind,
    FusionRef,
};
pub use ir::{
    DotEpiloguePayload, ElementwiseDagPayload, FusionCandidate, FusionDetail, FusionRejectReason,
    SelectedFusion,
};
pub use policy::{HintCostModel, HintLegalizer};
pub use rewrite::materialize_hints;
pub use select::{select_non_overlapping, SelectionResult};

pub const FUSION_ATTR_VERSION: &str = "fusion_version";
pub const FUSION_ATTR_KIND: &str = "fusion_kind";
pub const FUSION_ATTR_GENERATED: &str = "fusion_generated";
pub const FUSION_KIND_ELEMENTWISE_DAG_V1: &str = "elementwise_dag_v1";
pub const FUSION_KIND_DOT_EPILOGUE_V1: &str = "dot_epilogue_v1";
