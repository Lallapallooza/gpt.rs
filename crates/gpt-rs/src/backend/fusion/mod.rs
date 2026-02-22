mod elementwise_dag;

pub use elementwise_dag::{
    binary_opcode, build_elementwise_dag, unary_opcode, ElementwiseDag, FusionNode, FusionNodeKind,
    FusionRef,
};

pub const FUSION_ATTR_VERSION: &str = "fusion_version";
pub const FUSION_ATTR_KIND: &str = "fusion_kind";
pub const FUSION_KIND_ELEMENTWISE_DAG_V1: &str = "elementwise_dag_v1";
