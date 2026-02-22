use crate::backend::optimizer::OptimizeContext;
use crate::backend::spec::{Function, HintPolicy, PortableBackend};

use super::{FusionCandidate, FusionRejectReason};

pub trait HintLegalizer<B: PortableBackend + 'static>: Send + Sync {
    fn can_fuse(
        &self,
        function: &Function,
        candidate: &FusionCandidate,
        cx: &OptimizeContext<B>,
    ) -> Result<HintPolicy, FusionRejectReason>;
}

pub trait HintCostModel<B: PortableBackend + 'static>: Send + Sync {
    fn score(
        &self,
        function: &Function,
        candidate: &FusionCandidate,
        cx: &OptimizeContext<B>,
    ) -> i64;
}
