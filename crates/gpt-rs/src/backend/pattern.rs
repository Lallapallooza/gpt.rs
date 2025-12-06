use crate::backend::{index::InstId, rewriter::ProgramRewriter, spec::Operation};

/// Predicate used to restrict which operations a pattern should consider.
pub type OperationMatcher = fn(&Operation) -> bool;

pub trait OperationView: Clone {
    const MATCHER: OperationMatcher;
    fn extract(root: InstId, rewriter: &ProgramRewriter) -> Option<Self>;
}

mod pattern_views;
pub use pattern_views::*;

mod template;
pub use template::*;

mod capture;
pub use capture::*;

mod registry;
pub use registry::*;

/// Base trait for rewrite patterns operating on PTIR.
pub trait Pattern: Send + Sync {
    fn matches_operation(&self, op: &Operation) -> bool {
        let _ = op;
        true
    }
    fn benefit(&self) -> u16 {
        1
    }
    fn match_and_rewrite(&self, root: InstId, rewriter: &mut ProgramRewriter) -> bool;
}

/// Typed convenience trait mirroring MLIR's `OpRewritePattern`.
pub trait OpRewritePattern<T> {
    fn benefit(&self) -> u16 {
        1
    }
    fn may_match(&self, _op: &T, _rewriter: &ProgramRewriter) -> bool {
        true
    }
    fn match_and_rewrite(&self, op: T, rewriter: &mut ProgramRewriter) -> bool;
}

/// Adapter converting a typed pattern into a `Pattern`.
pub struct TypedPattern<P, T> {
    pattern: P,
    matcher: Option<OperationMatcher>,
    extractor: fn(InstId, &ProgramRewriter) -> Option<T>,
}

impl<P, T> TypedPattern<P, T> {
    fn from_parts(
        pattern: P,
        matcher: Option<OperationMatcher>,
        extractor: fn(InstId, &ProgramRewriter) -> Option<T>,
    ) -> Self {
        Self {
            pattern,
            matcher,
            extractor,
        }
    }

    pub fn with_operation_matcher(
        pattern: P,
        matcher: OperationMatcher,
        extractor: fn(InstId, &ProgramRewriter) -> Option<T>,
    ) -> Self {
        Self::from_parts(pattern, Some(matcher), extractor)
    }

    pub fn match_any(pattern: P, extractor: fn(InstId, &ProgramRewriter) -> Option<T>) -> Self {
        Self::from_parts(pattern, None, extractor)
    }
}

impl<P, V> TypedPattern<P, V>
where
    V: OperationView,
{
    pub fn from_view(pattern: P) -> Self {
        Self::with_operation_matcher(pattern, V::MATCHER, V::extract)
    }
}

impl<P, T> Pattern for TypedPattern<P, T>
where
    P: OpRewritePattern<T> + Send + Sync,
    T: Send,
{
    fn matches_operation(&self, op: &Operation) -> bool {
        match self.matcher {
            Some(matcher) => matcher(op),
            None => true,
        }
    }

    fn benefit(&self) -> u16 {
        self.pattern.benefit()
    }

    fn match_and_rewrite(&self, root: InstId, rewriter: &mut ProgramRewriter) -> bool {
        let Some(view) = (self.extractor)(root, rewriter) else {
            return false;
        };
        if !self.pattern.may_match(&view, rewriter) {
            return false;
        }
        self.pattern.match_and_rewrite(view, rewriter)
    }
}

/// Mutable set that collects rewrite patterns prior to freezing.
pub struct PatternSet {
    patterns: Vec<Box<dyn Pattern>>,
}

impl PatternSet {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    pub fn add<P>(&mut self, pattern: P) -> &mut Self
    where
        P: Pattern + 'static,
    {
        self.patterns.push(Box::new(pattern));
        self
    }

    pub fn add_typed<P, T>(&mut self, pattern: TypedPattern<P, T>) -> &mut Self
    where
        P: OpRewritePattern<T> + Send + Sync + 'static,
        T: Send + 'static,
    {
        self.patterns.push(Box::new(pattern));
        self
    }

    pub fn insert_view<V, P>(&mut self, pattern: P) -> &mut Self
    where
        V: OperationView + Send + 'static,
        P: OpRewritePattern<V> + Send + Sync + 'static,
    {
        self.add_typed(TypedPattern::<P, V>::from_view(pattern))
    }

    pub fn insert_many_view<V, P, I>(&mut self, patterns: I) -> &mut Self
    where
        V: OperationView + Send + 'static,
        P: OpRewritePattern<V> + Send + Sync + 'static,
        I: IntoIterator<Item = P>,
    {
        for pattern in patterns {
            self.insert_view::<V, P>(pattern);
        }
        self
    }

    pub fn insert_match_any<P, T>(
        &mut self,
        pattern: P,
        extractor: fn(InstId, &ProgramRewriter) -> Option<T>,
    ) -> &mut Self
    where
        P: OpRewritePattern<T> + Send + Sync + 'static,
        T: Send + 'static,
    {
        self.add_typed(TypedPattern::match_any(pattern, extractor))
    }

    pub fn freeze(mut self) -> FrozenPatternSet {
        self.patterns
            .sort_by_key(|pattern| std::cmp::Reverse(pattern.benefit()));
        FrozenPatternSet {
            patterns: self.patterns,
        }
    }
}

impl Default for PatternSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable collection of rewrite patterns ready for use by the driver.
pub struct FrozenPatternSet {
    patterns: Vec<Box<dyn Pattern>>,
}

impl FrozenPatternSet {
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &dyn Pattern)> {
        self.patterns.iter().enumerate().map(|(idx, p)| (idx, &**p))
    }

    pub fn matching<'a>(
        &'a self,
        op: &'a Operation,
    ) -> impl Iterator<Item = (usize, &'a dyn Pattern)> + 'a {
        self.patterns
            .iter()
            .enumerate()
            .filter(move |(_, pattern)| pattern.matches_operation(op))
            .map(|(idx, pattern)| (idx, &**pattern))
    }
}

/// Common operation matchers.
pub mod filters {
    use crate::backend::spec::{ElementwiseBinaryOp, ElementwiseUnaryOp, Operation, ReduceKind};

    pub const fn any(op: &Operation) -> bool {
        let _ = op;
        true
    }

    pub fn constant(op: &Operation) -> bool {
        matches!(op, Operation::Constant(_))
    }

    pub fn cast(op: &Operation) -> bool {
        matches!(op, Operation::Cast(_))
    }

    pub fn transpose(op: &Operation) -> bool {
        matches!(op, Operation::Transpose(_))
    }

    pub fn reshape(op: &Operation) -> bool {
        matches!(op, Operation::Reshape(_))
    }

    pub fn elementwise_unary(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseUnary(_))
    }

    pub fn exp(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseUnary(ElementwiseUnaryOp::Exp))
    }

    pub fn elementwise_binary(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseBinary(_))
    }

    pub fn add(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseBinary(ElementwiseBinaryOp::Add))
    }

    pub fn sub(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseBinary(ElementwiseBinaryOp::Sub))
    }

    pub fn mul(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseBinary(ElementwiseBinaryOp::Mul))
    }

    pub fn div(op: &Operation) -> bool {
        matches!(op, Operation::ElementwiseBinary(ElementwiseBinaryOp::Div))
    }

    pub fn maximum(op: &Operation) -> bool {
        matches!(
            op,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Maximum)
        )
    }

    pub fn minimum(op: &Operation) -> bool {
        matches!(
            op,
            Operation::ElementwiseBinary(ElementwiseBinaryOp::Minimum)
        )
    }

    pub fn broadcast_to(op: &Operation) -> bool {
        matches!(op, Operation::BroadcastTo(_))
    }

    pub fn slice(op: &Operation) -> bool {
        matches!(op, Operation::Slice(_))
    }

    pub fn reduce(op: &Operation) -> bool {
        matches!(op, Operation::Reduce(_))
    }

    pub fn reduce_sum(op: &Operation) -> bool {
        matches!(op, Operation::Reduce(spec) if spec.kind == ReduceKind::Sum)
    }

    pub fn reduce_max(op: &Operation) -> bool {
        matches!(op, Operation::Reduce(spec) if spec.kind == ReduceKind::Max)
    }

    pub fn reduce_min(op: &Operation) -> bool {
        matches!(op, Operation::Reduce(spec) if spec.kind == ReduceKind::Min)
    }

    pub fn stop_gradient(op: &Operation) -> bool {
        matches!(op, Operation::StopGradient)
    }

    pub fn dot_general(op: &Operation) -> bool {
        matches!(op, Operation::DotGeneral(_))
    }

    pub fn extract_patches(op: &Operation) -> bool {
        matches!(op, Operation::ExtractPatches(_))
    }

    pub fn custom_call(op: &Operation) -> bool {
        matches!(op, Operation::CustomCall(_))
    }
}

#[macro_export]
macro_rules! register_patterns_for_view {
    ($set:expr, $view:ty, $($pattern:expr),+ $(,)?) => {{
        $( $set.insert_view::<$view, _>($pattern); )+
    }};
}
