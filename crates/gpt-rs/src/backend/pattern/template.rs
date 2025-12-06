use std::collections::{HashSet, VecDeque};
use std::fmt;

use crate::backend::{
    index::InstId,
    rewriter::ProgramRewriter,
    spec::{
        ComparisonOp, ElementwiseBinaryOp, ElementwiseUnaryOp, Operand, Operation, ReduceKind,
        ScatterReduceKind, SegmentReduceKind, ValueId,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemplateNodeId(pub u32);

impl TemplateNodeId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemplateValueRef {
    Node(TemplateNodeId),
    Input(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateOperand {
    Value(TemplateValueRef),
    TupleElement {
        tuple: TemplateValueRef,
        index: usize,
    },
    Literal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationKey {
    discriminant: std::mem::Discriminant<Operation>,
    subkind: OperationSubkind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum OperationSubkind {
    None,
    ElementwiseUnary(ElementwiseUnaryOp),
    ElementwiseBinary(ElementwiseBinaryOp),
    Reduce(ReduceKind),
    Compare(ComparisonOp),
    ScatterReduce(ScatterReduceKind),
    SegmentReduce(SegmentReduceKind),
}

impl OperationKey {
    pub fn from_operation(op: &Operation) -> Self {
        let discriminant = std::mem::discriminant(op);
        let subkind = match op {
            Operation::ElementwiseUnary(op) => OperationSubkind::ElementwiseUnary(*op),
            Operation::ElementwiseBinary(op) => OperationSubkind::ElementwiseBinary(*op),
            Operation::Reduce(spec) => OperationSubkind::Reduce(spec.kind),
            Operation::ReduceWindow(spec) => OperationSubkind::Reduce(spec.reduce),
            Operation::Compare(spec) => OperationSubkind::Compare(spec.op),
            Operation::ScatterReduce(spec) => OperationSubkind::ScatterReduce(spec.reduce),
            Operation::SegmentReduce(spec) => OperationSubkind::SegmentReduce(spec.kind),
            _ => OperationSubkind::None,
        };
        Self {
            discriminant,
            subkind,
        }
    }

    fn matches(&self, op: &Operation) -> bool {
        *self == Self::from_operation(op)
    }

    fn is_commutative_binop(&self) -> bool {
        match self.subkind {
            OperationSubkind::ElementwiseBinary(op) => matches!(
                op,
                ElementwiseBinaryOp::Add
                    | ElementwiseBinaryOp::Mul
                    | ElementwiseBinaryOp::Maximum
                    | ElementwiseBinaryOp::Minimum
            ),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateNode {
    pub op: OperationKey,
    pub operands: Vec<TemplateOperand>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TemplateUse {
    user: TemplateNodeId,
    operand_index: usize,
}

#[derive(Debug, Clone)]
pub struct PatternTemplate {
    nodes: Vec<TemplateNode>,
    users: Vec<Vec<TemplateUse>>,
    pub anchor: TemplateNodeId,
    pub output: TemplateValueRef,
    input_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MatchConfig {
    pub allow_commutative_binops: bool,
    pub max_skip_through_depth: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            allow_commutative_binops: true,
            max_skip_through_depth: 0,
        }
    }
}

impl PatternTemplate {
    pub fn builder() -> PatternTemplateBuilder {
        PatternTemplateBuilder::new()
    }

    pub fn nodes(&self) -> &[TemplateNode] {
        &self.nodes
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn match_from_anchor(
        &self,
        anchor_inst: InstId,
        rewriter: &ProgramRewriter,
    ) -> Option<TemplateMatch> {
        self.match_from_anchor_with_config(anchor_inst, rewriter, MatchConfig::default())
    }

    pub fn match_from_anchor_with_config(
        &self,
        anchor_inst: InstId,
        rewriter: &ProgramRewriter,
        cfg: MatchConfig,
    ) -> Option<TemplateMatch> {
        if self.anchor.index() >= self.nodes.len() {
            return None;
        }

        let mut state = MatchState::new(self);
        state.bind_node(self, self.anchor, anchor_inst, rewriter, cfg)?;
        let state = state.solve(self, rewriter, cfg)?;

        if !inputs_are_external_to_match(&state, rewriter) {
            return None;
        }

        let output = resolve_value_ref(self.output, &state, rewriter)?;
        Some(TemplateMatch {
            node_insts: state.node_insts,
            input_values: state.input_values,
            anchor: anchor_inst,
            output,
        })
    }
}

fn inputs_are_external_to_match(state: &MatchState, rewriter: &ProgramRewriter) -> bool {
    let matched_insts: HashSet<InstId> = state.node_insts.iter().copied().flatten().collect();
    let matched_values: HashSet<ValueId> = matched_insts
        .iter()
        .copied()
        .map(|inst| rewriter.value_of(inst))
        .collect();

    for input in state.input_values.iter().copied().flatten() {
        let Some(producer) = rewriter.inst_of(input) else {
            continue;
        };
        if matched_insts.contains(&producer) {
            return false;
        }
        if rewriter
            .operands(producer)
            .iter()
            .any(|operand| operand_mentions_any(operand, &matched_values))
        {
            return false;
        }
    }

    true
}

fn operand_mentions_any(operand: &Operand, values: &HashSet<ValueId>) -> bool {
    match operand {
        Operand::Value(id) => values.contains(id),
        Operand::TupleElement { tuple, .. } => values.contains(tuple),
        Operand::Literal(_) => false,
    }
}

fn resolve_value_ref(
    value: TemplateValueRef,
    state: &MatchState,
    rewriter: &ProgramRewriter,
) -> Option<ValueId> {
    match value {
        TemplateValueRef::Input(index) => state.input_values.get(index as usize).copied().flatten(),
        TemplateValueRef::Node(node) => {
            let inst = state.node_insts.get(node.index()).copied().flatten()?;
            Some(rewriter.value_of(inst))
        }
    }
}

#[derive(Debug, Clone)]
struct MatchState {
    node_insts: Vec<Option<InstId>>,
    input_values: Vec<Option<ValueId>>,
    used_insts: HashSet<InstId>,
}

impl MatchState {
    fn new(template: &PatternTemplate) -> Self {
        Self {
            node_insts: vec![None; template.nodes.len()],
            input_values: vec![None; template.input_count],
            used_insts: HashSet::new(),
        }
    }

    fn bind_node(
        &mut self,
        template: &PatternTemplate,
        node: TemplateNodeId,
        inst: InstId,
        rewriter: &ProgramRewriter,
        cfg: MatchConfig,
    ) -> Option<()> {
        let node_index = node.index();
        let node_template = template.nodes.get(node_index)?;

        if let Some(existing) = self.node_insts[node_index] {
            return (existing == inst).then_some(());
        }

        if self.used_insts.contains(&inst) {
            return None;
        }

        if !node_template.op.matches(rewriter.op(inst)) {
            return None;
        }

        let operands = rewriter.operands(inst);
        if operands.len() != node_template.operands.len() {
            return None;
        }

        let is_commutative_binop = cfg.allow_commutative_binops
            && operands.len() == 2
            && node_template.op.is_commutative_binop();

        let mut base = self.clone();
        base.node_insts[node_index] = Some(inst);
        base.used_insts.insert(inst);

        if !is_commutative_binop {
            for (templ_operand, real_operand) in node_template.operands.iter().zip(operands.iter())
            {
                base.match_operand(template, templ_operand, real_operand, rewriter, cfg)?;
            }
            *self = base;
            return Some(());
        }

        for order in [[0, 1], [1, 0]] {
            let mut state = base.clone();
            let mut ok = true;
            for (templ_index, real_index) in order.into_iter().enumerate() {
                let templ_operand = match node_template.operands.get(templ_index) {
                    Some(value) => value,
                    None => {
                        ok = false;
                        break;
                    }
                };
                let real_operand = match operands.get(real_index) {
                    Some(value) => value,
                    None => {
                        ok = false;
                        break;
                    }
                };
                if state
                    .match_operand(template, templ_operand, real_operand, rewriter, cfg)
                    .is_none()
                {
                    ok = false;
                    break;
                }
            }

            if ok {
                *self = state;
                return Some(());
            }
        }

        None
    }

    fn match_operand(
        &mut self,
        template: &PatternTemplate,
        templ_operand: &TemplateOperand,
        real_operand: &Operand,
        rewriter: &ProgramRewriter,
        cfg: MatchConfig,
    ) -> Option<()> {
        match (templ_operand, real_operand) {
            (TemplateOperand::Literal, Operand::Literal(_)) => Some(()),
            (TemplateOperand::Value(value_ref), Operand::Value(real_value)) => {
                self.match_value_ref(template, *value_ref, *real_value, rewriter, cfg)
            }
            (
                TemplateOperand::TupleElement { tuple, index },
                Operand::TupleElement {
                    tuple: real_tuple,
                    index: real_index,
                },
            ) if *index == *real_index => {
                self.match_value_ref(template, *tuple, *real_tuple, rewriter, cfg)
            }
            _ => None,
        }
    }

    fn match_value_ref(
        &mut self,
        template: &PatternTemplate,
        value_ref: TemplateValueRef,
        real_value: ValueId,
        rewriter: &ProgramRewriter,
        cfg: MatchConfig,
    ) -> Option<()> {
        match value_ref {
            TemplateValueRef::Input(index) => {
                let slot = self.input_values.get_mut(index as usize)?;
                match *slot {
                    Some(existing) => (existing == real_value).then_some(()),
                    None => {
                        *slot = Some(real_value);
                        Some(())
                    }
                }
            }
            TemplateValueRef::Node(node) => {
                let mut value = real_value;
                for _ in 0..=cfg.max_skip_through_depth {
                    let producer = rewriter.inst_of(value)?;
                    let mut fork = self.clone();
                    if fork
                        .bind_node(template, node, producer, rewriter, cfg)
                        .is_some()
                    {
                        *self = fork;
                        return Some(());
                    }

                    if cfg.max_skip_through_depth == 0
                        || !is_skip_through_adapter(rewriter.op(producer))
                    {
                        break;
                    }
                    let [Operand::Value(next)] = rewriter.operands(producer) else {
                        break;
                    };
                    value = *next;
                }
                None
            }
        }
    }

    fn solve(
        &self,
        template: &PatternTemplate,
        rewriter: &ProgramRewriter,
        cfg: MatchConfig,
    ) -> Option<Self> {
        let next = self.clone();

        if next.all_nodes_bound() {
            return Some(next);
        }

        let (src, use_edge) = next.find_unmatched_user_edge(template)?;
        let src_inst = next.node_insts[src.index()]?;
        let src_value = rewriter.value_of(src_inst);

        let candidates = candidate_user_insts(
            template,
            use_edge.user,
            src_value,
            use_edge.operand_index,
            rewriter,
            cfg,
        )?;

        for candidate in candidates {
            if next.used_insts.contains(&candidate) {
                continue;
            }

            let mut fork = next.clone();
            if fork
                .bind_node(template, use_edge.user, candidate, rewriter, cfg)
                .is_none()
            {
                continue;
            }

            if let Some(solved) = fork.solve(template, rewriter, cfg) {
                return Some(solved);
            }
        }

        None
    }

    fn all_nodes_bound(&self) -> bool {
        self.node_insts.iter().all(|value| value.is_some())
    }

    fn find_unmatched_user_edge(
        &self,
        template: &PatternTemplate,
    ) -> Option<(TemplateNodeId, TemplateUse)> {
        for (node_index, maybe_inst) in self.node_insts.iter().enumerate() {
            if maybe_inst.is_none() {
                continue;
            }
            let node_id = TemplateNodeId(node_index as u32);
            for use_edge in template.users.get(node_index)?.iter().copied() {
                if self.node_insts.get(use_edge.user.index())?.is_none() {
                    return Some((node_id, use_edge));
                }
            }
        }
        None
    }
}

fn candidate_user_insts(
    template: &PatternTemplate,
    user_node: TemplateNodeId,
    src_value: ValueId,
    operand_index: usize,
    rewriter: &ProgramRewriter,
    cfg: MatchConfig,
) -> Option<Vec<InstId>> {
    let user_template = template.nodes.get(user_node.index())?;

    let mut candidates = Vec::new();
    let mut seen: HashSet<InstId> = HashSet::new();

    if cfg.max_skip_through_depth == 0 {
        for inst in rewriter.users_of(src_value) {
            if !user_template.op.matches(rewriter.op(*inst)) {
                continue;
            }
            let operands = rewriter.operands(*inst);
            if operands.len() != user_template.operands.len() {
                continue;
            }
            if !operand_edge_compatible(operands, src_value, operand_index, user_template, cfg) {
                continue;
            }
            if seen.insert(*inst) {
                candidates.push(*inst);
            }
        }
        return Some(candidates);
    }

    let mut queue: VecDeque<(ValueId, usize)> = VecDeque::new();
    let mut visited_values: HashSet<ValueId> = HashSet::new();
    queue.push_back((src_value, 0));
    visited_values.insert(src_value);

    while let Some((value, depth)) = queue.pop_front() {
        for inst in rewriter.users_of(value) {
            if user_template.op.matches(rewriter.op(*inst)) {
                let operands = rewriter.operands(*inst);
                if operands.len() == user_template.operands.len()
                    && operand_edge_compatible(operands, value, operand_index, user_template, cfg)
                    && seen.insert(*inst)
                {
                    candidates.push(*inst);
                }
            }

            if depth >= cfg.max_skip_through_depth {
                continue;
            }
            if !is_skip_through_adapter(rewriter.op(*inst)) {
                continue;
            }

            let next_value = rewriter.value_of(*inst);
            if rewriter.users_of(next_value).len() != 1 {
                continue;
            }
            if visited_values.insert(next_value) {
                queue.push_back((next_value, depth + 1));
            }
        }
    }

    Some(candidates)
}

fn operand_mentions_value(operand: &Operand, value: ValueId) -> bool {
    match operand {
        Operand::Value(id) => *id == value,
        Operand::TupleElement { tuple, .. } => *tuple == value,
        Operand::Literal(_) => false,
    }
}

fn operand_edge_compatible(
    operands: &[Operand],
    value: ValueId,
    operand_index: usize,
    user_template: &TemplateNode,
    cfg: MatchConfig,
) -> bool {
    if cfg.allow_commutative_binops
        && operands.len() == 2
        && user_template.op.is_commutative_binop()
    {
        operands
            .iter()
            .any(|operand| operand_mentions_value(operand, value))
    } else {
        operands
            .get(operand_index)
            .is_some_and(|operand| operand_mentions_value(operand, value))
    }
}

fn is_skip_through_adapter(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Reshape(_) | Operation::Transpose(_) | Operation::BroadcastTo(_)
    )
}

#[derive(Debug, Clone)]
pub struct TemplateMatch {
    node_insts: Vec<Option<InstId>>,
    input_values: Vec<Option<ValueId>>,
    pub anchor: InstId,
    pub output: ValueId,
}

impl TemplateMatch {
    pub fn inst(&self, node: TemplateNodeId) -> Option<InstId> {
        self.node_insts.get(node.index()).copied().flatten()
    }

    pub fn input(&self, index: u32) -> Option<ValueId> {
        self.input_values.get(index as usize).copied().flatten()
    }

    pub fn input_count(&self) -> usize {
        self.input_values.len()
    }

    pub fn closure_report(&self, rewriter: &ProgramRewriter) -> ClosureReport {
        let matched_insts: HashSet<InstId> = self.node_insts.iter().copied().flatten().collect();

        let mut external_uses = Vec::new();
        for inst in matched_insts.iter().copied() {
            let value = rewriter.value_of(inst);
            if value == self.output {
                continue;
            }
            let users = rewriter
                .users_of(value)
                .iter()
                .copied()
                .filter(|user| !matched_insts.contains(user))
                .collect::<Vec<_>>();
            if !users.is_empty() {
                external_uses.push(ExternalUse { value, users });
            }
        }

        ClosureReport { external_uses }
    }
}

#[derive(Debug, Clone)]
pub struct ExternalUse {
    pub value: ValueId,
    pub users: Vec<InstId>,
}

#[derive(Debug, Clone)]
pub struct ClosureReport {
    pub external_uses: Vec<ExternalUse>,
}

impl ClosureReport {
    pub fn is_closed(&self) -> bool {
        self.external_uses.is_empty()
    }
}

pub struct PatternTemplateBuilder {
    nodes: Vec<TemplateNode>,
}

impl PatternTemplateBuilder {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn node(
        &mut self,
        op: OperationKey,
        operands: impl Into<Vec<TemplateOperand>>,
    ) -> TemplateNodeId {
        let id = TemplateNodeId(self.nodes.len() as u32);
        self.nodes.push(TemplateNode {
            op,
            operands: operands.into(),
        });
        id
    }

    pub fn finish(self, anchor: TemplateNodeId, output: TemplateValueRef) -> PatternTemplate {
        let input_count = compute_input_count(&self.nodes, output);
        let users = compute_users(&self.nodes);
        PatternTemplate {
            nodes: self.nodes,
            users,
            anchor,
            output,
            input_count,
        }
    }
}

fn compute_input_count(nodes: &[TemplateNode], output: TemplateValueRef) -> usize {
    let mut max_input: Option<u32> = None;

    let mut consider_ref = |value_ref: TemplateValueRef| {
        if let TemplateValueRef::Input(index) = value_ref {
            max_input = Some(max_input.map_or(index, |prev| prev.max(index)));
        }
    };

    for node in nodes {
        for operand in &node.operands {
            match operand {
                TemplateOperand::Value(value) => consider_ref(*value),
                TemplateOperand::TupleElement { tuple, .. } => consider_ref(*tuple),
                TemplateOperand::Literal => {}
            }
        }
    }

    consider_ref(output);
    max_input.map_or(0, |value| value.saturating_add(1) as usize)
}

fn compute_users(nodes: &[TemplateNode]) -> Vec<Vec<TemplateUse>> {
    let mut users = vec![Vec::new(); nodes.len()];

    for (user_index, node) in nodes.iter().enumerate() {
        for (operand_index, operand) in node.operands.iter().enumerate() {
            let value_ref = match operand {
                TemplateOperand::Value(value) => *value,
                TemplateOperand::TupleElement { tuple, .. } => *tuple,
                TemplateOperand::Literal => continue,
            };
            if let TemplateValueRef::Node(def) = value_ref {
                if let Some(list) = users.get_mut(def.index()) {
                    list.push(TemplateUse {
                        user: TemplateNodeId(user_index as u32),
                        operand_index,
                    });
                }
            }
        }
    }

    users
}

impl fmt::Display for OperationKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
