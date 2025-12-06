use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::backend::pattern::{OperationKey, PatternTemplate, TemplateNodeId, TemplateOperand};
use crate::backend::spec::{Operand, Operation, ValueId};

#[derive(Debug, Clone)]
pub struct BindRecord {
    pub name: &'static str,
    pub value: ValueId,
}

#[derive(Debug, Clone)]
pub enum CapturedOperand {
    Value(ValueId),
    TupleElement { tuple: ValueId, index: usize },
    Literal,
}

#[derive(Debug, Clone)]
pub struct CapturedNode {
    pub value: ValueId,
    pub op: OperationKey,
    pub operands: Vec<CapturedOperand>,
}

struct CaptureCallState {
    nodes: Vec<CapturedNode>,
    binds: Vec<BindRecord>,
}

struct ActivePatternCapture {
    record: fn(u32, &[CapturedNode], &[BindRecord], &dyn Any),
    is_site_captured: fn(u32) -> bool,
}

thread_local! {
    static ACTIVE_PATTERN_STACK: RefCell<Vec<ActivePatternCapture>> = const { RefCell::new(Vec::new()) };
    static CAPTURE_SITE_STACK: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    static CAPTURE_CALL_STACK: RefCell<Vec<CaptureCallState>> = const { RefCell::new(Vec::new()) };
}

/// RAII guard that enables template capture for a `#[ptir_pattern]`-annotated function.
pub struct PatternCaptureGuard;

impl PatternCaptureGuard {
    pub fn push(
        record: fn(u32, &[CapturedNode], &[BindRecord], &dyn Any),
        is_site_captured: fn(u32) -> bool,
    ) -> Self {
        ACTIVE_PATTERN_STACK.with(|stack| {
            stack.borrow_mut().push(ActivePatternCapture {
                record,
                is_site_captured,
            })
        });
        PatternCaptureGuard
    }
}

impl Drop for PatternCaptureGuard {
    fn drop(&mut self) {
        ACTIVE_PATTERN_STACK.with(|stack| {
            let _ = stack
                .borrow_mut()
                .pop()
                .expect("pattern capture stack underflow: unmatched pop");
        });
    }
}

/// RAII guard that sets a stable capture-site id for a `capture_ptir!` invocation.
pub struct PatternCaptureSiteGuard;

impl PatternCaptureSiteGuard {
    pub fn push(site: u32) -> Self {
        CAPTURE_SITE_STACK.with(|stack| stack.borrow_mut().push(site));
        PatternCaptureSiteGuard
    }
}

impl Drop for PatternCaptureSiteGuard {
    fn drop(&mut self) {
        CAPTURE_SITE_STACK.with(|stack| {
            let _ = stack
                .borrow_mut()
                .pop()
                .expect("pattern capture site stack underflow: unmatched pop");
        });
    }
}

/// RAII guard that brackets a single `capture_ptir!` evaluation and collects emitted nodes.
pub struct CaptureCallGuard {
    active: bool,
    finished: bool,
}

impl CaptureCallGuard {
    pub fn begin() -> Self {
        let active = ACTIVE_PATTERN_STACK.with(|stack| {
            let stack = stack.borrow();
            stack.last().map(|capture| capture.is_site_captured)
        });
        let active = match (
            active,
            CAPTURE_SITE_STACK.with(|stack| stack.borrow().last().copied()),
        ) {
            (Some(is_site_captured), Some(site)) => !is_site_captured(site),
            _ => false,
        };
        if active {
            CAPTURE_CALL_STACK.with(|stack| {
                stack.borrow_mut().push(CaptureCallState {
                    nodes: Vec::new(),
                    binds: Vec::new(),
                });
            });
        }
        CaptureCallGuard {
            active,
            finished: false,
        }
    }

    pub fn finish(mut self, output: &dyn Any) {
        if !self.active || self.finished {
            return;
        }

        let site = CAPTURE_SITE_STACK.with(|stack| *stack.borrow().last().unwrap());
        let record = ACTIVE_PATTERN_STACK.with(|stack| stack.borrow().last().unwrap().record);

        let state = CAPTURE_CALL_STACK.with(|stack| {
            stack
                .borrow_mut()
                .pop()
                .expect("pattern capture call stack underflow: unmatched finish")
        });

        (record)(site, &state.nodes, &state.binds, output);
        self.finished = true;
    }
}

impl Drop for CaptureCallGuard {
    fn drop(&mut self) {
        if !self.active || self.finished {
            return;
        }
        CAPTURE_CALL_STACK.with(|stack| {
            let _ = stack
                .borrow_mut()
                .pop()
                .expect("pattern capture call stack underflow: unmatched drop");
        });
        self.finished = true;
    }
}

pub(crate) fn record_node(value: ValueId, op: &Operation, operands: &[Operand]) {
    let op_key = OperationKey::from_operation(op);
    let captured_operands = operands
        .iter()
        .map(|operand| match operand {
            Operand::Value(id) => CapturedOperand::Value(*id),
            Operand::TupleElement { tuple, index } => CapturedOperand::TupleElement {
                tuple: *tuple,
                index: *index,
            },
            Operand::Literal(_) => CapturedOperand::Literal,
        })
        .collect::<Vec<_>>();

    CAPTURE_CALL_STACK.with(|stack| {
        if let Some(state) = stack.borrow_mut().last_mut() {
            state.nodes.push(CapturedNode {
                value,
                op: op_key,
                operands: captured_operands,
            });
        }
    });
}

pub(crate) fn record_bind(name: &'static str, value: ValueId) {
    CAPTURE_CALL_STACK.with(|stack| {
        if let Some(state) = stack.borrow_mut().last_mut() {
            state.binds.push(BindRecord { name, value });
        }
    });
}

#[derive(Debug, Clone)]
pub struct BuiltTemplate {
    pub template: PatternTemplate,
    pub value_to_node: HashMap<ValueId, TemplateNodeId>,
}

pub fn build_template(
    nodes: &[CapturedNode],
    output: ValueId,
    anchor: ValueId,
) -> Option<BuiltTemplate> {
    let mut index_of_value: HashMap<ValueId, usize> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        index_of_value.insert(node.value, idx);
    }

    let mut reachable: HashSet<ValueId> = HashSet::new();
    let mut worklist = vec![output];
    while let Some(value) = worklist.pop() {
        let Some(&node_index) = index_of_value.get(&value) else {
            continue;
        };
        if !reachable.insert(value) {
            continue;
        }
        for operand in &nodes[node_index].operands {
            match operand {
                CapturedOperand::Value(id) => worklist.push(*id),
                CapturedOperand::TupleElement { tuple, .. } => worklist.push(*tuple),
                CapturedOperand::Literal => {}
            }
        }
    }

    if !reachable.contains(&anchor) {
        return None;
    }

    let mut builder = PatternTemplate::builder();
    let mut value_to_node: HashMap<ValueId, TemplateNodeId> = HashMap::new();
    let mut input_index_of_value: HashMap<ValueId, u32> = HashMap::new();
    let mut next_input = 0u32;

    let input_ref =
        |value: ValueId, input_index_of_value: &mut HashMap<ValueId, u32>, next_input: &mut u32| {
            let idx = match input_index_of_value.get(&value).copied() {
                Some(idx) => idx,
                None => {
                    let idx = *next_input;
                    *next_input = next_input.saturating_add(1);
                    input_index_of_value.insert(value, idx);
                    idx
                }
            };
            crate::backend::pattern::TemplateValueRef::Input(idx)
        };

    for node in nodes.iter() {
        if !reachable.contains(&node.value) {
            continue;
        }

        let templ_operands = node
            .operands
            .iter()
            .map(|operand| match operand {
                CapturedOperand::Literal => TemplateOperand::Literal,
                CapturedOperand::Value(value) => {
                    if let Some(node_id) = value_to_node.get(value).copied() {
                        TemplateOperand::Value(crate::backend::pattern::TemplateValueRef::Node(
                            node_id,
                        ))
                    } else {
                        TemplateOperand::Value(input_ref(
                            *value,
                            &mut input_index_of_value,
                            &mut next_input,
                        ))
                    }
                }
                CapturedOperand::TupleElement { tuple, index } => {
                    let tuple_ref = if let Some(node_id) = value_to_node.get(tuple).copied() {
                        crate::backend::pattern::TemplateValueRef::Node(node_id)
                    } else {
                        input_ref(*tuple, &mut input_index_of_value, &mut next_input)
                    };
                    TemplateOperand::TupleElement {
                        tuple: tuple_ref,
                        index: *index,
                    }
                }
            })
            .collect::<Vec<_>>();

        let node_id = builder.node(node.op, templ_operands);
        value_to_node.insert(node.value, node_id);
    }

    let anchor_node = value_to_node.get(&anchor).copied()?;

    let output_ref = if let Some(node_id) = value_to_node.get(&output).copied() {
        crate::backend::pattern::TemplateValueRef::Node(node_id)
    } else {
        crate::backend::pattern::TemplateValueRef::Input(
            *input_index_of_value.entry(output).or_insert_with(|| {
                let idx = next_input;
                next_input = next_input.saturating_add(1);
                idx
            }),
        )
    };

    let template = builder.finish(anchor_node, output_ref);
    Some(BuiltTemplate {
        template,
        value_to_node,
    })
}
