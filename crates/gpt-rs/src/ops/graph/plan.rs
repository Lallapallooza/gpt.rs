//! Plan keys and the program cache of lazy graph execution.

use std::any::Any;
use std::hash::{BuildHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use lru::LruCache;
use once_cell::sync::Lazy;

use gpt_rs_ir::module::Module;
use gpt_rs_ir::types::TensorType;
use gpt_rs_ir::Op;

use crate::ops::graph::ValueId;
use crate::tensor::InputRole;

/// Number of compiled programs the cache keeps before it evicts the least recently used one.
const PROGRAM_CACHE_CAPACITY: usize = 64;

/// Compiled programs of every arena, by plan key. A hit skips the pass pipeline and the backend
/// compile, which every decode step would otherwise repeat in its fresh arena.
static PROGRAM_CACHE: Lazy<Mutex<LruCache<PlanKey, CachedProgram>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(PROGRAM_CACHE_CAPACITY).expect("a nonzero capacity"),
    ))
});

/// A hash of everything a compiled program depends on: the backend, the graph structure (input
/// roles, ops with their attributes and literals, operand topology), and the types and parameter
/// ids it is specialized to. The values that a call requests are not part of it, because
/// programs bind outputs by node rank.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct PlanKey(pub(super) u64);

impl PlanKey {
    /// Computes the key of `graph` in one pass, without allocating.
    pub(super) fn new<'n, F>(backend: &str, graph: &PlanGraph<'_, F>) -> Result<Self>
    where
        F: Fn(ValueId) -> Option<PlanNodeView<'n>>,
    {
        debug_assert!(graph.nodes.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(graph
            .inputs
            .windows(2)
            .all(|pair| pair[0].value.0 < pair[1].value.0));

        let mut hasher = key_hasher();
        backend.hash(&mut hasher);
        hasher.write_usize(graph.inputs.len());
        for input in graph.inputs {
            input.role.hash(&mut hasher);
            input.stable_id.hash(&mut hasher);
            input.ty.hash(&mut hasher);
        }
        hasher.write_usize(graph.nodes.len());
        for &node_value in graph.nodes {
            let node = (graph.node_view)(node_value)
                .ok_or_else(|| anyhow::anyhow!("missing node for value {node_value:?}"))?;
            node.op.hash(&mut hasher);
            node.ty.hash(&mut hasher);
            hasher.write_usize(node.result);
            if node.result > 0 {
                write_ref(&mut hasher, graph.value_ref(node.first));
            }
            hasher.write_usize(node.operands.len());
            for &operand in node.operands {
                write_ref(&mut hasher, graph.value_ref(operand));
            }
        }
        Ok(PlanKey(hasher.finish()))
    }
}

/// One input of a plan: the value it binds, its role, its parameter id when it is a parameter,
/// and its type.
#[derive(Clone, Debug)]
pub(super) struct PlanInput {
    pub(super) value: ValueId,
    pub(super) role: InputRole,
    pub(super) stable_id: Option<u128>,
    pub(super) ty: TensorType,
}

/// What binds one parameter of a compiled program: a parameter by its stable id, or a run-time
/// argument by its position among the plan inputs.
#[derive(Clone, Copy, Debug)]
pub(super) enum InputBinding {
    Param(u128),
    Arg(usize),
}

/// A pending node of a plan: its value, op and operands, and which result of the op it is. The
/// results of one op share `first`, the value of result 0.
#[derive(Clone, Debug)]
pub(super) struct PlanNode {
    pub(super) value: ValueId,
    pub(super) op: Op,
    pub(super) operands: Vec<ValueId>,
    pub(super) first: ValueId,
    pub(super) result: usize,
}

/// Borrowed view of a pending node: its op, operands, type and result index.
#[derive(Clone, Copy)]
pub(super) struct PlanNodeView<'a> {
    pub(super) op: &'a Op,
    pub(super) operands: &'a [ValueId],
    pub(super) ty: &'a TensorType,
    pub(super) first: ValueId,
    pub(super) result: usize,
}

/// A compiled program, bound to the values of one arena.
#[derive(Debug)]
pub(super) struct Plan {
    /// The key of the program in the program cache.
    pub(super) key: PlanKey,
    /// The backend's compiled `@captured`, a `B::Executable`.
    pub(super) exe: Arc<dyn Any + Send + Sync>,
    /// The module the program was compiled from.
    pub(super) module: Arc<Module>,
    /// Whether the program came from the program cache.
    pub(super) program_cache_hit: bool,
    /// What binds each argument of the program, in order.
    pub(super) inputs: Vec<InputBinding>,
    /// The values that the run-time arguments of `inputs` bind, in order.
    pub(super) args: Vec<ValueId>,
    /// The values the run asks for: the targets and the exports, sorted by value id.
    pub(super) requested_outputs: Vec<ValueId>,
    /// Values that receive the program results, in result order.
    pub(super) program_outputs: Vec<ValueId>,
    /// The values marked for export, sorted by value id.
    pub(super) exports: Vec<ValueId>,
}

#[derive(Clone)]
pub(super) struct CachedProgram {
    pub(super) exe: Arc<dyn Any + Send + Sync>,
    pub(super) module: Arc<Module>,
    pub(super) inputs: Vec<InputBinding>,
    /// Positions of the program results in the pending nodes of the plan, followed by its
    /// parameters. Both lists are sorted by value id. The program addresses its outputs by these
    /// positions in any graph with the same key.
    pub(super) output_ranks: Vec<usize>,
}

pub(super) fn get_cached_program(key: &PlanKey) -> Option<CachedProgram> {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.get(key).cloned()
}

pub(super) fn insert_cached_program(key: PlanKey, program: CachedProgram) {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.put(key, program);
}

/// Borrowed view of the pending graph that a plan key is computed from.
///
/// `nodes` and `inputs` are sorted by value id. The key canonicalizes each value by its position
/// in those slices, which `refs` holds. Program parameters and results use the same order.
pub(super) struct PlanGraph<'a, F> {
    /// The inputs in parameter order.
    pub(super) inputs: &'a [PlanInput],
    pub(super) nodes: &'a [ValueId],
    /// The place of every value of the arena in the plan, by value id.
    pub(super) refs: &'a [ValueRef],
    /// Looks up a pending node by value id.
    pub(super) node_view: F,
}

/// The place of a value in a [`PlanGraph`]: its position among the pending nodes or the inputs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueRef {
    Node(u32),
    Input(u32),
    /// Neither a pending node nor an input, for example an export that is already materialized.
    External,
}

impl<F> PlanGraph<'_, F> {
    fn value_ref(&self, value: ValueId) -> ValueRef {
        self.refs
            .get(value.0 as usize)
            .copied()
            .unwrap_or(ValueRef::External)
    }
}

const PLAN_KEY_SEED: u64 = 0x5054_4952_706c_616e;

/// Plan-key hasher: foldhash that keeps every write in stream order.
///
/// foldhash buffers integer writes but folds byte slices at once, so an integer write and a
/// following byte write could swap places without a change in the hash. Only `write` is
/// overridden, so integers also go through it as bytes.
struct KeyHasher(foldhash::quality::FoldHasher);

fn key_hasher() -> KeyHasher {
    KeyHasher(foldhash::quality::FixedState::with_seed(PLAN_KEY_SEED).build_hasher())
}

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn write_ref(hasher: &mut KeyHasher, value: ValueRef) {
    match value {
        ValueRef::Node(index) => {
            hasher.write_u8(1);
            hasher.write_usize(index as usize);
        }
        ValueRef::Input(index) => {
            hasher.write_u8(2);
            hasher.write_usize(index as usize);
        }
        ValueRef::External => hasher.write_u8(3),
    }
}
