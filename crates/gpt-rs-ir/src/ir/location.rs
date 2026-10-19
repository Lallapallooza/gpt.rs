//! [`Location`]: where an op comes from, for messages, statistics and dumps. Locations never
//! change what an op computes: passes keep them, and CSE and the program caches ignore them.
//!
//! An op that a pass makes from another, as a rewrite or a lowering does, has that op's location
//! with one more [`Step`] on its trail: the pass, and the name and number of the op it was made
//! from. The number is the op's place in the module before the pass, as dumps print it
//! (`// #47`), so a trail leads from an op of a lowered program back through the dump of each
//! pass to the line of code that recorded it.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::ir::{AsInst, Inst, Module};

/// Where an op comes from: the module path of the layer that recorded it, such as
/// `model.layers.3.mlp`, the line of code that recorded it, and the passes that made it from the
/// op recorded there. Any may be missing.
#[derive(crate::Attr, Debug, Clone, Default, PartialEq, Eq, Hash)]
#[attr(via = LocationText)]
pub struct Location {
    /// The module path of the recording layer.
    pub scope: Option<Arc<str>>,
    /// The file, line and column of the recording code.
    pub source: Option<SourcePosition>,
    /// The passes that made the op, the first first.
    pub trail: Arc<[Step]>,
}

/// A pass that made an op from another: `"tile-to-loop"("tile.block":12)`, the pass, and the name
/// and number of the op it was made from in the module before the pass.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$pass `(` $op `` `:` `` $number `)`")]
pub struct Step {
    pub pass: Arc<str>,
    pub op: Arc<str>,
    pub number: u32,
}

/// A file, line and column: `"mlp.rs":42:9`.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "$file `` `:` `` $line `` `:` `` $column")]
pub struct SourcePosition {
    pub file: Arc<str>,
    pub line: u32,
    pub column: u32,
}

impl Location {
    /// The location of the code that called the `#[track_caller]` function this runs in, in the
    /// layer `scope`.
    #[track_caller]
    pub fn caller(scope: Option<Arc<str>>) -> Self {
        let caller = std::panic::Location::caller();
        Self {
            scope,
            source: Some(SourcePosition {
                file: Arc::from(caller.file()),
                line: caller.line(),
                column: caller.column(),
            }),
            trail: Arc::from([]),
        }
    }

    /// This location with `step` last on its trail.
    pub fn then(&self, step: Step) -> Self {
        Self {
            trail: [&self.trail[..], &[step]].concat().into(),
            ..self.clone()
        }
    }
}

/// The text form of a location: `loc("scope"("file":line:column))`, `loc("scope")` or
/// `loc("file":line:column)`. The trail, if any, comes before the closing parenthesis:
/// `loc("model.mlp"("mlp.rs":42:9) via [step, ...])`. The string after `loc(` is the scope, unless
/// a line and a column follow it, which make it the file of a location without a scope. It is
/// empty for a location with neither.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`loc` `(` $first (`` $rest^)? (`via` $trail^)? `)`")]
struct LocationText {
    first: Arc<str>,
    rest: Option<LocationRest>,
    #[attr(default)]
    trail: Vec<Step>,
}

/// What follows the first string of a location.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
enum LocationRest {
    /// The source of a location with a scope.
    #[attr(format = "`(` $0 `)`")]
    Source(SourcePosition),
    /// The line and the column of a location without a scope.
    #[attr(format = "`:` `` $line `` `:` `` $column")]
    Position { line: u32, column: u32 },
}

impl From<&Location> for LocationText {
    fn from(location: &Location) -> Self {
        let trail = location.trail.to_vec();
        match (&location.scope, &location.source) {
            (Some(scope), source) => LocationText {
                first: scope.clone(),
                rest: source.clone().map(LocationRest::Source),
                trail,
            },
            (None, Some(source)) => LocationText {
                first: source.file.clone(),
                rest: Some(LocationRest::Position {
                    line: source.line,
                    column: source.column,
                }),
                trail,
            },
            (None, None) => LocationText {
                first: Arc::from(""),
                rest: None,
                trail,
            },
        }
    }
}

impl TryFrom<LocationText> for Location {
    type Error = anyhow::Error;

    fn try_from(text: LocationText) -> anyhow::Result<Self> {
        let trail: Arc<[Step]> = text.trail.into();
        Ok(match text.rest {
            None if text.first.is_empty() => Location {
                scope: None,
                source: None,
                trail,
            },
            None => Location {
                scope: Some(text.first),
                source: None,
                trail,
            },
            Some(LocationRest::Source(source)) => Location {
                scope: Some(text.first),
                source: Some(source),
                trail,
            },
            Some(LocationRest::Position { line, column }) => Location {
                scope: None,
                source: Some(SourcePosition {
                    file: text.first,
                    line,
                    column,
                }),
                trail,
            },
        })
    }
}

/// The pass running on a module: its name, and the number of each op of the module when the pass
/// started. The module numbers its ops once, when a trail first needs the numbers or before an
/// edit first moves an op, so a pass that records no trail and moves nothing numbers nothing.
#[derive(Debug, Default)]
pub(crate) struct RunningPass(Option<PassTrail>);

#[derive(Debug)]
struct PassTrail {
    name: Arc<str>,
    numbers: OnceLock<HashMap<Inst, u32>>,
}

/// A copy of a module is not the module a pass runs on, so it records no trail.
impl Clone for RunningPass {
    fn clone(&self) -> Self {
        RunningPass(None)
    }
}

impl Module {
    /// Marks the pass `name` as running on the module, so the ops it makes record it on their
    /// trails ([`Module::made_from`]), and gives the pass that ran before, which
    /// [`Module::finish_pass`] restores.
    pub(crate) fn start_pass(&mut self, name: &str) -> RunningPass {
        let trail = PassTrail {
            name: Arc::from(name),
            numbers: OnceLock::new(),
        };
        std::mem::replace(&mut self.running, RunningPass(Some(trail)))
    }

    /// Ends the running pass, and restores `previous`, the pass that [`Module::start_pass`] gave.
    pub(crate) fn finish_pass(&mut self, previous: RunningPass) {
        self.running = previous;
    }

    /// Numbers the ops for the trails of the running pass, unless the module numbered them
    /// already. Called before an edit moves an op, so that the numbers are those of the module
    /// when the pass started.
    pub(super) fn number_for_trails(&self) {
        if let Some(trail) = &self.running.0 {
            trail.numbers.get_or_init(|| self.op_numbers());
        }
    }

    /// The location of an op that the running pass makes from `op`: the location of `op`, with
    /// the pass, its op and its number last on the trail. Outside a pass, or for an op the module
    /// did not hold when the pass started, the location of `op`.
    pub(crate) fn made_from(&self, op: impl AsInst) -> Option<Location> {
        let inst = op.inst();
        let location = self.location(inst);
        let Some(trail) = &self.running.0 else {
            return location.cloned();
        };
        let numbers = trail.numbers.get_or_init(|| self.op_numbers());
        let Some(&number) = numbers.get(&inst) else {
            return location.cloned();
        };
        let step = Step {
            pass: Arc::clone(&trail.name),
            op: Arc::from(self.op(inst).name().as_ref()),
            number,
        };
        Some(location.cloned().unwrap_or_default().then(step))
    }

    /// The number of each op: its place in the module, in the order the text form prints the
    /// ops, from 0.
    pub fn op_numbers(&self) -> HashMap<Inst, u32> {
        let mut numbers = HashMap::new();
        for (number, inst) in self.walk(self.body()).into_iter().enumerate() {
            numbers.insert(inst, number as u32);
        }
        numbers
    }
}
