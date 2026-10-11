//! [`Location`]: where an op comes from, for messages, statistics and dumps. Locations never
//! change what an op computes: passes keep them, and CSE and the program caches ignore them.

use std::fmt;
use std::sync::Arc;

use crate::attr::print_string;

/// Where an op comes from: the module path of the layer that recorded it, such as
/// `model.layers.3.mlp`, and the line of code that recorded it. Either may be missing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    /// The module path of the recording layer.
    pub scope: Option<Arc<str>>,
    /// The file, line and column of the recording code.
    pub source: Option<SourcePosition>,
}

/// A file, line and column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
        }
    }
}

/// The text form, `loc("scope"("file":line:column))`, `loc("scope")` or `loc("file":line:column)`,
/// with the strings quoted as the text form quotes them.
impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut text = String::from("loc(");
        if let Some(scope) = &self.scope {
            print_string(scope, &mut text);
        }
        if let Some(source) = &self.source {
            let mut position = String::new();
            print_string(&source.file, &mut position);
            position.push_str(&format!(":{}:{}", source.line, source.column));
            match self.scope {
                Some(_) => text.push_str(&format!("({position})")),
                None => text.push_str(&position),
            }
        }
        text.push(')');
        f.write_str(&text)
    }
}
