//! [`Location`]: where an op comes from, for messages, statistics and dumps. Locations never
//! change what an op computes: passes keep them, and CSE and the program caches ignore them.

use std::sync::Arc;

/// Where an op comes from: the module path of the layer that recorded it, such as
/// `model.layers.3.mlp`, and the line of code that recorded it. Either may be missing.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(via = LocationText)]
pub struct Location {
    /// The module path of the recording layer.
    pub scope: Option<Arc<str>>,
    /// The file, line and column of the recording code.
    pub source: Option<SourcePosition>,
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
        }
    }
}

/// The text form of a location: `loc("scope"("file":line:column))`, `loc("scope")` or
/// `loc("file":line:column)`. The string after `loc(` is the scope, unless a line and a column
/// follow it, which make it the file of a location without a scope.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash)]
#[attr(format = "`loc` `(` $first `` ($rest^)? `)`")]
struct LocationText {
    first: Arc<str>,
    rest: Option<LocationRest>,
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
        match (&location.scope, &location.source) {
            (Some(scope), source) => LocationText {
                first: scope.clone(),
                rest: source.clone().map(LocationRest::Source),
            },
            (None, Some(source)) => LocationText {
                first: source.file.clone(),
                rest: Some(LocationRest::Position {
                    line: source.line,
                    column: source.column,
                }),
            },
            // The recorder always knows the source; a location with neither reads as an empty
            // scope.
            (None, None) => LocationText {
                first: Arc::from(""),
                rest: None,
            },
        }
    }
}

impl TryFrom<LocationText> for Location {
    type Error = anyhow::Error;

    fn try_from(text: LocationText) -> anyhow::Result<Self> {
        Ok(match text.rest {
            None => Location {
                scope: Some(text.first),
                source: None,
            },
            Some(LocationRest::Source(source)) => Location {
                scope: Some(text.first),
                source: Some(source),
            },
            Some(LocationRest::Position { line, column }) => Location {
                scope: None,
                source: Some(SourcePosition {
                    file: text.first,
                    line,
                    column,
                }),
            },
        })
    }
}
