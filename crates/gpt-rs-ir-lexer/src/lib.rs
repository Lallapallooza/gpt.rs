//! The tokens of the `gpt-rs-ir` text form. The parser of `gpt-rs-ir` reads its input as these
//! tokens, and `gpt-rs-ir-macros` checks that each literal of an attribute's format is one of
//! them, so a format can only name tokens the parser reads.
//!
//! Whitespace and `//` comments separate tokens, and the lexer takes the longest token that
//! starts at a position: `->` rather than `-`, `<=` rather than `<`. Signs are tokens of their
//! own: `-3` is `-` and then `3`. A sigil followed by a name character starts a name, so `%`
//! the operator is followed by a space: `% 16`, where `%16` is a value.
//!
//! A shape such as `4x{T}xf32` lexes as `4`, then the identifier `x`, `{`, `T`, `}` and the
//! identifier `xf32`: the parser of a shape splits the `x` off the identifier after each extent.

/// What a token is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// A letter or `_`, then letters, digits, `_` and single `.`s: `tensor.add`, `f32`, `d0`,
    /// `none`. A name ends before `..`, so `d0..d3` is three tokens.
    Ident,
    /// `%` and a name of letters, digits, `_` and `.`: a value, `%x` or `%0`.
    Value,
    /// `^` and a name: the label of a region, `^body`.
    Label,
    /// `@` and a name: a symbol, `@main`.
    Symbol,
    /// `#` and a name, `#name`, which a declared attribute format may take as a literal.
    Hash,
    /// `!` and a name: a type of a dialect, `!loop.buf`.
    Bang,
    /// `$` and digits: an operand by position, `$2`.
    Operand,
    /// Digits: `42`.
    Int,
    /// Digits with a fraction, an exponent or both: `1.5`, `2e-3`, `1.0e10`.
    Float,
    /// A string in double quotes on one line, quotes included, in which `\` escapes the next
    /// character. The parser reads the escapes `\"`, `\\` and `\n`.
    String,
    /// One of [`PUNCTUATION`].
    Punct,
    /// A character that starts no token, or a string without its closing quote on its line, up
    /// to the end of the line.
    Invalid,
    /// The end of the input.
    End,
}

/// One token of a text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    /// Its text, as written.
    pub text: &'a str,
    /// The byte offset of its first character.
    pub start: usize,
}

impl Token<'_> {
    /// The byte offset after its last character.
    pub fn end(&self) -> usize {
        self.start + self.text.len()
    }
}

/// The punctuation tokens, each before the shorter ones it starts with.
pub const PUNCTUATION: [&str; 21] = [
    "->", "..", "==", "!=", "<=", ">=", "(", ")", "[", "]", "{", "}", "<", ">", ",", ":", "=", "+",
    "-", "*", "%",
];

/// Whether `character` may continue an identifier or the name after a sigil.
pub fn is_name_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_' || character == '.'
}

/// Whether `character` may start an identifier.
pub fn is_ident_start(character: char) -> bool {
    character.is_ascii_alphabetic() || character == '_'
}

/// The position of the first character at or after `pos` that is neither whitespace nor in a
/// comment.
pub fn skip_trivia(src: &str, mut pos: usize) -> usize {
    loop {
        let rest = &src[pos..];
        let trimmed = rest.trim_start();
        pos += rest.len() - trimmed.len();
        if !trimmed.starts_with("//") {
            return pos;
        }
        pos += trimmed.find('\n').unwrap_or(trimmed.len());
    }
}

/// The token at the first position at or after `pos` that is neither whitespace nor in a comment.
pub fn token(src: &str, pos: usize) -> Token<'_> {
    let start = skip_trivia(src, pos);
    let rest = &src[start..];
    let (kind, len) = match rest.chars().next() {
        None => (TokenKind::End, 0),
        Some(first) => lex(rest, first),
    };
    Token {
        kind,
        text: &rest[..len],
        start,
    }
}

/// The kind and length of the token at the start of `rest`, which starts with `first`.
fn lex(rest: &str, first: char) -> (TokenKind, usize) {
    if is_ident_start(first) {
        return (TokenKind::Ident, name_len(rest));
    }
    if first.is_ascii_digit() {
        return number(rest);
    }
    if first == '"' {
        return string(rest);
    }
    let sigil = match first {
        '%' => Some(TokenKind::Value),
        '^' => Some(TokenKind::Label),
        '@' => Some(TokenKind::Symbol),
        '#' => Some(TokenKind::Hash),
        '!' => Some(TokenKind::Bang),
        _ => None,
    };
    if let Some(kind) = sigil {
        let name = name_len(&rest[1..]);
        // `!` and a name is a type, but `!=` is punctuation, and so is `%` before a space.
        let starts_ident = rest[1..].starts_with(is_ident_start);
        let named = match kind {
            TokenKind::Bang => starts_ident,
            _ => name > 0,
        };
        if named {
            return (kind, 1 + name);
        }
    }
    if first == '$' {
        let digits = rest[1..]
            .find(|character: char| !character.is_ascii_digit())
            .unwrap_or(rest.len() - 1);
        if digits > 0 {
            return (TokenKind::Operand, 1 + digits);
        }
    }
    match PUNCTUATION.iter().find(|punct| rest.starts_with(**punct)) {
        Some(punct) => (TokenKind::Punct, punct.len()),
        None => (TokenKind::Invalid, first.len_utf8()),
    }
}

/// The length of the name at the start of `rest`, which ends before `..`.
fn name_len(rest: &str) -> usize {
    let end = rest
        .find(|character| !is_name_char(character))
        .unwrap_or(rest.len());
    rest[..end].find("..").unwrap_or(end)
}

/// The number at the start of `rest`: digits, then a fraction when a `.` follows that does not
/// start `..`, then an exponent when `e` or `E` and digits follow, with a sign or without.
fn number(rest: &str) -> (TokenKind, usize) {
    let bytes = rest.as_bytes();
    let digits_from = |from: usize| {
        bytes[from..]
            .iter()
            .position(|byte| !byte.is_ascii_digit())
            .map_or(bytes.len(), |len| from + len)
    };
    let mut end = digits_from(0);
    let mut kind = TokenKind::Int;
    if bytes.get(end) == Some(&b'.') && bytes.get(end + 1) != Some(&b'.') {
        end = digits_from(end + 1);
        kind = TokenKind::Float;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let sign = usize::from(matches!(bytes.get(end + 1), Some(b'+' | b'-')));
        let exponent = digits_from(end + 1 + sign);
        if exponent > end + 1 + sign {
            end = exponent;
            kind = TokenKind::Float;
        }
    }
    (kind, end)
}

/// The string at the start of `rest`, up to its closing quote, or an invalid token up to the end
/// of the line when the line has none.
fn string(rest: &str) -> (TokenKind, usize) {
    let mut chars = rest.char_indices().skip(1);
    while let Some((index, character)) = chars.next() {
        match character {
            '"' => return (TokenKind::String, index + 1),
            '\n' => return (TokenKind::Invalid, index),
            '\\' => {
                chars.next();
            }
            _ => {}
        }
    }
    (TokenKind::Invalid, rest.len())
}
