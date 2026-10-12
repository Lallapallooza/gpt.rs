//! The reader of tokens that the module parser and the text forms of attributes share.

use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use gpt_rs_ir_lexer::{token, Token, TokenKind};

/// A position in a text, read one token at a time.
pub struct Cursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    /// The next token, which stays unread.
    pub fn peek(&self) -> Token<'a> {
        token(self.src, self.pos)
    }

    /// The token after the next one, which stays unread too.
    pub fn peek_second(&self) -> Token<'a> {
        token(self.src, self.peek().end())
    }

    /// The next token after a `-`, if one comes first: what a signed number starts with.
    pub fn peek_unsigned(&self) -> Token<'a> {
        let next = self.peek();
        if next.text == "-" {
            self.peek_second()
        } else {
            next
        }
    }

    /// Whether the next token is `text`.
    pub fn peek_is(&self, text: &str) -> bool {
        let next = self.peek();
        next.text == text && next.kind != TokenKind::String
    }

    /// Whether the next token is of kind `kind`.
    pub fn peek_kind(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    pub fn at_end(&self) -> bool {
        self.peek_kind(TokenKind::End)
    }

    /// Reads the next token.
    pub fn next_token(&mut self) -> Token<'a> {
        let next = self.peek();
        self.pos = next.end();
        next
    }

    /// The byte offset of the next token.
    pub fn position(&self) -> usize {
        self.peek().start
    }

    /// Reads the next token if it is `text`.
    pub fn eat(&mut self, text: &str) -> bool {
        if !self.peek_is(text) {
            return false;
        }
        self.next_token();
        true
    }

    /// Reads the next token, which must be `text`.
    pub fn expect(&mut self, text: &str) -> Result<()> {
        if self.eat(text) {
            Ok(())
        } else {
            Err(self.unexpected(format!("`{text}`")))
        }
    }

    /// Reads the next token, which must be of kind `kind`; `what` names it in the error.
    pub fn token(&mut self, kind: TokenKind, what: &str) -> Result<&'a str> {
        if !self.peek_kind(kind) {
            return Err(self.unexpected(what));
        }
        Ok(self.next_token().text)
    }

    /// An identifier: `tensor.dot_general`, `f32`, `none`.
    pub fn ident(&mut self) -> Result<&'a str> {
        self.token(TokenKind::Ident, "a name")
    }

    /// The name after the sigil of a token of kind `kind`: `x` of `%x`, `body` of `^body`.
    pub fn sigil_name(&mut self, kind: TokenKind, what: &str) -> Result<&'a str> {
        Ok(&self.token(kind, what)?[1..])
    }

    /// The text of an integer, its sign included: `-3`.
    pub fn int_text(&mut self) -> Result<String> {
        if self.peek_unsigned().kind != TokenKind::Int {
            return Err(self.unexpected("an integer"));
        }
        let sign = if self.eat("-") { "-" } else { "" };
        Ok(format!("{sign}{}", self.next_token().text))
    }

    /// Whether a number comes next: an integer, a float, `inf` or `NaN`, with a sign or without.
    pub fn peek_number(&self) -> bool {
        let next = self.peek_unsigned();
        matches!(next.kind, TokenKind::Int | TokenKind::Float)
            || (next.kind == TokenKind::Ident && matches!(next.text, "inf" | "NaN"))
    }

    /// Whether a float comes next, written as one: with a fraction or an exponent, or `inf` or
    /// `NaN`, with a sign or without.
    pub fn peek_float(&self) -> bool {
        self.peek_number() && self.peek_unsigned().kind != TokenKind::Int
    }

    /// The text of the next number, its sign included: `-1.5`, `3`, `-inf`, `NaN`.
    pub fn number(&mut self) -> Result<String> {
        if !self.peek_number() {
            return Err(self.unexpected("a number"));
        }
        let sign = if self.eat("-") { "-" } else { "" };
        Ok(format!("{sign}{}", self.next_token().text))
    }

    /// A float, written as a float or an integer. The decimal is rounded once, to `F`.
    pub fn float<F: FromStr>(&mut self) -> Result<F> {
        let start = self.position();
        let text = self.number()?;
        text.parse()
            .map_err(|_| self.error_at(start, format!("`{text}` is not a float")))
    }

    /// A string, without its quotes and with its escapes, `\"`, `\\` and `\n`, replaced.
    pub fn string(&mut self) -> Result<String> {
        let start = self.position();
        let quoted = self.token(TokenKind::String, "a string")?;
        let mut out = String::new();
        let mut chars = quoted[1..quoted.len() - 1].chars();
        while let Some(character) = chars.next() {
            if character != '\\' {
                out.push(character);
                continue;
            }
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                _ => return Err(self.error_at(start, "bad escape in string")),
            }
        }
        Ok(out)
    }

    /// The `x` after an extent of a shape, which touches the extent and what follows it: the
    /// identifier after the extent starts with `x`, and only the `x` is read.
    pub fn shape_x(&mut self) -> Result<()> {
        let next = self.peek();
        if next.kind != TokenKind::Ident || !next.text.starts_with('x') {
            return Err(self.unexpected("`x` after an extent"));
        }
        let touches_after = next.text.len() > 1 || token(self.src, next.end()).start == next.end();
        if next.start != self.pos || !touches_after {
            return Err(self.error("a shape is written without spaces, as `4x{T}xf32`"));
        }
        self.pos = next.start + 1;
        Ok(())
    }

    /// Words joined by `-` with no space between, as pass names are: `outline-static`.
    pub fn hyphenated(&mut self) -> Result<String> {
        let mut name = self.ident()?.to_string();
        loop {
            let dash = self.peek();
            let word = self.peek_second();
            let joined = dash.text == "-"
                && dash.start == self.pos
                && word.kind == TokenKind::Ident
                && word.start == dash.end();
            if !joined {
                return Ok(name);
            }
            self.next_token();
            name.push('-');
            name.push_str(self.next_token().text);
        }
    }

    /// Items separated by `,`, up to and including `close`.
    pub fn list<T>(
        &mut self,
        close: &str,
        mut item: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut items = Vec::new();
        if self.eat(close) {
            return Ok(items);
        }
        loop {
            items.push(item(self)?);
            if self.eat(close) {
                return Ok(items);
            }
            if !self.eat(",") {
                return Err(self.unexpected(format!("`,` or `{close}`")));
            }
        }
    }

    /// The error that `what` was expected where the next token is.
    pub fn unexpected(&self, what: impl fmt::Display) -> anyhow::Error {
        let next = self.peek();
        let found = match next.kind {
            TokenKind::End => "the end".to_string(),
            TokenKind::String => next.text.to_string(),
            TokenKind::Invalid if next.text.starts_with('"') => {
                "a string without its closing quote".to_string()
            }
            _ => format!("`{}`", next.text),
        };
        self.error(format!("expected {what}, found {found}"))
    }

    /// An error at the next token.
    pub fn error(&self, msg: impl fmt::Display) -> anyhow::Error {
        self.error_at(self.position(), msg)
    }

    /// An error at byte offset `pos`, as `line:column: msg`.
    pub fn error_at(&self, pos: usize, msg: impl fmt::Display) -> anyhow::Error {
        let before = &self.src[..pos];
        let line = before.matches('\n').count() + 1;
        let column = before.len() - before.rfind('\n').map_or(0, |newline| newline + 1) + 1;
        anyhow!("{line}:{column}: {msg}")
    }
}
