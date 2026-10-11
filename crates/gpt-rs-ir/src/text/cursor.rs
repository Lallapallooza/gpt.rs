//! Character-level reader shared by the module parser and the `Attr` parsers.

use std::str::FromStr;

use anyhow::{anyhow, Result};

use crate::attr::Attr;

pub struct Cursor<'a> {
    src: &'a str,
    pos: usize,
    /// Set by the attribute dictionary of an op with more than one optional or variadic operand
    /// group.
    pub(crate) operand_segments: Option<Vec<usize>>,
    /// Set by the attribute dictionary of an op with more than one optional or variadic result
    /// group.
    pub(crate) result_segments: Option<Vec<usize>>,
}

pub(crate) fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

pub(crate) fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

impl<'a> Cursor<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            pos: 0,
            operand_segments: None,
            result_segments: None,
        }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    /// Skips whitespace and `//` comments.
    fn skip_ws(&mut self) {
        loop {
            let rest = self.rest();
            let trimmed = rest.trim_start();
            self.pos += rest.len() - trimmed.len();
            if self.rest().starts_with("//") {
                let line_end = self.rest().find('\n').unwrap_or(self.rest().len());
                self.pos += line_end;
            } else {
                return;
            }
        }
    }

    pub fn at_end(&mut self) -> bool {
        self.skip_ws();
        self.pos == self.src.len()
    }

    pub fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.rest().chars().next()
    }

    /// The character right after the next one, with no whitespace skipped between them.
    pub fn peek_second(&mut self) -> Option<char> {
        self.skip_ws();
        self.rest().chars().nth(1)
    }

    /// True when the next two tokens are `first` and `second`, with only whitespace between.
    pub fn peek_pair(&mut self, first: char, second: char) -> bool {
        self.skip_ws();
        let mut chars = self.rest().chars();
        chars.next() == Some(first) && chars.as_str().trim_start().starts_with(second)
    }

    /// True when an attribute dictionary comes next: `{` then a key and `=`, not `==`.
    pub fn peek_attr_dict(&mut self) -> bool {
        self.skip_ws();
        let Some(inner) = self.rest().strip_prefix('{') else {
            return false;
        };
        let inner = inner.trim_start();
        let end = inner.find(|c| !is_ident_char(c)).unwrap_or(inner.len());
        if end == 0 || !inner.starts_with(is_ident_start) {
            return false;
        }
        let after = inner[end..].trim_start();
        after.starts_with('=') && !after.starts_with("==")
    }

    /// True when the next token starts with `s`.
    pub fn peek_str(&mut self, s: &str) -> bool {
        self.skip_ws();
        self.rest().starts_with(s)
    }

    /// Consumes `s` if it comes next. When `s` ends in an identifier character, it matches only a
    /// whole word.
    pub fn eat(&mut self, s: &str) -> bool {
        self.skip_ws();
        let rest = self.rest();
        if !rest.starts_with(s) {
            return false;
        }
        if s.chars().last().is_some_and(is_ident_char)
            && rest[s.len()..].chars().next().is_some_and(is_ident_char)
        {
            return false;
        }
        self.pos += s.len();
        true
    }

    pub fn expect(&mut self, s: &str) -> Result<()> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(self.error(format!("expected `{s}`")))
        }
    }

    /// An identifier, which may contain dots: `tensor.dot_general`, `f32`, `none`.
    pub fn ident(&mut self) -> Result<&'a str> {
        self.skip_ws();
        let rest = self.rest();
        if !rest.starts_with(is_ident_start) {
            return Err(self.error("expected an identifier"));
        }
        let end = rest.find(|c| !is_ident_char(c)).unwrap_or(rest.len());
        self.pos += end;
        Ok(&rest[..end])
    }

    /// A word of letters, digits, `_` and `-`, such as a pass name: `outline-static`.
    pub fn word(&mut self) -> Result<&'a str> {
        self.skip_ws();
        let rest = self.rest();
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .unwrap_or(rest.len());
        if end == 0 {
            return Err(self.error("expected a name"));
        }
        self.pos += end;
        Ok(&rest[..end])
    }

    /// A name after `sigil`: `%3`, `%x`, `^body`, `@mlp`.
    pub fn sigil_name(&mut self, sigil: char) -> Result<&'a str> {
        self.skip_ws();
        let rest = self.rest();
        if !rest.starts_with(sigil) {
            return Err(self.error(format!("expected `{sigil}`")));
        }
        let body = &rest[sigil.len_utf8()..];
        let end = body.find(|c| !is_ident_char(c)).unwrap_or(body.len());
        if end == 0 {
            return Err(self.error(format!("expected a name after `{sigil}`")));
        }
        self.pos += sigil.len_utf8() + end;
        Ok(&body[..end])
    }

    /// An integer literal, which may be negative.
    pub fn int(&mut self) -> Result<i128> {
        self.skip_ws();
        let rest = self.rest();
        let digits_start = usize::from(rest.starts_with('-'));
        let end = rest[digits_start..]
            .find(|c: char| !c.is_ascii_digit())
            .map_or(rest.len(), |i| i + digits_start);
        // A `.` after the digits makes a float, but `..` ends the integer, as in `$1..$4`.
        let float = rest[end..].starts_with('.') && !rest[end..].starts_with("..");
        if end == digits_start || float {
            return Err(self.error("expected an integer"));
        }
        let value = rest[..end]
            .parse::<i128>()
            .map_err(|err| self.error(format!("bad integer: {err}")))?;
        self.pos += end;
        Ok(value)
    }

    pub fn uint128(&mut self) -> Result<u128> {
        self.skip_ws();
        let rest = self.rest();
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if end == 0 {
            return Err(self.error("expected an unsigned integer"));
        }
        let value = rest[..end]
            .parse::<u128>()
            .map_err(|err| self.error(format!("bad integer: {err}")))?;
        self.pos += end;
        Ok(value)
    }

    /// A float literal: `1.0`, `-2.5e-3`, `inf`, `-inf`, `NaN`. Integers are accepted too. The
    /// decimal is rounded once, to `F`.
    pub fn float<F: FromStr>(&mut self) -> Result<F> {
        self.skip_ws();
        let rest = self.rest();
        let is_word =
            |word: &&str| rest.starts_with(*word) && !rest[word.len()..].starts_with(is_ident_char);
        let end = match ["-inf", "inf", "NaN"].into_iter().find(is_word) {
            Some(word) => word.len(),
            None => rest
                .char_indices()
                .find(|&(i, c)| {
                    !(c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || ((c == '-' || c == '+')
                            && (i == 0 || matches!(&rest[i - 1..i], "e" | "E"))))
                })
                .map_or(rest.len(), |(i, _)| i),
        };
        let value = rest[..end]
            .parse()
            .map_err(|_| self.error("expected a float"))?;
        self.pos += end;
        Ok(value)
    }

    /// The text of the next number token, integer or float, as [`Cursor::float`] reads it.
    pub fn number(&mut self) -> Result<&'a str> {
        self.skip_ws();
        let rest = self.rest();
        let is_word =
            |word: &&str| rest.starts_with(*word) && !rest[word.len()..].starts_with(is_ident_char);
        let end = match ["-inf", "inf", "NaN", "-NaN"].into_iter().find(is_word) {
            Some(word) => word.len(),
            None => rest
                .char_indices()
                .find(|&(i, c)| {
                    !(c.is_ascii_digit()
                        || c == '.'
                        || c == 'e'
                        || c == 'E'
                        || ((c == '-' || c == '+')
                            && (i == 0 || matches!(&rest[i - 1..i], "e" | "E"))))
                })
                .map_or(rest.len(), |(i, _)| i),
        };
        if end == 0 {
            return Err(self.error("expected a number"));
        }
        self.pos += end;
        Ok(&rest[..end])
    }

    /// True when the next number token has a fractional part, an exponent or is `inf`/`NaN`.
    pub fn peek_is_float(&mut self) -> bool {
        self.skip_ws();
        let rest = self.rest();
        let rest = rest.strip_prefix('-').unwrap_or(rest);
        if rest.starts_with("inf") || rest.starts_with("NaN") {
            return true;
        }
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E'))
            .unwrap_or(rest.len());
        rest[..end].contains(['.', 'e', 'E'])
    }

    /// Hex bytes written as `0x0a1b...`, or `None` when the next token is not hex.
    pub fn hex_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        self.skip_ws();
        let Some(body) = self.rest().strip_prefix("0x") else {
            return Ok(None);
        };
        let end = body
            .find(|c: char| !c.is_ascii_hexdigit())
            .unwrap_or(body.len());
        let digits = &body[..end];
        if digits.len() % 2 != 0 {
            return Err(self.error("hex bytes need an even number of digits"));
        }
        let bytes = (0..digits.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).expect("hex digits"))
            .collect();
        self.pos += 2 + end;
        Ok(Some(bytes))
    }

    /// A double-quoted string with `\"`, `\\` and `\n` escapes.
    pub fn string(&mut self) -> Result<String> {
        self.expect("\"")?;
        let mut out = String::new();
        let mut chars = self.rest().char_indices();
        while let Some((i, c)) = chars.next() {
            match c {
                '"' => {
                    self.pos += i + 1;
                    return Ok(out);
                }
                '\\' => match chars.next() {
                    Some((_, '"')) => out.push('"'),
                    Some((_, '\\')) => out.push('\\'),
                    Some((_, 'n')) => out.push('\n'),
                    _ => return Err(self.error("bad escape in string")),
                },
                c => out.push(c),
            }
        }
        Err(self.error("unterminated string"))
    }

    /// Everything before the `>` that closes an already consumed `<`.
    pub fn until_close_angle(&mut self) -> Result<&'a str> {
        let rest = self.rest();
        let mut depth = 0usize;
        for (i, c) in rest.char_indices() {
            match c {
                '<' => depth += 1,
                '>' if depth == 0 => {
                    self.pos += i;
                    return Ok(&rest[..i]);
                }
                '>' => depth -= 1,
                _ => {}
            }
        }
        Err(self.error("unclosed `<`"))
    }

    fn line_col(&self) -> (usize, usize) {
        let before = &self.src[..self.pos];
        let line = before.matches('\n').count() + 1;
        let col = before.len() - before.rfind('\n').map_or(0, |i| i + 1) + 1;
        (line, col)
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
            self.expect(",")?;
        }
    }

    /// Parses the value of `key` in the attribute dictionary of `op` when `key` is
    /// `operand_segments` or `result_segments`, and keeps it for the module parser.
    pub fn segment_attr(&mut self, op: &str, key: &str) -> Result<()> {
        if key != "operand_segments" && key != "result_segments" {
            return Err(self.error(format!("{op} has no attribute `{key}`")));
        }
        let sizes = Vec::<usize>::parse(self)?;
        let slot = if key == "operand_segments" {
            &mut self.operand_segments
        } else {
            &mut self.result_segments
        };
        if slot.replace(sizes).is_some() {
            return Err(self.error(format!("duplicate attribute `{key}`")));
        }
        Ok(())
    }

    pub fn error(&self, msg: impl std::fmt::Display) -> anyhow::Error {
        let (line, col) = self.line_col();
        anyhow!("{line}:{col}: {msg}")
    }
}
