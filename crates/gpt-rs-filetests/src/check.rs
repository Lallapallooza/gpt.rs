//! CHECK lines and their match against a pass's output.

use std::collections::HashMap;

use regex::Regex;

fn pattern_regex(
    pattern: &str,
    vars: &HashMap<String, String>,
) -> Result<(Regex, Vec<String>), String> {
    let token =
        Regex::new(r"\[\[([A-Za-z_][A-Za-z0-9_]*)(?::(.*?))?\]\]|\{\{(.*?)\}\}").expect("valid");
    let mut out = String::new();
    let mut captures = Vec::new();
    let mut last = 0;
    for m in token.captures_iter(pattern) {
        let whole = m.get(0).expect("match");
        out.push_str(&regex::escape(&pattern[last..whole.start()]));
        last = whole.end();
        if let Some(re) = m.get(3) {
            out.push_str(&format!("(?:{})", re.as_str()));
        } else {
            let name = m[1].to_string();
            match m.get(2) {
                Some(re) => {
                    out.push_str(&format!("(?P<{name}>{})", re.as_str()));
                    captures.push(name);
                }
                None => {
                    let value = vars
                        .get(&name)
                        .ok_or_else(|| format!("[[{name}]] is used before it is captured"))?;
                    out.push_str(&regex::escape(value));
                }
            }
        }
    }
    out.push_str(&regex::escape(&pattern[last..]));
    Regex::new(&out)
        .map(|re| (re, captures))
        .map_err(|err| format!("bad CHECK pattern `{pattern}`: {err}"))
}

/// The case's CHECK lines as (directive, pattern).
pub fn check_lines(case: &str) -> Result<Vec<(&str, &str)>, String> {
    let mut checks = Vec::new();
    for line in case.lines().map(str::trim) {
        let Some(comment) = line.strip_prefix("//") else {
            continue;
        };
        if !comment.contains("CHECK") && !comment.trim_start().starts_with("check") {
            continue;
        }
        let parsed = comment.strip_prefix(' ').and_then(|c| c.split_once(':'));
        let Some((kind @ ("CHECK" | "CHECK-NEXT" | "CHECK-NOT"), pattern)) = parsed else {
            return Err(format!("malformed CHECK line `{line}`"));
        };
        checks.push((kind, pattern.trim()));
    }
    Ok(checks)
}

/// Checks `output` against `checks`.
pub fn file_check(checks: &[(&str, &str)], output: &str) -> Result<(), String> {
    let lines: Vec<&str> = output.lines().collect();
    let mut vars: HashMap<String, String> = HashMap::new();
    let mut cursor = 0usize; // the next line that may match
    let mut pending_not: Vec<&str> = Vec::new();
    let not_in =
        |pending: &[&str], range: &[&str], vars: &HashMap<String, String>| -> Result<(), String> {
            for pattern in pending {
                let (re, _) = pattern_regex(pattern, vars)?;
                if let Some(line) = range.iter().find(|l| re.is_match(l)) {
                    return Err(format!("CHECK-NOT: `{pattern}` matched `{line}`"));
                }
            }
            Ok(())
        };
    for &(kind, pattern) in checks {
        if kind == "CHECK-NOT" {
            pending_not.push(pattern);
            continue;
        }
        let (re, captures) = pattern_regex(pattern, &vars)?;
        let found = if kind == "CHECK-NEXT" {
            let index = cursor;
            lines
                .get(index)
                .and_then(|l| re.captures(l))
                .map(|c| (index, c))
                .ok_or_else(|| {
                    format!(
                        "CHECK-NEXT: `{pattern}` does not match the next line `{}`",
                        lines.get(index).unwrap_or(&"<end of output>")
                    )
                })?
        } else {
            (cursor..lines.len())
                .find_map(|i| re.captures(lines[i]).map(|c| (i, c)))
                .ok_or_else(|| format!("CHECK: `{pattern}` not found"))?
        };
        let (index, caps) = found;
        not_in(&pending_not, &lines[cursor..index], &vars)?;
        pending_not.clear();
        for name in captures {
            vars.insert(name.clone(), caps[name.as_str()].to_string());
        }
        cursor = index + 1;
    }
    not_in(&pending_not, &lines[cursor.min(lines.len())..], &vars)
}
