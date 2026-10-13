//! Outlining of per-op C code into shared static functions.
//!
//! Each op of the entry function is emitted with its buffer variables renamed to positional
//! parameters, so ops that differ only in their buffers, such as the same op in every decoder
//! layer, share one `static` function. Compile time then scales with the number of distinct ops,
//! not the number of layers; fully inlined, it grows superlinearly with model depth.

use std::collections::HashMap;

use anyhow::{ensure, Result};
use gpt_rs_ir::ir::Value;

use super::types::{Instruction, ValueInfo};
use super::utils::push_block;

pub(super) struct Outliner {
    param_types: HashMap<String, String>,
    definitions: String,
    by_body: HashMap<String, String>,
}

/// Value infos of one instruction with its buffer variables renamed to parameters.
pub(super) struct RenamedOperands {
    pub(super) value_infos: HashMap<Value, ValueInfo>,
    /// Original variable of each parameter, in parameter order.
    args: Vec<String>,
}

fn param_name(index: usize) -> String {
    format!("gpt_rs_p{index}")
}

impl Outliner {
    pub(super) fn new(param_types: HashMap<String, String>) -> Self {
        Self {
            param_types,
            definitions: String::new(),
            by_body: HashMap::new(),
        }
    }

    /// Copies the value infos of `inst`'s operands and results, and renames each distinct buffer
    /// variable to the next parameter.
    pub(super) fn rename(
        &self,
        inst: &Instruction<'_>,
        value_infos: &HashMap<Value, ValueInfo>,
    ) -> Result<RenamedOperands> {
        let values = inst.operands.iter().chain(&inst.results).copied();
        let mut renamed = HashMap::new();
        let mut args: Vec<String> = Vec::new();
        for value in values {
            let mut info = value_infos[&value].clone();
            ensure!(
                self.param_types.contains_key(&info.var),
                "buffer variable '{}' is not declared in the entry function",
                info.var
            );
            let index = match args.iter().position(|arg| *arg == info.var) {
                Some(index) => index,
                None => {
                    args.push(info.var.clone());
                    args.len() - 1
                }
            };
            info.var = param_name(index);
            renamed.insert(value, info);
        }
        Ok(RenamedOperands {
            value_infos: renamed,
            args,
        })
    }

    /// Returns the entry-function code for `body`, emitted against `operands`. The code is a call
    /// to a static function, which other ops may share.
    pub(super) fn outline(&mut self, body: &str, operands: &RenamedOperands) -> String {
        if body.trim().is_empty() {
            return String::new();
        }
        let args = &operands.args;
        let params = args
            .iter()
            .enumerate()
            .map(|(index, arg)| format!("{} {}", self.param_types[arg], param_name(index)))
            .collect::<Vec<_>>();
        let params = if params.is_empty() {
            "void".to_string()
        } else {
            params.join(", ")
        };
        let key = format!("{params}\n{body}");
        let name = match self.by_body.get(&key) {
            Some(name) => name.clone(),
            None => {
                let name = format!("gpt_rs_op_{}", self.by_body.len());
                push_block(
                    &mut self.definitions,
                    0,
                    &format!(
                        "static GPTRS_NOINLINE int {name}({params}) {{\n{body}\n  return 0;\n}}\n"
                    ),
                );
                self.by_body.insert(key, name.clone());
                name
            }
        };
        // Ops that validate indices return a nonzero status, which the entry function forwards.
        format!(
            "{{ const int gpt_rs_rc = {name}({}); if (gpt_rs_rc != 0) {{ return gpt_rs_rc; }} }}",
            args.join(", ")
        )
    }

    /// Definitions of all outlined functions.
    pub(super) fn finish(self) -> String {
        self.definitions
    }
}
