//! Outlining of per-op C code into shared static functions.
//!
//! Each op of the entry function is emitted with its buffer variables renamed to positional
//! parameters, so ops that differ only in their buffers, such as the same op in every decoder
//! layer, share one `static` function. Compile time then scales with the number of distinct ops,
//! not the number of layers; fully inlined, it grows superlinearly with model depth.

use std::collections::HashMap;

use gpt_rs::backend::conversion::{ConversionError, ConversionResult};
use gpt_rs::backend::spec::{Instruction, Operand, ValueId};

use super::types::{ValueInfo, ValueKey};
use super::utils::push_block;

pub(super) struct Outliner {
    param_types: HashMap<String, String>,
    keys_by_value: HashMap<ValueId, Vec<ValueKey>>,
    definitions: String,
    by_body: HashMap<String, String>,
}

/// Value infos of one instruction with its buffer variables renamed to parameters.
pub(super) struct RenamedOperands {
    pub(super) value_infos: HashMap<ValueKey, ValueInfo>,
    /// Original variable of each parameter, in parameter order.
    args: Vec<String>,
}

fn param_name(index: usize) -> String {
    format!("gpt_rs_p{index}")
}

impl Outliner {
    pub(super) fn new(
        param_types: HashMap<String, String>,
        value_infos: &HashMap<ValueKey, ValueInfo>,
    ) -> Self {
        let mut keys_by_value: HashMap<ValueId, Vec<ValueKey>> = HashMap::new();
        for key in value_infos.keys() {
            keys_by_value
                .entry(key.value)
                .or_default()
                .push(key.clone());
        }
        for keys in keys_by_value.values_mut() {
            keys.sort_by(|a, b| a.path.cmp(&b.path));
        }
        Self {
            param_types,
            keys_by_value,
            definitions: String::new(),
            by_body: HashMap::new(),
        }
    }

    /// Copies the value infos of `inst`'s operands and results, and renames each distinct buffer
    /// variable to the next parameter.
    pub(super) fn rename(
        &self,
        inst: &Instruction,
        value_infos: &HashMap<ValueKey, ValueInfo>,
    ) -> ConversionResult<RenamedOperands> {
        let values = inst
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(value) | Operand::TupleElement { tuple: value, .. } => Some(*value),
                Operand::Literal(_) => None,
            })
            .chain(std::iter::once(inst.id));
        let mut renamed = HashMap::new();
        let mut args: Vec<String> = Vec::new();
        for value in values {
            for key in self.keys_by_value.get(&value).into_iter().flatten() {
                let mut info = value_infos[key].clone();
                if !self.param_types.contains_key(&info.var) {
                    return Err(ConversionError::new(format!(
                        "buffer variable '{}' is not declared in the entry function",
                        info.var
                    )));
                }
                let index = match args.iter().position(|arg| *arg == info.var) {
                    Some(index) => index,
                    None => {
                        args.push(info.var.clone());
                        args.len() - 1
                    }
                };
                info.var = param_name(index);
                renamed.insert(key.clone(), info);
            }
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
