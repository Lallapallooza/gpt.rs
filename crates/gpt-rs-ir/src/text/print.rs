use std::collections::HashMap;
use std::fmt::Write as _;

use crate::module::{Inst, Module, Region, Value};
use crate::op::{OpDyn, SymbolOp};
use crate::traits::IsolatedFromAbove;
use crate::types::{type_list, Type};

struct Printer<'m> {
    module: &'m Module,
    out: String,
    names: HashMap<Value, usize>,
    next: usize,
    /// Whether each op with a location ends with it.
    locations: bool,
}

/// Result types as an op's text writes them: one type alone, several in parentheses.
pub(crate) fn results_text(types: &[&Type]) -> String {
    match types {
        [single] => single.to_string(),
        _ => type_list(types),
    }
}

impl<'m> Printer<'m> {
    fn define(&mut self, value: Value) -> String {
        let number = self.next;
        self.next += 1;
        self.names.insert(value, number);
        format!("%{number}")
    }

    fn name(&self, value: Value) -> String {
        match self.names.get(&value) {
            Some(number) => format!("%{number}"),
            None => format!("%<unknown {value:?}>"),
        }
    }

    fn indent(&mut self, depth: usize) {
        for _ in 0..depth {
            self.out.push_str("  ");
        }
    }

    /// Runs `f` with a fresh name scope, as an isolated op's regions see no outer values.
    fn isolated<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let names = std::mem::take(&mut self.names);
        let next = std::mem::replace(&mut self.next, 0);
        let result = f(self);
        self.names = names;
        self.next = next;
        result
    }

    fn print_insts(&mut self, region: Region, depth: usize) {
        for inst in self.module.insts(region).collect::<Vec<_>>() {
            self.indent(depth);
            self.print_inst(inst, depth);
            self.out.push('\n');
        }
    }

    /// ` ({...}, {...})`: the regions of `inst`, each with its block arguments.
    fn print_regions(&mut self, inst: Inst, depth: usize) {
        let module = self.module;
        let labels = module.op(inst).region_names();
        self.out.push_str(" (");
        for (i, &region) in module.regions(inst).iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            self.print_region(region, labels.get(i).copied().unwrap_or("bb"), depth);
        }
        self.out.push(')');
    }

    fn print_region(&mut self, region: Region, label: &str, depth: usize) {
        let module = self.module;
        self.out.push_str("{\n");
        if !module.args(region).is_empty() {
            self.indent(depth);
            let args: Vec<String> = module
                .args(region)
                .iter()
                .map(|&arg| format!("{}: {}", self.define(arg), module.ty(arg)))
                .collect();
            let _ = writeln!(self.out, "^{label}({}):", args.join(", "));
        }
        self.print_insts(region, depth + 1);
        self.indent(depth);
        self.out.push('}');
    }

    /// A symbol op: `{op} @{name}(%a: T{arg suffix}, ...) -> {results} {attrs} { body }`.
    fn print_symbol(&mut self, inst: Inst, depth: usize, symbol: &dyn SymbolOp) {
        let module = self.module;
        let body = module.regions(inst)[0];
        self.isolated(|p| {
            let _ = write!(p.out, "{} @{}(", module.op(inst).name(), symbol.symbol());
            for (index, &arg) in module.args(body).iter().enumerate() {
                if index > 0 {
                    p.out.push_str(", ");
                }
                let arg_name = p.define(arg);
                let _ = write!(
                    p.out,
                    "{arg_name}: {}{}",
                    module.ty(arg),
                    symbol.arg_suffix(index)
                );
            }
            let _ = write!(p.out, ") -> {}", symbol.results_text());
            let attrs = symbol.attr_entries();
            if !attrs.is_empty() {
                let _ = write!(p.out, " {{{}}}", attrs.join(", "));
            }
            p.out.push_str(" {\n");
            p.print_insts(body, depth + 1);
            p.indent(depth);
            p.out.push('}');
        });
    }

    fn print_inst(&mut self, inst: Inst, depth: usize) {
        let module = self.module;
        let op = module.op(inst);
        if let Some(symbol) = op.dyn_symbol() {
            self.print_symbol(inst, depth, symbol);
            return;
        }

        let results = module.results(inst).to_vec();
        if !results.is_empty() {
            let names: Vec<String> = results.iter().map(|&v| self.define(v)).collect();
            let _ = write!(self.out, "{} = ", names.join(", "));
        }
        let operands: Vec<String> = module
            .operands(inst)
            .iter()
            .map(|&v| self.name(v))
            .collect();
        let _ = write!(self.out, "{}({})", op.name(), operands.join(", "));

        let mut attrs = Vec::new();
        op.print_attrs(&mut attrs);
        let seg_text = |segs: Option<&[usize]>| {
            segs.map(|s| {
                let items: Vec<String> = s.iter().map(|n| n.to_string()).collect();
                format!("[{}]", items.join(", "))
            })
        };
        if let Some(text) = seg_text(module.operand_segments(inst)) {
            attrs.push(("operand_segments", text));
        }
        if let Some(text) = seg_text(module.result_segments(inst)) {
            attrs.push(("result_segments", text));
        }
        if !attrs.is_empty() {
            let items: Vec<String> = attrs.iter().map(|(k, v)| format!("{k} = {v}")).collect();
            let _ = write!(self.out, " {{{}}}", items.join(", "));
        }

        if !module.regions(inst).is_empty() {
            if op.has_trait::<IsolatedFromAbove>() {
                self.isolated(|p| p.print_regions(inst, depth));
            } else {
                self.print_regions(inst, depth);
            }
        }

        let operand_types = module.operand_types(inst);
        let result_tys = module.result_types(inst);
        let _ = write!(
            self.out,
            " : {} -> {}",
            type_list(&operand_types),
            results_text(&result_tys)
        );
        if let Some(location) = module.location(inst).filter(|_| self.locations) {
            let _ = write!(self.out, " {location}");
        }
    }
}

/// What the text form shows besides the ops.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextOptions {
    /// The version of the spec the module follows, first: the form of a program stored for later.
    pub versioned: bool,
    /// The location of each op that has one, after its signature, as dumps show them.
    pub locations: bool,
}

impl Module {
    /// The module in its text form.
    pub fn to_text(&self) -> String {
        self.print(TextOptions::default())
    }

    /// The module in its text form, after the version of the spec it follows: the form of a
    /// program stored for later.
    pub fn to_versioned_text(&self) -> String {
        self.print(TextOptions {
            versioned: true,
            ..TextOptions::default()
        })
    }

    /// The module in its text form, with what `options` adds.
    pub fn print(&self, options: TextOptions) -> String {
        let mut printer = Printer {
            module: self,
            out: String::new(),
            names: HashMap::new(),
            next: 0,
            locations: options.locations,
        };
        if options.versioned {
            let _ = writeln!(printer.out, "spec_version = \"{}\"", super::SPEC_VERSION);
        }
        printer.print_insts(self.body(), 0);
        printer.out
    }
}
