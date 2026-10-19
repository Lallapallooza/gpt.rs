use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ir::{print_tuple, type_list, AsInst, Inst, Module, OpDyn, Region, SymbolOp, Value};
use crate::traits::IsolatedFromAbove;

struct Printer<'m> {
    module: &'m Module,
    out: String,
    names: HashMap<Value, usize>,
    next: usize,
    /// Whether each op with a location ends with it.
    locations: bool,
    /// The number of each op, which ends its first line, when the options ask for numbers.
    numbers: Option<HashMap<Inst, u32>>,
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

    /// Runs `body` with a fresh name scope, as an isolated op's regions see no outer values.
    fn isolated<T>(&mut self, body: impl FnOnce(&mut Self) -> T) -> T {
        let names = std::mem::take(&mut self.names);
        let next = std::mem::replace(&mut self.next, 0);
        let result = body(self);
        self.names = names;
        self.next = next;
        result
    }

    fn print_insts(&mut self, region: Region, depth: usize) {
        for inst in self.module.insts(region).collect::<Vec<_>>() {
            self.indent(depth);
            let start = self.out.len();
            self.print_inst(inst, depth);
            self.number(inst, start);
            self.out.push('\n');
        }
    }

    /// Ends the first line of the op of `inst`, printed from `start`, with its number.
    fn number(&mut self, inst: Inst, start: usize) {
        let Some(number) = self.numbers.as_ref().and_then(|numbers| numbers.get(&inst)) else {
            return;
        };
        let comment = format!(" // #{number}");
        let end = self.out[start..]
            .find('\n')
            .map_or(self.out.len(), |offset| start + offset);
        self.out.insert_str(end, &comment);
    }

    /// ` ({...}, {...})`: the regions of `inst`, each with its block arguments.
    fn print_regions(&mut self, inst: Inst, depth: usize) {
        let module = self.module;
        let labels = module.op(inst).region_names();
        self.out.push_str(" (");
        for (position, &region) in module.regions(inst).iter().enumerate() {
            if position > 0 {
                self.out.push_str(", ");
            }
            self.print_region(region, labels.get(position).copied().unwrap_or("bb"), depth);
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

    /// A symbol op: its name, its signature, which the op writes, then its body.
    fn print_symbol(&mut self, inst: Inst, depth: usize, symbol: &dyn SymbolOp) {
        let module = self.module;
        let body = module.regions(inst)[0];
        self.isolated(|p| {
            let mut arg_names = Vec::with_capacity(module.args(body).len());
            for &arg in module.args(body) {
                arg_names.push(p.define(arg));
            }
            let _ = write!(p.out, "{} ", module.op(inst).name());
            symbol.print_signature(&arg_names, &mut p.out);
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
            let names: Vec<String> = results.iter().map(|&result| self.define(result)).collect();
            let _ = write!(self.out, "{} = ", names.join(", "));
        }
        let operands: Vec<String> = module
            .operands(inst)
            .iter()
            .map(|&operand| self.name(operand))
            .collect();
        let _ = write!(self.out, "{}({})", op.name(), operands.join(", "));

        let mut attrs = Vec::new();
        op.print_attrs(&mut attrs);
        let seg_text = |segs: Option<&[usize]>| {
            segs.map(|sizes| {
                let items: Vec<String> = sizes.iter().map(|size| size.to_string()).collect();
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
            let items: Vec<String> = attrs
                .iter()
                .map(|(key, value)| format!("{key} = {value}"))
                .collect();
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
        let _ = write!(self.out, " : {} -> ", type_list(&operand_types));
        print_tuple(module.result_types(inst), &mut self.out);
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
    /// The number of each op, its place in print order, in a comment at the end of its first
    /// line: `// #47`, the number the trails of locations name.
    pub numbers: bool,
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

    /// The op of `op`, with its regions, in its text form, its values named from `%0`. The op has
    /// no operands and is isolated from above, so the text names only values it defines.
    pub fn inst_text(&self, op: impl AsInst) -> String {
        let inst = op.inst();
        assert!(
            self.operands(inst).is_empty() && self.op(inst).has_trait::<IsolatedFromAbove>(),
            "{} reads values from outside it, which its text cannot name",
            self.op(inst).name()
        );
        let mut printer = Printer {
            module: self,
            out: String::new(),
            names: HashMap::new(),
            next: 0,
            locations: false,
            numbers: None,
        };
        printer.print_inst(inst, 0);
        printer.out
    }

    /// The module in its text form, with what `options` adds.
    pub fn print(&self, options: TextOptions) -> String {
        let mut printer = Printer {
            module: self,
            out: String::new(),
            names: HashMap::new(),
            next: 0,
            locations: options.locations,
            numbers: options.numbers.then(|| self.op_numbers()),
        };
        if options.versioned {
            let _ = writeln!(printer.out, "spec_version = \"{}\"", super::SPEC_VERSION);
        }
        printer.print_insts(self.body(), 0);
        printer.out
    }
}
