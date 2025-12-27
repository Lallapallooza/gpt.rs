#[derive(Debug, Clone, Copy)]
pub struct PatternField {
    pub name: &'static str,
    pub view: &'static str,
    pub optional: bool,
}

pub struct PatternDef {
    pub target: &'static str,
    pub module_path: &'static str,
    pub view_name: &'static str,
    pub fields: &'static [PatternField],
}

#[linkme::distributed_slice]
pub static PATTERN_DEFS: [PatternDef] = [..];

pub fn all_pattern_defs() -> &'static [PatternDef] {
    &PATTERN_DEFS
}
