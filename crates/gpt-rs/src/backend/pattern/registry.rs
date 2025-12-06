use crate::backend::pattern::PatternSet;
use crate::backend::spec::PortableBackend;

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
    pub insert: fn(&mut PatternSet),
}

#[linkme::distributed_slice]
pub static PATTERN_DEFS: [PatternDef] = [..];

pub fn all_pattern_defs() -> &'static [PatternDef] {
    &PATTERN_DEFS
}

pub fn register_default_patterns<B: PortableBackend + 'static>(set: &mut PatternSet, backend: &B) {
    for def in all_pattern_defs() {
        if backend.supports_custom_call(def.target) {
            (def.insert)(set);
        }
    }
}
