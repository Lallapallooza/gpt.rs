//! The presets: the target files of `targets/`, each named by its stem.

/// Each preset's name and TOML text.
pub const PRESETS: &[(&str, &str)] = &[
    (
        "x86-64-v2",
        include_str!("../../../../targets/x86-64-v2.toml"),
    ),
    (
        "x86-64-v3",
        include_str!("../../../../targets/x86-64-v3.toml"),
    ),
    (
        "x86-64-v4",
        include_str!("../../../../targets/x86-64-v4.toml"),
    ),
    ("znver5", include_str!("../../../../targets/znver5.toml")),
];

/// The TOML text of the preset `name`.
pub(super) fn preset(name: &str) -> Option<&'static str> {
    PRESETS
        .iter()
        .find(|(preset, _)| *preset == name)
        .map(|(_, text)| *text)
}

/// The names of the presets.
pub(super) fn names() -> Vec<&'static str> {
    PRESETS.iter().map(|(name, _)| *name).collect()
}
