//! Targets as `gpt_rs_ir::targets` documents them: the presets, target files that start from a
//! base, how a target reference resolves, the messages that reject a target whose facts
//! disagree, and the one that names a fact it lacks.

use std::fs;
use std::path::PathBuf;

use gpt_rs_ir::ir::Level;
use gpt_rs_ir::targets::{TargetDesc, TargetRef};

/// The facts of a one-core x86-64 part, with `extra` appended: a target file to break.
fn target_file(extra: &str) -> String {
    format!(
        r#"[isa]
arch = "x86_64"
features = ["avx", "avx2", "fma"]
vector_bits = [128, 256]
vector_registers = 16
cc_march = "x86-64-v3"

[[level]]
name = "l3"
units = 1

[[level]]
name = "core"
units = 1

[[memory]]
name = "registers"
kind = "registers"

[[memory]]
name = "dram"
kind = "main"
{extra}"#
    )
}

fn rejection(text: &str) -> String {
    format!("{:#}", TargetDesc::from_toml("part", text).unwrap_err())
}

/// The path of a target file named `name` in this test's directory, holding `text`.
fn write_target(test: &str, name: &str, text: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(test);
    fs::create_dir_all(&dir).expect("the test directory is made");
    let path = dir.join(name);
    fs::write(&path, text).expect("the target file is written");
    path.to_str().expect("a UTF-8 path").to_string()
}

#[test]
fn a_target_file_states_what_differs_from_its_base() {
    let base = TargetDesc::resolve("x86-64-v4").expect("the preset loads");
    let text = r#"base = "x86-64-v4"
description = "One core of an x86-64-v4 part"

[[level]]
name = "l3"
units = 1

[[level]]
name = "core"
units = 1
"#;
    let path = write_target("target_base", "one-core.toml", text);
    let target = TargetDesc::resolve(&path).expect("the target file loads");
    assert_eq!(target.description, "One core of an x86-64-v4 part");
    // The `[[level]]` list replaces the base's, and what the file leaves out is the base's.
    assert_eq!(target.units(&Level::new("core")), Some(1));
    assert_eq!(base.units(&Level::new("core")), Some(28));
    assert_eq!(target.isa, base.isa);
    assert_eq!(target.memories, base.memories);
}

#[test]
fn target_files_that_start_from_one_another_are_rejected() {
    let first_path = write_target(
        "target_base_cycle",
        "first.toml",
        "base = \"../target_base_cycle/second.toml\"\n",
    );
    let second_path = write_target(
        "target_base_cycle",
        "second.toml",
        "base = \"first.toml\"\n",
    );
    let canonical = |path: &str| {
        let resolved = fs::canonicalize(path).expect("the target file exists");
        resolved.display().to_string()
    };
    let (first, second) = (canonical(&first_path), canonical(&second_path));
    let err = format!("{:#}", TargetDesc::resolve(&first_path).unwrap_err());
    assert_eq!(
        err,
        format!("the target files start from one another: {first} -> {second} -> {first}")
    );
}

#[test]
fn an_unknown_name_lists_the_presets() {
    let err = TargetDesc::resolve("pentium").unwrap_err();
    assert_eq!(
        format!("{err:#}"),
        "no target `pentium`: name a preset (x86-64-v2, x86-64-v3, x86-64-v4, znver5), `host`, or \
         a target file"
    );
}

#[test]
fn a_reference_to_facts_that_changed_is_rejected() {
    let stale = TargetRef::try_from("x86-64-v4@00000000".to_string()).expect("a reference");
    let now = TargetDesc::resolve("x86-64-v4")
        .expect("a preset")
        .fingerprint();
    assert_eq!(
        format!("{:#}", stale.resolve().unwrap_err()),
        format!(
            "the target x86-64-v4@00000000 is now x86-64-v4@{now}: the schedule was chosen for \
             other facts"
        )
    );
}

#[test]
fn a_reference_without_a_fingerprint_is_rejected() {
    let err = TargetRef::try_from("znver5".to_string()).unwrap_err();
    assert_eq!(
        err.to_string(),
        "a target reference is `name@fingerprint`, not `znver5`"
    );
}

#[test]
fn units_that_the_level_before_does_not_share_evenly_are_rejected() {
    let text = target_file("")
        .replace("name = \"l3\"\nunits = 1", "name = \"l3\"\nunits = 2")
        .replace("name = \"core\"\nunits = 1", "name = \"core\"\nunits = 3");
    assert_eq!(
        rejection(&text),
        "level core has 3 units, which the 2 units of level l3 do not share evenly"
    );
}

#[test]
fn a_target_whose_first_memory_is_not_its_registers_is_rejected() {
    let text = target_file("").replace("kind = \"registers\"", "kind = \"scratch\"");
    assert_eq!(rejection(&text), "a target's first memory is its registers");
}

#[test]
fn a_cost_of_an_op_that_isa_lacks_is_rejected() {
    let text = target_file("\n[[cost]]\nop = \"isa.sqrt\"\nwidth = 256\nissue = 2\n");
    assert_eq!(
        rejection(&text),
        "a cost names isa.sqrt, which is not an op of isa"
    );
}

#[test]
fn a_target_without_an_alignment_of_buffers_names_the_missing_fact() {
    let target = TargetDesc::from_toml("part", &target_file("")).expect("the target loads");
    assert_eq!(
        format!("{:#}", target.buffer_alignment().unwrap_err()),
        "part states no alignment of the buffers in its main memory, dram"
    );
}

#[test]
fn an_alignment_of_buffers_that_is_not_a_power_of_two_is_rejected() {
    let text = target_file("alignment = 48\n");
    assert_eq!(
        rejection(&text),
        "memory dram aligns buffers to 48 bytes, not a power of two"
    );
}

#[test]
fn an_alignment_of_buffers_outside_main_memory_is_rejected() {
    let text = target_file("alignment = 64\n").replace(
        "kind = \"registers\"",
        "kind = \"registers\"\nalignment = 64",
    );
    assert_eq!(
        rejection(&text),
        "memory registers states an alignment of buffers, which only main memory holds"
    );
}
