//! Runs the filetests of each dialect's directory under `filetests/`. `gpt_rs_filetests` gives
//! their format.

#[test]
fn tensor_filetests() {
    gpt_rs_filetests::run("tensor", None);
}

#[test]
fn arith_filetests() {
    gpt_rs_filetests::run("arith", None);
}

#[test]
fn tile_filetests() {
    gpt_rs_filetests::run("tile", None);
}

#[test]
fn loop_filetests() {
    gpt_rs_filetests::run("loop", None);
}

#[test]
fn isa_filetests() {
    gpt_rs_filetests::run("isa", None);
}

#[test]
fn host_filetests() {
    gpt_rs_filetests::run("host", None);
}
