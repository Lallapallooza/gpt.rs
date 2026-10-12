//! Runs the filetests of the IR passes, under `filetests/tensor/`, `filetests/tile/` and
//! `filetests/loop/`. `gpt_rs_filetests` gives their format.

#[test]
fn tensor_filetests() {
    gpt_rs_filetests::run("tensor", None);
}

#[test]
fn tile_filetests() {
    gpt_rs_filetests::run("tile", None);
}

#[test]
fn loop_filetests() {
    gpt_rs_filetests::run("loop", None);
}
