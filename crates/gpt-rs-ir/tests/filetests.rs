//! Runs the filetests of the IR passes, under `filetests/tensor/` and `filetests/tile/`.
//! `gpt_rs_filetests` gives their format.

#[test]
fn tensor_filetests() {
    gpt_rs_filetests::run("tensor", None);
}

#[test]
fn tile_filetests() {
    gpt_rs_filetests::run("tile", None);
}
