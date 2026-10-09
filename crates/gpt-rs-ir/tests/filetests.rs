//! Runs the filetests of the IR passes, under `filetests/tensor/`. `gpt_rs_filetests` gives their
//! format.

#[test]
fn filetests() {
    gpt_rs_filetests::run("tensor", None);
}
