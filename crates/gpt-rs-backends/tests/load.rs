//! Choosing a backend by name.

use gpt_rs::runtime::LoadOptions;

#[test]
fn an_unknown_backend_name_lists_the_backends_of_the_build() {
    let err = gpt_rs_backends::load_model("tpu", "unused", LoadOptions::default())
        .err()
        .expect("no backend is named tpu");
    let expected = format!(
        "unknown backend 'tpu'; this build has {}",
        gpt_rs_backends::names().join(", ")
    );
    assert_eq!(err.to_string(), expected);
}
