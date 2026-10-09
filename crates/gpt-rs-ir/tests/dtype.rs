//! Parsing a dtype from its name.

use gpt_rs_ir::types::DType;

#[test]
fn a_dtype_parses_from_its_mnemonic_in_any_case() {
    assert_eq!(" BF16\n".parse::<DType>().unwrap(), DType::Bf16);
    assert_eq!("f8E4M3".parse::<DType>().unwrap(), DType::F8E4M3);
    let err = "i32".parse::<DType>().unwrap_err();
    assert_eq!(err.to_string(), "unknown dtype `i32`");
}
