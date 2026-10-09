//! The type rule of `tensor.constant` rejects bytes that do not hold the elements. Only code can
//! build such a constant: the parser checks the payload of the text form first.

use gpt_rs_ir::dense::Dense;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::types::DType;

#[test]
fn constant_bytes_must_hold_its_elements() {
    let dense = Dense {
        dtype: DType::F32,
        dims: vec![4],
        bytes: vec![0u8; 8].into(),
    };
    let err = Module::new()
        .build_func("f", Vec::new(), Vec::new(), |b, _| {
            Ok(vec![b.constant(dense)?])
        })
        .unwrap_err();
    assert_eq!(
        format!("{err:#}"),
        "tensor.constant: 8 bytes do not hold 4 f32 elements"
    );
}
