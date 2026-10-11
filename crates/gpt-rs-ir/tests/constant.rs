//! The type rule of `tensor.constant` rejects bytes that do not hold the elements. Only code can
//! build such a constant: the parser checks the payload of the text form first. Building an op
//! its type rule rejects is a bug in the code that builds it, so the builder panics.

use gpt_rs_ir::dense::Dense;
use gpt_rs_ir::dialects::tensor::Constant;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::types::DType;

#[test]
#[should_panic(expected = "building tensor.constant: 8 bytes do not hold 4 f32 elements")]
fn constant_bytes_must_hold_its_elements() {
    let value = Dense {
        dtype: DType::F32,
        dims: vec![4],
        bytes: vec![0u8; 8].into(),
    };
    let _ = Module::new().build_func("f", Vec::new(), Vec::new(), |builder, _| {
        Ok(vec![builder.create(Constant { value }, ())])
    });
}
