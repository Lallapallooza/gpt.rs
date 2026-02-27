use gpt_rs::backend::spec::{DecodeSampleRequest, PortableBackend, TensorInit};
use gpt_rs::tensor::{Shape as HostShape, Tensor};
use gpt_rs_backend_triton::TritonBackend;

#[test]
fn triton_decode_sample_greedy_returns_argmax() -> anyhow::Result<()> {
    if !TritonBackend::is_available() {
        return Ok(());
    }

    let backend = TritonBackend::new();
    let logits = Tensor::from_vec(HostShape::new([1, 4]), vec![-1.0, 0.1, 2.5, 0.4])?;
    let literal = logits.to_literal();
    let spec = literal.spec.clone();
    let handle = backend.materialize(TensorInit::Literal(literal))?;

    let token = backend.sample_decode_token(&handle, &spec, DecodeSampleRequest::greedy())?;
    assert_eq!(token, Some(2));
    Ok(())
}

#[test]
fn triton_decode_sample_temperature_returns_in_range_token() -> anyhow::Result<()> {
    if !TritonBackend::is_available() {
        return Ok(());
    }

    let backend = TritonBackend::new();
    let logits = Tensor::from_vec(HostShape::new([1, 4]), vec![0.1, 0.2, 0.3, 0.4])?;
    let literal = logits.to_literal();
    let spec = literal.spec.clone();
    let handle = backend.materialize(TensorInit::Literal(literal))?;

    let token = backend.sample_decode_token(
        &handle,
        &spec,
        DecodeSampleRequest {
            temperature: 0.8,
            top_k: None,
            random_u: Some(0.42),
        },
    )?;
    let sampled = token.expect("triton decode sampling should return a token");
    assert!(sampled < 4);
    Ok(())
}
