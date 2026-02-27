use std::collections::VecDeque;
use std::sync::Mutex;

use anyhow::Result;
use gpt_rs::backend::spec::{
    BackendError, BackendResult, DecodeSampleRequest, Instruction, PortableBackend, Program,
    TensorInit, TensorLiteral,
};
use gpt_rs::inference::generate::Generator;
use gpt_rs::inference::sampler::Sampler;
use gpt_rs::inference::CausalLanguageModel;
use gpt_rs::ops::functional::DecodeKvCache;
use gpt_rs::tensor::{Shape, Tensor};

#[derive(Clone)]
struct MockBackend;

impl PortableBackend for MockBackend {
    type TensorHandle = ();

    fn backend_name(&self) -> &str {
        "mock"
    }

    fn materialize(&self, _init: TensorInit) -> BackendResult<Self::TensorHandle> {
        Err(BackendError::execution(
            "mock backend does not materialize tensors",
        ))
    }

    fn to_literal(&self, _tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        Err(BackendError::execution(
            "mock backend does not read literals",
        ))
    }

    fn execute_instruction(
        &self,
        _instruction: &Instruction,
        _inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        Err(BackendError::execution(
            "mock backend does not execute instructions",
        ))
    }

    fn run_program(
        &self,
        _program: &Program,
        _entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        Err(BackendError::execution(
            "mock backend does not run programs",
        ))
    }
}

#[derive(Default, Clone, Copy)]
struct CallCounts {
    decode_calls: usize,
    sample_next_calls: usize,
}

struct MockModel {
    calls: Mutex<CallCounts>,
    sample_tokens: Mutex<VecDeque<Option<usize>>>,
}

impl MockModel {
    fn new(sample_tokens: Vec<Option<usize>>) -> Self {
        Self {
            calls: Mutex::new(CallCounts::default()),
            sample_tokens: Mutex::new(VecDeque::from(sample_tokens)),
        }
    }

    fn calls(&self) -> CallCounts {
        *self.calls.lock().expect("call count mutex poisoned")
    }

    fn logits() -> Result<Tensor> {
        Tensor::from_vec(Shape::new([1, 3]), vec![0.1, 0.2, 0.9])
    }
}

impl CausalLanguageModel<MockBackend> for MockModel {
    fn context_length(&self) -> usize {
        16
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn forward(&self, _tokens: &[usize]) -> Result<Tensor> {
        Self::logits()
    }

    fn forward_with_decode_cache(
        &self,
        _tokens: &[usize],
        _position_offset: usize,
        _caches: &mut [Option<DecodeKvCache<MockBackend>>],
    ) -> Result<Tensor> {
        let mut calls = self.calls.lock().expect("call count mutex poisoned");
        calls.decode_calls += 1;
        Self::logits()
    }

    fn forward_with_decode_cache_with_capacity(
        &self,
        _tokens: &[usize],
        _position_offset: usize,
        _caches: &mut [Option<DecodeKvCache<MockBackend>>],
        _capacity: usize,
    ) -> Result<Tensor> {
        let mut calls = self.calls.lock().expect("call count mutex poisoned");
        calls.decode_calls += 1;
        Self::logits()
    }

    fn forward_with_decode_cache_sample_next(
        &self,
        _tokens: &[usize],
        _position_offset: usize,
        _caches: &mut [Option<DecodeKvCache<MockBackend>>],
        _request: DecodeSampleRequest,
    ) -> Result<Option<usize>> {
        let mut calls = self.calls.lock().expect("call count mutex poisoned");
        calls.sample_next_calls += 1;
        drop(calls);
        Ok(self
            .sample_tokens
            .lock()
            .expect("sample token mutex poisoned")
            .pop_front()
            .unwrap_or(None))
    }
}

#[test]
fn generator_uses_pending_gpu_sample_for_final_step() -> Result<()> {
    let model = MockModel::new(vec![Some(1)]);
    let sampler = Sampler::new(0.0);
    let mut generator = Generator::new(&model, &sampler, &[0], true)?;

    let first = generator.step()?;
    let second = generator.step_final()?;

    assert_eq!(first, 2);
    assert_eq!(second, 1);

    let calls = model.calls();
    assert_eq!(calls.decode_calls, 1);
    assert_eq!(calls.sample_next_calls, 1);
    Ok(())
}

#[test]
fn generator_uses_gpu_sample_with_fixed_kv_capacity() -> Result<()> {
    let model = MockModel::new(vec![Some(1)]);
    let sampler = Sampler::new(0.0);
    let mut generator =
        Generator::new_with_kv_cache_capacity(&model, &sampler, &[0], true, Some(8))?;

    let first = generator.step()?;
    let second = generator.step_final()?;

    assert_eq!(first, 2);
    assert_eq!(second, 1);

    let calls = model.calls();
    assert_eq!(calls.decode_calls, 1);
    assert_eq!(calls.sample_next_calls, 1);
    Ok(())
}

#[test]
fn generator_falls_back_to_host_logits_when_gpu_sampling_unavailable() -> Result<()> {
    let model = MockModel::new(vec![None]);
    let sampler = Sampler::new(0.0);
    let mut generator = Generator::new(&model, &sampler, &[0], true)?;

    let _ = generator.step()?;

    let calls = model.calls();
    assert_eq!(calls.decode_calls, 2);
    assert_eq!(calls.sample_next_calls, 1);
    Ok(())
}

#[test]
fn sampler_backend_decode_sampling_rejects_top_k() {
    let greedy = Sampler::new(0.0);
    assert!(greedy.supports_backend_decode_sampling());

    let temp = Sampler::new(0.8);
    assert!(temp.supports_backend_decode_sampling());

    let with_top_k = Sampler::new(0.8).with_top_k(40);
    assert!(!with_top_k.supports_backend_decode_sampling());
}
