use crate::backend::spec::PortableBackend;
use crate::model::{Gpt, GptForwardState};
use crate::tensor::{Shape, Tensor};
use anyhow::{bail, Result};

pub struct Trainer<B: PortableBackend + 'static> {
    pub model: Gpt<B>,
    pub learning_rate: f32,
    pub checkpoint_interval: Option<usize>,
    pub step: usize,
}

impl<B: PortableBackend + 'static> Trainer<B> {
    pub fn new(model: Gpt<B>, learning_rate: f32) -> Self {
        Trainer {
            model,
            learning_rate,
            checkpoint_interval: None,
            step: 0,
        }
    }

    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = Some(interval);
        self
    }

    pub fn train_step(&mut self, tokens: &[usize], targets: &[usize]) -> Result<Tensor> {
        if tokens.len() != targets.len() {
            bail!("tokens and targets must have same length");
        }
        let (logits, state) = self.model.forward_with_state(tokens)?;
        let dims = logits.shape().dims();
        let seq_len = dims[0];
        let vocab_size = dims[1];
        let mut loss = 0.0;
        let mut grad_logits = vec![0.0; seq_len * vocab_size];
        for (t, &target) in targets.iter().enumerate() {
            if target >= vocab_size {
                bail!("target {} out of range {}", target, vocab_size);
            }
            let row = &logits.data()[t * vocab_size..(t + 1) * vocab_size];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum: f32 = 0.0;
            let mut probs = vec![0.0; vocab_size];
            for (i, &logit) in row.iter().enumerate() {
                let val = (logit - max).exp();
                probs[i] = val;
                exp_sum += val;
            }
            let inv_sum = 1.0f32 / exp_sum.max(1e-9f32);
            for (i, prob) in probs.iter_mut().enumerate() {
                *prob *= inv_sum;
                let index = t * vocab_size + i;
                grad_logits[index] = *prob;
                if i == target {
                    grad_logits[index] -= 1.0;
                }
                grad_logits[index] /= seq_len as f32;
            }
            let log_prob = probs[target].max(1e-9f32).ln();
            loss -= log_prob;
        }

        self.backward_and_update(&state, &grad_logits)?;
        self.step += 1;

        Tensor::from_vec(Shape::new([1]), vec![loss / seq_len as f32])
    }

    fn backward_and_update(
        &mut self,
        _state: &GptForwardState<B>,
        _grad_logits: &[f32],
    ) -> Result<()> {
        bail!("optimizer path is not implemented for device-backed tensors yet")
    }
}
