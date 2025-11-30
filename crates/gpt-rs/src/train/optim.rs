use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);
    fn zero_grad(&mut self, params: &mut [Tensor]) {
        for p in params {
            if let Some(grad) = p.grad_mut() {
                grad.fill(0.0);
            }
        }
    }
}

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, _params: &mut [Tensor], _grads: &[Tensor]) {
        // Implementation deferred to later step.
    }
}
