use crate::tensor::Tensor;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<usize>,
}

impl Sampler {
    pub fn new(temperature: f32) -> Self {
        Sampler {
            temperature,
            top_k: None,
        }
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k.max(1));
        self
    }

    pub fn sample(&self, logits: &Tensor) -> usize {
        let data = logits.data();
        if data.is_empty() {
            panic!("cannot sample from empty logits");
        }

        if self.temperature <= 0.0 {
            return Self::argmax(data);
        }

        let mut adjusted: Vec<f32> = data.iter().map(|&v| v / self.temperature).collect();
        if let Some(k) = self.top_k {
            if k < adjusted.len() {
                let mut indices: Vec<usize> = (0..adjusted.len()).collect();
                indices.select_nth_unstable_by(k - 1, |&a, &b| {
                    adjusted[b]
                        .partial_cmp(&adjusted[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let kth_value = adjusted[*indices.get(k - 1).unwrap()];
                for value in &mut adjusted {
                    if *value < kth_value {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let max = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = adjusted.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = probs.iter().sum();

        if !(sum.is_finite()) || sum <= 0.0 {
            return Self::argmax(data);
        }

        let inv_sum = 1.0 / sum;
        for p in &mut probs {
            *p *= inv_sum;
        }

        let dist = WeightedIndex::new(&probs).expect("invalid probabilities");
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
