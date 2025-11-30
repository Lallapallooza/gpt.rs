pub trait LrSchedule {
    fn learning_rate(&self, step: usize) -> f32;
}

pub struct ConstantSchedule {
    pub lr: f32,
}

impl LrSchedule for ConstantSchedule {
    fn learning_rate(&self, _step: usize) -> f32 {
        self.lr
    }
}
