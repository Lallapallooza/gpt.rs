use crate::backend::spec::PortableBackend;
use crate::tensor::{DeviceTensor, Tensor};

pub enum ModelInput<B: PortableBackend + 'static> {
    Tokens(Vec<usize>),
    Vision(DeviceTensor<B>),
}

pub enum ModelOutput {
    Tensor(Tensor),
}
