//! Data in host memory: tensors, their element storage, and scalars. The interpreter computes on
//! them, and the CPU backends share the storage.

mod scalar;
mod tensor;

pub use self::scalar::*;
pub use self::tensor::*;
