//! Backwards-compatible module for image models and building blocks.
//!
//! Prefer importing from `gpt_rs::model::*` for all models, including ResNet and MobileNet.

pub use crate::model::conv as layers;
pub use crate::model::mobilenet_v2;
pub use crate::model::resnet;

pub use crate::model::{MobileNetV2, ResNet34};
