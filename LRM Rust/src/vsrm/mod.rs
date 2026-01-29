// src/vsrm/mod.rs

//! Visual Speech Recognition Model components
//!
//! - `vsrm`: core model architecture and forward pass
//! - `tcn`: Temporal Convolutional Network blocks

pub mod tcn;
pub mod vsrm;

pub use vsrm::{VsrModel, VsrModelConfig};