// src/vsrm/mod.rs

//! Visual Speech Recognition Model components
//!
//! - `summary`: module summary visitor for architecture inspection
//! - `vsrm`: core model architecture and forward pass
//! - `residual`: Residual Blocks
//! - `tcn`: Temporal Convolutional Network Blocks

pub mod summary;
pub mod vsrm;
pub mod residual;
pub mod tcn;

pub use summary::SummaryVisitor;
pub use vsrm::{VsrModel, VsrModelConfig};