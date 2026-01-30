// src/training/mod.rs

//! Training and evaluation orchestration
//!
//! - `learner`: formal Burn training/validation pipeline
//! - `metrics`: CTC-compatible error rate calculations (CER/WER)
//! - `trainer`: manual training implementations and model-specific training logic

pub mod learner;
pub mod metrics;
pub mod trainer;

pub use learner::{VsrmLearnerConfig, train};
pub use metrics::{CtcCharErrorRate, CtcWordErrorRate};