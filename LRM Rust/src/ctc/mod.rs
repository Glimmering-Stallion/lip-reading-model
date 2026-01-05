// src/ctc/mod.rs

//! Connectionist Temporal Classification (CTC) components
//!
//! - `ctc_loss`: training loss function
//! - `ctc_decode`: greedy/beam search decoders
//! - `lm`: language model modules for shallow fusion

pub mod ctc_decode;
pub mod ctc_loss;
pub mod lm;
