//! Inference module for loading trained VSRM and running prediction on video input.
//!
//! - `predictor`: VsrmPredictorConfig, InferenceSession, and SlidingWindow
//! - `loader`: video file loading with tracker and live camera capture
//! - `overlay`: visualization overlays for live inference mode

pub mod predictor;
pub mod loader;
pub mod overlay;

pub use predictor::{VsrmPredictorConfig, InferenceSession, SlidingWindow, infer};
pub use overlay::{FrameAnnotator, LiveWindow};
