//! Inference module for loading trained VSRM and running prediction on video input.
//!
//! - `predictor`: VsrmPredictorConfig, InferenceSession, and SlidingWindow
//! - `loader`: video file loading with tracker and live camera capture
//! - `overlay`: visualization overlays for live inference mode
//! - `speech_gate`: speech-activity hysteresis from tracker lock + lip-motion flags

pub mod predictor;
pub mod loader;
pub mod overlay;
pub mod speech_gate;

pub use predictor::{VsrmPredictorConfig, InferenceSession, SlidingWindow, infer};
pub use overlay::{FrameAnnotator, LiveWindow};
