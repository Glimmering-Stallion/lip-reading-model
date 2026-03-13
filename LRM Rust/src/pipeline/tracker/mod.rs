//! Trait-based temporal mouth tracking and Region of Interest (ROI) extraction.
//!
//! - `backend`: tracker trait, shared types, configuration dispatch, and TLS helpers
//! - `haar`: Haar cascade face/mouth detection with EMA smoothing (default)

pub mod tracker;
pub mod haar;

pub use tracker::{
    LipTrackerBackend,
    TrackerConfig,
    TrackerResult,
    VizMetadata,
    with_local_tracker,
};
pub use haar::HaarTrackerConfig;
