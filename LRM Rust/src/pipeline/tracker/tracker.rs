//! Backend-agnostic tracker trait, shared types, configuration dispatch,
//! and Thread Local Storage (TLS) helpers for per-worker tracker instances.



use std::{
    cell::RefCell,
    error::Error,
};
use opencv::core::{
    Mat,
    Point2f,
    Rect,
};

use super::haar::HaarTrackerConfig;



thread_local! {
    static TRACKER_TLS: RefCell<Option<Box<dyn LipTrackerBackend>>> = RefCell::new(None);
}



/// Standardized output from any tracker backend.
///
/// Contains the mouth crop tensor (for the model) alongside optional
/// visualization metadata (for the display fork).
pub struct TrackerResult {
    pub crop: Mat,
    pub metadata: VizMetadata,
}



/// Backend-agnostic visualization metadata.
///
/// Each tracker backend populates the fields it can produce;
/// the visualization module draws whatever is `Some`.
pub struct VizMetadata {
    pub face_rect: Option<Rect>,
    pub mouth_rect: Option<Rect>,
    pub landmarks: Option<Vec<Point2f>>,
    pub stabilized_center: Option<Point2f>,
}



/// Common interface for all tracker backends.
///
/// A tracker takes a raw video frame and returns a standardized
/// mouth crop plus visualization metadata.
pub trait LipTrackerBackend: Send {
    /// Processes a single video frame and returns a cropped, stabilized
    /// mouth image for the model along with visualization metadata.
    ///
    /// ### Params:
    /// - `frame`: The raw grayscale video frame.
    ///
    /// ### Returns:
    /// A [`TrackerResult`] containing the mouth crop and visualization metadata.
    fn process_frame(&mut self, frame: &Mat) -> Result<TrackerResult, Box<dyn Error>>;

    /// Resets temporal smoothing state for processing a new video.
    fn reset_state(&mut self);

    /// Returns the target output dimensions `(height, width)` for the mouth crop.
    fn target_dims(&self) -> (usize, usize);
}



/// Configuration enum for backend dispatch.
///
/// Each variant wraps the backend-specific config struct.
/// Call [`TrackerConfig::build`] to instantiate the corresponding tracker.
#[derive(Debug, Clone)]
pub enum TrackerConfig {
    Haar(HaarTrackerConfig),
    // MediaPipe(MediaPipeTrackerConfig),  // future
}



impl TrackerConfig {
    /// Instantiates the tracker backend corresponding to this config variant.
    ///
    /// ### Returns:
    /// A boxed trait object implementing [`LipTrackerBackend`].
    pub fn build(&self) -> Box<dyn LipTrackerBackend> {
        match self {
            TrackerConfig::Haar(c) => Box::new(c.init()),
        }
    }

    /// Returns the target output dimensions `(height, width)` for the configured backend.
    pub fn target_dims(&self) -> (usize, usize) {
        match self {
            TrackerConfig::Haar(c) => c.target_dims,
        }
    }
}



/// Executes a closure with the current thread's tracker instance.
///
/// Creates and caches a tracker on first call per thread using the given config.
/// Subsequent calls on the same thread reuse the cached instance.
///
/// ### Params:
/// - `config`: The tracker configuration used to build the instance on first access.
/// - `f`: A closure receiving a mutable reference to the tracker backend.
///
/// ### Returns:
/// The result of the closure `f`.
pub fn with_local_tracker<F, R>(config: &TrackerConfig, f: F) -> R
where
    F: FnOnce(&mut dyn LipTrackerBackend) -> R,
{
    TRACKER_TLS.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(config.build());
        }
        f(&mut **opt.as_mut().unwrap())
    })
}
