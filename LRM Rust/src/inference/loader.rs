//! Video loading utilities for inference.
//!
//! - `load_video`: loads a standardized video file for inference by running mouth tracking on each frame, and returning a `FramesBuffer`
//! - `open_camera`: opens a camera device for live capture
//! - `load_frame`: reads and converts a single frame from any `VideoCapture` source



use crate::{
    pipeline::{
        tracker::LipTrackerBackend,
        FramesBuffer,
    },
    prelude::{io_err, ESS},
};
use opencv::{
    core::{
        AlgorithmHint,
        Mat,
        MatTraitConst,
        MatTraitConstManual,
    },
    imgproc,
    videoio::{
        VideoCapture,
        VideoCaptureTrait,
        VideoCaptureTraitConst,
        CAP_ANY,
    },
};
use std::{
    io::ErrorKind,
    path::Path,
};



/// Loads a video file, runs mouth tracking on each frame, and collects
/// the resulting crops into a contiguous grayscale `FramesBuffer`.
///
/// Mirrors the slow path in `GridDataset::load_frames` but accepts an
/// arbitrary file path instead of a GRID corpus entry.
///
/// ### Params:
/// - `path`: Path to the video file (.mpg, .avi, .mp4, etc.).
/// - `tracker`: Mutable reference to any lip tracker backend.
///
/// ### Returns:
/// A `FramesBuffer` of tracked and cropped mouth regions, or an error
/// if the video cannot be opened or has no decodable frames.
pub fn load_video(
    path: &Path,
    tracker: &mut dyn LipTrackerBackend,
) -> Result<FramesBuffer, ESS> {
    let path_str = path.to_str().ok_or_else(|| io_err("invalid video path", ErrorKind::InvalidInput))?;
    let mut cap = VideoCapture::from_file(path_str, CAP_ANY)
        .map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;

    if !cap.is_opened().map_err(|e| io_err(e.to_string(), ErrorKind::Other))? {
        return Err(io_err(format!("failed to open video: {}", path_str), ErrorKind::Other));
    }

    tracker.reset_state();

    let mut frames: Vec<u8> = Vec::new();
    let (mut frame_h, mut frame_w) = (0usize, 0usize);
    let mut orig_frame = Mat::default();
    let mut gray_frame = Mat::default();

    while cap.read(&mut orig_frame).map_err(|e| io_err(e.to_string(), ErrorKind::Other))? {
        if orig_frame.empty() { break; }

        imgproc::cvt_color(
            &orig_frame,
            &mut gray_frame,
            imgproc::COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        ).map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;

        let result = tracker.process_frame(&gray_frame)?;
        let crop = result.crop;
        let size = crop.size().map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;
        frame_h = size.height as usize;
        frame_w = size.width as usize;
        frames.extend(crop.data_bytes()?);
    }

    if frames.is_empty() {
        return Err(io_err(format!("no frames decoded from video: {}", path_str), ErrorKind::InvalidData));
    }

    Ok(FramesBuffer {
        data: frames,
        height: frame_h,
        width: frame_w,
    })
}



/// Opens a camera device for live video capture.
///
/// ### Params:
/// - `device_id`: Camera device index (0 = default webcam, 1+ = other devices).
///
/// ### Returns:
/// An opened `VideoCapture` handle, or an error if the device cannot be opened.
pub fn open_camera(device_id: i32) -> Result<VideoCapture, ESS> {
    let cap = VideoCapture::new(device_id, CAP_ANY)
        .map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;

    if !cap.is_opened().map_err(|e| io_err(e.to_string(), ErrorKind::Other))? {
        return Err(io_err(
            format!("no camera available at device index {}; ensure a webcam is connected and not in use by another application", device_id),
            ErrorKind::NotFound,
        ));
    }

    Ok(cap)
}



/// Reads and converts one frame from a `VideoCapture` source to grayscale.
///
/// ### Params:
/// - `cap`: Mutable reference to an open `VideoCapture` (camera or file).
///
/// ### Returns:
/// `Ok(Some(gray_frame))` on success, `Ok(None)` if the stream has ended,
/// or an error on capture failure.
pub fn load_frame(cap: &mut VideoCapture) -> Result<Option<Mat>, ESS> {
    let mut frame = Mat::default();
    if !cap.read(&mut frame).map_err(|e| io_err(e.to_string(), ErrorKind::Other))? || frame.empty() {
        return Ok(None);
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(
        &frame,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    ).map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;

    Ok(Some(gray))
}
