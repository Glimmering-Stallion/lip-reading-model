//! Temporal mouth tracking and Region of Interest (ROI) extraction.
//!
//! This module provides the ```LipTracker```, which uses Haar cascades to detect
//! mouth positions and applies Alpha smoothing to maintain a stable bounding 
//! box across video frames. It makes use of Thread Local Storage (TLS) to manage 
//! expensive classifier instances across multiple threads safely.



use std::{
    error::Error,
    cell::RefCell,
    path::{Path, PathBuf},
};
use burn::config::Config;
use opencv::{
    core::{Mat, Rect, Size, Vector},
    imgproc::{self, INTER_LINEAR},
    objdetect::CascadeClassifier,
    prelude::*,
};



// thread local storage (TLS) for a tracker instance
// tracker is contained in a RefCell for interior mutabiliity of the tracker,
// while the TLS item itself remains immutable
thread_local! {
    static TRACKER_TLS: RefCell<Option<LipTracker>> = RefCell::new(None);
}



#[derive(Debug, Config)]
pub struct LipTrackerConfig {
    pub cascade_path: PathBuf,
    pub target_dims: (i32, i32), // (height, width)
}



pub struct LipTracker {
    pub mouth_cascade: CascadeClassifier,
    pub target_dims: (i32, i32), // (height, width)
    pub prev_center: Option<(f32, f32)>, //(x-position, y-position)
}



impl LipTrackerConfig {
    pub fn init(&self) -> LipTracker {
        LipTracker::new(
            self.cascade_path.clone(),
            self.target_dims,
        )
    }
}



impl LipTracker {
    pub fn new<P: AsRef<Path>>(cascade_path: P, target_dims: (i32, i32)) -> Self {
        let cascade_path = cascade_path.as_ref().to_str().expect("Invalid cascade path");
        let mouth_cascade = CascadeClassifier::new(cascade_path)
            .expect("Failed to load mouth cascade XML");

        Self { mouth_cascade, target_dims, prev_center: None }
    }

    /// function to process a given frame
    /// works by:
    /// - finding a valid ROI box over a detected mouth
    /// - applying necessary scaling and cropping on that frame to match a given target dim
    /// params:
    /// - frame: the frame to process
    /// returns: a frame cropped to the detected mouth region
    pub fn process_frame(&mut self, frame: &Mat) -> Result<Mat, Box<dyn Error>> {
        // obtain ROI detection box from given frame
        let roi = self.detect_mouth_roi(frame).ok_or("No mouth found")?;

        // find center of ROI box position
        let curr_center = (
            (roi.x + roi.width / 2) as f32,
            (roi.y + roi.height / 2) as f32,
        );

        // positional smoothing across frames
        let alpha = 0.6;
        let (smooth_x, smooth_y) = match self.prev_center {
            Some(prev_center) => (
                alpha * prev_center.0 + (1.0 - alpha) * curr_center.0,
                alpha * prev_center.1 + (1.0 - alpha) * curr_center.1,
            ),
            None => curr_center,
        };
        self.prev_center = Some((smooth_x, smooth_y));

        // create updated bounding box ROI at a position smoothed across frames
        let (h, w) = self.target_dims;
        let roi = Rect::new(
            (smooth_x as i32) - (w as i32 / 2),
            (smooth_y as i32) - (h as i32 / 2),
            w as i32,
            h as i32,
        );

        // apply rescale logic on frame based on ROI box
        Ok(self.rescale_to_target_dims(frame, roi))
    }

    /// looks for a mouth within a given frame and applies a bounding box to it
    /// this provides position invariance
    /// params:
    /// - frame: the given frame to determine the bounding box
    /// returns: the best found bounding box rectangle
    pub fn detect_mouth_roi(&mut self, frame: &Mat) -> Option<Rect> {
        let mut detections = Vector::<Rect>::new();  // container for detected candidate mouth positions, given current frame
        let scale_factor = 1.1;                                     // how much image size is reduced at varying scales
        let min_neighbors = 8;                                      // how many pre-detections must be made on the same area to consider that area a positive (higher values reduce false positives)
        let flags = 0;
        let (min_size, max_size) = (
            Size::new(30, 30),                             // min size of objects to consider detecting
            Size::new(300, 300),                           // max size of objects to consider detecting
        );

        let _ = self.mouth_cascade.detect_multi_scale(
            frame,
            &mut detections,
            scale_factor,
            min_neighbors,
            flags,
            min_size,
            max_size,
        );

        // find best candidate ROI box and return
        if detections.is_empty() { return None };
        let rect = detections.iter().max_by_key(|r| r.width * r.height).unwrap();
        Some(rect)
    }

    /// takes a frame and a rectangle, crops, clamps, and resizes it to the tracker struct's given target dims
    /// this provides scale invariance
    /// params:
    /// - frame: the frame contents to apply processing to
    /// - roi: the bounding box used to base the processing on
    /// returns: a Mat array representing the newly processed frame
    pub fn rescale_to_target_dims(&self, frame: &Mat, roi: Rect) -> Mat {
        // obtain frame dims
        // clamp ROI to prevent OOB sampling on given frame
        let frame_w = frame.cols();
        let frame_h = frame.rows();
        let (x1, y1, x2, y2) = (
            roi.x.clamp(0, frame_w),
            roi.y.clamp(0, frame_h),
            (roi.x + roi.width).clamp(0, frame_w),
            (roi.y + roi.height).clamp(0, frame_h),
        );

        // obtain ROI dims and define ROI rectangle
        let roi_w = x2 - x1;
        let roi_h = y2 - y1;
        let cropped_roi = Rect::new(
            x1,
            y1,
            roi_w,
            roi_h,
        );

        // define cropped ROI frame and resized buffer
        let cropped_frame = Mat::roi(frame, cropped_roi)
            .expect("ROI out of bounds");
        let mut resized_frame = Mat::default();

        // resize clamped ROI to match target ROI dims if they differ
        if cropped_roi.width != self.target_dims.1 || cropped_roi.height != self.target_dims.0 {
            imgproc::resize(
                &cropped_frame,
                &mut resized_frame,
                Size::new(self.target_dims.1, self.target_dims.0),
                0.0,
                0.0,
                INTER_LINEAR,
            ).expect("Frame to target dims resize failed");
        } else {
            cropped_frame.copy_to(&mut resized_frame).expect("Cropped to resized frame copy failed");
        }

        resized_frame
    }

    /// executes a given closure operation using current thread's local tracker
    /// this hides the RefCell and thread_local logic from other files
    /// params:
    /// - tracker_config: the current thread's lip tracker config to initialize if it doesn't exist
    /// - f: the closure to execute with a mutable reference to the tracker
    pub fn with_local<F, R>(tracker_config: &LipTrackerConfig, f: F) -> R
    where
        F: FnOnce(&mut LipTracker) -> R,
    {
        TRACKER_TLS.with(|cell| {
            let mut tracker_opt = cell.borrow_mut();
            if tracker_opt.is_none() {
                *tracker_opt = Some(tracker_config.init());
            }
            f(tracker_opt.as_mut().unwrap())
        })
    }

    /// resets temporal smoothing state for a new video
    pub fn reset_state(&mut self) { self.prev_center = None; }
}



#[cfg(test)]
mod tests {
    use super::*;
    use opencv::{
        core::{
            CV_8UC1,
            Mat,
            Scalar,
        },
    };

    #[test]
    fn test_centered_roi_dimensions() {
        let target_dims = (50, 150);

        // create dummy detection box and get center of box values
        let detection = Rect::new(100, 100, 50, 40);
        let center_x = detection.x + detection.width / 2;
        let center_y = detection.y + detection.height / 2;

        let x = center_x - target_dims.1 / 2;
        let y = center_y - target_dims.0 / 2;

        let roi = Rect::new(x, y, target_dims.1, target_dims.0);

        println!("\nROI height & width: ({}, {})", roi.height, roi.width);
        println!("Target height & width: ({}, {})\n", target_dims.0, target_dims.1);

        assert_eq!(roi.width, target_dims.1);
        assert_eq!(roi.height, target_dims.0);
    }

    #[test]
    fn test_roi_clamps_to_frame_bounds() {
        let target_dims = (50, 150);

        // create dummy frame
        let frame = Mat::new_rows_cols_with_default(
            200,
            200,
            CV_8UC1,
            Scalar::all(0.0),
        ).unwrap();

        let tracker = LipTracker {
            mouth_cascade: CascadeClassifier::default().unwrap(),
            target_dims,
            prev_center: None,
        };

        // intentionally push ROI partially outside frame
        let roi = Rect::new(180, 180, 100, 100);
        let output = tracker.rescale_to_target_dims(&frame, roi);

        println!("\nTracker output height & width: ({}, {})", output.rows(), output.cols());
        println!("Target height & width: ({}, {})\n", target_dims.0, target_dims.1);

        // must still return target dims
        assert_eq!(output.cols(), target_dims.1);
        assert_eq!(output.rows(), target_dims.0);
    }
}
