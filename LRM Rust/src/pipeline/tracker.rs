//! Temporal mouth tracking and Region of Interest (ROI) extraction.
//!
//! This module provides the `LipTracker`, which uses Haar cascades to detect
//! mouth positions and applies Alpha smoothing to maintain a stable bounding 
//! box across video frames. It makes use of Thread Local Storage (TLS) to manage 
//! expensive classifier instances across multiple threads safely.



use std::{
    error::Error,
    cell::RefCell,
    path::PathBuf,
};
use burn::config::Config;
use opencv::{
    core::{Mat, Point2f, Rect, Size, Vector},
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
    pub face_cascade_path: PathBuf,       // face cascade .xml file path
    pub mouth_cascade_path: PathBuf,      // mouth cascade .xml file path
    pub target_dims: (usize, usize),      // final target dimensions to rescale base frame to for mouth ROI (height, width)

    #[config(default = "25.0")]
    pub max_gating_threshold: f32,        // max allowable Euclidean pixel distance that the mouth ROI center position can move between frames before we reject it as a glitch

    #[config(default = "3.0")]
    pub min_gating_threshold: f32,        // min allowable Euclidean pixel distance that the mouth ROI center position can move between frames before we reject it as noise

    #[config(default = "0.5")]
    pub smoothing_alpha: f32,             // smoothing factor to control amount of weight to most recent frame versus last frame (higher values mean smoother but slower averages)
}



pub struct LipTracker {
    pub face_cascade: CascadeClassifier,  // face cascade struct
    pub mouth_cascade: CascadeClassifier, // mouth cascade struct
    pub prev_center: Option<Point2f>,     // positional coordinates of the center of mouth ROI (x-position, y-position)
    config: LipTrackerConfig,             // config settings
}



impl LipTrackerConfig {
    pub fn init(&self) -> LipTracker {
        // convert paths from PathBuf to string slices
        let face_cascade_path = self.face_cascade_path.to_str().expect("Invalid face cascade path");
        let mouth_cascade_path = self.mouth_cascade_path.to_str().expect("Invalid mouth cascade path");

        // init face and mouth cascade instances
        let face_cascade = CascadeClassifier::new(face_cascade_path)
            .expect("Failed to load face cascade XML");
        let mouth_cascade = CascadeClassifier::new(mouth_cascade_path)
            .expect("Failed to load mouth cascade XML");

        LipTracker {
            face_cascade,
            mouth_cascade,
            prev_center: None,
            config: self.clone(),
        }
    }
}



impl LipTracker {
    /// Hierarchically crops a given frame down to mouth region.
    /// 
    /// Works by:
    /// - finding a valid ROI box over a detected face in the full frame,
    /// - reducing frame by narrowing search region to lower half of face,
    /// - finding a valid ROI box over a detected mouth in the reduced frame,
    /// - applying necessary scaling and cropping on that detected mouth region to match a given target dim.
    ///
    /// ### Params:
    /// - `frame`: The frame to process.
    ///
    /// ### Returns:
    /// A frame cropped to the detected mouth region.
    pub fn process_frame(&mut self, frame: &Mat) -> Result<Mat, Box<dyn Error>> {
        // obtain face ROI detection box from given frame
        let face_roi = match self.detect_face_roi(frame) {
            Some(roi) => roi,
            None => {
                let center = self.prev_center.unwrap_or_else(|| {
                    Point2f::new(frame.cols() as f32 / 2.0, frame.rows() as f32 / 2.0)
                });
                self.prev_center = Some(center);
                let fallback_roi = Rect::new(
                    (center.x as i32) - (self.config.target_dims.1 as i32 / 2),
                    (center.y as i32) - (self.config.target_dims.0 as i32 / 2),
                    self.config.target_dims.1 as i32,
                    self.config.target_dims.0 as i32,
                );
                return Ok(self.rescale_to_target_dims(frame, fallback_roi));
            }
        };

        // shrink face ROI to lower half of face
        // (for reducing search region for mouth detection)
        let half_face_roi = Rect::new(
            face_roi.x,
            face_roi.y + (face_roi.height / 2),
            face_roi.width,
            face_roi.height / 2,
        );
        let half_face_frame = Mat::roi(frame, half_face_roi)?.clone_pointee();

        // obtain mouth ROI detection box from given frame
        let mouth_roi = self.detect_mouth_roi(&half_face_frame);

        // find absolute center of mouth ROI box position from its position relative to the reduced box position
        let curr_center = if let Some(mouth_roi) = mouth_roi {
            let abs_center = Point2f::new(
                (half_face_roi.x + mouth_roi.x + (mouth_roi.width / 2)) as f32,
                (half_face_roi.y + mouth_roi.y + (mouth_roi.height / 2)) as f32,
            );
            self.stabilize_position(abs_center)
        } else {
            self.prev_center.unwrap_or_else(|| {
                Point2f::new(frame.cols() as f32 / 2.0, frame.rows() as f32 / 2.0)
            })
        };

        // update mouth ROI position history
        self.prev_center = Some(curr_center);

        // create updated bounding box ROI
        let final_roi = Rect::new(
            (curr_center.x as i32) - (self.config.target_dims.1 as i32 / 2),
            (curr_center.y as i32) - (self.config.target_dims.0 as i32 / 2),
            self.config.target_dims.1 as i32,
            self.config.target_dims.0 as i32,
        );

        // rescale base frame to target dimensions
        Ok(self.rescale_to_target_dims(frame, final_roi))
    }

    /// Applies Kalman gating and temporal smoothing (EMA) to a given position in relation to a previous position in time to reduce small jittering and large jumps.
    /// 
    /// First filters out a given position if the positional change relative to the previous change is beyond a threshold.
    /// 
    /// Then applies smoothing to the positional changes between current point and previous point using Exponential Moving Average.
    ///
    /// ### Params:
    /// - `curr_point`: The current position to evaluate.
    ///
    /// ### Returns:
    /// A filtered position.
    fn stabilize_position(&mut self, curr_point: Point2f) -> Point2f {
        match self.prev_center {
            Some(prev_point) => {
                // find distance from curr position to prev position
                let (dx, dy) = (
                    curr_point.x - prev_point.x,
                    curr_point.y - prev_point.y,
                );
                let dist = (dx * dx + dy * dy).sqrt();

                // Kalman gating:
                // - if change in position is too large, likely a detection error
                // - if change in position is too small, likely detection noise
                if dist > self.config.max_gating_threshold { return prev_point }
                if dist < self.config.min_gating_threshold { return prev_point }

                // EMA (exponential moving average) smoothing
                let alpha = self.config.smoothing_alpha;
                Point2f::new(
                    alpha * prev_point.x + (1.0 - alpha) * curr_point.x,
                    alpha * prev_point.y + (1.0 - alpha) * curr_point.y,
                )
            }
            None => curr_point,
        }
    }

    /// Looks for a face within a given full-sized frame and applies a bounding box to it.
    /// 
    /// This provides a reduced search region for the mouth ROI detection.
    ///
    /// ### Params:
    /// - `frame`: The given frame to determine the bounding box.
    ///
    /// ### Returns:
    /// The best found bounding box rectangle, or `None` if no face detected.
    pub fn detect_face_roi(&mut self, frame: &Mat) -> Option<Rect> {
        let mut face_detections = Vector::<Rect>::new();   // container for detected candidate face positions, given current frame
        let scale_factor = 1.1;                                           // how much image size is reduced at varying scales
        let min_neighbors = 8;                                            // how many pre-detections must be made on the same area to consider that area a positive (higher values reduce false positives)
        let flags = 0;                                                    // something...
        let (min_size, max_size) = (
            Size::new(80, 80),                                   // min size of face objects to consider detecting
            Size::default(),                                                   // max size of face objects to consider detecting
        );

        let _ = self.face_cascade.detect_multi_scale(
            frame,
            &mut face_detections,
            scale_factor,
            min_neighbors,
            flags,
            min_size,
            max_size,
        );

        // find best candidate ROI box and return
        if face_detections.is_empty() { return None };
        face_detections.iter().max_by_key(|r| r.width * r.height)
    }

    /// Looks for a mouth within a given face frame and applies a bounding box to it.
    /// 
    /// This enables position invariance for the video data that will be fed to the VSRM.
    ///
    /// ### Params:
    /// - `frame`: The given frame to determine the bounding box.
    ///
    /// ### Returns:
    /// The best found bounding box rectangle, or `None` if no mouth detected.
    pub fn detect_mouth_roi(&mut self, frame: &Mat) -> Option<Rect> {
        let mut mouth_detections = Vector::<Rect>::new();  // container for detected candidate mouth positions, given current frame
        let scale_factor = 1.1;                                           // how much image size is reduced at varying scales
        let min_neighbors = 8;                                            // how many pre-detections must be made on the same area to consider that area a positive (higher values reduce false positives)
        let flags = 0;                                                    // something...
        let (min_size, max_size) = (
            Size::new(30, 30),                                   // min size of mouth objects to consider detecting
            Size::new(300, 300),                                 // max size of mouth objects to consider detecting
        );

        let _ = self.mouth_cascade.detect_multi_scale(
            frame,
            &mut mouth_detections,
            scale_factor,
            min_neighbors,
            flags,
            min_size,
            max_size,
        );

        // find best candidate ROI box and return
        if mouth_detections.is_empty() { return None };
        mouth_detections.iter().max_by_key(|r| r.width * r.height)
    }

    /// Takes a frame and a rectangle, crops, clamps, and resizes it to the tracker struct's given target dims.
    /// 
    /// This provides scale invariance.
    ///
    /// ### Params:
    /// - `frame`: The frame contents to apply processing to.
    /// - `roi`: The bounding box used to base the processing on.
    ///
    /// ### Returns:
    /// A `Mat` array representing the newly processed frame.
    pub fn rescale_to_target_dims(&self, frame: &Mat, roi: Rect) -> Mat {
        // obtain base frame dims and target dims
        let (frame_height, frame_width, target_height, target_width) = (
            frame.rows(),
            frame.cols(),
            self.config.target_dims.0 as i32,
            self.config.target_dims.1 as i32,
        );

        // clamp ROI to prevent OOB sampling on given frame
        let (x1, y1, x2, y2) = (
            roi.x.clamp(0, frame_width),
            roi.y.clamp(0, frame_height),
            (roi.x + roi.width).clamp(0, frame_width),
            (roi.y + roi.height).clamp(0, frame_height),
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
        if cropped_roi.width != target_width || cropped_roi.height != target_height {
            imgproc::resize(
                &cropped_frame,
                &mut resized_frame,
                Size::new(target_width, target_height),
                0.0,
                0.0,
                INTER_LINEAR,
            ).expect("Frame to target dims resize failed");
        } else {
            cropped_frame.copy_to(&mut resized_frame).expect("Cropped to resized frame copy failed");
        }

        resized_frame
    }

    /// Executes a given closure operation using current thread's local tracker.
    /// 
    /// This hides the `RefCell` and `thread_local` logic from other files.
    ///
    /// ### Params:
    /// - `tracker_config`: The current thread's lip tracker config to initialize if it doesn't exist.
    /// - `f`: The closure to execute with a mutable reference to the tracker.
    ///
    /// ### Returns:
    /// The result of the closure `f`.
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

    /// Resets temporal smoothing state for a new video.
    pub fn reset_state(&mut self) { self.prev_center = None; }
}



#[cfg(test)]
mod tests {
    use crate::context::Context;

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
        let context = Context::new();
        let face_cascade_path = context.models_path.join("haarcascade_frontalface_alt2.xml");
        let mouth_cascade_path = context.models_path.join("haarcascade_mcs_mouth.xml");
        let target_dims = (50, 150);

        // create dummy frame
        let frame = Mat::new_rows_cols_with_default(
            200,
            200,
            CV_8UC1,
            Scalar::all(0.0),
        ).unwrap();

        let tracker = LipTrackerConfig::new(
            face_cascade_path,
            mouth_cascade_path,
            target_dims,
        ).init();

        // intentionally push ROI partially outside frame
        let roi = Rect::new(180, 180, 100, 100);
        let output = tracker.rescale_to_target_dims(&frame, roi);

        println!("\nTracker output height & width: ({}, {})", output.rows(), output.cols());
        println!("Target height & width: ({}, {})\n", target_dims.0, target_dims.1);

        // must still return target dims
        assert_eq!(output.cols(), target_dims.1 as i32);
        assert_eq!(output.rows(), target_dims.0 as i32);
    }
}
