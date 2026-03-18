//! Haar cascade tracker backend.
//!
//! Uses OpenCV Haar cascade classifiers for hierarchical face and mouth
//! detection, with distance-gated EMA smoothing for temporal stability.



use burn::config::Config;
use opencv::{
    core::{Mat, Point2f, Rect, Size, Vector},
    imgproc::{self, INTER_LINEAR},
    objdetect::CascadeClassifier,
    prelude::*,
};
use crate::prelude::{io_err, ESS};
use std::{
    io::ErrorKind,
    path::PathBuf,
};

use super::tracker::{LipTrackerBackend, TrackerResult, VizMetadata};



#[derive(Debug, Config)]
pub struct HaarTrackerConfig {
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



pub struct HaarTracker {
    pub face_cascade: CascadeClassifier,
    pub mouth_cascade: CascadeClassifier,
    pub prev_center: Option<Point2f>,
    config: HaarTrackerConfig,
}



impl HaarTrackerConfig {
    /// Initializes a [`HaarTracker`] from this configuration.
    ///
    /// ### Returns:
    /// A configured `HaarTracker` ready to process frames.
    pub fn init(&self) -> HaarTracker {
        let face_cascade_path = self.face_cascade_path.to_str().expect("invalid face cascade path");
        let mouth_cascade_path = self.mouth_cascade_path.to_str().expect("invalid mouth cascade path");

        let face_cascade = CascadeClassifier::new(face_cascade_path)
            .expect("failed to load face cascade XML");
        let mouth_cascade = CascadeClassifier::new(mouth_cascade_path)
            .expect("failed to load mouth cascade XML");

        HaarTracker {
            face_cascade,
            mouth_cascade,
            prev_center: None,
            config: self.clone(),
        }
    }
}



impl LipTrackerBackend for HaarTracker {
    /// Hierarchically crops a given frame down to the mouth region.
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
    /// A [`TrackerResult`] containing the mouth crop and visualization metadata.
    fn process_frame(&mut self, frame: &Mat) -> Result<TrackerResult, ESS> {
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
                return Ok(TrackerResult {
                    crop: self.rescale_to_target_dims(frame, fallback_roi),
                    metadata: VizMetadata {
                        face_rect: None,
                        mouth_rect: None,
                        landmarks: None,
                        stabilized_center: Some(center),
                    },
                });
            }
        };

        // shrink face ROI to lower half of face
        let half_face_roi = Rect::new(
            face_roi.x,
            face_roi.y + (face_roi.height / 2),
            face_roi.width,
            face_roi.height / 2,
        );
        let half_face_frame = Mat::roi(frame, half_face_roi)
            .map_err(|e| io_err(e.to_string(), ErrorKind::Other))?
            .clone_pointee();

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

        self.prev_center = Some(curr_center);

        let final_roi = Rect::new(
            (curr_center.x as i32) - (self.config.target_dims.1 as i32 / 2),
            (curr_center.y as i32) - (self.config.target_dims.0 as i32 / 2),
            self.config.target_dims.1 as i32,
            self.config.target_dims.0 as i32,
        );

        // convert mouth_roi from half-face-relative to frame-absolute for visualization
        let abs_mouth_rect = mouth_roi.map(|mr| Rect::new(
            half_face_roi.x + mr.x,
            half_face_roi.y + mr.y,
            mr.width,
            mr.height,
        ));

        Ok(TrackerResult {
            crop: self.rescale_to_target_dims(frame, final_roi),
            metadata: VizMetadata {
                face_rect: Some(face_roi),
                mouth_rect: abs_mouth_rect,
                landmarks: None,
                stabilized_center: Some(curr_center),
            },
        })
    }

    /// Resets temporal smoothing state for a new video.
    fn reset_state(&mut self) { self.prev_center = None; }

    /// Returns the target output dimensions `(height, width)` for the mouth crop.
    fn target_dims(&self) -> (usize, usize) { self.config.target_dims }
}



impl HaarTracker {
    /// Applies Kalman gating and temporal smoothing (EMA) to a given position
    /// in relation to a previous position in time to reduce small jittering
    /// and large jumps.
    ///
    /// First filters out a given position if the positional change relative
    /// to the previous change is beyond a threshold.
    ///
    /// Then applies smoothing to the positional changes between current point
    /// and previous point using Exponential Moving Average.
    ///
    /// ### Params:
    /// - `curr_point`: The current position to evaluate.
    ///
    /// ### Returns:
    /// A filtered position.
    fn stabilize_position(&mut self, curr_point: Point2f) -> Point2f {
        match self.prev_center {
            Some(prev_point) => {
                let (dx, dy) = (
                    curr_point.x - prev_point.x,
                    curr_point.y - prev_point.y,
                );
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > self.config.max_gating_threshold { return prev_point }
                if dist < self.config.min_gating_threshold { return prev_point }

                let alpha = self.config.smoothing_alpha;
                Point2f::new(
                    alpha * prev_point.x + (1.0 - alpha) * curr_point.x,
                    alpha * prev_point.y + (1.0 - alpha) * curr_point.y,
                )
            }
            None => curr_point,
        }
    }

    /// Looks for a face within a given full-sized frame and applies a bounding
    /// box to it.
    ///
    /// This provides a reduced search region for the mouth ROI detection.
    ///
    /// ### Params:
    /// - `frame`: The given frame to determine the bounding box.
    ///
    /// ### Returns:
    /// The best found bounding box rectangle, or `None` if no face detected.
    pub fn detect_face_roi(&mut self, frame: &Mat) -> Option<Rect> {
        let mut face_detections = Vector::<Rect>::new();
        let scale_factor = 1.1;
        let min_neighbors = 8;
        let flags = 0;
        let (min_size, max_size) = (
            Size::new(80, 80),
            Size::default(),
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

        if face_detections.is_empty() { return None };
        face_detections.iter().max_by_key(|r| r.width * r.height)
    }

    /// Looks for a mouth within a given face frame and applies a bounding
    /// box to it.
    ///
    /// This enables position invariance for the video data that will be fed
    /// to the VSRM.
    ///
    /// ### Params:
    /// - `frame`: The given frame to determine the bounding box.
    ///
    /// ### Returns:
    /// The best found bounding box rectangle, or `None` if no mouth detected.
    pub fn detect_mouth_roi(&mut self, frame: &Mat) -> Option<Rect> {
        let mut mouth_detections = Vector::<Rect>::new();
        let scale_factor = 1.1;
        let min_neighbors = 8;
        let flags = 0;
        let (min_size, max_size) = (
            Size::new(30, 30),
            Size::new(300, 300),
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

        if mouth_detections.is_empty() { return None };
        mouth_detections.iter().max_by_key(|r| r.width * r.height)
    }

    /// Takes a frame and a rectangle, crops, clamps, and resizes it to the
    /// tracker's configured target dims.
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
        let (frame_height, frame_width, target_height, target_width) = (
            frame.rows(),
            frame.cols(),
            self.config.target_dims.0 as i32,
            self.config.target_dims.1 as i32,
        );

        let (x1, y1, x2, y2) = (
            roi.x.clamp(0, frame_width),
            roi.y.clamp(0, frame_height),
            (roi.x + roi.width).clamp(0, frame_width),
            (roi.y + roi.height).clamp(0, frame_height),
        );

        let roi_w = x2 - x1;
        let roi_h = y2 - y1;
        let cropped_roi = Rect::new(x1, y1, roi_w, roi_h);

        let cropped_frame = Mat::roi(frame, cropped_roi)
            .expect("ROI out of bounds");
        let mut resized_frame = Mat::default();

        if cropped_roi.width != target_width || cropped_roi.height != target_height {
            imgproc::resize(
                &cropped_frame,
                &mut resized_frame,
                Size::new(target_width, target_height),
                0.0,
                0.0,
                INTER_LINEAR,
            ).expect("frame to target dims resize failed");
        } else {
            cropped_frame.copy_to(&mut resized_frame).expect("cropped to resized frame copy failed");
        }

        resized_frame
    }
}



#[cfg(test)]
mod tests {
    use crate::context::Context;

    use super::*;
    use opencv::core::{CV_8UC1, Mat, Scalar};

    #[test]
    fn test_centered_roi_dimensions() {
        let target_dims = (50, 150);

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

        let frame = Mat::new_rows_cols_with_default(
            200,
            200,
            CV_8UC1,
            Scalar::all(0.0),
        ).unwrap();

        let tracker = HaarTrackerConfig::new(
            face_cascade_path,
            mouth_cascade_path,
            target_dims,
        ).init();

        let roi = Rect::new(180, 180, 100, 100);
        let output = tracker.rescale_to_target_dims(&frame, roi);

        println!("\nTracker output height & width: ({}, {})", output.rows(), output.cols());
        println!("Target height & width: ({}, {})\n", target_dims.0, target_dims.1);

        assert_eq!(output.cols(), target_dims.1 as i32);
        assert_eq!(output.rows(), target_dims.0 as i32);
    }
}
