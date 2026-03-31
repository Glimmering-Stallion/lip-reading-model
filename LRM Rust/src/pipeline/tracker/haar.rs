//! Haar cascade tracker backend.
//!
//! Uses OpenCV Haar cascade classifiers for hierarchical face and mouth
//! detection, with distance-gated EMA smoothing for temporal stability.



use burn::config::Config;
use opencv::{
    Result,
    core::{
        self as cv_core,
        AlgorithmHint,
        Mat,
        Point2f,
        Rect,
        Size,
        Vector,
    },
    imgproc::{self, INTER_LINEAR},
    objdetect::CascadeClassifier,
    prelude::*
};
use crate::prelude::{io_err, ESS};
use std::{
    io::ErrorKind,
    path::PathBuf,
};

use super::tracker::{
    LipTrackerBackend,
    TrackerResult,
    VizMetadata,
};



#[derive(Debug, Config)]
pub struct HaarTrackerConfig {
    pub face_cascade_path: PathBuf,       // face cascade .xml file path
    pub mouth_cascade_path: PathBuf,      // mouth cascade .xml file path
    pub target_dims: (usize, usize),      // final target dimensions to rescale base frame to for mouth ROI (height, width)

    #[config(default = "2500.0")]
    pub max_gating_threshold: f32,        // max allowable Euclidean pixel distance that the mouth ROI center position can move between frames before we reject it as a glitch

    #[config(default = "3.0")]
    pub min_gating_threshold: f32,        // min allowable Euclidean pixel distance that the mouth ROI center position can move between frames before we reject it as noise

    #[config(default = "0.5")]
    pub smoothing_alpha: f32,             // smoothing factor to control amount of weight to most recent frame versus last frame (higher values mean smoother but slower averages)

    // abstract speech activity detection parameters

    #[config(default = "2.0")]
    pub energy_threshold: f32,            // min temporal energy threshold to consider the mouth as in motion, determined using MAD (Mean Absolute Deviation) of float gradient magnitudes

    #[config(default = "0.5")]
    pub mouth_activity_zone_scale: f32,   // size of inner mouth crop region to measure lip motion (fraction of crop width/height) where articulation is expected to dominate here and rest of crop is considered periphery

    #[config(default = "1.35")]
    pub mouth_isolation_ratio: f32,       // mouth to periphery motion ratio, which determines how much stronger the motion in the inner region must be vs. the peripheral region to be considered mouth activity

    #[config(default = "1e-6")]
    pub min_periphery_motion: f32,        // min value used for periphery mean when forming the mouth isolation ratio (purely numerical)
}



pub struct HaarTracker {
    pub face_cascade: CascadeClassifier,
    pub mouth_cascade: CascadeClassifier,
    pub prev_center: Option<Point2f>,
    pub prev_magnitude: Option<Mat>,
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
            prev_magnitude: None,
            config: self.clone(),
        }
    }
}



impl LipTrackerBackend for HaarTracker {
    /// Hierarchically crops a given frame down to the mouth region.
    ///
    /// Works by:
    /// - finding a valid ROI box over a detected face in the full frame,
    /// - reducing frame by narrowing search region to the lower third of the face ROI,
    /// - finding a valid ROI box over a detected mouth in the reduced frame,
    /// - applying necessary scaling and cropping on that detected mouth region to match a given target dim.
    ///
    /// ### Params:
    /// - `frame`: The frame to process.
    ///
    /// ### Returns:
    /// A [`TrackerResult`] containing the mouth crop, visualization metadata, and tracker/speech statuses.
    fn process_frame(&mut self, frame: &Mat) -> Result<TrackerResult, ESS> {
        let face_roi = match self.detect_face_roi(frame) {
            Some(roi) => roi,
            None => {
                let center = self.prev_center
                    .unwrap_or_else(|| { Point2f::new(frame.cols() as f32 / 2.0, frame.rows() as f32 / 2.0) });
                self.prev_center = Some(center);
                self.prev_magnitude = None;
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
                    has_lock: false,
                    has_lip_motion: false,
                });
            }
        };

        // mouth search region: lower third of the face box (avoids nose / upper face in Haar mouth pass)
        let lower_third_y = face_roi.y + (2 * face_roi.height / 3);
        let lower_third_h = (face_roi.height - (2 * face_roi.height / 3)).max(1);
        let lower_third_face_roi = Rect::new(
            face_roi.x,
            lower_third_y,
            face_roi.width,
            lower_third_h,
        );
        let lower_third_face_frame = Mat::roi(frame, lower_third_face_roi)
            .map_err(|e| io_err(e.to_string(), ErrorKind::Other))?
            .clone_pointee();

        let mouth_roi = self.detect_mouth_roi(&lower_third_face_frame);

        // absolute mouth center: mouth detection coords are relative to the lower-third subframe
        let curr_center = if let Some(mouth_roi) = mouth_roi {
            let abs_center = Point2f::new(
                (lower_third_face_roi.x + mouth_roi.x + (mouth_roi.width / 2)) as f32,
                (lower_third_face_roi.y + mouth_roi.y + (mouth_roi.height / 2)) as f32,
            );
            self.stabilize_position(abs_center)
        } else {
            self.prev_center
                .unwrap_or_else(|| { Point2f::new(frame.cols() as f32 / 2.0, frame.rows() as f32 / 2.0) })
        };
        self.prev_center = Some(curr_center);

        // use face width as reference to dynamically size final roi
        let aspect_ratio = self.config.target_dims.0 as f32 / self.config.target_dims.1 as f32;
        let dyn_w = face_roi.width as f32 * 0.45;
        let dyn_h = dyn_w * aspect_ratio;

        let final_roi = Rect::new(
            (curr_center.x as i32) - (dyn_w as i32 / 2),
            (curr_center.y as i32) - (dyn_h as i32 / 2),
            dyn_w as i32,
            dyn_h as i32,
        );

        let final_crop = self.rescale_to_target_dims(frame, final_roi);

        // mouth box from lower-third-relative to full-frame coordinates for visualization
        let abs_mouth_rect = mouth_roi.map(|mr| Rect::new(
            lower_third_face_roi.x + mr.x,
            lower_third_face_roi.y + mr.y,
            mr.width,
            mr.height,
        ));

        let metadata = VizMetadata {
            face_rect: Some(face_roi),
            mouth_rect: abs_mouth_rect,
            landmarks: None,
            stabilized_center: Some(curr_center),
        };

        // tracker lock and per-frame lip-motion proxy (MAD on consecutive crops)
        let has_lock = self.has_lock(&metadata);
        let has_lip_motion = if has_lock { self.has_lip_motion(&final_crop) }
        else { self.prev_magnitude = None; false };

        Ok(TrackerResult {
            crop: final_crop,
            metadata,
            has_lock,
            has_lip_motion,
        })
    }

    /// Returns `true` when temporal gradient change is strong in the **inner** mouth zone and
    /// dominates the **periphery** (donut-style core vs complement), using [`HaarTrackerConfig::energy_threshold`],
    /// [`HaarTrackerConfig::mouth_isolation_ratio`], and related fields.
    ///
    /// This is a **visual lip-motion** cue, not linguistic “talking.” Updates the lip-motion baseline for the next frame.
    fn has_lip_motion(&mut self, curr_crop: &Mat) -> bool {
        let mut is_moving = false;
        let curr_magnitude = Self::calc_gradient_magnitudes(curr_crop, 1.5);

        // obtain prev gradient magnitude
        if let Some(ref prev_magnitude) = self.prev_magnitude
        && prev_magnitude.size().unwrap() == curr_crop.size().unwrap() {
            // calc mean absolute deviation between prev and curr gradient magnitudes
            let mut diff = Mat::default();
            if cv_core::absdiff(prev_magnitude, &curr_magnitude, &mut diff).is_ok()
            && let Ok(diff_dims) = diff.size() {
                if let Some(core) = Self::calc_centered_rect_scaled(diff_dims.width, diff_dims.height, self.config.mouth_activity_zone_scale)
                && let Ok((inner_mean, periphery_mean)) = Self::calc_core_and_border_means(&diff, core) {
                    let epsilon = self.config.min_periphery_motion as f64;
                    let periphery_mean = periphery_mean.max(epsilon);
                    let ratio = inner_mean / periphery_mean;
                    let thr = self.config.energy_threshold as f64;
                    is_moving = (inner_mean > thr) && (ratio >= f64::from(self.config.mouth_isolation_ratio));
                }
            }
        }

        // save current crop for next frame comparison
        self.prev_magnitude = Some(curr_magnitude);
        is_moving
    }

    /// Returns `true` when tracking has lock on a valid face and mouth region to feed a cropped frame to the VSRM sliding window.
    fn has_lock(&self, metadata: &VizMetadata) -> bool
    { metadata.face_rect.is_some() && metadata.mouth_rect.is_some() }

    /// Resets temporal smoothing and lip-motion baseline states for a new video.
    fn reset_state(&mut self) {
        self.prev_center = None;
        self.prev_magnitude = None;
    }

    /// Returns the target output dimensions `(height, width)` for the mouth crop.
    fn target_dims(&self) -> (usize, usize) { self.config.target_dims }

    /// Returns a processed visualization crop of the cropped mouth region with edge gradient magnitudes (Haar specific) calculated,
    /// for visualizing motion cues in the mouth region.
    fn mouth_crop_inset(&self, crop: &Mat, _metadata: &VizMetadata) -> Option<Mat> {
        if crop.empty() { return None; }
        let mag = Self::calc_gradient_magnitudes(crop, 1.0);
        let mut u8_out = Mat::default();

        cv_core::normalize(
            &mag,
            &mut u8_out,
            0.0,
            255.0,
            cv_core::NORM_MINMAX,
            cv_core::CV_8U,
            &cv_core::no_array(),
        ).ok()?;

        Some(u8_out)
    }
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

                // if dist > self.config.max_gating_threshold { return prev_point }
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
        } else { cropped_frame.copy_to(&mut resized_frame).expect("cropped to resized frame copy failed"); }

        resized_frame
    }

    /// Takes a frame and calculates the combined horizontal/vertical edge gradient magnitudes.
    /// 
    /// ### Params:
    /// - `frame`: Input frame to calculate element-wise magnitudes for.
    /// - `blur_sigma`: Gaussian blurring spread factor to apply to the input frame.
    /// 
    /// ### Returns:
    /// A `Mat` array containing edge magnitudes per pixel position.
    fn calc_gradient_magnitudes(frame: &Mat, blur_sigma: f64) -> Mat {
        let mut src = Mat::default();
        if blur_sigma > 0.0 {
            let kernel_size = Size::new(0, 0);
            imgproc::gaussian_blur(
                frame,
                &mut src,
                kernel_size,
                blur_sigma,
                blur_sigma,
                cv_core::BORDER_DEFAULT,
                AlgorithmHint::ALGO_HINT_DEFAULT,
            ).expect("gaussian blur failed");
        } else { src = frame.clone() }

        let mut gx = Mat::default();
        let mut gy = Mat::default();

        imgproc::sobel(
            &src,
            &mut gx,
            cv_core::CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            cv_core::BORDER_DEFAULT,
        ).expect("sobel x failed");

        imgproc::sobel(
            &src,
            &mut gy,
            cv_core::CV_32F,
            0,
            1,
            3,
            1.0,
            0.0,
            cv_core::BORDER_DEFAULT,
        ).expect("sobel y failed");

        let mut magnitude = Mat::default();
        cv_core::magnitude(&gx, &gy, &mut magnitude).expect("magnitude calculation failed");

        magnitude
    }

    /// Finds a centered rectangle within given dimensions scaled by a given factor.
    /// 
    /// For inner vs. outer zone motion stats on the mouth crop,
    /// we want a centered rectangle covering a fraction of the full crop's width and height to define the inner zone,
    /// and rest of crop is periphery.
    /// 
    /// ### Params:
    /// - `width`: Width of the full crop.
    /// - `height`: Height of the full crop.
    /// - `scale`: Fraction of width and height to use for the inner rectangle (between 0 and 1).
    /// 
    /// ### Returns:
    /// A centered `Rect` with the given scale, or `None` if the resulting rectangle would have non-positive dimensions.
    fn calc_centered_rect_scaled(width: i32, height: i32, scale: f32) -> Option<Rect> {
        let scale = scale.clamp(0.05, 1.0);

        let scaled_width = ((width as f32) * scale).round() as i32;
        let scaled_height = ((height as f32) * scale).round() as i32;
        if scaled_width < 1 || scaled_height < 1 { return None; }

        // top left origin coordinates for centered and scaled rectangle
        let x = (width - scaled_width) / 2;
        let y = (height - scaled_height) / 2;

        Some(Rect::new(x, y, scaled_width, scaled_height))
    }

    /// Calculates mean difference in a given core and periphery regions of a given cropped mouth frame.
    /// 
    /// The core is defined by a given rectangle and the periphery is defined as the complementary pixels in the frame.
    /// 
    /// ### Params:
    /// - `frame`: The given cropped mouth frame to calculate the means on.
    /// - `core`: The rectangle defining the core region to calculate the inner mean on.
    /// 
    /// ### Returns:
    /// A tuple of `(core_mean, periphery_mean)`.
    fn calc_core_and_border_means(frame: &Mat, core: Rect) -> Result<(f64, f64)> {
        let (frame_h, frame_w) = (frame.rows(), frame.cols());
        let frame_area = (frame_h * frame_w) as f64;

        let core_roi = Mat::roi(frame, core)?;
        let core_area = (core.width * core.height) as f64;
        
        let global_mean = cv_core::mean(frame, &cv_core::no_array())?[0];
        let core_mean = cv_core::mean(&core_roi, &cv_core::no_array())?[0];

        let global_sum = global_mean * frame_area;
        let core_sum = core_mean * core_area;

        let periphery_area = (frame_area - core_area).max(1.0);
        let periphery_mean = (global_sum - core_sum) / periphery_area;

        Ok((core_mean, periphery_mean))
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
