//! Live inference visualization overlay.
//!
//! - [`FrameAnnotator`]: stateless drawing of ML metadata on any `Mat` (live window, file export, etc.)
//! - [`LiveWindow`]: OpenCV HighGUI window for displaying frames and handling user input (e.g., ESC)



use crate::{
    pipeline::tracker::{LipTrackerBackend, TrackerResult, VizMetadata},
    prelude::{io_err, ESS},
};
use opencv::{
    core::{
        Mat,
        Point,
        Rect,
        Scalar,
        Size,
    },
    highgui,
    imgproc,
};
#[allow(unused_imports)]
use opencv::prelude::*;



/// Stateless overlay drawing (backend-agnostic [`VizMetadata`] + prediction text).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameAnnotator;



/// Per-frame HUD scaling derived from frame size (for both live cam feed and static video file exports)
#[derive(Debug, Clone, Copy)]
pub struct OverlayLayout {
    pub margin: i32,                // margins from the edges of the frame to the overlay element
    pub text_line_spacing: i32,     // spacing between text lines
    pub text_size: f64,             // size of the text
    pub text_thickness: i32,        // thickness of the text
    pub roi_thickness: i32,         // thickness of the ROI boxes
    pub dot_radius: i32,            // radius of any type of filled points
    pub pip_max_dim: i32,           // max size that the longest dim in the picture-in-picture overlay could take
    pub pip_border_thickness: i32,  // thickness of the border for the picture-in-picture overlay
}



/// Manages the OpenCV HighGUI display window for live inference.
///
/// Handles `named_window`, `imshow`, `wait_key`, and destroys the window on drop.
pub struct LiveWindow { window_name: String }



impl FrameAnnotator {
    /// Draws tracker bounding boxes and stabilized center onto a frame (backend-specific).
    ///
    /// Renders whatever metadata the tracker backend populated:
    /// - Face bounding box (green)
    /// - Mouth bounding box (blue)
    /// - Landmark points (cyan dots), if available
    /// - Stabilized center (red dot)
    ///
    /// ### Params:
    /// - `frame`: The image to draw on (mutated in place).
    /// - `metadata`: Visualization metadata from the tracker backend.
    /// - `layout`: The overlay element scaling context.
    pub fn draw_tracker_info(
        &self,
        frame: &mut Mat,
        metadata: &VizMetadata,
        layout: &OverlayLayout,
    ) {
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let blue = Scalar::new(255.0, 0.0, 0.0, 0.0);
        let cyan = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let red = Scalar::new(0.0, 0.0, 255.0, 0.0);

        if let Some(face_rect) = metadata.face_rect {
            let _ = imgproc::rectangle(frame, face_rect, green, layout.roi_thickness, imgproc::LINE_8, 0);
        }

        if let Some(mouth_rect) = metadata.mouth_rect {
            let _ = imgproc::rectangle(frame, mouth_rect, blue, layout.roi_thickness, imgproc::LINE_8, 0);
        }

        if let Some(ref landmarks) = metadata.landmarks {
            for pt in landmarks {
                let center = Point::new(pt.x as i32, pt.y as i32);
                let _ = imgproc::circle(frame, center, layout.dot_radius, cyan, -1, imgproc::LINE_8, 0);
            }
        }

        if let Some(center) = metadata.stabilized_center {
            let pt = Point::new(center.x as i32, center.y as i32);
            let _ = imgproc::circle(frame, pt, layout.dot_radius, red, -1, imgproc::LINE_8, 0);
        }
    }

    /// Draws bottom-right picture-in-picture mouth crop debug inset onto a frame (backend-specific).
    ///
    /// ### Params:
    /// - `frame`: The image frame to draw on (mutated in place).
    /// - `tracker`: Active tracker (Haar, future landmark backend, …).
    /// - `result`: Latest [`TrackerResult`] (`crop` + `metadata` passed into the trait hook).
    /// - `layout`: The overlay element scaling context.
    pub fn draw_mouth_crop_inset(
        &self,
        frame: &mut Mat,
        tracker: &dyn LipTrackerBackend,
        result: &TrackerResult,
        layout: &OverlayLayout,
    ) {
        if result.crop.empty() { return; }
        let Some(pip) = tracker.mouth_crop_inset(&result.crop, &result.metadata) else { return; };

        let margin = layout.margin;
        let (frame_h, frame_w) = (frame.rows(), frame.cols());
        let (pip_h, pip_w) = (pip.rows(), pip.cols());

        if frame_h < 24 || frame_w < 24 { return; }
        if pip_h < 1 || pip_w < 1 { return; }

        // find available space for pip overlay
        let avail_w = (frame_w - 2 * margin).max(1) as f64;
        let avail_h = (frame_h - 2 * margin).max(1) as f64;
        let bigger_pip_dim = pip_h.max(pip_w) as f64;

        // scale pip overlay with a constraint on pip overlay's longest dim relative to the available space in the frame's shortest dim
        let target_limit_ratio = layout.pip_max_dim as f64 / bigger_pip_dim;
        let horiz_limit_ratio = avail_w / (pip_w as f64);
        let vert_limit_ratio = avail_h / (pip_h as f64);
        let scale = target_limit_ratio.min(horiz_limit_ratio).min(vert_limit_ratio);

        let dst_h = ((pip_h as f64 * scale).round() as i32).max(1);
        let dst_w = ((pip_w as f64 * scale).round() as i32).max(1);

        let mut resized = Mat::default();
        if imgproc::resize(
            &pip,
            &mut resized,
            Size::new(dst_w, dst_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        ).is_err() { return; }

        let border = Scalar::new(255.0, 255.0, 255.0, 0.0);
        let _ = imgproc::rectangle(
            &mut resized,
            Rect::new(0, 0, dst_w, dst_h),
            border,
            layout.pip_border_thickness,
            imgproc::LINE_8,
            0,
        );

        let x0 = frame_w - margin - dst_w;
        let y0 = frame_h - margin - dst_h;
        if x0 < 0 || y0 < 0 { return; }
        let dst_rect = Rect::new(x0, y0, dst_w, dst_h);
        if let Ok(mut roi) = frame.roi_mut(dst_rect)
        { let _ = resized.copy_to(&mut roi); }
    }

    /// Draws bottom-left text status lines onto a frame (backend-specific).
    ///
    /// ### Params:
    /// - `frame`: The image frame to draw on (mutated in place).
    /// - `text_lines`: The lines of different text status lines to draw into the image frame.
    /// - `layout`: The overlay element scaling context.
    pub fn draw_status_block(
        &self,
        frame: &mut Mat,
        text_lines: &[&str],
        layout: &OverlayLayout,
    ) {
        let frame_h = frame.rows();
        let x = layout.margin;

        for (i, text) in text_lines
            .iter()
            .rev()
            .filter(|t| !t.is_empty())
            .enumerate()
        {
            let y = frame_h - layout.margin - (i as i32) * layout.text_line_spacing;

            let position = Point::new(x, y);
            self.draw_text(frame, text, position, layout.text_size, layout.text_thickness);
        }
    }

    /// Draws a given text at a given position within a given frame.
    ///
    /// ### Params:
    /// - `frame`: The image frame to draw on (mutated in place).
    /// - `text`: The text string to display (if empty, does nothing).
    /// - `position`: The origin position in the frame to draw the text string at.
    /// - `scale`: The size of the text string.
    /// - `thickness`: Thickness of the text string.
    pub fn draw_text(
        &self,
        frame: &mut Mat,
        text: &str,
        position: Point,
        scale: f64,
        thickness: i32,
    ) {
        if text.is_empty() { return; }

        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        let white = Scalar::new(255.0, 255.0, 255.0, 0.0);
        let black = Scalar::new(0.0, 0.0, 0.0, 0.0);

        let _ = imgproc::put_text(frame, text, position, font, scale, black, thickness, imgproc::LINE_8, false);
        let _ = imgproc::put_text(frame, text, position, font, scale, white, thickness, imgproc::LINE_8, false);
    }
}



impl OverlayLayout {
    // these are the formatting knobs relative to the provided frame dims
    // adjust these to change the overall aesthetic
    const FONT_SCALE_RATIO: f64 = 0.0015; 
    const MARGIN_RATIO: f64     = 0.05;
    const TEXT_LINE_SPACING_RATIO: f64    = 0.055;
    const PIP_MAX_DIM_RATIO: f64        = 0.35;

    pub fn from_frame(frame: &Mat) -> Self {
        let (frame_h, frame_w) = (frame.rows(), frame.cols());
        let smaller_frame_dim = frame_h.min(frame_w).max(1) as f64; // smaller of the height/width dim of the given frame

        Self {
            margin: ((Self::MARGIN_RATIO * smaller_frame_dim).round() as i32).clamp(4, 48),
            text_line_spacing: ((Self::TEXT_LINE_SPACING_RATIO * smaller_frame_dim).round() as i32).clamp(12, 50),
            text_size: (Self::FONT_SCALE_RATIO * smaller_frame_dim).clamp(0.3, 1.2),
            text_thickness: ((smaller_frame_dim / 600.0).round() as i32).clamp(1, 3),
            roi_thickness: ((smaller_frame_dim / 350.0).round() as i32).clamp(1, 4),
            dot_radius: ((smaller_frame_dim * 0.005).round() as i32).clamp(1, 6),
            pip_max_dim: ((Self::PIP_MAX_DIM_RATIO * smaller_frame_dim).round() as i32).clamp(48, 240),
            pip_border_thickness: ((smaller_frame_dim / 400.0).round() as i32).clamp(1, 2),
        }
    }
}



impl LiveWindow {
    /// Creates a new display window.
    ///
    /// ### Params:
    /// - `window_name`: Title for the OpenCV display window.
    ///
    /// ### Returns:
    /// An initialized [`LiveWindow`], or an error if the window could not be created.
    pub fn new(window_name: &str) -> Result<Self, ESS> {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)
            .map_err(|e| io_err(e.to_string(), std::io::ErrorKind::Other))?;
        Ok(Self { window_name: window_name.to_string() })
    }

    /// Displays the frame in the window and waits briefly for key input.
    ///
    /// ### Params:
    /// - `frame`: The frame to display (already annotated if desired).
    ///
    /// ### Returns:
    /// `false` if ESC was pressed or the window was closed, signaling the
    /// live loop should exit. `true` otherwise.
    pub fn show(&self, frame: &Mat) -> Result<bool, ESS> {
        highgui::imshow(&self.window_name, frame)
            .map_err(|e| io_err(e.to_string(), std::io::ErrorKind::Other))?;
        let key = highgui::wait_key(1)
            .map_err(|e| io_err(e.to_string(), std::io::ErrorKind::Other))?;
        Ok(key != 27) // ESC = 27
    }
}



impl Drop for LiveWindow {
    fn drop(&mut self) { let _ = highgui::destroy_window(&self.window_name); }
}
