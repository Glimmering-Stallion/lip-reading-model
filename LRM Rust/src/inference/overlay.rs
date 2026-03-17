//! Live inference visualization overlay.
//!
//! Draws tracker bounding boxes, landmark points, and prediction text
//! onto display frames for real-time visual feedback during webcam inference.



use crate::{
    pipeline::tracker::VizMetadata,
    prelude::{io_err, ESS},
};
use opencv::{
    core::{Mat, Point, Scalar},
    highgui,
    imgproc,
    prelude::*,
};



/// Renders visualization overlays onto display frames for live inference.
///
/// Draws tracker metadata (bounding boxes, landmarks) and prediction text
/// onto cloned display frames. Uses `opencv::highgui` for window management.
pub struct OverlayRenderer {
    window_name: String,
}



impl OverlayRenderer {
    /// Creates a new overlay renderer and initializes the display window.
    ///
    /// ### Params:
    /// - `window_name`: Title for the OpenCV display window.
    pub fn new(window_name: &str) -> Result<Self, ESS> {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)
            .map_err(|e| io_err(e.to_string(), std::io::ErrorKind::Other))?;
        Ok(Self {
            window_name: window_name.to_string(),
        })
    }

    /// Draws tracker bounding boxes and stabilized center onto a display frame.
    ///
    /// Renders whatever metadata the tracker backend populated:
    /// - Face bounding box (green)
    /// - Mouth bounding box (blue)
    /// - Landmark points (cyan dots), if available
    /// - Stabilized center (red dot)
    ///
    /// ### Params:
    /// - `frame`: The display frame to draw on (mutated in place).
    /// - `metadata`: Visualization metadata from the tracker backend.
    pub fn draw_tracker_info(&self, frame: &mut Mat, metadata: &VizMetadata) {
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let blue = Scalar::new(255.0, 0.0, 0.0, 0.0);
        let cyan = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let red = Scalar::new(0.0, 0.0, 255.0, 0.0);

        if let Some(face_rect) = metadata.face_rect {
            let _ = imgproc::rectangle(frame, face_rect, green, 2, imgproc::LINE_8, 0);
        }

        if let Some(mouth_rect) = metadata.mouth_rect {
            let _ = imgproc::rectangle(frame, mouth_rect, blue, 2, imgproc::LINE_8, 0);
        }

        if let Some(ref landmarks) = metadata.landmarks {
            for pt in landmarks {
                let center = Point::new(pt.x as i32, pt.y as i32);
                let _ = imgproc::circle(frame, center, 2, cyan, -1, imgproc::LINE_8, 0);
            }
        }

        if let Some(center) = metadata.stabilized_center {
            let pt = Point::new(center.x as i32, center.y as i32);
            let _ = imgproc::circle(frame, pt, 4, red, -1, imgproc::LINE_8, 0);
        }
    }

    /// Draws the current prediction text onto the bottom of the display frame.
    ///
    /// ### Params:
    /// - `frame`: The display frame to draw on (mutated in place).
    /// - `text`: The predicted text string to display.
    pub fn draw_prediction(&self, frame: &mut Mat, text: &str) {
        if text.is_empty() { return; }

        let white = Scalar::new(255.0, 255.0, 255.0, 0.0);
        let black = Scalar::new(0.0, 0.0, 0.0, 0.0);

        let rows = frame.rows();
        let origin = Point::new(10, rows - 20);

        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        let _ = imgproc::put_text(frame, text, origin, font, 0.8, black, 3, imgproc::LINE_8, false);
        let _ = imgproc::put_text(frame, text, origin, font, 0.8, white, 2, imgproc::LINE_8, false);
    }

    /// Displays the frame in the window and waits briefly for key input.
    ///
    /// ### Params:
    /// - `frame`: The frame to display.
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



impl Drop for OverlayRenderer {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(&self.window_name);
    }
}
