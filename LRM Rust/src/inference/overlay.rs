//! Live inference visualization overlay.
//!
//! - [`FrameAnnotator`]: stateless drawing of ML metadata on any `Mat` (live window, file export, etc.)
//! - [`LiveWindow`]: OpenCV HighGUI window for displaying frames and handling user input (e.g. ESC)



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



/// Stateless overlay drawing (backend-agnostic [`VizMetadata`] + prediction text).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameAnnotator;



/// Manages the OpenCV HighGUI display window for live inference.
///
/// Handles `named_window`, `imshow`, `wait_key`, and destroys the window on drop.
pub struct LiveWindow { window_name: String }



impl FrameAnnotator {
    /// Draws tracker bounding boxes and stabilized center onto a frame.
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
            let _ = imgproc::circle(frame, pt, 2, red, -1, imgproc::LINE_8, 0);
        }
    }

    /// Draws a given text at a given position within a given frame.
    ///
    /// ### Params:
    /// - `frame`: The image frame to draw on (mutated in place).
    /// - `text`: The text string to display (if empty, does nothing).
    /// - `position`: The origin position in the frame to draw the text string at.
    pub fn draw_text(&self, frame: &mut Mat, text: &str, position: Point) {
        if text.is_empty() { return; }

        let font = imgproc::FONT_HERSHEY_SIMPLEX;
        let white = Scalar::new(255.0, 255.0, 255.0, 0.0);
        let black = Scalar::new(0.0, 0.0, 0.0, 0.0);

        let _ = imgproc::put_text(frame, text, position, font, 0.8, black, 1, imgproc::LINE_8, false);
        let _ = imgproc::put_text(frame, text, position, font, 0.8, white, 1, imgproc::LINE_8, false);
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
