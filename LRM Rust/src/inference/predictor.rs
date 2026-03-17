//! Inference predictor for loading a trained VSRM and running prediction on video input.
//!
//! Provides [`VsrmPredictorConfig`] for serializable inference knobs and [`InferenceSession`]
//! for checkpoint loading, single-video prediction, and live sliding-window inference.



// custom imports
use crate::{
    cli::find_latest_checkpoint_epoch,
    context::Context,
    ctc::ctc_decode::{
        CtcDecoder,
        CtcDecoderConfig,
        CtcDecodeType,
    },
    inference::{
        loader::{
            load_video,
            load_frame,
            open_camera,
        },
        overlay::OverlayRenderer,
    },
    
    
    pipeline::{
        FramesBuffer,
        batcher::{VsrmBatcher, VsrmItem},
        dataset::DatasetStats,
        tracker::{HaarTrackerConfig, TrackerConfig},
    },
    prelude::{ESS, io_err},
    vocab::{BLANK_ID, TokenMap},
    vsrm::{VsrModel, VsrModelConfig},
};

// imports
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    prelude::Backend,
    record::CompactRecorder,
    tensor::TensorData,
};
use opencv::core::{Mat, MatTraitConstManual};
use std::{
    io::ErrorKind,
    collections::VecDeque,
    path::Path,
};



/// Serializable configuration knobs for inference.
///
/// `rf_window_size = 0` means auto-derive from `model.total_receptive_field()`.
#[derive(Config, Debug)]
pub struct VsrmPredictorConfig {
    pub model_id: String,

    #[config(default = "(50, 100)")]
    pub frame_dims: (usize, usize),

    #[config(default = "0")]
    pub rf_window_size: usize,

    #[config(default = "10")]
    pub rf_window_stride: usize,

    #[config(default = "CtcDecodeType::GreedySearch")]
    pub search_type: CtcDecodeType,
}



/// Loaded model session for running predictions.
///
/// Holds the trained model, CTC decoder, frame batcher, vocabulary map.
pub struct InferenceSession<B: Backend> {
    model: VsrModel<B>,
    decoder: CtcDecoder,
    batcher: VsrmBatcher<B>,
    token_map: TokenMap,
    device: B::Device,
}



/// Sliding window buffer for accumulating mouth crop frames in live mode.
///
/// Frames are pushed one at a time from the live camera loop. When the
/// window reaches its capacity, it can be flushed to a `FramesBuffer`
/// for prediction, then shifted forward by a configurable stride.
pub struct SlidingWindow {
    frames: VecDeque<Vec<u8>>,
    height: usize,
    width: usize,
    capacity: usize,
}



impl<B: Backend> InferenceSession<B> {
    /// Builds an inference session from pre-loaded configs.
    ///
    /// ### Params:
    /// - `device`: Burn backend device.
    /// - `model_config`, `norm_stats`: Pre-loaded from main.
    /// - `model_path`: Path to model directory (parent of `checkpoint/`).
    /// - `predictor_config`: Inference knobs (window params, decoder type).
    /// - `token_map`: Bidirectional char-to-ID mapping.
    ///
    /// ### Returns:
    /// An initialized `InferenceSession` ready for prediction.
    pub fn new(
        device: B::Device,
        model_path: &Path,
        model_config: VsrModelConfig,
        predictor_config: &VsrmPredictorConfig,
        norm_stats: DatasetStats,
        token_map: TokenMap,
    ) -> Result<Self, ESS> {
        let checkpoint_epoch = find_latest_checkpoint_epoch(model_path)
            .ok_or_else(|| io_err(format!("No checkpoints found in {:?}", model_path), ErrorKind::NotFound))?;
        let checkpoint_path = model_path
            .join("checkpoint")
            .join(format!("model-{}", checkpoint_epoch));

        let model = model_config.init::<B>(&device)
            .load_file(checkpoint_path, &CompactRecorder::new(), &device)
            .map_err(|e| format!("Failed to load model checkpoint: {}", e))?;

        let decoder = CtcDecoderConfig::new()
            .with_search_type(predictor_config.search_type)
            .with_blank_id(BLANK_ID)
            .init();

        let batcher = VsrmBatcher::<B>::new(device.clone(), Some(norm_stats));

        println!("=== Inference session loaded ===");
        println!("  Model:           {}",          predictor_config.model_id);
        println!("  Receptive field: {} frames",   model.total_receptive_field());
        println!("  Window size:     {} frames",   predictor_config.rf_window_size);
        println!("  Window shift:    {} frames",   predictor_config.rf_window_stride);
        println!("  Decoder:         {:?}",        predictor_config.search_type);
        println!("================================\n");

        Ok(Self {
            model,
            decoder,
            batcher,
            token_map,
            device,
        })
    }

    /// Runs prediction on a pre-loaded buffer of mouth crop frames.
    ///
    /// Builds a batch of 1, runs the forward pass, decodes CTC output,
    /// and returns the predicted text string.
    ///
    /// ### Params:
    /// - `frames`: Contiguous grayscale mouth crop frames.
    ///
    /// ### Returns:
    /// The decoded prediction string.
    pub fn predict_frames(&self, frames: FramesBuffer) -> Result<String, ESS> {
        let (h, w) = (frames.height, frames.width);
        let t = frames.data.len() / (h * w);
        if t == 0 {
            return Err(io_err("Empty frame buffer", ErrorKind::InvalidInput));
        }

        let item = VsrmItem {
            frames: TensorData::new(frames.data, vec![1, t, h, w]),
            transcript_ids: vec![],
            item_id: "inference".to_string(),
        };

        let batch = self.batcher.batch(vec![item], &self.device);
        let logits = self.model.forward(batch.inputs);
        let decoded = self.decoder.forward(logits);

        let text = if let Some(token_ids) = decoded.first() {
            token_ids
                .iter()
                .filter_map(|&id| self.token_map.char_of(id as usize))
                .collect::<String>()
        } else { String::new() };

        Ok(text)
    }

    /// Runs prediction on a video file by tracking and cropping each frame,
    /// then delegating to `predict_frames`.
    ///
    /// ### Params:
    /// - `video_path`: Path to the video file.
    /// - `tracker_config`: Configuration for the tracker backend to use.
    ///
    /// ### Returns:
    /// The decoded prediction string.
    pub fn predict_file(
        &self,
        video_path: &Path,
        tracker_config: &TrackerConfig,
    ) -> Result<String, ESS> {
        let mut tracker = tracker_config.init();
        let frames = load_video(video_path, tracker.as_mut())?;
        self.predict_frames(frames)
    }
}



impl SlidingWindow {
    /// Creates a new sliding window with the given capacity and frame dimensions.
    ///
    /// ### Params:
    /// - `capacity`: Maximum number of frames the window holds before it is full.
    /// - `height`: Height of each frame in pixels.
    /// - `width`: Width of each frame in pixels.
    pub fn new(capacity: usize, height: usize, width: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            height,
            width,
            capacity,
        }
    }

    /// Pushes a single tracked mouth crop frame into the window.
    ///
    /// ### Params:
    /// - `crop`: The mouth crop `Mat` from the tracker.
    pub fn push(&mut self, crop: &Mat) {
        if let Ok(bytes) = crop.data_bytes() {
            self.frames.push_back(bytes.to_vec());
        }
    }

    /// Returns `true` when the window has accumulated enough frames for prediction.
    pub fn is_full(&self) -> bool {
        self.frames.len() >= self.capacity
    }

    /// Flattens the current window contents into a contiguous `FramesBuffer`
    /// suitable for `InferenceSession::predict_frames`.
    ///
    /// ### Returns:
    /// A `FramesBuffer` containing all frames currently in the window.
    pub fn to_buffer(&self) -> FramesBuffer {
        let data: Vec<u8> = self.frames.iter().flat_map(|f| f.iter().copied()).collect();
        FramesBuffer {
            data,
            height: self.height,
            width: self.width,
        }
    }

    /// Shifts the window forward by removing the oldest `n` frames,
    /// making room for new frames from the live stream.
    ///
    /// ### Params:
    /// - `n`: Number of frames to drop from the front.
    pub fn shift(&mut self, n: usize) {
        for _ in 0..n.min(self.frames.len()) {
            self.frames.pop_front();
        }
    }
}



/// Runs inference in file mode or live webcam mode.
///
/// Loads model from checkpoint, builds session, then runs the appropriate loop.
/// Mirrors `train()` in learner.rs: receives configs, builds session internally.
///
/// ### Params:
/// - `device`, `context`: Backend and filesystem context.
/// - `model_config`, `norm_stats`: Pre-loaded from main.
/// - `predictor_config`: Inference knobs (frame_dims, rf_window_size, search_type).
/// - `model_path`: Path to model directory.
/// - `token_map`: Bidirectional char-to-ID mapping.
/// - `input`: Video file path for file mode; `None` for live webcam mode.
/// - `camera`: Camera device index (when live).
pub fn infer<B: Backend>(
    device: B::Device,
    context: &Context,
    model_path: &Path,
    model_config: VsrModelConfig,
    predictor_config: VsrmPredictorConfig,
    norm_stats: DatasetStats,
    token_map: TokenMap,
    input: Option<&Path>,
    camera: i32,
) -> Result<(), ESS> {
    let session = InferenceSession::<B>::new(
        device,
        model_path,
        model_config,
        &predictor_config,
        norm_stats,
        token_map,
    )?;

    let tracker_config = TrackerConfig::Haar(
        HaarTrackerConfig::new(
            context.models_path.join("haarcascade_frontalface_alt2.xml"),
            context.models_path.join("haarcascade_mcs_mouth.xml"),
            predictor_config.frame_dims,
        ),
    );

    if let Some(video_path) = input {
        // ----------------- mode (A): static video inference ----------------
        let prediction = session.predict_file(video_path, &tracker_config)?;
        println!("Prediction: {}\n", prediction);
    } else {
        // ------------------ mode (B): live camera inference ----------------
        let mut tracker = tracker_config.init();
        let mut cap = open_camera(camera)?;
        let renderer = OverlayRenderer::new("LRM Live Inference")?;
        let (h, w) = predictor_config.frame_dims;

        let mut window = SlidingWindow::new(
            predictor_config.rf_window_size,
            h,
            w,
        );

        println!("Live inference started, press ESC to quit\n");

        let mut last_prediction = String::new();
        loop {
            let frame = match load_frame(&mut cap)? {
                Some(f) => f,
                None => break,
            };

            let mut display = frame.clone();
            let result = tracker.process_frame(&frame)?;
            renderer.draw_tracker_info(&mut display, &result.metadata);

            window.push(&result.crop);
            if window.is_full() {
                match session.predict_frames(window.to_buffer()) {
                    Ok(text) => {
                        if text != last_prediction {
                            println!(">> {}", text);
                            last_prediction = text;
                        }
                    }
                    Err(e) => eprintln!("Prediction error: {}", e),
                }
                window.shift(predictor_config.rf_window_stride);
            }

            renderer.draw_prediction(&mut display, &last_prediction);
            if !renderer.show(&display)? { break; }
        }

        println!("\nLive inference ended\n");
    }

    Ok(())
}
