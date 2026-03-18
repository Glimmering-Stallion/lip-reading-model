//! Inference predictor for loading a trained VSRM and running prediction on video input.
//!
//! Provides [`VsrmPredictorConfig`] for serializable inference knobs and [`InferenceSession`]
//! for checkpoint loading, single-video prediction, and live sliding-window inference.



// custom imports
use crate::{
    cli::find_latest_checkpoint_epoch,
    context::Context,
    ctc::ctc_decode::{CtcDecodeType, CtcDecoder, CtcDecoderConfig},
    inference::{
        loader::{load_frame, load_video, open_camera},
        overlay::OverlayRenderer,
    },
    pipeline::{
        batcher::{VsrmBatcher, VsrmItem},
        dataset::DatasetStats,
        tracker::{HaarTrackerConfig, TrackerConfig},
        FramesBuffer,
    },
    prelude::{io_err, ESS},
    vocab::{TokenMap, BLANK_ID},
    vsrm::{VsrModel, VsrModelConfig},
};
// imports
use burn::{
    config::Config, data::dataloader::batcher::Batcher, module::Module, prelude::Backend,
    record::CompactRecorder, tensor::TensorData,
};
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use opencv::core::{Mat, MatTraitConstManual};
use std::{
    collections::VecDeque,
    io::ErrorKind,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};



/// Sent from main (UI) thread to worker as a `FramesBuffer` ready for inference.
pub type InferenceRequest = FramesBuffer;
/// Sent from worker to main (UI) as decoded `String` prediction.
pub type InferenceResponse = String;



/// Serializable configuration knobs for inference with the VSR model.
#[derive(Config, Debug)]
pub struct VsrmPredictorConfig {
    pub model_id: String,            // model name to be saved as

    #[config(default = "(50, 100)")]
    pub frame_dims: (usize, usize),  // input video frame dimensions (height, width)

    #[config(default = "0")]
    pub temporal_window: usize,      // sliding window capacity (default `0` value means auto-derive from `model.total_receptive_field()`)

    #[config(default = "10")]
    pub temporal_stride: usize,      // stride length of the sliding window

    #[config(default = "CtcDecodeType::GreedySearch")]
    pub search_type: CtcDecodeType,  // decoding search type strategy (Greedy vs. Prefix Beam)
}



/// Loaded model session that holds necessary components for running predictions.
///
/// Holds the trained model, CTC decoder, frame batcher, vocabulary map, and device backend.
pub struct InferenceSession<B: Backend> {
    model: VsrModel<B>,
    decoder: CtcDecoder,
    batcher: VsrmBatcher<B>,
    token_map: TokenMap,
    device: B::Device,
}



/// Sliding window buffer for accumulating mouth crop frames in live mode.
///
/// Frames are pushed one at a time from the live camera loop.
/// 
/// When the window reaches its capacity, it can be flushed to a
/// `FramesBuffer` for prediction, then shifted forward by a
/// configurable stride.
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
        model_config: &VsrModelConfig,
        predictor_config: &VsrmPredictorConfig,
        norm_stats: DatasetStats,
        token_map: TokenMap,
    ) -> Result<Self, ESS> {
        let checkpoint_epoch = find_latest_checkpoint_epoch(model_path).ok_or_else(|| {
            io_err(
                format!("No checkpoints found in {:?}", model_path),
                ErrorKind::NotFound,
            )
        })?;
        let checkpoint_path = model_path
            .join("checkpoint")
            .join(format!("model-{}", checkpoint_epoch));

        let model = model_config
            .init::<B>(&device)
            .load_file(checkpoint_path, &CompactRecorder::new(), &device)
            .map_err(|e| format!("Failed to load model checkpoint: {}", e))?;

        let resolved_temporal_window = if predictor_config.temporal_window == 0 { model.total_receptive_field() }
        else { predictor_config.temporal_window };

        let decoder = CtcDecoderConfig::new()
            .with_search_type(predictor_config.search_type)
            .with_blank_id(BLANK_ID)
            .init();

        let batcher = VsrmBatcher::<B>::new(device.clone(), Some(norm_stats));

        println!("=== Inference session loaded ===");
        println!("  Model:           {}",          predictor_config.model_id);
        println!("  Receptive field: {} frames",   model.total_receptive_field());
        println!("  Window size:     {} frames",   resolved_temporal_window);
        println!("  Window stride:   {} frames",   predictor_config.temporal_stride);
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
        if t == 0 { return Err(io_err("empty frame buffer", ErrorKind::InvalidInput)); }

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
/// Expects an already-constructed [`InferenceSession`]; builds tracker config from
/// `context` and `predictor_config`, then runs the file or live loop.
///
/// ### Params:
/// - `session`: Pre-built inference session (model, decoder, batcher, vocab).
/// - `context`: Filesystem context (used for tracker cascade paths).
/// - `predictor_config`: Inference knobs (frame_dims, temporal_window, search_type).
/// - `input`: Video file path for file mode; `None` for live webcam mode.
/// - `camera`: Camera device index (when live).
pub fn infer<B: Backend>(
    session: InferenceSession<B>,
    context: &Context,
    predictor_config: &VsrmPredictorConfig,
    input: Option<&Path>,
    camera: i32,
) -> Result<(), ESS> {
    let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
        context.models_path.join("haarcascade_frontalface_alt2.xml"),
        context.models_path.join("haarcascade_mcs_mouth.xml"),
        predictor_config.frame_dims,
    ));

    if let Some(video_path) = input {
        // ----------------- mode (A): static video inference ----------------
        infer_file(video_path, session, &tracker_config)?;
    } else {
        // ------------------ mode (B): live camera inference ----------------
        infer_live(camera, session, &tracker_config, predictor_config)?;
    }

    Ok(())
}



/// Worker thread orchestrator:
/// - receives buffers,
/// - runs model forward passing,
/// - sends prediction output strings.
///
/// Exits when `rx` is disconnected (main dropped sender) or `shutdown` is set.
///
/// ### Params:
/// - `session`: Initialized session engine holding inference-related components.
/// - `rx`: Receives inference input requests from main thread. Worker blocks on this until a request arrives or the channel is disconnected.
/// - `tx`: Sends prediction results back to main thread. Main thread uses `try_recv` so it never blocks.
/// - `shutdown`: Shared flag so main thread can request exit (ESC press or window close); worker checks it each loop step and breaks when set.
fn inference_worker<B>(
    session: InferenceSession<B>,
    rx: Receiver<InferenceRequest>,
    tx: Sender<InferenceResponse>,
    shutdown: Arc<AtomicBool>,
) where
    B: Backend + Send,
    B::Device: Send,
{
    while !shutdown.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => {
                if let Ok(pred) = session.predict_frames(req) {
                    let _ = tx.send(pred);
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}



/// Runs prediction on a video file by tracking and cropping each frame,
/// then delegating to `predict_frames`.
///
/// ### Params:
/// - `session`: Initialized session engine holding inference-related components.
/// - `video_path`: Path to the video file.
/// - `tracker_config`: Configuration for the tracker backend to use.
///
/// ### Returns:
/// The decoded prediction string.
fn infer_file<B: Backend>(
    video_path: &Path,
    session: InferenceSession<B>,
    tracker_config: &TrackerConfig,
) -> Result<(), ESS> {
    let mut tracker = tracker_config.init();
    let frames = load_video(video_path, tracker.as_mut())?;
    let prediction = session.predict_frames(frames)?;

    println!("Prediction: {}\n", prediction);

    Ok(())
}



/// Runs live webcam inference with a dedicated inference worker thread.
/// 
/// Coordinates a producer-consumer pipeline where the main thread handles 
/// high-frequency UI tasks (capture, tracking, visualization) and a 
/// background worker handles expensive model forward passes.
///
/// ### Params:
/// - `session`: Initialized session engine holding inference-related components.
/// - `tracker_config`: Configuration for the face/mouth tracking backend.
/// - `predictor_config`: User-defined inference knobs (stride, dims, etc).
/// - `camera`: The hardware index of the camera to open.
///
/// ### Returns:
/// `Ok(())` on clean exit (ESC pressed), or an Error String if hardware/tracking fails.
fn infer_live<B>(
    camera: i32,
    session: InferenceSession<B>,
    tracker_config: &TrackerConfig,
    predictor_config: &VsrmPredictorConfig,
) -> Result<(), ESS>
where
    B: Backend + Send,
    B::Device: Send,
{
    // initialize CV tracker, camera capture, and OpenCV high-gui renderer
    let mut tracker = tracker_config.init();
    let mut cap = open_camera(camera)?;
    let renderer = OverlayRenderer::new("LRM Live Inference")?;

    // create sliding window using session's resolved receptive field
    let (h, w) = predictor_config.frame_dims;
    let t = if predictor_config.temporal_window == 0 { session.model.total_receptive_field() }
    else { predictor_config.temporal_window };

    let mut window = SlidingWindow::new(t, h, w);

    // req: main (UI) --> worker (infer) (bounded 1 to prevent latency if model falls behind)
    // res: worker (infer) --> main (UI) (bounded 1 as we only care about most recent prediction)
    let (req_tx, req_rx) = bounded::<InferenceRequest>(1);
    let (res_tx, res_rx) = bounded::<InferenceResponse>(1);

    // atomic flag to signal worker to stop if UI loop breaks (such as from ESC press)
    let shutdown_main = Arc::new(AtomicBool::new(false));
    let shutdown_worker = shutdown_main.clone();

    // spawn worker thread
    let join_handle = thread::spawn(move || {
        inference_worker(session, req_rx, res_tx, shutdown_worker);
    });

    println!("Live inference started, press ESC to quit\n");

    // main UI loop
    println!("Predictions:");
    let mut last_prediction = String::new();
    let result = loop {
        // grab frame from camera
        let frame = match load_frame(&mut cap)? {
            Some(f) => f,
            None => break Ok(()),
        };

        // detect face/mouth, overlay on display only frame clone
        let mut display = frame.clone();
        let result = tracker.process_frame(&frame)?;
        renderer.draw_tracker_info(&mut display, &result.metadata);

        // push current crop into sliding window buffer
        window.push(&result.crop);
        if window.is_full() {
            // if buffer ready, try to hand off to worker (infer) thread
            // if worker busy, send fails and window buffer dropped
            let buffer = window.to_buffer();
            let _ = req_tx.try_send(buffer);
            window.shift(predictor_config.temporal_stride);
        }

        // check if worker (infer) thread has finished a prediction
        if let Ok(pred) = res_rx.try_recv() {
            last_prediction = pred.clone();
            if !last_prediction.is_empty()
            { println!(">> {}", last_prediction); }
        }

        renderer.draw_prediction(&mut display, &last_prediction);
        if !renderer.show(&display)? { break Ok(()); }
    };

    // signal exit to worker (infer) thread and drop sender to close channel
    // wait for worker to finish its last pass and shut down
    shutdown_main.store(true, Ordering::Relaxed);
    drop(req_tx);
    join_handle
        .join()
        .map_err(|_| io_err("inference worker thread panicked", ErrorKind::Other))?;

    println!("\nLive inference ended\n");

    result
}
