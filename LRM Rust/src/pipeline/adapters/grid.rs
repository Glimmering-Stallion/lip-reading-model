//! GRID-specific corpus adapter for audio-visual sentence processing.
//! 
//! This module implements the ```GridDataset``` adapter, which standardizes raw
//! GRID videos (as .mpg files) and alignments (as .align files) into a common
//! ```VsrmItem``` format. It also orchestrates frame loading, grayscale conversion,
//! and lip-region extraction.



// custom imports
use crate::{
    context::Context,
    pipeline::{
        FramesBuffer,
        batcher::VsrmItem,
        tracker::{LipTracker, LipTrackerConfig}
    },
    vocab::TokenMap
};

// imports
use burn::{
    data::dataset::Dataset,
    tensor::TensorData,
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
        CAP_ANY,
        VideoCapture,
        VideoCaptureTrait,
    }
};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::{
        PathBuf,
    },
    fs::read_dir,
};



pub struct GridDataset {
    grid_path: PathBuf,
    entries: Vec<String>,
    token_map: TokenMap,
    tracker_config: Option<LipTrackerConfig>,
}



impl GridDataset {
    /// constructor for GRID dataset adapter
    /// scans disk for available video samples and their corresponding alignment files
    /// stores valid entries as "speaker_id/item_id" (e.g., "s1/bbaf2n")
    /// params:
    /// - context: filesystem context (that should contain "data/grid-lr-corpus" subdirectory with frames and alignments)
    /// - token_map: bidirectional mapping of chars to IDs for transcript encoding
    /// returns: initialized GridDataset instance with valid entries loaded from disk
    pub fn new(
        context: &Context,
        token_map: TokenMap,
        tracker_config: Option<LipTrackerConfig>,
    ) -> Self {
        let grid_path = context.data_path.join("grid-lr-corpus");
        let frames_path = grid_path.join("frames");
        let alignments_path = grid_path.join("alignments");

        assert!(grid_path.exists(), "GRID corpus directory does not exist at {:?}", grid_path);
        assert!(frames_path.exists(), "GRID frames directory does not exist at {:?}", frames_path);
        assert!(alignments_path.exists(), "GRID alignments directory does not exist at {:?}", alignments_path);

        // identify all speakers available on disk (s1, s2, ..., s34)
        let mut avail_speakers = Vec::new();
        if let Ok(speaker_paths) = read_dir(&frames_path) {
            for speaker_path in speaker_paths.flatten() {
                let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
                if speaker_str.starts_with('s') && speaker_path.path().is_dir() {
                    avail_speakers.push(speaker_str);
                }
            }
        }
        assert!(!avail_speakers.is_empty(), "No speaker directories found in {:?}", frames_path);

        // sort s1, s2, ..., s34
        avail_speakers.sort_by_key(|s| s[1..].parse::<i32>().unwrap_or(1));

        // scan disk for only selected speakers and store
        let mut entries = Vec::new();
        for speaker in &avail_speakers { // loop through dirs of selected speakers
            let video_path = frames_path.join(speaker);
            let alignment_path = alignments_path.join(speaker);

            if let Ok(items) = read_dir(&video_path) {
                for item in items.flatten() { // loop through data items of each speakers' dirs
                    if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {

                        // validity flags
                        // only if a valid video file has a corresponding alignment file do we consider it a valid entry
                        let is_video = item.path().extension().is_some_and(|ext| ext == "mpg");
                        let has_alignment = alignment_path.join(stem).with_extension("align").exists();

                        // store speaker/data (e.g., "s1/bbaf2n")
                        if is_video && has_alignment { entries.push(format!("{}/{}", speaker, stem)); }
                        else { /* println!("Skipping {}/{}: missing alignment file.", speaker, stem); */ }
                    }
                }
            }
        }
        entries.sort(); // sort for deterministic order
        assert!(!entries.is_empty(), "Dataset instance resulted in 0 samples\nCheck if path {:?} contains .mpg files", grid_path);
        println!("\nInitialized GridDataset instance with {} samples from speakers {:?}\n", entries.len(), avail_speakers);

        Self {
            grid_path,
            entries,
            token_map,
            tracker_config,
        }
    }

    /// helper for parsing a GRID specific .align transcript file
    /// params:
    /// - entry: the unique GRID dataset entry ID to parse its alignments from (in the form of "s1/bbaf2n")
    /// returns: a list of corresponding char IDs
    fn load_alignment(&self, entry: &str) -> Result<Vec<usize>, Box<dyn Error>> {
        let alignment_path = self.grid_path
            .join("alignments")
            .join(entry)
            .with_extension("align");
        assert!(alignment_path.exists(), "Alignment file {} for GRID entry {} not found", alignment_path.to_string_lossy(), entry);

        match File::open(alignment_path) {
            Ok(file) => {
                let mut tokens: Vec<String> = vec![];
                let lines = BufReader::new(file).lines();

                for line in lines.map_while(Result::ok) {
                    let line_group = line.split_whitespace().collect::<Vec<_>>();
                    assert!(line_group.len() >= 3, "Malformed alignment line: {:?}", line_group);
                    if line_group[2] != "sil" { tokens.push(line_group[2].to_string()); }
                }
                assert!(!tokens.is_empty(), "No non-silence tokens found in alignment file");

                Ok(tokens
                    .iter()
                    .flat_map(|token| token.chars())
                    .filter_map(|ch| self.token_map.id_of(ch))
                    .collect())
            }
            Err(e) => {
                eprintln!("Error opening alignments file: {}", e);
                Err(Box::new(e))
            }
        }
    }

    /// helper for processing a GRID specific .mpg video file
    /// params:
    /// - entry: the unique GRID dataset entry ID to load and process its frames from (in the form of "s1/bbaf2n")
    /// - process: the process to apply to the given frame
    /// returns: a FramesBuffer containing the corresponding flattened vector of frames along with frame dimensions
    fn load_frames<F>(
        &self,
        entry: &str,
        mut process: F,
    ) -> Result<FramesBuffer, Box<dyn Error>>
    where F: FnMut(&Mat) -> Result<Mat, Box<dyn Error>>
    {
        let frames_path: PathBuf = self.grid_path
            .join("frames")
            .join(entry)
            .with_extension("mpg");
        assert!(frames_path.exists(), "Video file {} for GRID entry {} not found", frames_path.to_string_lossy(), entry);

        // frames container, dims, and single frame buffers
        let mut frames: Vec<u8> = Vec::new();
        let mut frame_dims: (usize, usize) = (0, 0); // (height, width)
        let (mut orig_frame, mut gray_frame) = (Mat::default(), Mat::default());

        match VideoCapture::from_file(frames_path.to_str().ok_or("Invalid path")?, CAP_ANY) {
            Ok(mut cap) => {
                while cap.read(&mut orig_frame).expect("Error reading frame") {
                    if orig_frame.empty() { break; }

                    // convert frame to grayscale
                    imgproc::cvt_color(
                        &orig_frame,
                        &mut gray_frame,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        AlgorithmHint::ALGO_HINT_DEFAULT,
                    ).expect("Failed to convert frame to grayscale");

                    // process frame, obtain frame dims, and add to frames container
                    let proc_frame = process(&gray_frame)?;
                    let size = proc_frame.size()?;
                    (frame_dims.0, frame_dims.1) = (
                        size.height as usize,
                        size.width as usize,
                    );
                    frames.extend(proc_frame.data_bytes()?);
                }

                Ok(FramesBuffer {
                    data: frames,
                    height: frame_dims.0,
                    width: frame_dims.1,
                })
            }
            Err(e) => {
                eprintln!("Error opening video file: {}", e);
                Err(Box::new(e))
            }
        }

    }
}



impl Dataset<VsrmItem> for GridDataset {
    /// load and normalize a specific dataset sample obtained by index
    /// file IO and ROI cropping performed by a loader helper function from "io.rs"
    /// performs per-sample pixel normalization [0, 1] and tensorization
    /// params:
    /// - index: dataset entry index
    /// returns: standardized item with [C, T, H, W] frames and transcript IDs, or None if invalid
    fn get(&self, index: usize) -> Option<VsrmItem> {
        let entry = self.entries.get(index)?;

        let transcript_ids = self.load_alignment(entry).ok()?;
        let frames = match &self.tracker_config {
            Some(config) => {
                // --------------- mode (A): lip tracking and cropping ---------------
                LipTracker::with_local(config, |tracker| {
                    tracker.reset_state(); // clear smoothing state from last video
                    self.load_frames(entry, |frame| tracker.process_frame(frame))
                }).ok()?
            }
            None => {
                // -------------------- mode (B) full sized frames -------------------
                self.load_frames(entry, |frames| {
                    Ok(frames.clone())
                }).ok()?
            }
        };

        assert!(!frames.data.is_empty(), "No frames loaded");
        assert!(! transcript_ids.is_empty(), "No transcripts loaded");

        // isolate dims
        let (c, h, w) = (1, frames.height, frames.width);
        let (t, l) = (
            frames.data.len() / (c * h * w),
            transcript_ids.len(),
        );
        assert!(frames.data.len().is_multiple_of(c * h * w), "Frame buffer size {} is not divisible by frame dimensions", frames.data.len());
        assert!(t > 0, "Computed zero frames for item {}", entry);

        // enforce CTC constraint: T must be greater than L
        // (ideally 2x greater, for chars plus possible blanks)
        if t < (2 * l) {
            // eprintln!("\nSkipping sample {}: T = {} is too short for L = {}", entry_name, t, l);
            return None;
        }

        assert!(frames.data.len() == (c * t * h * w), "Tensor shape mismatch: len={} expected={}", frames.data.len(), (c * t * h * w));

        // convert frames into 4D tensor
        let frames = TensorData::new(
            frames.data,
            // frames.into_iter().map(|b| b as f32).collect(), // temp f32 conversion if Burn doesn't support u8 tensors yet
            vec![c, t, h, w],
        );

        Some(VsrmItem {
            frames,
            transcript_ids,
            item_id: entry.clone(),
        })
    }

    /// get the total number of samples in the dataset split
    /// params: none
    /// returns: count of valid video entries
    fn len(&self) -> usize {
        self.entries.len()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        vocab::VOCAB,
    };
    use image::{GrayImage, Luma};
    use std::fs;
    use rand::{
        Rng,
        SeedableRng,
        rngs::StdRng,
    };

    const SEED: u64 = 70;

    fn save_item_frames(item: &VsrmItem, context: &Context, prefix: &str) {
        let item_id = item.item_id.replace("/", "_");
        let output_dir = context.tests_path.join(format!("{}_{}", prefix, &item_id));
        if !output_dir.exists() { fs::create_dir_all(&output_dir).expect(&format!("Failed to create output directory for frames of item {}", item.item_id)); }

        // extract frame dims and convert frames to vec for slicing
        let (c, t, h, w) = (
            item.frames.shape[0],
            item.frames.shape[1],
            item.frames.shape[2],
            item.frames.shape[3],
        );

        println!("Exporting {} frames for item: {}\n", t, item.item_id);

        let frames = item.frames.as_slice::<u8>().expect("Failed to convert frames to slice");
        for t_idx in 0..t {
            let start_idx = t_idx * c * h * w;
            let end_idx = start_idx + (c * h * w);
            let frame_slice = &frames[start_idx..end_idx];
            
            // create grayscale image
            let mut img_buffer = GrayImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let pixel_value = (frame_slice[y * w + x]).clamp(0, 255) as u8;
                    img_buffer.put_pixel(x as u32, y as u32, Luma([pixel_value]));
                }
            }

            // save image
            let frame_path = output_dir.join(format!("{}_frame_{:03}.png", item_id, t_idx));
            img_buffer.save(&frame_path).expect("Failed to save extracted frame image");
        }
    }

    #[test]
    fn test_extract_full_frames_from_grid_dataset_item() {
        // test if we can load a GRID dataset item video at all
        // extract frames, and save them as individual PNG images for visual inspection

        let context = Context::new();
        let mut rng = StdRng::seed_from_u64(SEED);

        let tracker_config = None;

        // GRID dataset instance
        let dataset = GridDataset::new(&context, TokenMap::new(VOCAB), tracker_config);

        // obtain first valid GRID dataset item
        let item = dataset.get(rng.random_range(0..dataset.len()))
            .expect("Failed to extract a valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        // save collection of extracted frames as pngs
        save_item_frames(&item, &context, "full");
    }

    #[test]
    fn test_extract_cropped_frames_from_grid_dataset_item() {
        // test if provided mouth tracker is properly trackng and cropping to mouth region for GRID dataset item video
        // extract frames, and save them as individual PNG images for visual inspection

        let context = Context::new();
        let mut rng = StdRng::seed_from_u64(SEED);

        let tracker_config = Some(LipTrackerConfig::new(
            context.models_path.join("haarcascade_mcs_mouth.xml"),
            (50, 100),
        ));

        // GRID dataset instance
        let dataset = GridDataset::new(&context, TokenMap::new(VOCAB), tracker_config);

        // obtain first valid GRID dataset item
        let item = dataset.get(rng.random_range(0..dataset.len()))
            .expect("Failed to extract a valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        // save collection of extracted frames as pngs
        save_item_frames(&item, &context, "cropped");
    }
}