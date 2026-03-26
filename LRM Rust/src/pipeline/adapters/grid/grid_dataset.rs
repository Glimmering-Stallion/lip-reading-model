//! GRID-specific corpus adapter for audio-visual sentence processing.
//!
//! This module implements the `GridDataset` adapter, which standardizes raw
//! GRID videos (as .mpg files) and alignments (as .align files) into a common
//! `VsrmItem` format. It also orchestrates frame loading, grayscale conversion,
//! and lip-region extraction.



// custom imports
use crate::{
    context::Context,
    prelude::{io_err, ESS},
    pipeline::{
        FramesBuffer,
        batcher::VsrmItem,
        dataset::sample_subset_entries,
        io::{
            file_nonempty,
            load_json,
            read_tensor_3d,
            save_json,
            write_tensor_3d,
        },
        tracker::{
            LipTrackerBackend,
            TrackerConfig,
            with_local_tracker,
        }
    },
    vocab::{SPACE_ID, TokenMap},
};

// imports
use serde::{Deserialize, Serialize};
use burn::{
    data::dataset::Dataset,
    tensor::TensorData,
};
use indicatif::{ProgressBar, ProgressStyle};
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
    collections::HashMap,
    fs::{File, read_dir},
    path::{Path, PathBuf},
    io::{
        BufRead,
        BufReader,
        ErrorKind,
    },
};

/// Subdirectory of `grid-lr-corpus` for pre-extracted mouth-crop `.bin` tensors per dataset entry.
const CROPPED_FRAMES_DIR: &str = "cropped_frames";



pub struct GridDataset {
    pub grid_path: PathBuf,
    entries: Vec<String>,
    token_map: TokenMap,
    tracker_config: Option<TrackerConfig>,
}



impl GridDataset {
    /// Constructor for GRID dataset adapter.
    ///
    /// Scans disk for available video samples and their corresponding transcript files.
    ///
    /// Stores valid entries as "speaker_id/item_id" (e.g., "s1/bbaf2n").
    ///
    /// When `active_subset` is `Some((pct, seed))`, uses only fraction `pct` of entries (sampled with `seed`).
    ///
    /// ### Params:
    /// - `context`: Filesystem context (that should contain `data/grid-lr-corpus/<speaker>/<utterance_id>/`
    /// - `token_map`: Bidirectional mapping of tokens to IDs for transcript encoding.
    /// - `tracker_config`: Optional lip tracker config for on-the-fly cropping when pre-extracted `.bin` crops are missing.
    /// - `active_subset`: Optional `(fraction, seed)` for subsetting (e.g. `Some((0.1, 69))` = 10% with seed 69). `None` = full dataset.
    ///
    /// ### Returns:
    /// Initialized `GridDataset` instance with valid entries loaded from disk.
    pub fn new(
        context: &Context,
        token_map: &TokenMap,
        tracker_config: Option<TrackerConfig>,
        active_subset: Option<(f32, u64)>,
    ) -> Self {
        let grid_path = context.data_path.join("grid-lr-corpus");
        assert!(grid_path.exists(), "GRID corpus directory does not exist at {:?}", grid_path);

        // identify all speakers available on disk (s1, s2, ..., s34) in bundled layout
        let mut avail_speakers = Vec::new();
        if let Ok(speaker_paths) = read_dir(&grid_path) {
            for speaker_path in speaker_paths.flatten() {
                let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
                if speaker_str.starts_with('s') && speaker_path.path().is_dir()
                { avail_speakers.push(speaker_str); }
            }
        }
        assert!(!avail_speakers.is_empty(), "no bundled speaker directories found in {:?}", grid_path);

        // sort s1, s2, ..., s34
        avail_speakers.sort_by_key(|s| s[1..].parse::<i32>().unwrap_or(1));

        // scan disk for only selected speakers and store utterance entries
        let mut utterance_entries = Vec::new();
        for speaker in &avail_speakers {
            let speaker_path = grid_path.join(speaker);
            if let Ok(entries) = read_dir(&speaker_path) {
                for entry in entries.flatten() {
                    if !entry.path().is_dir() { continue; }
                    let utterance_id = entry.file_name().to_string_lossy().to_string();
                    let utterance_path = speaker_path.join(&utterance_id);

                    let mpg_path = utterance_path.join(format!("{}.mpg", utterance_id));
                    let mp4_path = utterance_path.join(format!("{}.mp4", utterance_id));
                    let video_ok = file_nonempty(&mp4_path) || file_nonempty(&mpg_path);

                    let align_path = utterance_path.join(format!("{}.align", utterance_id));
                    let txt_path = utterance_path.join(format!("{}.txt", utterance_id));
                    let transcript_ok = align_path.is_file() || txt_path.is_file();

                    if video_ok && transcript_ok { utterance_entries.push(format!("{}/{}", speaker, utterance_id)); }
                }
            }
        }
        utterance_entries.sort(); // sort for deterministic order
        assert!(!utterance_entries.is_empty(), "dataset instance resulted in 0 samples\ncheck if path {:?} contains .mpg files", grid_path);

        // apply active_subset if specified
        if let Some((pct, subset_seed)) = active_subset {
            utterance_entries = sample_subset_entries(utterance_entries, pct, subset_seed);
            println!("Using active subset: {} samples ({:.1}% of full dataset)\n", utterance_entries.len(), (pct * 100.0));
        }

        // per-speaker entry count diagnostic
        let mut per_speaker: HashMap<&str, usize> = HashMap::new();
        for e in &utterance_entries {
            let speaker = e.split('/').next().unwrap_or("");
            *per_speaker.entry(speaker).or_insert(0) += 1;
        }

        // speakers with zero entries
        let zeros: Vec<_> = avail_speakers
            .iter()
            .filter(|s| *per_speaker.get(s.as_str()).unwrap_or(&0) == 0)
            .collect();
        if !zeros.is_empty() {
            eprintln!("WARNING: {} speaker(s) have 0 valid entries (no frame + alignment match): {:?}\n", zeros.len(), zeros);
            eprintln!("Per-speaker counts: {:?}", per_speaker);
        }

        println!("Initialized GridDataset: {} samples from speakers {:?}\n", utterance_entries.len(), avail_speakers);

        Self {
            grid_path,
            entries: utterance_entries,
            token_map: token_map.clone(),
            tracker_config,
        }
    }

    /// Attempts to load a single dataset entry by index.
    ///
    /// Fast path: loads video frames from pre-extracted `.bin` mouth crops in `cropped_frames/` if present.
    /// Slow path: decodes video and runs `LipTracker` (or full frames if no tracker).
    ///
    /// ### Params:
    /// - `index`: Dataset entry index.
    ///
    /// ### Returns:
    /// Standardized `VsrmItem` with [C, T, H, W] frames / transcript IDs, or `None` on any failure.
    fn try_load(&self, index: usize) -> Option<VsrmItem> {
        let entry = self.entries.get(index)?;
        let pre_extract_path = self.grid_path.join(CROPPED_FRAMES_DIR);
        let bin_path = pre_extract_path.join(entry).with_extension("bin");

        // load GRID transcripts and video frames
        let transcript_ids = self.load_transcript(entry).ok()?;
        // cropped_frames holds mouth crops; only use fast path when we want cropped (tracker_config is Some)
        let frames = if bin_path.exists() && self.tracker_config.is_some() {
            // fast path: load from pre-extracted binary (mouth crops)
            let (data, (h, w, t)) = read_tensor_3d::<u8, _>(&bin_path).ok()?;
            if data.is_empty() || t == 0 { return None; }
            TensorData::new(data, vec![1, t, h, w])
        } else {
            // slow path: video decode + tracker (or full frames)
            let frames_buffer = match &self.tracker_config {
                // --------------- mode (A): lip tracking and cropping ---------------
                Some(config) => with_local_tracker(config, |tracker: &mut dyn LipTrackerBackend| {
                    tracker.reset_state(); // clear smoothing state from last video to prevent drift
                    self.load_video(entry, |frame| { tracker.process_frame(frame).map(|result| result.crop) })
                }).ok()?,
                // -------------------- mode (B) full sized frames -------------------
                None => self.load_video(entry, |f| Ok(f.clone())).ok()?,
            };

            // internal filtering
            if frames_buffer.data.is_empty() { return None; }
            let (c, h, w) = (1, frames_buffer.height, frames_buffer.width);
            if !frames_buffer.data.len().is_multiple_of(c * h * w) { return None; }
            let t = frames_buffer.data.len() / (c * h * w);

            // convert frames into 4D TensorData buffer
            TensorData::new(frames_buffer.data, vec![c, t, h, w])
        };

        // external filtering
        let t = frames.shape[1];
        let l = transcript_ids.len();
        if t == 0 || l == 0 { return None; }
        if t < (2 * l) { return None; } // CTC constraint

        Some(VsrmItem {
            frames,
            transcript_ids,
            item_id: entry.clone(),
        })
    }

    /// Helper for parsing a transcript file for the GRID dataset.
    /// 
    /// Prefers to parse '.txt' else parses '.align'.
    ///
    /// ### Params:
    /// - `entry`: Unique GRID dataset entry ID to parse alignments from (in the form of "s1/bbaf2n").
    ///
    /// ### Returns:
    /// A sequence of corresponding token IDs.
    fn load_transcript(&self, entry: &str) -> Result<Vec<usize>, ESS> {
        let (speaker, stem) = entry.split_once('/').unwrap_or((entry, ""));
        let utterance_path = self.grid_path.join(speaker).join(stem);
        let txt_path = utterance_path.join(stem).with_extension("txt");
        let align_path = utterance_path.join(stem).with_extension("align");

        if txt_path.is_file() { return Self::load_transcript_from_txt_file(&txt_path, &self.token_map); }
        if align_path.is_file() { return Self::load_transcript_from_align_file(&align_path, &self.token_map); }

        Err(io_err(
            format!("no transcript found for GRID entry {} (expected {:?} or {:?})", entry, align_path, txt_path),
            ErrorKind::NotFound,
        ))
    }

    /// Fallback helper for parsing a GRID-specific `.align` transcript file.
    /// 
    /// `.align` files are in the form of stacked start-end timestamp word lines.
    ///
    /// ### Params:
    /// - `path`: Path to the `.align` file.
    /// - `token_map`: Bidirectional mapping of tokens to IDs for transcript encoding.
    ///
    /// ### Returns:
    /// A sequence of corresponding token IDs.
    fn load_transcript_from_align_file(path: &Path, token_map: &TokenMap) -> Result<Vec<usize>, ESS> {
        let file = File::open(path).map_err(|e| { io_err(format!("failed to open alignment file {:?}: {}", path, e), ErrorKind::Other) })?;
        let mut sequence: Vec<usize> = vec![];

        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let line_group = line.split_whitespace().collect::<Vec<_>>();
            if line_group.len() < 3 { continue; }

            let word = line_group[2];
            if word != "sil" && word != "sp" {
                if !sequence.is_empty() { sequence.push(SPACE_ID); }
                let char_ids = word.chars().filter_map(|c| token_map.id_of(c));
                sequence.extend(char_ids);
            }
        }

        if sequence.is_empty() { return Err(io_err(format!("no non-silence tokens found in {:?}", path), ErrorKind::InvalidData)); }
        Ok(sequence)
    }

    /// Helper for parsing a standardized `.txt` transcript file.
    /// 
    /// `.txt` files are in the form of single-line, whitespace-separated word sequences.
    ///
    /// ### Params:
    /// - `path`: Path to the `.txt` file.
    /// - `token_map`: Bidirectional mapping of tokens to IDs for transcript encoding.
    ///
    /// ### Returns:
    /// A sequence of corresponding token IDs.
    fn load_transcript_from_txt_file(path: &Path, token_map: &TokenMap) -> Result<Vec<usize>, ESS> {
        let content = std::fs::read_to_string(path).map_err(|e| { io_err(format!("failed to read transcript txt {:?}: {}", path, e), ErrorKind::Other) })?;

        let mut sequence: Vec<usize> = vec![];
        for word in content.split_whitespace() {
            if word.is_empty() { continue; }
            if !sequence.is_empty() { sequence.push(SPACE_ID); }
            sequence.extend(word.chars().filter_map(|c| token_map.id_of(c)));
        }

        if sequence.is_empty() { return Err(io_err(format!("empty or unmapped transcript in {:?}", path), ErrorKind::InvalidData)); }
        Ok(sequence)
    }

    /// Helper for processing a video file for the GRID dataset.
    /// 
    /// Prefers to process '.mp4' else processes '.mpg'.
    ///
    /// ### Params:
    /// - `entry`: Unique GRID dataset entry ID to load and process frames from (in the form of "s1/bbaf2n").
    /// - `process`: The processing to apply to the given frame.
    ///
    /// ### Returns:
    /// A `FramesBuffer` containing the flattened vector of frames along with frame dimensions.
    fn load_video<F>(&self, entry: &str, mut process: F) -> Result<FramesBuffer, ESS>
    where F: FnMut(&Mat) -> Result<Mat, ESS>
    {
        let (speaker, stem) = entry.split_once('/').unwrap_or((entry, ""));
        let utterance_path = self.grid_path.join(speaker).join(stem);
        let mpg_path = utterance_path.join(stem).with_extension("mpg");
        let mp4_path = utterance_path.join(stem).with_extension("mp4");

        let frames_path = if file_nonempty(&mp4_path) { mp4_path }
        else if mpg_path.is_file() { mpg_path }
        else {
            return Err(io_err(
                format!(
                    "no video for GRID entry {} (expected {:?} or {:?})",
                    entry,
                    utterance_path.join(stem).with_extension("mp4"),
                    utterance_path.join(stem).with_extension("mpg")
                ),
                ErrorKind::NotFound,
            ));
        };

        let mut frames: Vec<u8> = Vec::new();
        let mut frame_dims: (usize, usize) = (0, 0);
        let (mut orig_frame, mut gray_frame) = (Mat::default(), Mat::default());

        let path_str = frames_path
            .to_str()
            .ok_or_else(|| io_err("invalid path", ErrorKind::InvalidInput))?;

        match VideoCapture::from_file(path_str, CAP_ANY) {
            Ok(mut cap) => {
                while cap
                    .read(&mut orig_frame)
                    .map_err(|e| io_err(e.to_string(), ErrorKind::Other))?
                {
                    if orig_frame.empty() { break; }

                    imgproc::cvt_color(
                        &orig_frame,
                        &mut gray_frame,
                        imgproc::COLOR_BGR2GRAY,
                        0,
                        AlgorithmHint::ALGO_HINT_DEFAULT,
                    ).map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;

                    let proc_frame = process(&gray_frame)?;
                    let size = proc_frame.size().map_err(|e| io_err(e.to_string(), ErrorKind::Other))?;
                    (frame_dims.0, frame_dims.1) = (size.height as usize, size.width as usize);
                    frames.extend(proc_frame.data_bytes().map_err(|e| io_err(e.to_string(), ErrorKind::Other))?);
                }

                Ok(FramesBuffer {
                    data: frames,
                    height: frame_dims.0,
                    width: frame_dims.1,
                })
            }
            Err(e) => {
                eprintln!("error opening video file: {}", e);
                Err(io_err(e.to_string(), ErrorKind::Other))
            }
        }
    }

    /// Iterates through the entire GRID corpus to calculate global mean and std dev of video pixel values for input normalization.
    /// 
    /// ### Returns:
    /// The global mean and std dev values found from the GRID corpus.
    pub fn calc_global_stats(&self) -> (f32, f32) {
        let mut total_sum = 0.0f64;
        let mut total_sum_sq = 0.0f64;
        let mut total_pix = 0u64;
        let entry_count = self.len();

        println!("Calculating global stats for {} samples from the GRID corpus", entry_count);

        let prog_bar = ProgressBar::new(entry_count as u64);
        prog_bar.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) (ETA: {eta})")
                .unwrap()
                .progress_chars("#>-")
        );

        for i in 0..entry_count {
            if let Some(entry) = self.get(i) {
                if i.is_multiple_of(100) { prog_bar.set_message(format!("Processing: {}", entry.item_id)); }

                let data = entry.frames.as_slice::<u8>().expect("failed to get u8 frame pixel data");
                for &pixel in data {
                    let p = (pixel as f64) / 255.0;
                    total_sum += p;
                    total_sum_sq += p * p;
                    total_pix += 1;
                }
            }

            prog_bar.inc(1);
        }

        let mean = total_sum / (total_pix as f64);
        let var = (total_sum_sq / (total_pix as f64)) - (mean * mean);
        let std_dev = var.sqrt();

        prog_bar.finish_with_message("Global stats calculated successfully\n");

        (mean as f32, std_dev as f32)
    }

    /// Pre-extracts mouth-crop frames to disk (`cropped_frames/{entry}.bin`) for faster training.
    ///
    /// Skips if `cropped_frames/manifest.json` exists and `num_entries == len()` (run-once-and-skip).
    ///
    /// Otherwise iterates entries, loads from video when bin missing, saves to `cropped_frames/{entry}.bin`.
    pub fn pre_extract_all(&self) {
        let pre_extract_path = self.grid_path.join(CROPPED_FRAMES_DIR);
        let manifest_path = pre_extract_path.join("manifest.json");
        std::fs::create_dir_all(&pre_extract_path).expect("failed to create cropped_frames directory");

        #[derive(Serialize, Deserialize)]
        struct PreExtractManifest { num_entries: usize }

        if manifest_path.exists() {
            if let Ok(manifest) = load_json::<_, PreExtractManifest>(&manifest_path) {
                if manifest.num_entries == self.len() {
                    println!("Pre-extracted crops already complete ({} entries)\n", manifest.num_entries);
                    return;
                }
            }
        }

        println!("Pre-extracting GRID mouth regions for {} samples...", self.len());

        let prog_bar = ProgressBar::new(self.len() as u64);
        prog_bar.set_style(
            ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) (ETA: {eta})\n")
                .unwrap()
                .progress_chars("#>-"),
        );

        for i in 0..self.len() {
            let entry = &self.entries[i];
            let bin_path = pre_extract_path.join(entry).with_extension("bin");

            if !bin_path.exists() {
                if let Some(item) = self.try_load(i) {
                    let pixel_data = item.frames.as_slice::<u8>().expect("u8 frame data");
                    let (h, w, t) = (item.frames.shape[2], item.frames.shape[3], item.frames.shape[1]);

                    if write_tensor_3d(&bin_path, pixel_data, (h, w, t)).is_err()
                    { eprintln!("Failed to save {}", bin_path.display()); }
                }
            }

            prog_bar.set_message(format!("{}", entry));
            prog_bar.inc(1);
        }

        let manifest = PreExtractManifest { num_entries: self.len() };
        save_json(&manifest_path, &manifest).expect("failed to write pre_extract manifest");
        prog_bar.finish_with_message("Pre-extraction complete");
        println!("\n");

    }
}



impl Dataset<VsrmItem> for GridDataset {
    /// Loads a dataset sample by index with fallback to adjacent entries.
    /// 
    /// Only returns `None` when `index >= len()`, so Burn's dataloader never sees a mid-dataset `None` that would terminate epoch early.
    ///
    /// ### Params:
    /// - `index`: Dataset entry index.
    ///
    /// ### Returns:
    /// A `VsrmItem` or `None`.
    fn get(&self, index: usize) -> Option<VsrmItem> {
        if index >= self.entries.len() { return None; }

        for offset in 0..self.entries.len() {
            let try_idx = (index + offset) % self.entries.len();
            if let Some(item) = self.try_load(try_idx) { return Some(item); }
        }

        None
    }

    /// Gets total number of samples in the dataset split.
    ///
    /// ### Returns:
    /// Count of valid video entries.
    fn len(&self) -> usize { self.entries.len() }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        pipeline::{
            adapters::grid::{align_grid_directories, bundle_grid_utterances},
            tracker::HaarTrackerConfig,
        },
        vocab::VOCAB,
    };
    use image::{GrayImage, Luma};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::fs;

    const SEED: u64 = 69;

    fn save_item_frames(item: &VsrmItem, context: &Context, prefix: &str) {
        let item_id = item.item_id.replace("/", "_");
        let output_dir = context.outputs_path.join(format!("{}_{}", prefix, &item_id));
        if !output_dir.exists() { fs::create_dir_all(&output_dir).expect(&format!("Failed to create output directory for frames of item {}", item.item_id)); }

        let (c, t, h, w) = (
            item.frames.shape[0],
            item.frames.shape[1],
            item.frames.shape[2],
            item.frames.shape[3],
        );

        println!("Exporting {} frames for item: {}\n", t, item.item_id);

        let frames = item
            .frames
            .as_slice::<u8>()
            .expect("failed to convert frames to slice");
        for t_idx in 0..t {
            let start_idx = t_idx * c * h * w;
            let end_idx = start_idx + (c * h * w);
            let frame_slice = &frames[start_idx..end_idx];

            let mut img_buffer = GrayImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let pixel_value = (frame_slice[y * w + x]).clamp(0, 255);
                    img_buffer.put_pixel(x as u32, y as u32, Luma([pixel_value]));
                }
            }

            let frame_path = output_dir.join(format!("{}_frame_{:03}.png", item_id, t_idx));
            img_buffer.save(&frame_path).expect("failed to save extracted frame image");
        }
    }

    #[test]
    fn test_extract_full_frames_from_grid_dataset_item() {
        let context = Context::new();
        let token_map = TokenMap::new(VOCAB);
        let mut rng = StdRng::seed_from_u64(SEED);

        align_grid_directories(&context, false).expect("failed to validate GRID speaker mapping");
        bundle_grid_utterances(&context).expect("failed to bundle GRID utterances");

        let dataset = GridDataset::new(&context, &token_map, None, None);

        let item = dataset
            .get(rng.random_range(0..dataset.len()))
            .expect("failed to extract a valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        save_item_frames(&item, &context, "full");
    }

    #[test]
    fn test_extract_cropped_frames_from_grid_dataset_item() {
        let context = Context::new();
        let token_map = TokenMap::new(VOCAB);
        let mut rng = StdRng::seed_from_u64(SEED);

        align_grid_directories(&context, false).expect("failed to validate GRID speaker mapping");
        bundle_grid_utterances(&context).expect("failed to bundle GRID utterances");

        let face_cascade_path = context.models_path.join("haarcascade_frontalface_alt2.xml");
        let mouth_cascade_path = context.models_path.join("haarcascade_mcs_mouth.xml");
        let target_dims = (50, 100);

        let tracker_config = TrackerConfig::Haar(
            HaarTrackerConfig::new(face_cascade_path, mouth_cascade_path, target_dims)
                .with_smoothing_alpha(0.8),
        );

        let dataset = GridDataset::new(&context, &token_map, Some(tracker_config), None);

        let item = dataset
            .get(rng.random_range(0..dataset.len()))
            .expect("failed to extract a valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        save_item_frames(&item, &context, "cropped");
    }
}
