//! GRID-specific corpus adapter for audio-visual sentence processing.
//! 
//! This module implements the `GridDataset` adapter, which standardizes raw
//! GRID videos (as .mpg files) and alignments (as .align files) into a common
//! `VsrmItem` format. It also orchestrates frame loading, grayscale conversion,
//! and lip-region extraction.



// custom imports
use crate::{
    context::Context,
    pipeline::{
        FramesBuffer,
        batcher::VsrmItem,
        dataset::sample_subset_entries,
        io::{
            load_json,
            read_tensor_3d,
            save_json,
            write_tensor_3d,
        },
        tracker::
        {
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
    collections::{HashMap, HashSet},
    error::Error,
    fs::{read_dir, rename, File},
    io::{BufRead, BufReader},
    path::PathBuf,
};



pub struct GridDataset {
    pub grid_path: PathBuf,
    entries: Vec<String>,
    token_map: TokenMap,
    tracker_config: Option<TrackerConfig>,
    frames_to_alignment: HashMap<String, String>,
}



impl GridDataset {
    /// Constructor for GRID dataset adapter.
    /// 
    /// Scans disk for available video samples and their corresponding alignment files.
    /// 
    /// Stores valid entries as "speaker_id/item_id" (e.g., "s1/bbaf2n").
    /// 
    /// When `active_subset` is `Some((pct, seed))`, uses only fraction `pct` of entries (sampled with `seed`).
    ///
    /// ### Params:
    /// - `context`: Filesystem context (that should contain "data/grid-lr-corpus" subdirectory with frames and alignments).
    /// - `token_map`: Bidirectional mapping of chars to IDs for transcript encoding.
    /// - `tracker_config`: Optional lip tracker config for on-the-fly cropping when preproc bins are missing.
    /// - `active_subset`: Optional `(fraction, seed)` for subsetting (e.g. `Some((0.1, 69))` = 10% with seed 69). `None` = full dataset.
    ///
    /// ### Returns:
    /// Initialized `GridDataset` instance with valid entries loaded from disk.
    pub fn new(
        context: &Context,
        token_map: TokenMap,
        tracker_config: Option<TrackerConfig>,
        active_subset: Option<(f32, u64)>,
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

        let frames_to_alignment = determine_frames_to_alignment_mapping(
            &frames_path,
            &alignments_path,
            &avail_speakers,
        );

        // log non-identity mappings for debugging
        let non_identity: Vec<_> = frames_to_alignment
            .iter()
            .filter(|(k, v)| k != v)
            .collect();
        if !non_identity.is_empty() {
            println!("Discovered frame --> alignment mappings (non-identity): {:?}", non_identity);
        }

        // scan disk for only selected speakers and store
        let mut entries = Vec::new();
        for speaker in &avail_speakers { // loop through dirs of selected speakers
            let video_path = frames_path.join(speaker);
            let alignment_speaker = frames_to_alignment.get(speaker).map(|s| s.as_str()).unwrap_or(speaker.as_str());
            let alignment_path = alignments_path.join(alignment_speaker);

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

        // apply active_subset if specified
        if let Some((pct, subset_seed)) = active_subset {
            entries = sample_subset_entries(entries, pct, subset_seed);
            println!("Using active subset: {} samples ({:.1}% of full dataset)\n", entries.len(), pct * 100.0);
        }

        // per-speaker entry count diagnostic
        let mut per_speaker: HashMap<&str, usize> = HashMap::new();
        for e in &entries {
            let speaker = e.split('/').next().unwrap_or("");
            *per_speaker.entry(speaker).or_insert(0) += 1;
        }

        // speakers with zero entries
        let zeros: Vec<_> = avail_speakers
            .iter()
            .filter(|s| *per_speaker.get(s.as_str()).unwrap_or(&0) == 0)
            .collect();
        if !zeros.is_empty() {
            eprintln!(
                "WARNING: {} speaker(s) have 0 valid entries (no frame + alignment match): {:?}\n",
                zeros.len(),
                zeros
            );
            eprintln!("Per-speaker counts: {:?}", per_speaker);
        }

        println!("\nInitialized GridDataset: {} samples from speakers {:?}\n", entries.len(), avail_speakers);

        Self {
            grid_path,
            entries,
            token_map,
            tracker_config,
            frames_to_alignment,
        }
    }

    /// Attempts to load a single dataset entry by index.
    ///
    /// Fast path: loads video frames from preprocessed `.bin` in `preproc_frames/` if present.
    /// Slow path: decodes video and runs `LipTracker` (or full frames if no tracker).
    ///
    /// ### Params:
    /// - `index`: Dataset entry index.
    ///
    /// ### Returns:
    /// Standardized `VsrmItem` with [C, T, H, W] frames / transcript IDs, or `None` on any failure.
    fn try_load(&self, index: usize) -> Option<VsrmItem> {
        let entry = self.entries.get(index)?;
        let preproc_path = self.grid_path.join("preproc_frames");       // path to preprocessed collection of videos for each speaker
        let bin_path = preproc_path.join(entry).with_extension("bin");  // path to individual video frames as a binary file

        // load GRID transcripts and video frames
        let transcript_ids = self.load_alignment(entry).ok()?;
        // preproc_frames contain mouth crops; only use fast path when we want cropped (tracker_config is Some)
        let frames = if bin_path.exists() && self.tracker_config.is_some() {
            // fast path: load from preprocessed binary (mouth crops)
            let (data, (h, w, t)) = read_tensor_3d::<u8, _>(&bin_path).ok()?;
            if data.is_empty() || t == 0 { return None; }
            TensorData::new(data, vec![1, t, h, w])
        } else {
            // slow path: video decode + tracker (or full frames)
            let frames_buffer = match &self.tracker_config {
                // --------------- mode (A): lip tracking and cropping ---------------
                Some(config) => with_local_tracker(config, |tracker: &mut dyn LipTrackerBackend| {
                    tracker.reset_state();
                    self.load_frames(entry, |frame| {
                        tracker.process_frame(frame).map(|result| result.crop)
                    })
                }).ok()?,
                // -------------------- mode (B) full sized frames -------------------
                None => self.load_frames(entry, |f| Ok(f.clone())).ok()?,
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

                let data = entry.frames.as_slice::<u8>().expect("Failed to get u8 frame pixel data");
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

    /// Preprocesses video frames as mouth crops to disk for faster training.
    ///
    /// Skips if `preproc_frames/manifest.json` exists and `num_entries == len()` (run-once-and-skip).
    /// 
    /// Otherwise iterates entries, loads from video when bin missing, saves to `preproc_frames/{entry}.bin`.
    pub fn preprocess_all(&self) {
        let preproc_path = self.grid_path.join("preproc_frames");
        let manifest_path = preproc_path.join("manifest.json");
        std::fs::create_dir_all(&preproc_path).expect("Failed to create preproc_frames directory");

        #[derive(Serialize, Deserialize)]
        struct PreprocManifest { num_entries: usize }

        // skip preprocessing run if manifest marker file exists and entry count matches
        if manifest_path.exists() {
            if let Ok(manifest) = load_json::<_, PreprocManifest>(&manifest_path) {
                if manifest.num_entries == self.len() {
                    println!("Pre-extracted crops already complete ({} entries)\n", manifest.num_entries);
                    return;
                }
            }
        }

        println!("Preprocessing mouth regions for {} samples...", self.len());

        // setup progress bar
        let prog_bar = ProgressBar::new(self.len() as u64);
        prog_bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({msg}) (ETA: {eta})\n",
            )
            .unwrap()
            .progress_chars("#>-"),
        );

        // process each entry if not processed already
        for i in 0..self.len() {
            let entry = &self.entries[i];
            let bin_path = preproc_path.join(entry).with_extension("bin");

            if !bin_path.exists() {
                if let Some(item) = self.try_load(i) {
                    let pixel_data = item.frames.as_slice::<u8>().expect("u8 frame data");
                    let (h, w, t) = (
                        item.frames.shape[2],
                        item.frames.shape[3],
                        item.frames.shape[1],
                    );

                    if write_tensor_3d(&bin_path, pixel_data, (h, w, t)).is_err() {
                        eprintln!("Failed to save {}", bin_path.display());
                    }
                }
            }

            prog_bar.set_message(format!("{}", entry));
            prog_bar.inc(1);
        }

        // persist JSON manifest marker so future runs can skip this process
        let manifest = PreprocManifest { num_entries: self.len() };
        save_json(&manifest_path, &manifest).expect("Failed to write preproc manifest");
        prog_bar.finish_with_message("Preprocessing complete");
    }

    /// Diagnostic: report per-speaker entry counts and frame/alignment stem mismatch for zero-entry speakers.
    ///
    /// Helps identify why some speakers contribute 0 entries (e.g., frame and alignment stems do not overlap).
    pub fn diagnose_entry_mismatch(&self) {
        use std::collections::{HashMap, HashSet};

        let frames_path = self.grid_path.join("frames");
        let alignments_path = self.grid_path.join("alignments");

        // collect avail_speakers from frames dir
        let mut avail_speakers = Vec::new();
        if let Ok(speaker_paths) = read_dir(&frames_path) {
            for speaker_path in speaker_paths.flatten() {
                let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
                if speaker_str.starts_with('s') && speaker_path.path().is_dir() {
                    avail_speakers.push(speaker_str);
                }
            }
        }
        avail_speakers.sort_by_key(|s| s[1..].parse::<i32>().unwrap_or(1));

        // per-speaker entry counts from self.entries
        let mut per_speaker: HashMap<String, usize> = HashMap::new();
        for e in &self.entries {
            let speaker = e.split('/').next().unwrap_or("").to_string();
            *per_speaker.entry(speaker).or_insert(0) += 1;
        }

        let zero_speakers: Vec<_> = avail_speakers
            .iter()
            .filter(|s| *per_speaker.get(*s).unwrap_or(&0) == 0)
            .collect();

        println!("Total entries: {}", self.len());
        println!("Speakers with 0 entries: {} ({:?})\n", zero_speakers.len(), zero_speakers);

        for speaker in zero_speakers {
            let video_path = frames_path.join(speaker);
            let alignment_speaker = self.frames_to_alignment.get(speaker).map(|s| s.as_str()).unwrap_or(speaker.as_str());
            let alignment_path = alignments_path.join(alignment_speaker);

            let mut frame_stems: HashSet<String> = HashSet::new();
            if let Ok(items) = read_dir(&video_path) {
                for item in items.flatten() {
                    if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {
                        if item.path().extension().is_some_and(|ext| ext == "mpg") {
                            frame_stems.insert(stem.to_string());
                        }
                    }
                }
            }

            let mut align_stems: HashSet<String> = HashSet::new();
            if let Ok(items) = read_dir(&alignment_path) {
                for item in items.flatten() {
                    if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {
                        if item.path().extension().is_some_and(|ext| ext == "align") {
                            align_stems.insert(stem.to_string());
                        }
                    }
                }
            }

            let intersection: HashSet<_> = frame_stems.intersection(&align_stems).collect();
            let only_in_frames: Vec<_> = frame_stems.difference(&align_stems).take(5).cloned().collect();
            let only_in_align: Vec<_> = align_stems.difference(&frame_stems).take(5).cloned().collect();

            println!("Speaker {}: 0 entries", speaker);
            println!(
                "  Frames: {} files | Alignments: {} files",
                frame_stems.len(),
                align_stems.len()
            );
            println!("  Intersection size: {}", intersection.len());
            println!("  Sample stems only in frames: {:?}", only_in_frames);
            println!("  Sample stems only in alignments: {:?}", only_in_align);
            println!();
        }
    }

    /// Helper for parsing a GRID-specific `.align` transcript file.
    ///
    /// ### Params:
    /// - `entry`: Unique GRID dataset entry ID to parse alignments from (in the form of "s1/bbaf2n").
    ///
    /// ### Returns:
    /// A sequence of corresponding char IDs.
    fn load_alignment(&self, entry: &str) -> Result<Vec<usize>, Box<dyn Error>> {
        let (speaker, stem) = entry.split_once('/').unwrap_or((entry, ""));
        let alignment_speaker = self.frames_to_alignment.get(speaker).map(|s| s.as_str()).unwrap_or(speaker);
        let alignment_path = self.grid_path
            .join("alignments")
            .join(alignment_speaker)
            .join(stem)
            .with_extension("align");
        assert!(alignment_path.exists(), "Alignment file {} for GRID entry {} not found", alignment_path.to_string_lossy(), entry);

        match File::open(alignment_path) {
            Ok(file) => {
                let mut sequence: Vec<usize> = vec![];
                let lines = BufReader::new(file).lines();

                for line in lines.map_while(Result::ok) {
                    let line_group = line.split_whitespace().collect::<Vec<_>>();
                    assert!(line_group.len() >= 3, "Malformed alignment line: {:?}", line_group);

                    let word = line_group[2];
                    if word != "sil" && word != "sp" {
                        if !sequence.is_empty() { sequence.push(SPACE_ID); }

                        let char_ids = word.chars().filter_map(|char| self.token_map.id_of(char));
                        sequence.extend(char_ids);
                    }
                }
                assert!(!sequence.is_empty(), "No non-silence tokens found in alignment file");

                Ok(sequence)
            }
            Err(e) => {
                eprintln!("Error opening alignments file: {}", e);
                Err(Box::new(e))
            }
        }
    }

    /// Helper for processing a GRID-specific `.mpg` video file.
    ///
    /// ### Params:
    /// - `entry`: Unique GRID dataset entry ID to load and process frames from (in the form of "s1/bbaf2n").
    /// - `process`: The process to apply to the given frame.
    ///
    /// ### Returns:
    /// A `FramesBuffer` containing the flattened vector of frames along with frame dimensions.
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
    fn len(&self) -> usize {
        self.entries.len()
    }
}



/// Discovers the correct alignment speaker folder for each frames speaker by maximizing stem overlap.
/// 
/// On ties, prefers identity (same name).
/// 
/// ### Returns:
/// frames_speaker -> alignment_speaker.
fn determine_frames_to_alignment_mapping(
    frames_path: &PathBuf,
    alignments_path: &PathBuf,
    avail_speakers: &[String],
) -> HashMap<String, String> {
    // 1. collect alignment stems per alignment speaker
    let mut align_stems: HashMap<String, HashSet<String>> = HashMap::new();
    if let Ok(speaker_paths) = read_dir(alignments_path) {
        for speaker_path in speaker_paths.flatten() {
            let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
            if speaker_str.starts_with('s') && speaker_path.path().is_dir() {
                let mut stems = HashSet::new();
                if let Ok(items) = read_dir(speaker_path.path()) {
                    for item in items.flatten() {
                        if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {
                            if item.path().extension().is_some_and(|ext| ext == "align") {
                                stems.insert(stem.to_string());
                            }
                        }
                    }
                }
                align_stems.insert(speaker_str, stems);
            }
        }
    }

    // 2. for each frames speaker, find alignment speaker with max overlap; tie-break: prefer identity
    let mut map = HashMap::new();
    for speaker in avail_speakers {
        let video_path = frames_path.join(speaker);
        let mut frame_stems = HashSet::new();
        if let Ok(items) = read_dir(&video_path) {
            for item in items.flatten() {
                if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {
                    if item.path().extension().is_some_and(|ext| ext == "mpg") {
                        frame_stems.insert(stem.to_string());
                    }
                }
            }
        }

        let best = align_stems
            .iter()
            .map(|(align_sp, align_set)| {
                let overlap = frame_stems.intersection(align_set).count();
                let same_name = align_sp == speaker;
                (overlap, same_name, align_sp.clone())
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
            .map(|(_, _, sp)| sp)
            .unwrap_or_else(|| speaker.clone());

        map.insert(speaker.clone(), best);
    }
    map
}



/// Since the GRID dataset speaker dirs in `grid-lr-corpus/frames/` might not be correctly
/// mapped to the same speaker dir in `grid-lr-corpus/alignments/`, we use this helper to
/// correct the mappings.
/// 
/// Physically renames alignment directories to match frames speakers.
///
/// Uses a two-phase rename (via temp dirs) to handle permutations. On `dry_run`,
/// only prints planned renames without modifying the filesystem.
///
/// ### Params:
/// - `context`: Filesystem context (data path must contain `grid-lr-corpus/frames` and `grid-lr-corpus/alignments`).
/// - `dry_run`: If true, only print planned renames.
///
/// ### Returns:
/// `Ok(())` on success, or an error if mapping is ambiguous (many-to-one) or I/O fails.
pub fn align_grid_directories(context: &crate::context::Context, dry_run: bool) -> Result<(), Box<dyn Error>> {
    let grid_path = context.data_path.join("grid-lr-corpus");
    let frames_path = grid_path.join("frames");
    let alignments_path = grid_path.join("alignments");

    if !grid_path.exists() {
        return Err(format!("GRID corpus directory does not exist at {:?}", grid_path).into());
    }
    if !frames_path.exists() {
        return Err(format!("GRID frames directory does not exist at {:?}", frames_path).into());
    }
    if !alignments_path.exists() {
        return Err(format!("GRID alignments directory does not exist at {:?}", alignments_path).into());
    }

    // Collect frames speakers
    let mut avail_speakers = Vec::new();
    if let Ok(speaker_paths) = read_dir(&frames_path) {
        for speaker_path in speaker_paths.flatten() {
            let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
            if speaker_str.starts_with('s') && speaker_path.path().is_dir() {
                avail_speakers.push(speaker_str);
            }
        }
    }
    avail_speakers.sort_by_key(|s| s[1..].parse::<i32>().unwrap_or(1));

    let mapping = determine_frames_to_alignment_mapping(&frames_path, &alignments_path, &avail_speakers);

    // Many-to-one check: each alignment speaker must map to at most one frames speaker
    let mut align_to_frames: HashMap<String, Vec<String>> = HashMap::new();
    for (f, a) in &mapping {
        align_to_frames.entry(a.clone()).or_default().push(f.clone());
    }
    for (a, fs) in &align_to_frames {
        if fs.len() > 1 {
            return Err(format!(
                "Ambiguous mapping: alignment dir {} is best match for multiple frames dirs: {:?}",
                a, fs
            )
            .into());
        }
    }

    // collect alignment dirs (s1, s2, ...)
    let mut align_dirs: Vec<String> = Vec::new();
    if let Ok(speaker_paths) = read_dir(&alignments_path) {
        for speaker_path in speaker_paths.flatten() {
            let speaker_str = speaker_path.file_name().to_string_lossy().to_string();
            if speaker_str.starts_with('s') && !speaker_str.starts_with("_temp_") && speaker_path.path().is_dir() {
                align_dirs.push(speaker_str);
            }
        }
    }

    if dry_run {
        println!("[DRY RUN] Planned renames:");
        let non_identity: Vec<_> = mapping.iter().filter(|(k, v)| k != v).collect();
        if non_identity.is_empty() {
            println!("  No non-identity mappings; all alignments already match frames.");
        } else {
            for (f, a) in non_identity {
                println!("  alignments/{} -> alignments/{} (via _temp)", a, f);
            }
        }
        return Ok(());
    }

    // phase 1: rename all alignment dirs to _temp_X
    for name in &align_dirs {
        let src = alignments_path.join(name);
        let dst = alignments_path.join(format!("_temp_{}", name));
        if src.exists() {
            rename(&src, &dst)?;
        }
    }

    // phase 2: for each frames speaker F, move _temp_{mapping[F]} to F
    for f in &avail_speakers {
        let a = mapping.get(f).map(|s| s.as_str()).unwrap_or(f.as_str());
        let src = alignments_path.join(format!("_temp_{}", a));
        let dst = alignments_path.join(f);
        if src.exists() {
            rename(&src, &dst)?;
        }
    }

    // phase 3: restore any remaining _temp_X (alignment dirs not in mapping range)
    if let Ok(entries) = read_dir(&alignments_path) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("_temp_") {
                let orig = name.strip_prefix("_temp_").unwrap_or(&name).to_string();
                let src = alignments_path.join(&name);
                let dst = alignments_path.join(&orig);
                if src.exists() {
                    rename(&src, &dst)?;
                }
            }
        }
    }

    println!("Alignment directories renamed to match frames.");
    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        pipeline::tracker::HaarTrackerConfig,
        vocab::VOCAB,
    };
    use image::{GrayImage, Luma};
    use std::fs;
    use rand::{
        Rng,
        SeedableRng,
        rngs::StdRng,
    };

    const SEED: u64 = 69;

    // helper function for saving frames
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

        // GRID dataset instance
        let dataset = GridDataset::new(
            &context,
            TokenMap::new(VOCAB),
            None,
            None,
        );

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

        let face_cascade_path = context.models_path.join("haarcascade_frontalface_alt2.xml");
        let mouth_cascade_path = context.models_path.join("haarcascade_mcs_mouth.xml");
        let target_dims = (50, 100);

        let tracker_config = TrackerConfig::Haar(
            HaarTrackerConfig::new(
                face_cascade_path,
                mouth_cascade_path,
                target_dims,
            ).with_smoothing_alpha(0.8)
        );

        // GRID dataset instance
        let dataset = GridDataset::new(
            &context,
            TokenMap::new(VOCAB),
            Some(tracker_config),
            None,
        );

        // obtain first valid GRID dataset item
        let item = dataset.get(rng.random_range(0..dataset.len()))
            .expect("Failed to extract a valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        // save collection of extracted frames as pngs
        save_item_frames(&item, &context, "cropped");
    }
}