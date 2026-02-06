// Data preprocessing/standardization for GRID audio-visual sentence corpus



// custom imports
use crate::{
    vocab::TokenMap,
    pipeline::{
        batcher::VsrmItem,
        io::{
            load_grid_corpus,
        },
        DatasetSplit,
    },
};

// imports
use burn::{
    data::dataset::Dataset,
    tensor::TensorData,
};
use image::{GrayImage, Luma};
use std::{
    path::{
        Path,
        PathBuf,
    },
    fs::read_dir,
};



pub struct GridDataset {
    root_path: PathBuf,
    entries: Vec<String>,
    token_map: TokenMap,
}



impl GridDataset {
    pub fn new<P: AsRef<Path>>(
        root_path: P,
        split: DatasetSplit,
        split_thresholds: (f32, f32),
        token_map: TokenMap
    ) -> Self {
        let root_path = root_path.as_ref();
        let grid_dir = root_path
            .join("data")
            .join("grid-lr-corpus");
        assert!(grid_dir.exists(), "GRID corpus directory does not exist at {:?}", grid_dir);

        // identify all speakers available on disk (s1, s2, ..., s34)
        let mut avail_speakers = Vec::new();
        if let Ok(speaker_dirs) = read_dir(&grid_dir) {
            for speaker_dir in speaker_dirs.flatten() {
                let speaker_str = speaker_dir.file_name().to_string_lossy().to_string();
                if speaker_str.starts_with('s') && speaker_dir.path().is_dir() {
                    avail_speakers.push(speaker_str);
                }
            }
        }
        assert!(!avail_speakers.is_empty(), "No speaker directories found in {:?}", grid_dir);

        // sort s1, s2, ..., s34
        avail_speakers.sort_by_key(|s| s[1..].parse::<i32>().unwrap_or(1));

        // calculate split points
        // dataset: [train|val|test]
        let total = avail_speakers.len() as f32;
        let (train_test_threshold, val_test_threshold) = split_thresholds;
        let (train_end, val_end) = (
            (total * train_test_threshold).round() as usize,
            (total * val_test_threshold).round() as usize,
        );

        assert!(total > 1.0, "Total number of speakers must be more than one");
        assert!(train_test_threshold < val_test_threshold || train_end < val_end, "Train threshold must be before Val threshold");
        assert!(train_test_threshold > 0.0 && val_test_threshold <= 1.0, "Split thresholds must be in (0, 1]");

        // partition speakers based on split points and select based on which split given
        let selected_speakers = match split {
            DatasetSplit::Train => &avail_speakers[0..train_end],
            DatasetSplit::Val => &avail_speakers[train_end..val_end],
            DatasetSplit::Test => &avail_speakers[val_end..],
        };

        // scan disk for only selected speakers and store
        let mut entries = Vec::new();
        for speaker in selected_speakers { // loop through dirs of selected speakers
            let video_dir = grid_dir.join(speaker);
            if let Ok(items) = read_dir(&video_dir) {
                for item in items.flatten() { // loop through data items of each speakers' dirs
                    if item.path().extension().is_some_and(|ext| ext == "mpg") {
                        if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {
                            entries.push(format!("{}/{}", speaker, stem)); // store speaker/data (e.g., "s1/bbaf2n")
                        }
                    }
                }
            }
        }
        entries.sort(); // sort for deterministic order
        assert!(!entries.is_empty(), "Dataset split {:?} resulted in 0 samples\nCheck if path {:?} contains .mpg files", split, grid_dir);
        println!("Initialized GridDataset ({:?}) with {} samples from {:?}", split, entries.len(), selected_speakers);

        Self {
            root_path: root_path.to_path_buf(),
            entries,
            token_map,
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
        let entry_name = self.entries.get(index)?;
        
        // frames is currently a flattened Vec<32>
        let (mut frames, transcript_ids) = load_grid_corpus(
            &self.root_path,
            entry_name,
            &self.token_map,
        ).ok()?;

        assert!(!frames.is_empty(), "No frames loaded");
        assert!(!transcript_ids.is_empty(), "No transcripts loaded");

        // isolate dims
        let (c, h, w) = (1, 50, 150);
        let t: usize = frames.len() / (c * h * w);
        assert!(frames.len().is_multiple_of(c * h * w), "Frame buffer size {} is not divisible by frame dimensions", frames.len());
        assert!(t > 0, "Computed zero frames for item {}", entry_name);
        if t == 0 { return None; }

        // // obtain min and max pixel values (fold approach)
        // let (norm_min, norm_max) = frames
        //     .iter()
        //     .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
        //         (min.min(x), max.max(x))
        //     });
        // assert!(norm_max.is_finite() && norm_min.is_finite(), "Non-finite pixel values detected");

        // obtain min and max pixel values (loop approach)
        let (mut norm_min, mut norm_max) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in frames.iter() {
            if x < norm_min { norm_min = x; }
            if x > norm_max { norm_max = x; }
        }
        assert!(norm_max.is_finite() && norm_min.is_finite(), "Non-finite pixel values detected");

        // normalize pixel values to within [0, 1] (adaptive approach)
        let range = norm_max - norm_min;
        let divisor = if range == 0.0 { 1.0 } else { range };
        for x in frames.iter_mut() { *x = ((*x - norm_min) / divisor).clamp(0.0, 1.0); }

        // // normalize pixel values to within [0, 1] (global approach)
        // for x in frames.iter_mut() {
        //     *x /= 255.0; 
        // }

        assert!(frames.len() == (c * t * h * w), "Tensor shape mismatch: len={} expected={}", frames.len(), (c * t * h * w));

        // convert frames into 4D tensor
        let frames = TensorData::new(
            frames,
            vec![c, t, h, w],
        );

        Some(VsrmItem {
            frames,
            transcript_ids,
            item_id: entry_name.clone(),
        })
    }

    /// get the total number of samples in the dataset split
    /// params: none
    /// returns: count of valid video entries
    fn len(&self) -> usize {
        self.entries.len()
    }
}