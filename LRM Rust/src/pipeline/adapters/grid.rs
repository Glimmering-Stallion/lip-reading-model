// Data adaptation/standardization for GRID audio-visual sentence corpus



// custom imports
use crate::{
    vocab::TokenMap,
    pipeline::{
        batcher::VsrmItem,
        io::load_grid_corpus,
        DatasetSplit,
    },
};

// imports
use burn::{
    data::dataset::Dataset,
    tensor::TensorData,
};
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
        let frames_dir = grid_dir.join("frames");
        let alignments_dir = grid_dir.join("alignments");

        assert!(grid_dir.exists(), "GRID corpus directory does not exist at {:?}", grid_dir);
        assert!(frames_dir.exists(), "GRID frames directory does not exist at {:?}", frames_dir);
        assert!(alignments_dir.exists(), "GRID alignments directory does not exist at {:?}", alignments_dir);

        // identify all speakers available on disk (s1, s2, ..., s34)
        let mut avail_speakers = Vec::new();
        if let Ok(speaker_dirs) = read_dir(&frames_dir) {
            for speaker_dir in speaker_dirs.flatten() {
                let speaker_str = speaker_dir.file_name().to_string_lossy().to_string();
                if speaker_str.starts_with('s') && speaker_dir.path().is_dir() {
                    avail_speakers.push(speaker_str);
                }
            }
        }
        assert!(!avail_speakers.is_empty(), "No speaker directories found in {:?}", frames_dir);

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
            let video_dir = frames_dir.join(speaker);
            let alignment_dir = alignments_dir.join(speaker);

            if let Ok(items) = read_dir(&video_dir) {
                for item in items.flatten() { // loop through data items of each speakers' dirs
                    if let Some(stem) = item.path().file_stem().and_then(|s| s.to_str()) {

                        let is_video = item.path().extension().is_some_and(|ext| ext == "mpg");
                        let has_alignment = alignment_dir.join(stem).with_extension("align").exists();

                        // store speaker/data (e.g., "s1/bbaf2n")
                        if is_video && has_alignment { entries.push(format!("{}/{}", speaker, stem)); }
                        else { /* println!("Skipping {}/{}: missing alignment file.", speaker, stem); */ }
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
        
        // frames is currently a flattened Vec<u8>
        let (frames, transcript_ids) = load_grid_corpus(
            &self.root_path,
            entry_name,
            &self.token_map,
        ).ok()?;

        assert!(!frames.is_empty(), "No frames loaded");
        assert!(!transcript_ids.is_empty(), "No transcripts loaded");

        // isolate dims
        let (c, h, w) = (1, 50, 150);
        let t = frames.len() / (c * h * w);
        let l = transcript_ids.len();
        assert!(frames.len().is_multiple_of(c * h * w), "Frame buffer size {} is not divisible by frame dimensions", frames.len());
        assert!(t > 0, "Computed zero frames for item {}", entry_name);

        // enforce CTC constraint: T must be greater than L
        // (ideally 2x greater, for chars plus possible blanks)
        if t < (2 * l) {
            // eprintln!("\nSkipping sample {}: T = {} is too short for L = {}", entry_name, t, l);
            return None;
        }

        assert!(frames.len() == (c * t * h * w), "Tensor shape mismatch: len={} expected={}", frames.len(), (c * t * h * w));

        // convert frames into 4D tensor
        let frames = TensorData::new(
            frames,
            // frames.into_iter().map(|b| b as f32).collect(), // temp f32 conversion if Burn doesn't support u8 tensors yet
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



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        pipeline::DatasetSplit,
        vocab::VOCAB,
    };
    use image::{GrayImage, Luma};
    use std::{
        path::PathBuf,
        fs,
    };

    #[test]
    fn test_extract_frames_from_grid_dataset_item() {
        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tests_path = Path::new(&root_path).join("tests");
        if !tests_path.exists() { fs::create_dir(&tests_path).expect("Failed to create tests directory") }

        let token_map = TokenMap::new(VOCAB);
        let split_thresholds = (0.1, 1.0);

        // train/validation dataset instances
        let dataset = GridDataset::new(
            root_path, 
            DatasetSplit::Train,
            split_thresholds,
            token_map.clone(),
        );

        // obtain first valid GRID dataset item
        let mut item = None;
        for i in 0..10 {
            if let Some(entry) = dataset.get(i) {
                item = Some(entry);
                break;
            }
        }
        let item = item.expect("Failed to extract any valid dataset item");
        println!("Obtained item ID: {}", item.item_id);

        // create output dir to hold collection of extracted frames as pngs
        let item_id = item.item_id.replace("/", "_");
        let output_dir = tests_path.join(&item_id);
        if !output_dir.exists() { fs::create_dir(&output_dir).expect(&format!("Failed to create output directory for frames of item {}", item_id)); }

        // extract frames
        let frames = item.frames.as_slice::<u8>().expect("Failed to convert frames to slice");
        let (c, h, w) = (1, 50, 150);
        let t = frames.len() / (c * h * w);

        println!("Exporting {} frames for item: {}", t, item.item_id);

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
}