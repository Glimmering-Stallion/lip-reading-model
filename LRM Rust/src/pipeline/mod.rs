// src/pipeline/mod.rs

//! Data ingestion and processing pipeline
//!
//! - `io`: raw filesystem, networking, and decompression utilities
//! - `tracker`: dynamic mouth tracking and fixed box cropping using Haar cascade logic
//! - `dataset`: high-level dataset source abstractions (GRID/LRW/others)
//! - `batcher`: tensor grouping and padding logic for CTC-style rectangular batches
//! - `adapters`: source-specific logic to map specific datasets to standardized 'VsrmItem' format for batch collation
//! - `video`: (stub) video loading with LipTracker for inference; code commented out pending integration

pub mod io;
pub mod tracker;
pub mod dataset;
pub mod batcher;
pub mod adapters;
pub mod video;

pub use dataset::{DatasetSource, DatasetSplit};
pub use batcher::{VsrmBatcher, Batch};



/// Flattened grayscale frame buffer for a video clip.
///
/// `data` is row-major, contiguous: `[frame0_pixels | frame1_pixels | ...]`, where each frame has `height * width` bytes.
pub struct FramesBuffer {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
}