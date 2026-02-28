// src/pipeline/mod.rs

//! Data ingestion and processing pipeline
//!
//! - `io`: raw filesystem, networking, and decompression utilities
//! - `tracker`: dynamic mouth tracking and fixed box cropping using Haar cascade logic
//! - `dataset`: high-level dataset source abstractions (GRID/LRW/others)
//! - `batcher`: tensor grouping and padding logic for CTC-style rectangular batches
//! - `adapters`: source-specific logic to map specific datasets to standardized 'VsrmItem' format for batch collation

pub mod io;
pub mod tracker;
pub mod dataset;
pub mod batcher;
pub mod adapters;

pub use dataset::{DatasetSource, DatasetSplit};
pub use batcher::{VsrmBatcher, Batch};



pub struct FramesBuffer {
    pub data: Vec<u8>, // frame represented as flattened vector of u8 pixel data
    pub height: usize,
    pub width: usize,
}