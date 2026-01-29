// src/pipeline/mod.rs

//! Data ingestion and processing pipeline
//!
//! - `io`: raw filesystem, networking, and decompression utilities
//! - `dataset`: high-level dataset source abstractions (GRID/LRW/others)
//! - `batcher`: tensor grouping and padding logic for CTC-style rectangular batches
//! - `preprocessors`: source-specific dataset transformations (video cropping, normalization)

pub mod io;
pub mod dataset;
pub mod batcher;
pub mod preprocessors;

pub use dataset::{DatasetSource, DatasetSplit};
pub use batcher::{VsrmBatcher, Batch};