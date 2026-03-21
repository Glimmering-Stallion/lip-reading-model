//! GRID adapter + preprocessing utilities.
//!
//! - `grid_dataset`: dataset adapter (loading, alignment parsing, video decode)
//! - `grid_adapter`: corpus layout normalization (mapping, bundling, `.mpg`/`.align` → `.mp4`/`.txt`, `clean_corpus`)

pub mod grid_dataset;
pub mod grid_adapter;

pub use grid_dataset::GridDataset;
pub use grid_adapter::{
    align_grid_directories,
    bundle_grid_utterances,
    convert_to_standard_mp4,
    convert_to_standard_txt,
    normalize_to_standard_formats,
    clean_corpus,
};
