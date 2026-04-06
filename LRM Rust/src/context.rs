//! Global project configuration and filesystem context.
//!
//! This module provides a centralized `Context` struct that acts as the source of truth for
//! relevant directory paths (data, models, outputs, exports) and handles the initialization
//! of the project workspace.

use std::{env, fs, path::PathBuf};

pub struct Context {
    pub rust_root: PathBuf,
    pub models_path: PathBuf,
    pub data_path: PathBuf,
    pub outputs_path: PathBuf,
    pub exports_path: PathBuf,
}

impl Context {
    /// Creates a new `Context` with paths derived from `CARGO_MANIFEST_DIR`.
    ///
    /// Makes sure `models/`, `data/`, `outputs/`, and `exports/` exist under the crate root.
    pub fn new() -> Self {
        // dynamically get Rust project root and relevant dir paths
        let rust_root =
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));

        let models_path = rust_root.join("models");
        let data_path = rust_root.join("data");
        let outputs_path = rust_root.join("outputs");
        let exports_path = rust_root.join("exports");

        if !models_path.exists() { fs::create_dir_all(&models_path).expect("failed to create output directory for models"); }
        if !data_path.exists() { fs::create_dir_all(&data_path).expect("failed to create output directory for data"); }
        if !outputs_path.exists() { fs::create_dir_all(&outputs_path).expect("failed to create outputs directory"); }
        if !exports_path.exists() { fs::create_dir_all(&exports_path).expect("failed to create exports directory"); }

        Context {
            rust_root,
            models_path,
            data_path,
            outputs_path,
            exports_path,
        }
    }
}
