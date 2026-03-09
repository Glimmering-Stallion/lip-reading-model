//! Global project configuration and filesystem context.
//! 
//! This module provides a centralized `Context` struct that acts as the source of truth for
//! relevant directory paths (data, models, tests) and handles the initialization
//! of the project workspace.



use std::{
    env,
    fs,
    path::PathBuf,
};



pub struct Context {
    pub rust_root: PathBuf,
    pub models_path: PathBuf,
    pub data_path: PathBuf,
    pub tests_path: PathBuf,
}



impl Context {
    /// Creates a new `Context` with paths derived from `CARGO_MANIFEST_DIR`.
    ///
    /// Makes sure `models/`, `data/`, and `tests/` directories exist under project root.
    pub fn new() -> Self {
        // dynamically get Rust project root and relevant dir paths
        let rust_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));

        let models_path = rust_root.join("models");
        let data_path = rust_root.join("data");
        let tests_path = rust_root.join("tests");

        if !models_path.exists() { fs::create_dir_all(&models_path).expect("Failed to create output directory for models"); }
        if !data_path.exists() { fs::create_dir_all(&data_path).expect("Failed to create output directory for data"); }
        if !tests_path.exists() { fs::create_dir_all(&tests_path).expect("Failed to create tests directory"); }

        Context {
            rust_root,
            models_path,
            data_path,
            tests_path,
        }
    }
}