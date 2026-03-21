//! I/O utilities for high-level filesystem operations and extraction tasks.
//! 
//! This module provides high-level handlers for downloading external corpora (SLR),
//! decompressing common archive formats (Zip/Gzip), and streaming processed text 
//! data with integrated progress tracking and sampling.



use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use reqwest::blocking::Client;
use serde::{
    Serialize,
    de::DeserializeOwned,
};
use crate::prelude::ESS;
use std::{
    fs::{
        self,
        File,
        metadata,
    },
    io::{
        self,
        BufRead,
        BufReader,
        Read,
        Write,
    },
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use zip::ZipArchive;



/// Helper to check if a given path to a file is non-empty.
pub fn file_nonempty(path: &Path) -> bool {
    path.is_file() && metadata(path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}



/// Generic 3D tensor serializer.
///
/// Saves a 3D sequence (e.g., video, mouth crops) to a structured binary file.
/// Format: [u32: H] [u32: W] [u32: T] [raw bytes...]
///
/// T can be u8, f32, i32, etc. (must implement `bytemuck::Pod`).
pub fn write_tensor_3d<T, P>(
    path: P,
    data: &[T],
    shape: (usize, usize, usize),
) -> Result<(), ESS>
where
    T: bytemuck::Pod,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    if let Some(parent) = path.parent()
    { fs::create_dir_all(parent)?; }

    let mut file = File::create(path)?;
    let (h, w, t) = shape;

    // write header metadata ([H, W, T] dims)
    file.write_all(&(h as u32).to_le_bytes())?;
    file.write_all(&(w as u32).to_le_bytes())?;
    file.write_all(&(t as u32).to_le_bytes())?;

    // write raw bytes of generic tensor data slice
    let bytes: &[u8] = bytemuck::cast_slice(data);
    file.write_all(bytes)?;

    Ok(())
}



/// Generic 3D tensor deserializer.
///
/// Loads a 3D sequence from a structured binary file.
/// Returns (data, (H, W, T)).
pub fn read_tensor_3d<T, P>(path: P) -> Result<(Vec<T>, (usize, usize, usize)), ESS>
where
    T: bytemuck::Pod,
    P: AsRef<Path>,
{
    let mut file = File::open(path)?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;

    let h = u32::from_le_bytes(header[0..4].try_into()?) as usize;
    let w = u32::from_le_bytes(header[4..8].try_into()?) as usize;
    let t = u32::from_le_bytes(header[8..12].try_into()?) as usize;

    let expected_bytes = h * w * t * std::mem::size_of::<T>();
    let mut buf = vec![0u8; expected_bytes];
    file.read_exact(&mut buf)?;

    let data: Vec<T> = bytemuck::cast_slice(buf.as_slice()).to_vec();

    Ok((data, (h, w, t)))
}



/// General-purpose JSON saver.
///
/// ### Params:
/// - `path`: Path to write the JSON file.
/// - `data`: Serializable data to save.
///
/// ### Returns:
/// `Ok(())` on success, or an error on I/O or serialization failure.
pub fn save_json<P: AsRef<Path>, T: Serialize>(path: P, data: &T) -> Result<(), ESS> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, data)?;
    Ok(())
}



/// General-purpose JSON loader.
///
/// ### Params:
/// - `path`: Path to the JSON file to read.
///
/// ### Returns:
/// Deserialized value of type `T`, or an error on I/O or deserialization failure.
pub fn load_json<P: AsRef<Path>, T: DeserializeOwned>(path: P) -> Result<T, ESS> {
    let file = File::open(path)?;
    let data = serde_json::from_reader(file)?;
    Ok(data)
}



/// Streams all text lines under a corpus line by line while applying sampling and basic preprocessing.
///
/// ### Params:
/// - `file_path`: Path to the corpus file.
/// - `sample_rate`: Fraction of lines to keep (0.0, 1.0]; each line is kept with this probability.
///
/// ### Returns:
/// An iterator yielding preprocessed `String` lines.
pub fn stream_corpus_lines<P: AsRef<Path>>(
    file_path: P,
    sample_rate: f64
) -> impl Iterator<Item = String> {
    let file_path = file_path.as_ref();
    assert!(file_path.exists(), "corpus file {:?} does not exist", file_path);
    assert!(sample_rate > 0.0 && sample_rate <= 1.0, "sample_rate must be in (0, 1]");
    println!("Streaming corpus lines from: {}", file_path.to_string_lossy());

    let file = File::open(file_path).expect("failed to open corpus file");
    let metadata = file.metadata().expect("failed to get file metadata");
    let file_size = metadata.len();

    let prog_bar = ProgressBar::new(file_size);
    prog_bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({msg}) (ETA: {eta})")
            .unwrap()
            .progress_chars("#>-")
    );
    let kept_count = Arc::new(AtomicU64::new(0));

    let pb_inspect = prog_bar.clone();
    let pb_filter = prog_bar.clone();
    let pb_final = prog_bar.clone();
    let count_filter = kept_count.clone();

    let reader = BufReader::new(file);
    let mut rng = rand::rng();

    reader
        .lines()
        .filter_map(|line| line.ok())
        .inspect(move |line| pb_inspect.inc(line.len() as u64 + 1)) // update bar per line read
        .filter(move |_| rng.random::<f64>() < sample_rate) // keep line with certain prob
        .inspect(move |_| {
            let count = count_filter.fetch_add(1, Ordering::SeqCst);
            if count.is_multiple_of(1000) {
                // update message every 1000 lines to save CPU
                pb_filter.set_message(format!("{} lines kept", count));
            }
        })
        .map(|line| {
            line.to_lowercase()
                .replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), "") // strip non-vocab chars
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|line| !line.is_empty())
        .inspect(move |_| {
            if pb_final.position() >= pb_final.length().unwrap_or(u64::MAX) {
                pb_final.finish_with_message("Done");
            }
        })
}



/// Extracts a zip file to a given path.
///
/// ### Params:
/// - `zip_path`: Path to the zip file.
/// - `extract_to`: Destination directory for extracted contents.
pub fn extract_zip<P: AsRef<Path>, Q: AsRef<Path>>(zip_path: P, extract_to: Q) {
    let zip_path = zip_path.as_ref();
    let extract_to = extract_to.as_ref();
    assert!(zip_path.exists(), "zip file {:?} does not exist", zip_path);
    
    let input_file = File::open(zip_path).expect("failed to open zip file");
    let mut archive = ZipArchive::new(input_file).expect("failed to read zip file");
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("failed to read file from zip");
        let out_path = match file.enclosed_name() {
            Some(path) => extract_to.join(path),
            None => continue, // skip files with invalid names if need be
        };
        
        if file.name().ends_with('/') {
            fs::create_dir_all(&out_path).expect("failed to create directory");
        } else {
            if let Some(p) = out_path.parent() {
                fs::create_dir_all(p).expect("failed to create parent directory");
            }
            let mut outfile = File::create(&out_path).expect("failed to create file");
            io::copy(&mut file, &mut outfile).expect("failed to write file");
        }
    }
    assert!(extract_to.exists(), "zip destination does not exist");
    println!("Extracted zip file to {}", extract_to.to_string_lossy());
}



/// Extracts a gzip file to a given path.
///
/// ### Params:
/// - `gzip_path`: Path to the gzip file.
/// - `extract_to`: Destination path for the decompressed file.
pub fn extract_gzip<P: AsRef<Path>, Q: AsRef<Path>>(gzip_path: P, extract_to: Q) {
    let gzip_path = gzip_path.as_ref();
    let extract_to = extract_to.as_ref();
    assert!(gzip_path.exists(), "gzip file {:?} does not exist", gzip_path);
    
    let input_file = File::open(gzip_path).expect("failed to open gzip file");
    let mut decoder = GzDecoder::new(input_file);
    
    // get folder path from the 'extract_to' string and create it
    if let Some(parent) = extract_to.parent() {
        fs::create_dir_all(parent).expect("failed to create parent directory for extraction");
    }

    let mut out_file = File::create(extract_to).expect("failed to create output file");
    io::copy(&mut decoder, &mut out_file).expect("failed to decompress gzip content");

    assert!(extract_to.exists(), "gzip destination does not exist");
    println!("Extracted gzip file to {}", extract_to.to_string_lossy());
}



/// Extracts SLR corpus externally to a given path.
/// Downloads from OpenSLR if not already present, then decompresses to `data/librispeech-lm-norm/librispeech-lm-norm.txt`.
///
/// ### Params:
/// - `root_path`: Project root path (must contain or will create `data/librispeech-lm-norm`).
pub fn extract_slr_corpus<P: AsRef<Path>>(root_path: P) {
    let root_path = root_path.as_ref();
    let data_dir = root_path.join("data");
    let slr_dir = data_dir.join("librispeech-lm-norm");
    let final_path = slr_dir.join("librispeech-lm-norm.txt");
    assert!(root_path.exists(), "root path {:?} does not exist", root_path);

    // check if the SLR corpus exists at the given path
    if !final_path.exists() {
        println!("\nSLR corpus not found, downloading...");

        fs::create_dir_all(&slr_dir).expect("failed to create SLR directory");

        // use client with NO timeout for large files
        let client = Client::builder()
            .timeout(None)
            .build()
            .expect("failed to create HTTP client");

        let url = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt";
        let output = data_dir.join("librispeech-lm-norm.gz");

        // download and extract corpus
        match client.get(url).send() {
            Ok(mut response) => {
                if response.status().is_success() {
                    let mut file = File::create(&output).expect("failed to create file");
                    response
                        .copy_to(&mut file)
                        .expect("failed to write to file");
                    println!(
                        "SLR corpus downloaded successfully to {}\n",
                        slr_dir.to_string_lossy()
                    );

                    // extract/remove gzip file
                    extract_gzip(&output, &final_path);
                    fs::remove_file(&output).expect("failed to delete gzip file");

                    assert!(final_path.exists(), "SLR corpus file missing after extraction");
                    assert!(final_path.metadata().unwrap().len() > 0, "SLR corpus file is empty");
                } else { eprintln!("Failed to download SLR corpus: {}", response.status()); }
            }
            Err(e) => { eprintln!("Error parsing URL: {}", e); }
        }
    } else { println!("SLR corpus already exists, downloading skipped\n"); }
}
