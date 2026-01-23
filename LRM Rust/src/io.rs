// I/O handler for high-level fs and networking tasks (loading, streaming, preprocessing)



use crate::prelude::*;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use opencv::{
    self,
    // core::{MatTrait, Size},
    prelude::*,
    videoio::VideoCaptureTrait, // for CV tasks
    photo::fast_nl_means_denoising_vec_def,
};
use rand::Rng;
use reqwest::blocking::{get, Client};
// use serde_json::Value;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader, Cursor},
    path::Path,
    sync::{atomic, Arc},
};
use tar::Archive;



/// stream plain .txt lines from disk
pub fn stream_txt_lines<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut out = Vec::new();
    for line in reader.lines().flatten() {
        let text = line.trim().to_string();
        if !text.is_empty() {
            out.push(text);
        }
    }

    Ok(out)
}



/// stream "text" field from remote JSONL.gz shard (e.g., C4)
pub fn stream_jsonl_gz(url: String) -> Result<Vec<String>, Box<dyn Error>> {
    let client = Client::new();
    let resp = client.get(&url).send()?.error_for_status()?;
    let decoder = GzDecoder::new(resp);
    let reader = BufReader::new(decoder);

    let mut out = Vec::new();
    for line in reader.lines().flatten() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(text) = val.get("text").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    out.push(text.to_string());
                }
            }
        }
    }

    Ok(out)
}



/// stream all text lines under a corpus line by line while applying sampling and basic preprocessing (takes in a file path as String for ownership)
pub fn stream_corpus_lines(file_path: String, sample_rate: f64) -> impl Iterator<Item = String> {
    println!("Streaming corpus lines from: {}", file_path);
    let file = File::open(&file_path).expect("Failed to open corpus file");
    let metadata = file.metadata().expect("Failed to get file metadata");
    let file_size = metadata.len();

    let prog_bar = ProgressBar::new(file_size);
    prog_bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({msg}) (ETA: {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    let kept_count = Arc::new(atomic::AtomicU64::new(0));

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
            let count = count_filter.fetch_add(1, atomic::Ordering::SeqCst);
            if count % 1000 == 0 {
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



/// extract zip file to a given path
pub fn extract_zip(zip_path: &str, extract_to: &str) {
    let input_file = File::open(zip_path).expect("Failed to open zip file.");
    let mut archive = zip::ZipArchive::new(input_file).expect("Failed to read zip file.");

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Failed to read file from zip.");
        let out_path = match file.enclosed_name() {
            Some(path) => Path::new(extract_to).join(path),
            None => continue, // skip files with invalid names if need be
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&out_path).expect("Failed to create directory.");
        } else {
            if let Some(p) = out_path.parent() {
                fs::create_dir_all(p).expect("Failed to create parent directory.");
            }
            let mut outfile = File::create(&out_path).expect("Failed to create file.");
            io::copy(&mut file, &mut outfile).expect("Failed to write file.");
        }
    }
    println!("Extracted zip file to {}", extract_to);
}



/// extract gzip file to a given path
pub fn extract_gzip(gzip_path: &str, extract_to: &str) {
    let input_file = File::open(gzip_path).expect("Failed to open gzip file.");
    let mut decoder = GzDecoder::new(input_file);

    // get folder path from the 'extract_to' string and create it
    let path = Path::new(extract_to);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent directory for extraction.");
    }

    let mut out_file = File::create(path).expect("Failed to create output file.");

    io::copy(&mut decoder, &mut out_file).expect("Failed to decompress gzip content.");
    println!("Extracted gzip file to {}", extract_to);
}



/// extract GRID corpus externally to a given path
pub fn extract_grid_corpus(root_path: &str) {
    let root_path = Path::new(root_path);
    let data_dir = root_path.join("data");
    let grid_dir = data_dir.join("grid-lr-corpus");

    // check if the GRID corpus exists at the given path
    if !grid_dir.exists() {
        println!("Grid corpus not found, downloading...");

        // use client with NO timeout for large files
        let client = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .expect("Failed to create HTTP client");

        let url = "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL&confirm=t&export=download";
        let output = root_path.join("data.zip");

        // download file from URL
        match client.get(url).send() {
            Ok(mut response) => {
                if response.status().is_success() {
                    let mut file = File::create(&output).expect("Failed to create file.");
                    response
                        .copy_to(&mut file)
                        .expect("Failed to write to file.");
                    println!(
                        "File downloaded successfully to {}",
                        grid_dir.to_string_lossy()
                    );

                    // extract zip file
                    extract_zip(&output.to_string_lossy(), &root_path.to_string_lossy());

                    // rename extracted subdir to grid-lr-corpus
                    let nested_dir = data_dir.join("data");
                    if nested_dir.exists() {
                        fs::rename(&nested_dir, &grid_dir)
                            .expect("Failed to rename extracted directory.");
                    }

                    // clean up zip file
                    fs::remove_file(&output).expect("Failed to delete zip file.");
                } else {
                    eprintln!("Failed to download GRID corpus: {}", response.status());
                    return;
                }
            }
            Err(e) => {
                eprintln!("Error parsing URL: {}", e);
                return;
            }
        }
    } else {
        println!("GRID corpus already exists, downloading skipped");
    }
}



/// extract SLR corpus externally to a given path
pub fn extract_slr_corpus(root_path: &str) {
    let root_path = Path::new(root_path);
    let data_dir = root_path.join("data");
    let slr_dir = data_dir.join("librispeech-lm-norm");
    let final_path = slr_dir.join("librispeech-lm-norm.txt");

    // check if the SLR corpus exists at the given path
    if !final_path.exists() {
        println!("SLR corpus not found, downloading...");

        fs::create_dir_all(&slr_dir).expect("Failed to create SLR directory");

        // use client with NO timeout for large files
        let client = reqwest::blocking::Client::builder()
            .timeout(None)
            .build()
            .expect("Failed to create HTTP client");

        let url = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt";
        let output = data_dir.join("librispeech-lm-norm.gz");

        // download and extract corpus
        match client.get(url).send() {
            Ok(mut response) => {
                if response.status().is_success() {
                    let mut file = File::create(&output).expect("Failed to create file.");
                    response
                        .copy_to(&mut file)
                        .expect("Failed to write to file.");
                    println!(
                        "SLR corpus downloaded successfully to {}",
                        slr_dir.to_string_lossy()
                    );

                    // extract gzip file
                    extract_gzip(&output.to_string_lossy(), &final_path.to_string_lossy());

                    // clean up gzip file
                    fs::remove_file(&output).expect("Failed to delete gzip file.");
                } else {
                    eprintln!("Failed to download SLR corpus: {}", response.status());
                    return;
                }
            }
            Err(e) => {
                eprintln!("Error parsing URL: {}", e);
                return;
            }
        }
    } else {
        println!("SLR corpus already exists, downloading skipped");
    }
}



/// takes in a video path and outputs a list of floats
pub fn load_grid_video(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match opencv::videoio::VideoCapture::from_file(path, opencv::videoio::CAP_ANY) {
        Ok(mut cap) => {
            let mut frames: Vec<f32> = vec![];

            let mut frame = opencv::core::Mat::default();
            while cap.read(&mut frame).expect("Error reading frame") {
                let size = frame.size().expect("Failed to get frame size");
                if size.width == 0 || size.height == 0 {
                    break; // End of video
                }

                // convert frame to grayscale
                let mut gray_frame = opencv::core::Mat::default();
                opencv::imgproc::cvt_color(
                    &frame,
                    &mut gray_frame,
                    opencv::imgproc::COLOR_BGR2GRAY,
                    0,
                    opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
                )
                .expect("Failed to convert frame to grayscale");

                // crop frame to isolate region of interest (where the mouth is)
                let roi = opencv::core::Rect::new(80, 190, 150, 50);
                let temp = opencv::core::Mat::roi(&gray_frame, roi).expect("Failed to crop frame");
                let mut mouth_frame = opencv::core::Mat::default();
                temp.copy_to(&mut mouth_frame)
                    .expect("Failed to copy ROI to a continuous Mat");

                // flatten and store
                let flattened_frame: Vec<f32> = mouth_frame
                    .data_bytes()
                    .expect("Failed to get frame data")
                    .iter()
                    .map(|&pixel| pixel as f32)
                    .collect();
                frames.extend(flattened_frame);
            }
            // standardize frames (by centering to zero mean and scaling to unit variance)
            let mean = mean(&frames);
            let std_dev = std_dev(&frames);

            frames = frames
                .iter()
                .map(|&x| (x - mean) / std_dev)
                .collect::<Vec<f32>>(); // frames as a vector of pixels as floats

            Ok(frames)
        }
        Err(e) => {
            eprintln!("Error opening video file: {}", e);
            Err(Box::new(e))
        }
    }
}



/// takes in an alignments path (as well as TokenMap struct) and outputs a list of char indices
pub fn load_grid_alignments(
    path: &str,
    token_map: &TokenMap,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    match std::fs::File::open(&path) {
        Ok(file) => {
            let mut tokens: Vec<String> = vec![];
            let lines = io::BufReader::new(file).lines();

            for line in lines.flatten() {
                let line_group = line.split_whitespace().collect::<Vec<_>>();
                if line_group[2] != "sil" {
                    tokens.push(line_group[2].to_string());
                }
            }

            Ok(tokens
                .iter()
                .flat_map(|token| token.chars())
                .filter_map(|ch| token_map.id_of(ch))
                .collect())
        }
        Err(e) => {
            eprintln!("Error opening alignments file: {}", e);
            Err(Box::new(e))
        }
    }
}



/// function to load GRID data (takes in a data path and outputs frames and alignments)
pub fn load_grid_corpus(
    root_path: &str,
    entry_name: &str,
    token_map: &TokenMap,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
     // entry_name in the form of "s1/bbaf2n"
    if entry_name.trim().is_empty() {
        eprintln!("Error: Provided entry_name is empty.");
        return Err("Failed to extract filename: entry_name was empty.".into());
    }

    // join project root with LR data path for an absolute path
    let grid_dataset_path = Path::new(root_path).join("data/grid-lr-corpus");

    let video_path = grid_dataset_path
        .join(entry_name)
        .with_extension("mpg");

    let alignments_path = grid_dataset_path
        .join("alignments")
        .join(entry_name)
        .with_extension("align");

    let frames = load_grid_video(&video_path.to_string_lossy())?;
    let alignments = load_grid_alignments(&alignments_path.to_string_lossy(), token_map)?;

    Ok((frames, alignments))
}
