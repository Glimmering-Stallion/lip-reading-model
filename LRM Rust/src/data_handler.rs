// Data handler for loading, streaming, preprocessing



use crate::prelude::*;
use flate2::read::GzDecoder;
use reqwest::blocking::{get, Client};
// use serde_json::Value;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader, Cursor},
    path::Path,
};
use opencv::{
    self,
    // core::{MatTrait, Size},
    prelude::*,
    videoio::VideoCaptureTrait, // for CV tasks
};
use tar::Archive;



/// stream plain .txt lines from disk
pub fn stream_txt_lines<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut out = Vec::new();
    for line in reader.lines().flatten() {
        let text = line.trim().to_string();
        if !text.is_empty() { out.push(text); }
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
                if !text.is_empty() { out.push(text.to_string()); }
            }
        }
    }

    Ok(out)
}



// download a corpus .tgz and extract to out_dir
pub fn import_corpus(url: &str, out_dir: &str) -> Result<(), Box<dyn Error>> {
    // http get
    let resp = get(url)?.error_for_status()?;
    let bytes = resp.bytes()?;
    let cursor = Cursor::new(bytes);

    // gzip decode and tar extract
    let gz = GzDecoder::new(cursor);
    let mut archive = Archive::new(gz);
    archive.unpack(out_dir)?;

    Ok(())
}



// stream all .txt files under a corpus dir line by line
pub fn stream_corpus_lines(corpus_dir: &str) -> impl Iterator<Item = String> {
    fs::read_dir(corpus_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt"))
        .flat_map(|path| {
            let content = fs::read_to_string(path).unwrap_or_default();
            content.lines()
                .map(|line| {
                    line.to_lowercase()
                        .replace(|c: char| !c.is_alphanumeric() && !c.is_whitespace(), "") // strip non-vocab chars
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
        })
}



/* -------------------------------------------------- Old Data Loading/Processing Functions -------------------------------------------------- */



pub fn extract_zip(zip_path: &str, extract_to: &str) {
    let mut archive =
        zip::ZipArchive::new(std::fs::File::open(zip_path).expect("Failed to open zip file."))
            .expect("Failed to read zip file.");

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Failed to read file from zip.");
        let out_path = std::path::Path::new(extract_to).join(file.sanitized_name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&out_path).expect("Failed to create directory.");
        } else {
            if let Some(p) = out_path.parent() {
                std::fs::create_dir_all(p).expect("Failed to create parent directory.");
            }
            let mut outfile = std::fs::File::create(&out_path).expect("Failed to create file.");
            std::io::copy(&mut file, &mut outfile).expect("Failed to write file.");
        }
    }
    println!("Extracted zip file to {}", extract_to);
}



// extract data externally to a given path
pub fn extract_data(path: &str) {
    // check if the video file exists at the given path
    if !std::path::Path::new(path).exists() {
        println!("Data directory not found, downloading...");

        let url = "https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL";
        let output = "../data.zip";

        // download file from URL
        match reqwest::blocking::get(url) {
            Ok(mut response) => {
                if response.status().is_success() {
                    let mut file = std::fs::File::create(output).expect("Failed to create file.");
                    response
                        .copy_to(&mut file)
                        .expect("Failed to write to file.");
                    println!("File downloaded successfully to {}", output);

                    // extract zip file
                    extract_zip(output, "../data");
                } else {
                    eprintln!("Failed to download file: {}", response.status());
                    return;
                }
            }
            Err(e) => {
                eprintln!("Error parsing URL: {}", e);
                return;
            }
        }
    } else {
        println!("Data directory already exists, downloading skipped.");
    }
}



// takes in a video path and outputs a list of floats
pub fn load_video(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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



// takes in an alignments path (as well as TokenMap struct) and outputs a list of char indices
pub fn load_alignments(
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



// function to load data (takes in a data path and outputs frames and alignments)
pub fn load_data(
    path: &str,
    token_map: &TokenMap,
) -> Result<(Vec<f32>, Vec<usize>), Box<dyn std::error::Error>> {
    let filename = std::path::Path::new(&path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|s| s.to_string());

    match filename {
        Some(name) => {
            let video_path = format!("../data/s1/{}.mpg", name);
            let alignments_path = format!("../data/alignments/s1/{}.align", name);

            let frames = load_video(&video_path)?;
            let alignments = load_alignments(&alignments_path, token_map)?;

            Ok((frames, alignments))
        }
        None => {
            eprintln!("Failed to extract filename from path: {}", path);
            Err("Failed to extract filename.".into())
        }
    }
}
