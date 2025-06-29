// create new Rust project with cargo (separate dir name from package name):              cargo new "[dir name]" --name [package_name]
// create new Rust project with cargo (but without auto creating new Git repo):           cargo new [dir name] --vcs none

// for big projects:

// compile project with cargo:                                                            cargo build
// compile project with cargo with optimizations:                                         cargo build --release
// compile and run project with cargo:                                                    cargo run

// for small experiments:

// compile single Rust file manually with rustc:                                          rustc [file name]
// run compiled binary (in same folder):                                                  .\[file name]



// imports
use std::vec;
use ndarray;                    // Rust's NumPy equivalent (for numerical operations)
use image;                      // for image processing
use opencv::{self, videoio::VideoCaptureTrait, core::{MatTrait, Size}, prelude::*};                     // for CV tasks
use clap;                       // for terminal arg parsing
use tract_onnx::prelude::*;     // for ONNX model inference
use reqwest;                    // for HTTP requests to download data
use zip;                        // for extracting zip files
// use onnxruntime;



fn main() {
    // debugging
    // let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // println!("Input: {:?}, Mean: {}, Std Dev: {}", vector, mean(&vector), std_dev(&vector));

    // load data
    load_data("../data");
}



// ------------------------------------------------------------ Utility Functions ------------------------------------------------------------



fn mean(data: &Vec<f32>) -> f32 {
    let count = data.len() as f32;
    let sum: f32 = data.iter().sum();
    sum / count
}

fn std_dev(data: &Vec<f32>) -> f32 {
    let count = data.len();
    let mean = mean(data);
    let variance: f32 = data
        .iter()
        .map(|x| x - mean)
        .map(|x| x * x)
        .sum::<f32>() / count as f32;
    variance.sqrt()
}

fn extract_zip(zip_path: &str, extract_to: &str) {
    let mut archive = zip::ZipArchive::new(
        std::fs::File::open(zip_path).expect("Failed to open zip file.")
    ).expect("Failed to read zip file.");

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



// ---------------------------------------------------- Data Loading/Processing Functions ----------------------------------------------------



// load data externally to a given path
fn load_data(path: &str) {
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
                    response.copy_to(&mut file).expect("Failed to write to file.");
                    println!("File downloaded successfully to {}", output);

                    // extract zip file
                    extract_zip(output, "../data");
                } else {
                    eprintln!("Failed to download file: {}", response.status());
                    return;
                }
            },
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
fn load_video(path: &str) -> Vec<f32> {
    // let mut cap = opencv::videoio::VideoCapture::from_file(path, videoio::CAP_ANY)
    //     .expect("Failed to open video file");

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
                ).expect("Failed to convert frame to grayscale");

                // crop frame to isolate region of interest (where the mouth is)
                let roi = opencv::core::Rect::new(80, 190, 150, 50);
                let mouth_frame = opencv::core::Mat::roi(&gray_frame, roi).expect("Failed to crop frame");

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
            frames = frames.iter().map(|&x| (x - mean) / std_dev).collect::<Vec<f32>>();

            return frames;
        },
        Err(e) => {
            eprintln!("Error opening video file: {}", e);
            return vec![];
        }
    }
}