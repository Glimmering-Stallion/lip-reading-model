// create new Rust project with cargo (separate dir name from package name):              cargo new "[dir name]" --name [package_name]
// create new Rust project with cargo (but without auto creating new Git repo):           cargo new [dir name] --vcs none

// for big projects:

// compile project with cargo:                                                            cargo build
// compile project with cargo with optimizations:                                         cargo build --release
// compile and run project with cargo:                                                    cargo run

// for small experiments:

// compile single Rust file manually with rustc:                                          rustc [file name]
// run compiled binary (in same folder):                                                  .\[file name]

// for crate imports:

// import crate with cargo:                                                              cargo add [crate name]



// modules
mod ctc;
mod model;
mod train;
mod utils;



// custom imports
use model::LRModel;
use crate::utils::{mean, std_dev, extract_zip};



// imports
use clap; // for terminal arg parsing
use image::{GrayImage, Luma}; // for image processing
use ndarray; // Rust's NumPy equivalent (for numerical operations)
use opencv::{
    self,
    core::{MatTrait, Size},
    prelude::*,
    videoio::VideoCaptureTrait,
}; // for CV tasks
// use tract_onnx::prelude::*;     // for ONNX model inference
use reqwest; // for HTTP requests to download data
use std::{
    collections::HashMap, // for bidirectional token-id mapping
    fs::File,
    io::{self, BufRead},
    vec,
};
use burn::{
    nn::loss::Reduction,
};



struct TokenMap {
    char_to_num: HashMap<char, usize>,
    num_to_char: Vec<char>,
}

impl TokenMap {
    fn new(vocab: &str) -> Self {
        let vocab: Vec<char> = vocab.chars().collect();

        // character to numerical index map and vice versa
        let mut char_to_num = HashMap::new();
        for (idx, ch) in vocab.iter().enumerate() {
            char_to_num.insert(*ch, idx);
        }

        Self {
            char_to_num,
            num_to_char: vocab,
        }
    }

    fn char_to_num(&self, ch: char) -> Option<usize> {
        self.char_to_num.get(&ch).copied()
    }
    fn num_to_char(&self, num: usize) -> Option<char> {
        self.num_to_char.get(num).copied()
    }
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    // debugging
    // let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // println!("Input: {:?}, Mean: {}, Std Dev: {}", vector, mean(&vector), std_dev(&vector));

    // obtain data (if data isn't already loaded)
    // extract_data("../data");

    // ------ Define vocabulary and token map ------

    let vocab = "abcdefghijklmnopqrstuvwxyz'?!0123456789 _";
    let vocab_size = vocab.chars().count();
    let token_map = TokenMap::new(vocab);

    println!("Vocabulary: {:?}", token_map.num_to_char);

    // ----------------- Load data -----------------

    // height and width of cropped ROI (mouth region)
    let width: u32 = 150;
    let height: u32 = 50;
    let dim = (width * height) as usize;

    let test_path = "../data/s1/bbal6n.mpg";
    let (test_frames, test_alignments) = load_data(test_path, &token_map)?;

    // debugging
    let norm_min = test_frames.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm_max = test_frames
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    // println!("Normalized range: min = {:.3}, max = {:.3}", norm_min, norm_max);

    // extract an arbitrary frame (f32) and rescale to [0, 255] range for u8
    let frame: Vec<u8> = test_frames[0..dim]
        .iter()
        .map(|x| {
            ((x - norm_min) / (norm_max - norm_min) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect::<Vec<u8>>();

    // debugging
    // println!("test_frames.len(): {}", test_frames.len());
    // println!("frame.len(): {}", frame.len());
    // println!("Expected per-frame size: {}", width * height);
    // assert_eq!(frame.len(), dim, "Frame length doesn't match image dimensions!");

    let img_buffer: GrayImage =
        GrayImage::from_vec(width, height, frame).expect("Failed to create image buffer");
    img_buffer
        .save("test_frame.png")
        .expect("Failed to save image");

    // ----------------- Model training -----------------
    
    let blank_id = vocab_size - 1;
    // let loader_factory = || dataloader::DataLoader::new("/path/to/data").iter();
    // let model = LRModel::<train::AD>::new(c, out_channels, (h, w), vocab_size, &device);
    // let (_model, losses) = train::train_loop(model, epochs, learning_rate, loader_factory, blank_index);

    Ok(())
}



/* ---------------------------------------------------- Data Loading/Processing Functions ---------------------------------------------------- */



// extract data externally to a given path
fn extract_data(path: &str) {
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
fn load_video(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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
fn load_alignments(
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
                .filter_map(|ch| token_map.char_to_num(ch))
                .collect())
        }
        Err(e) => {
            eprintln!("Error opening alignments file: {}", e);
            Err(Box::new(e))
        }
    }
}

// function to load data (takes in a data path and outputs frames and alignments)
fn load_data(
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



// ----------------------------------------------------------- Model Architecture ------------------------------------------------------------



// fn load_model(path: &str) -> TractResult<SimplePlan<TypedFact, Box<dyn TypedOp>>> {
//     tract_onnx::onnx()
//         .model_for_path(path)?
//         .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), shape![1, 75, 50, 150, 1]))?
//         .into_optimized()?
//         .into_runnable()
// }

// fn ctc_loss<B: Backend>(
//     log_probs: Tensor<B, 3>,
//     targets: Tensor<B, 2, Int>,
//     input_lens: Tensor<B, 1, Int>,
//     target: Tensor<B, 1, Int>,
//     blank: usize,
// ) -> Tensor<B, 1> {
// }
