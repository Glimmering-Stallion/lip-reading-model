// I/O handler for high-level fs and networking tasks (data loading, streaming, extracting, etc.)



use crate::prelude::*;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use opencv::{
    self,
    core::{Mat, AlgorithmHint, Rect},
    imgproc,
    prelude::*,
    videoio::{
        VideoCapture,
        VideoCaptureTrait,
        CAP_ANY,
    },
};
use rand::Rng;
use reqwest::blocking::Client;
// use serde_json::Value;
use std::{
    error::Error,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    path::Path,
    sync::{
        atomic::{
            AtomicU64,
            Ordering,
        },
        Arc,
    },
};
use zip::ZipArchive;
use tar::Archive;



/// stream plain .txt lines from disk
pub fn stream_txt_lines<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Box<dyn Error>> {
    assert!(path.as_ref().exists(), "Text file {} does not exist", path.as_ref().to_string_lossy());

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut out = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let text = line.trim().to_string();
        if !text.is_empty() {
            out.push(text);
        }
    }

    Ok(out)
}



/// stream "text" field from remote JSONL.gz shard (e.g., C4)
pub fn stream_jsonl_gz(url: String) -> Result<Vec<String>, Box<dyn Error>> {
    assert!(!url.is_empty(), "URL is empty");

    let client = Client::new();
    let resp = client.get(&url).send()?.error_for_status()?;
    let decoder = GzDecoder::new(resp);
    let reader = BufReader::new(decoder);

    let mut out = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
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
pub fn stream_corpus_lines<P: AsRef<Path>>(
    file_path: P,
    sample_rate: f64
) -> impl Iterator<Item = String> {
    let file_path = file_path.as_ref();
    assert!(file_path.exists(), "Corpus file {:?} does not exist", file_path);
    assert!(sample_rate > 0.0 && sample_rate <= 1.0, "sample_rate must be in (0, 1]");
    println!("Streaming corpus lines from: {}", file_path.to_string_lossy());

    let file = File::open(file_path).expect("Failed to open corpus file");
    let metadata = file.metadata().expect("Failed to get file metadata");
    let file_size = metadata.len();

    let prog_bar = ProgressBar::new(file_size);
    prog_bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({msg}) (ETA: {eta})")
            .unwrap()
            .progress_chars("#>-"),
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



/// extract zip file to a given path
pub fn extract_zip<P: AsRef<Path>, Q: AsRef<Path>>(zip_path: P, extract_to: Q) {
    let zip_path = zip_path.as_ref();
    let extract_to = extract_to.as_ref();
    assert!(zip_path.exists(), "Zip file {:?} does not exist", zip_path);
    
    let input_file = File::open(zip_path).expect("Failed to open zip file.");
    let mut archive = ZipArchive::new(input_file).expect("Failed to read zip file.");
    
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).expect("Failed to read file from zip.");
        let out_path = match file.enclosed_name() {
            Some(path) => extract_to.join(path),
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
    assert!(extract_to.exists(), "Zip destination does not exist");
    println!("Extracted zip file to {}", extract_to.to_string_lossy());
}



/// extract gzip file to a given path
pub fn extract_gzip<P: AsRef<Path>, Q: AsRef<Path>>(gzip_path: P, extract_to: Q) {
    let gzip_path = gzip_path.as_ref();
    let extract_to = extract_to.as_ref();
    assert!(gzip_path.exists(), "GZip file {:?} does not exist", gzip_path);
    
    let input_file = File::open(gzip_path).expect("Failed to open gzip file.");
    let mut decoder = GzDecoder::new(input_file);
    
    // get folder path from the 'extract_to' string and create it
    if let Some(parent) = extract_to.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent directory for extraction.");
    }

    let mut out_file = File::create(extract_to).expect("Failed to create output file.");
    io::copy(&mut decoder, &mut out_file).expect("Failed to decompress gzip content.");

    assert!(extract_to.exists(), "GZip destination does not exist");
    println!("Extracted gzip file to {}", extract_to.to_string_lossy());
}



/// extract GRID corpus externally to a given path
/// (deprecated in favor of manual download due to Google Drive download complexities)
pub fn extract_grid_corpus<P: AsRef<Path>>(root_path: P) {
    let root_path = root_path.as_ref();
    let data_dir = root_path.join("data");
    let grid_dir = data_dir.join("grid-lr-corpus");
    assert!(root_path.exists(), "Root path {:?} does not exist", root_path);

    // check if the GRID corpus exists at the given path
    if !grid_dir.exists() {
        println!("Grid corpus not found, downloading...");

        // use client with NO timeout for large files
        let client = Client::builder()
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
                    extract_zip(&output, &root_path);

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
                }
            }
            Err(e) => { eprintln!("Error parsing URL: {}", e); }
        }
    } else { println!("GRID corpus already exists, downloading skipped"); }
}



/// extract SLR corpus externally to a given path
pub fn extract_slr_corpus<P: AsRef<Path>>(root_path: P) {
    let root_path = root_path.as_ref();
    let data_dir = root_path.join("data");
    let slr_dir = data_dir.join("librispeech-lm-norm");
    let final_path = slr_dir.join("librispeech-lm-norm.txt");
    assert!(root_path.exists(), "Root path {:?} does not exist", root_path);

    // check if the SLR corpus exists at the given path
    if !final_path.exists() {
        println!("\nSLR corpus not found, downloading...");

        fs::create_dir_all(&slr_dir).expect("Failed to create SLR directory");

        // use client with NO timeout for large files
        let client = Client::builder()
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

                    // extract/remove gzip file
                    extract_gzip(&output, &final_path);
                    fs::remove_file(&output).expect("Failed to delete gzip file.");

                    assert!(final_path.exists(), "SLR corpus file missing after extraction");
                    assert!(final_path.metadata().unwrap().len() > 0, "SLR corpus file is empty");
                } else { eprintln!("Failed to download SLR corpus: {}", response.status()); }
            }
            Err(e) => { eprintln!("Error parsing URL: {}", e); }
        }
    } else { println!("SLR corpus already exists, downloading skipped"); }
}



/// takes in a frames path and outputs a list of floats
pub fn load_grid_frames<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, Box<dyn Error>> {
    let path = path.as_ref().to_str().ok_or("Non-UTF8 path")?;

    match VideoCapture::from_file(path, CAP_ANY) {
        Ok(mut cap) => {
            let mut frames: Vec<u8> = Vec::with_capacity(75 * 150 * 50);
            let mut frame = Mat::default();
            let mut gray_frame = Mat::default();

            while cap.read(&mut frame).expect("Error reading frame") {
                let size = frame.size().expect("Failed to get frame size");
                if size.width == 0 || size.height == 0 { break; } // end of frames

                // convert frame to grayscale
                imgproc::cvt_color(
                    &frame,
                    &mut gray_frame,
                    imgproc::COLOR_BGR2GRAY,
                    0,
                    AlgorithmHint::ALGO_HINT_DEFAULT,
                )
                .expect("Failed to convert frame to grayscale");

                // crop frame to where mouth is (in future this will be replaced by dynamic tracking)
                let roi = Rect::new(80, 190, 150, 50);
                let temp = Mat::roi(&gray_frame, roi).expect("Failed to crop frame");
                let mut mouth_frame = Mat::default();
                temp.copy_to(&mut mouth_frame)
                    .expect("Failed to copy ROI to a continuous Mat");

                // store frames as bytes (u8) to save memory
                // will be converted to f32 and standardized later in the pipeline
                frames.extend(mouth_frame.data_bytes()?);
            }

            assert!(!frames.is_empty(), "No frames read from video");
            Ok(frames)
        }
        Err(e) => {
            eprintln!("Error opening video file: {}", e);
            Err(Box::new(e))
        }
    }
}



/// takes in an alignments path (as well as TokenMap struct) and outputs a list of char IDs (as indices)
pub fn load_grid_alignments<P: AsRef<Path>>(
    path: P,
    token_map: &TokenMap,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let path = path.as_ref();

    match File::open(path) {
        Ok(file) => {
            let mut tokens: Vec<String> = vec![];
            let lines = BufReader::new(file).lines();

            for line in lines.map_while(Result::ok) {
                let line_group = line.split_whitespace().collect::<Vec<_>>();
                assert!(line_group.len() >= 3, "Malformed alignment line: {:?}", line_group);
                if line_group[2] != "sil" { tokens.push(line_group[2].to_string()); }
            }
            assert!(!tokens.is_empty(), "No non-silence tokens found in alignment file");

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
pub fn load_grid_corpus<P: AsRef<Path>>(
    root_path: P,
    entry_name: &str,
    token_map: &TokenMap,
) -> Result<(Vec<u8>, Vec<usize>), Box<dyn Error>> {
    let root_path = root_path.as_ref();
    assert!(root_path.exists(), "Root path {:?} does not exist", root_path);

     // entry_name in the form of "s1/bbaf2n"
    if entry_name.trim().is_empty() {
        eprintln!("Error: Provided entry_name is empty.");
        return Err("Failed to extract filename: entry_name was empty.".into());
    }

    // join project root with LR data path for an absolute path
    let grid_dataset_path = root_path.join("data/grid-lr-corpus");

    let frames_path = grid_dataset_path
        .join("frames")
        .join(entry_name)
        .with_extension("mpg");
    assert!(frames_path.exists(), "Video file not found: {}", frames_path.to_string_lossy());

    let alignments_path = grid_dataset_path
        .join("alignments")
        .join(entry_name)
        .with_extension("align");
    assert!(alignments_path.exists(), "Alignment file not found: {}", alignments_path.to_string_lossy());

    let frames = load_grid_frames(&frames_path)?;
    let alignments = load_grid_alignments(&alignments_path, token_map)?;
    assert!(!frames.is_empty() && !alignments.is_empty(), "Loaded empty GRID sample");

    Ok((frames, alignments))
}
