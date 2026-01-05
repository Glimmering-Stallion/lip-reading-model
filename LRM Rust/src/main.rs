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



use burn::module::Module;
// imports
use lrm_rust::prelude::*;
use clap::Parser;
use image::{GrayImage, Luma};
// use opencv::{
//     self,
//     core::{MatTrait, Size},
//     videoio::VideoCaptureTrait,
// };
use std::{
    error::Error,
    // io::{self, BufRead},
};



#[derive(Parser, Debug)]
#[command(name = "lm_build")]
#[command(about = "Train an N-gram language model and save it to disk")]
struct Args {
    /// path to local text file corpus (mutually exclusive with --url)
    #[arg(long)]
    corpus: Option<String>,

    /// URL to remote JSONL.gz shard (mutually exclusive with --corpus)
    #[arg(long)]
    url: Option<String>,

    /// path to output model file
    #[arg(long, default_value = "models/ngram_lm.bin")]
    output: String,

    /// N-gram size
    #[arg(long, default_value_t = 3)]
    n: usize,
}



fn main() -> Result<(), Box<dyn Error>> {

    let args = Args::parse();

    // debugging
    // let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // println!("Input: {:?}, Mean: {}, Std Dev: {}", vector, mean(&vector), std_dev(&vector));

    // obtain data (if data isn't already loaded)
    // extract_data("../data");

    // ------ Obtain vocabulary and token map ------

    let blank_id = BLANK_ID;
    let token_map = TokenMap::new(VOCAB);

    println!("Vocabulary: {:?}", VOCAB);

    // ----------------- Load data -----------------

    // create data dir if it doesn't exist
    std::fs::create_dir_all("data")?;

    // height and width of cropped ROI (mouth region)
    // TODO: make this region adaptively tracking based on face detection
    let width: u32 = 150;
    let height: u32 = 50;
    let dim = (width * height) as usize;

    let test_path = "../data/grid-lr-dataset/s1/bbal6n.mpg";
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

    // ------------- N-Gram Model training --------------
    
    // collect training sequences as Vec<usize>
    // TODO: check if this is right path
    let corpus_dir = "../data/librispeech-lm-corpus/corpus";
    if !std::path::Path::new(corpus_dir).exists() {
        println!("Corpus not found locally — downloading (this may take a while)...");
        let url = "https://www.openslr.org/resources/11/librispeech-lm-corpus.tgz";
        import_corpus(url, "data")?;
        println!("Download + extract complete");
    }

    // now stream lines from local files
    // convert lines -> ids -> feed LM.train(...)
    let sequences = Box::new(
        stream_corpus_lines(corpus_dir).filter_map(move |line| {
            token_map.chars_to_ids(line.chars().collect())
        })
    );

    // does lm model already exist?
    if !std::path::Path::new(&args.output).exists() {
        // init LM, train, and save
        let mut lm = NgramLMConfig::new()
            .with_n(args.n)
            .init();
        lm.train(sequences);
        lm.save(&args.output)?;
        println!("Saved N-gram LM to {}", args.output);
    } else {
        println!("N-gram LM already exists at {}, skipping training", args.output);
    }

    // ------------- LipNet Model training --------------

    // let loader_factory = || dataloader::DataLoader::new("/path/to/data").iter();
    // let model = LRModel::<train::AD>::new(c, out_channels, (h, w), vocab_size, &device);
    // let (_model, losses) = train::train_loop(model, epochs, learning_rate, loader_factory, blank_index);

    Ok(())
}
