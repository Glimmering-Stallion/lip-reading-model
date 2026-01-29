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



// imports
use burn::{
    module::Module,
    data::dataloader::DataLoaderBuilder,
};
use lrm_rust::{ctc::lm::{self, LanguageModel}, prelude::*};
use clap::Parser;
use image::{GrayImage, Luma};
// use opencv::{
//     self,
//     core::{MatTrait, Size},
//     videoio::VideoCaptureTrait,
// };
use std::{
    sync::{atomic, Arc},
    error::Error,
    path::Path,
    env,
    fs,
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
    #[arg(long, default_value = "ngram_lm.bin")]
    
    output: String,

    /// N-gram size
    #[arg(long, default_value_t = 3)]
    n: usize,
}



fn main() -> Result<(), Box<dyn Error>> {

    // ------------------------------------------- Initial setup --------------------------------------------

    // debugging
    // let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // println!("Input: {:?}, Mean: {}, Std Dev: {}", vector, mean(&vector), std_dev(&vector));

    // obtain data (if data isn't already loaded)
    // extract_data("../data");
    
    let args = Args::parse();

    // create data dir if it doesn't exist
    fs::create_dir_all("data")?;

    // dynamically get Rust project root and relevant dir paths
    let rust_root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let models_path = Path::new(&rust_root).join("models");
    let data_path = Path::new(&rust_root).join("data");

    if !models_path.exists() { fs::create_dir(&models_path).expect("Failed to create output directory for models") }
    if !data_path.exists() { fs::create_dir(&data_path).expect("Failed to create output directory for data") }

    let token_map = Arc::new(TokenMap::new(VOCAB)); // bidirectional char to ID mapping

    // debugging
    println!("Vocabulary: {:?}", VOCAB);
    println!("Vocabulary size: {}", VOCAB_SIZE);
    println!("Blank token ID: {}", BLANK_ID);

    // ------------------------------------- Load data for N-gram model -------------------------------------

    let corpus_path = data_path
        .join("librispeech-lm-norm")
        .join("librispeech-lm-norm.txt");

    let corpus = corpus_path.to_string_lossy().to_string();

    // using extract_slr_dataset function in data_handler.rs to download + extract N-Gram corpus if needed
    extract_slr_corpus(rust_root.as_str());

    // --------------------------------- N-Gram Model training/evaluation -----------------------------------

    let output_path = models_path.join(&args.output); // output path for where LM model resides
    
    // does an LM model already exist?
    let lm = if !output_path.exists() {
        println!("N-gram LM model not found at {}, proceeding to train fresh model", output_path.to_string_lossy());

        
        // now stream lines from local files (5% sampling rate, which is ~200MG of the 4GB corpus)
        // convert lines -> IDs -> feed LM.train(...)
        let train_token_map = Arc::clone(&token_map);
        let train_sequences = stream_corpus_lines(corpus.clone(), 0.05)
            .filter_map(move |line| {
                let chars = line.chars().collect::<Vec<char>>();
                train_token_map.chars_to_ids(&chars)
            });

        // init, train, and save N-gram LM
        let mut lm = NgramConfig::new()
            .with_n(args.n)
            .with_vocab_size(VOCAB_SIZE)
            .init();

        // safety check: make sure parent dir exists one more time just in case
        if let Some(parent) = output_path.parent() { fs::create_dir_all(parent).ok(); }

        lm.train(Box::new(train_sequences));
        lm.save(output_path.to_str().unwrap())?;
        println!("Saved N-gram LM to {}", output_path.to_string_lossy());

        lm
    } else {
        println!("N-gram LM model already exists at {}, skipping corpus streaming and training", output_path.to_string_lossy());

        // load existing N-gram LM
        let lm = Ngram::load(output_path.to_str().unwrap()).unwrap();
        println!("Loaded N-gram LM from {}", output_path.to_string_lossy());

        lm
    };

    // evaluate N-gram LM on held-out eval set (0.1% sampling rate)
    let eval_token_map = Arc::clone(&token_map);
    let eval_sequences = stream_corpus_lines(corpus.clone(), 0.05)
        .filter_map(move |line| {
            let chars = line.chars().collect::<Vec<char>>();
            eval_token_map.chars_to_ids(&chars)
        })
        .take(100000);

    let perplexity = lm.perplexity(Box::new(eval_sequences));
    println!("N-gram LM perplexity on eval set: {:.3}", perplexity);

    // ----------------------------------------- Load data for VSRM -----------------------------------------

    // let data_path = root_path.join("data/grid-lr-dataset");

    // height and width of cropped ROI (mouth region)
    // TODO: make this region adaptively tracking based on face detection
    let width: u32 = 150;
    let height: u32 = 50;
    let dim = (width * height) as usize;

    let test_input = "s1/bbal6n";
    let (test_frames, test_alignments) = load_grid_corpus(&rust_root, test_input, &token_map)?;
    println!("Loaded {} frames for {}", test_frames.len() / dim, test_input);

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

    let img_buffer: GrayImage = GrayImage::from_vec(width, height, frame).expect("Failed to create image buffer");
    img_buffer
        .save("test_frame.png")
        .expect("Failed to save image");

    // --------------------------------------- LipNet Model training ----------------------------------------

    // define hyperparameters
    let epochs = 100;
    let learning_rate = 0.0001;
    let batch_size = 8;
    let num_workers = 4;
    let seed = 42;
    // let device = 

    // let grid_data = GridDataset::new();

    // let batcher = VsrmBatcher::new(device);

    // let loader_factory = || DataLoaderBuilder::new(batcher)
    //     .batch_size(batch_size)
    //     .num_workers(num_workers)
    //     .shuffle(seed)
    //     .build(grid_data);

    // let model = VsrModel::<train::AD>::new(c, out_channels, (h, w), VOCAB_SIZE, &device);
    // let (_model, losses) = train_loop(model, epochs, learning_rate, loader_factory, BLANK_ID);

    Ok(())
}
