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
    optim::AdamConfig,
    backend::{
        {Autodiff, Wgpu},
        wgpu::WgpuDevice::DefaultDevice,
    },
};
use lrm_rust::{
    ctc::lm::{LanguageModel},
    pipeline::DatasetSource,
    prelude::*,
};
use clap::Parser;
use std::{
    sync::Arc,
    error::Error,
    path::Path,
    env,
    fs,
};



// Put this at the absolute top of main.rs or lib.rs
type MyBackend = Autodiff<Wgpu>;



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
    
    let args = Args::parse();

    // create data dir if it doesn't exist
    fs::create_dir_all("data")?;

    // dynamically get Rust project root and relevant dir paths
    let rust_root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let models_path = Path::new(&rust_root).join("models");
    let data_path = Path::new(&rust_root).join("data");
    let tests_path = Path::new(&rust_root).join("tests");

    if !models_path.exists() { fs::create_dir(&models_path).expect("Failed to create output directory for models") }
    if !data_path.exists() { fs::create_dir(&data_path).expect("Failed to create output directory for data") }
    if !tests_path.exists() { fs::create_dir(&tests_path).expect("Failed to create tests directory") }

    let token_map = Arc::new(TokenMap::new(VOCAB)); // bidirectional char to ID mapping
    // let token_map = TokenMap::new(VOCAB); // bidirectional char to ID mapping

    // debugging
    println!("\nVocabulary: {:?}", VOCAB);
    println!("Vocabulary size: {}", VOCAB_SIZE);
    println!("Blank token ID: {}\n", BLANK_ID);
    assert!(BLANK_ID < VOCAB_SIZE, "Blank ID ({}) is out of vocabulary size bounds ({})", BLANK_ID, VOCAB_SIZE);
    assert!(args.n > 0, "N-gram order ({}) must be greater than one", args.n);

    // ------------------------------------- Load data for N-gram model -------------------------------------

    let corpus_path = data_path
        .join("librispeech-lm-norm")
        .join("librispeech-lm-norm.txt");

    let corpus = corpus_path.to_string_lossy().to_string();

    // using extract_slr_dataset function in data_handler.rs to download + extract N-Gram corpus if needed
    extract_slr_corpus(rust_root.as_str());

    // --------------------------------- N-Gram model training/evaluation -----------------------------------

    let lm_output_path = models_path.join(&args.output); // output path for where LM resides
    
    // does an LM already exist?
    let lm = if !lm_output_path.exists() {
        println!("N-gram LM not found at {}, proceeding to train fresh model", lm_output_path.to_string_lossy());

        // now stream lines from local files (5% sampling rate, which is ~200MG of the 4GB corpus)
        // convert lines -> IDs -> feed LM.train(...)
        let train_token_map = Arc::clone(&token_map);
        let train_sequences = stream_corpus_lines(corpus.clone(), 0.05)
            .filter_map(move |line| {
                let chars = line.chars().collect::<Vec<char>>();
                train_token_map.clone().chars_to_ids(&chars)
            });

        // init, train, and save N-gram LM
        let mut lm = NgramConfig::new()
            .with_n(args.n)
            .with_vocab_size(VOCAB_SIZE)
            .init();

        // safety check: make sure parent dir exists one more time just in case
        if let Some(parent) = lm_output_path.parent() { fs::create_dir_all(parent).ok(); }

        lm.train(Box::new(train_sequences));
        lm.save(lm_output_path.to_str().unwrap())?;
        println!("Saved N-gram LM to {}\n", lm_output_path.to_string_lossy());

        lm
    } else {
        println!("N-gram LM already exists at {}, skipping corpus streaming and training", lm_output_path.to_string_lossy());

        // load existing N-gram LM
        let lm = Ngram::load(lm_output_path.to_str().unwrap()).unwrap();
        println!("Loaded N-gram LM from {}\n", lm_output_path.to_string_lossy());

        lm
    };

    // evaluate N-gram LM on held-out eval set (0.1% sampling rate)
    let eval_token_map = Arc::clone(&token_map);
    let eval_sequences = stream_corpus_lines(corpus.clone(), 0.05)
        .filter_map(move |line| {
            let chars = line.chars().collect::<Vec<char>>();
            eval_token_map.clone().chars_to_ids(&chars)
        })
        .take(10000);

    let perplexity = lm.perplexity(Box::new(eval_sequences));
    println!("N-gram LM perplexity on eval set: {:.3}\n", perplexity);
    assert!(perplexity.is_finite(), "LM perplexity ({}) is non-finite", perplexity);

    // ------------------------------------------- VSRM training --------------------------------------------

    // define hyperparameters
    let frame_dims = (50, 150); // height, width
    let num_epochs = 1;
    let batch_size = 1;
    let learning_rate = 1e-3;
    let num_workers = 4;
    let accumulation = 4;
    let seed = 42;
    let device = DefaultDevice;
    let root_path = rust_root;
    let vsrm_output_path = models_path;

    let dataset_src = DatasetSource::Grid;

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);

    let model_config = VsrModelConfig::new(frame_dims)
        .with_vocab_size(VOCAB_SIZE);

    let learner_config = VsrmLearnerConfig {
        num_epochs,
        batch_size,
        learning_rate,
        optimizer: optimizer_config,
        num_workers,
        accumulation,
        seed,
    };

    train::<MyBackend, _, _>(
        device,
        dataset_src,
        model_config,
        learner_config,
        (*token_map).clone(),
        root_path,
        vsrm_output_path,
    );

    Ok(())
}
