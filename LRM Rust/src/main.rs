// create new Rust project with cargo (separate dir name from package name):             cargo new "[dir name]" --name [package_name]
// create new Rust project with cargo (but without auto creating new Git repo):          cargo new [dir name] --vcs none

// for big projects:

// compile project with cargo:                                                           cargo build
// compile project with cargo with optimizations:                                        cargo build --release
// compile and run project with cargo:                                                   cargo run
// compile and run all tests (while allowing prints):                                    cargo test --nocapture
// compile and run specific unit test(while allowing prints):                            cargo test -- [test name] --nocapture

// for small experiments:

// compile single Rust file manually with rustc:                                         rustc [file name]
// run compiled binary (in same folder):                                                 .\[file name]

// for crate imports:

// import crate with cargo:                                                              cargo add [crate name]

// for this project:

// build LM:                                                                             cargo run -- build-lm --corpus data/librispeech-lm-norm
// train VSRM:                                                                           cargo run -- train-vsrm --epochs [num epochs]


// imports
use burn::{
    optim::{
        AdamConfig,
        optim::decay::WeightDecayConfig,
    },
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

    /// path to output LM file
    #[arg(long, default_value = "ngram_lm.bin")]
    output: String,

    /// N-gram size
    #[arg(long, default_value_t = 3)]
    n: usize,
}



fn main() -> Result<(), Box<dyn Error>> {

    // ------------------------------------------- Initial setup --------------------------------------------
    
    // obtain terminal arg values, filesystem context, and token map
    let args = Args::parse();
    let context = Context::new();
    let token_map = Arc::new(TokenMap::new(VOCAB)); // bidirectional char to ID mapping
    // let token_map = TokenMap::new(VOCAB); // bidirectional char to ID mapping

    // debugging
    println!("\nVocabulary: {:?}", VOCAB);
    println!("Vocabulary size: {}", VOCAB_SIZE);
    println!("Blank token ID: {}\n", BLANK_ID);
    assert!(BLANK_ID < VOCAB_SIZE, "Blank ID ({}) is out of vocabulary size bounds ({})", BLANK_ID, VOCAB_SIZE);
    assert!(args.n > 0, "N-gram order ({}) must be greater than one", args.n);

    // ------------------------------------- Load data for N-gram model -------------------------------------

    let corpus_path = context.data_path
        .join("librispeech-lm-norm")
        .join("librispeech-lm-norm.txt");

    let corpus = corpus_path.to_string_lossy().to_string();

    // using extract_slr_dataset function in data_handler.rs to download + extract N-Gram corpus if needed
    extract_slr_corpus(&context.rust_root);

    // --------------------------------- N-Gram model training/evaluation -----------------------------------

    let lm_output_path = context.models_path.join(&args.output); // output path for where LM resides
    
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
    println!("N-gram LM perplexity on eval set: {:.3}", perplexity);
    assert!(perplexity.is_finite(), "LM perplexity ({}) is non-finite", perplexity);

    // ------------------------------------------- VSRM training --------------------------------------------

    // define hyperparameters
    let vocab_size = VOCAB_SIZE;
    let blank_id = BLANK_ID;
    let frame_dims = (50, 150); // height, width
    let num_epochs = 30;
    let batch_size = 8;
    let learning_rate = 1e-4;
    let num_workers = 4;
    let accumulation = 1;
    let seed = 42;
    let device = DefaultDevice;

    let dataset_src = DatasetSource::Grid;

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)));

    let model_config = VsrModelConfig::new(frame_dims)
        .with_vocab_size(vocab_size)
        .with_blank_id(blank_id);

    let learner_config = VsrmLearnerConfig {
        num_epochs,
        batch_size,
        learning_rate,
        optimizer: optimizer_config,
        num_workers,
        accumulation,
        seed,
    };

    train::<MyBackend>(
        device,
        &context,
        dataset_src,
        model_config,
        learner_config,
        (*token_map).clone(),
    );

    Ok(())
}
