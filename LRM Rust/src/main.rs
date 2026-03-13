#![recursion_limit = "2048"]

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

// build N-gram LM (train if missing, else load and eval):                               cargo run -- build-lm --model [lm.bin] --corpus [path/to/corpus] --n [N-gram order]
// preprocess a specific dataset for the VSRM::                                          cargo run -- preprocess --dataset [dataset_src]
// train new VSRM with default model ID `vsrm_{dataset_src}` (error if ID alr exists):   cargo run -- train --model
// train new VSRM with custom model ID (error if ID exists):                             cargo run -- train --model [my_vsrm]
// resume training from latest checkpoint (uses last completed epoch):                   cargo run -- train --model [...] --resume
// resume training from specified checkpoint:                                            cargo run -- train --model [...] --resume [epoch]
// train using a subset of the dataset (e.g. fraction = 0.1 for 10%):                    cargo run -- train --model [...] --subset [fraction]
// toggle keep-all-checkpoints during training (default: keep most recent only):         cargo run -- train --model [...] --keep-all-checkpoints [on|off]
// run inference for default model ID on a video file:                                   cargo run -- infer --model --input [path/to/video.mpg]
// run inference for custom model ID on a video file:                                    cargo run -- infer --model [my_vsrm] --input [path/to/video.mpg]
// run inference on a video file with visulization overlay:                              cargo run -- infer --model [...] --input [...] --visualize
// run real-time live inference from webcam:                                             cargo run -- infer --model [...] --live
// run real-time live inference from specified webcam:                                   cargo run -- infer --model [...] --live --camera [my_camera]


// imports
use burn::{
    backend::{
        Autodiff,
        Wgpu,
        wgpu::WgpuDevice::DefaultDevice,
    },
    grad_clipping::GradientClippingConfig,
    optim::{AdamConfig, decay::WeightDecayConfig}, prelude::Backend,
};
use lrm_rust::{
    cli,
    ctc::lm::LanguageModel,
    pipeline::{
        DatasetSource,
        tracker::{TrackerConfig, HaarTrackerConfig},
    },
    prelude::*,
    vocab::SPACE_ID,
};
use clap::{Parser, Subcommand};
use std::{
    sync::Arc,
    path::PathBuf,
    error::Error,
};



type MyBackend = Autodiff<Wgpu>;



#[derive(Parser, Debug)]
#[command(name = "lrm")]
struct Args {
    #[command(subcommand)]
    command: Command,
}



#[derive(Subcommand, Debug)]
enum Command {
    /// Builds N-gram LM from corpus or loads pretrained N-gram LM.
    BuildLm {
        #[arg(long, default_value = "ngram_lm.bin")]
        model: String,

        #[arg(long)]
        corpus: Option<String>,

        #[arg(long, default_value_t = 3)]
        n: usize,
    },
    /// Trains VSRM from scratch or resumes from checkpoint.
    Train {
        #[arg(long, num_args = 0..=1, value_name = "MODEL")]
        model: Option<Option<String>>,

        #[arg(long, num_args = 0..=1, value_name = "EPOCH")]
        resume: Option<Option<usize>>,

        #[arg(long, value_name = "FRACTION")]
        subset: Option<f32>,

        #[arg(long, num_args = 0..=1, value_name = "ON_OFF", default_missing_value = "on", value_parser = clap::builder::PossibleValuesParser::new(["on", "off"]))]
        keep_all_checkpoints: Option<Option<String>>,
    },
    /// Loads trained VSRM and runs inference on video(s).
    Infer {
        #[arg(long)]
        model: String,

        #[arg(long)]
        input: PathBuf,

        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Pre-extracts mouth crops to disk for faster training.
    Preprocess {
        #[arg(long, default_value = "grid")]
        dataset: String,
    },
}



fn main() -> Result<(), Box<dyn Error>> {
    // obtain terminal arg values, filesystem context, and token map
    let args = Args::parse();
    let context = Context::new();
    let token_map = Arc::new(TokenMap::new(VOCAB));

    // debugging
    println!("\nVocabulary: {:?}", VOCAB);
    println!("Vocabulary size: {}", VOCAB_SIZE);
    println!("Blank token ID: {}", BLANK_ID);
    println!("Blank token char: {}\n", VOCAB.chars().nth(BLANK_ID).unwrap());
    println!("Space token ID: {}", SPACE_ID);
    println!("Blank token char: {}\n", VOCAB.chars().nth(SPACE_ID).unwrap());
    assert!(BLANK_ID < VOCAB_SIZE, "Blank ID ({}) is out of vocabulary size bounds ({})", BLANK_ID, VOCAB_SIZE);

    // CLI control flow
    match &args.command {
        Command::BuildLm { corpus, model, n } => {
            run_build_lm(&context, corpus.as_deref(), model, *n, &token_map)?;
        }
        Command::Train { model, resume, subset, keep_all_checkpoints } => {
            run_train_vsrm(&context, model.as_ref().and_then(|m| m.as_deref()), *resume, *subset, keep_all_checkpoints.as_ref().map(|o| o.as_deref()), &token_map)?;
        }
        Command::Infer { model, input, output } => {
            // run_infer_vsrm(&context, model.as_ref(), input, &(*token_map).clone())?;
        }
        Command::Preprocess { dataset } => {
            run_preprocess(&context, dataset, &token_map)?;
        }
    }

    Ok(())
}



/// Builds or loads N-gram LM, then evaluates perplexity.
///
/// ### Params:
/// - `context`: Filesystem context for paths.
/// - `corpus`: Optional path to corpus file; if `None`, uses default LibriSpeech path.
/// - `model`: Output filename for the LM (e.g. `"ngram_lm.bin"`).
/// - `n`: N-gram order.
/// - `token_map`: Bidirectional char-to-ID mapping for encoding sequences.
///
/// ### Returns:
/// `Ok(())` on success, or an error on I/O or LM build failure.
fn run_build_lm(
    context: &Context,
    corpus: Option<&str>,
    model: &str,
    n: usize,
    token_map: &Arc<TokenMap>,
) -> Result<(), Box<dyn Error>> {
    extract_slr_corpus(&context.rust_root);

    let corpus_path = corpus
        .map(PathBuf::from)
        .unwrap_or_else(|| context.data_path.join("librispeech-lm-norm").join("librispeech-lm-norm.txt"));

    if !corpus_path.exists() {
        return Err(format!("Corpus path {:?} does not exist", corpus_path).into());
    }

    let lm_output_path = context.models_path.join(model);
    let corpus_str = corpus_path.to_string_lossy().to_string();
    let train_token_map = Arc::clone(token_map);
    let train_sequences: Vec<Vec<usize>> = if lm_output_path.exists() {
        vec![]
    } else {
        stream_corpus_lines(&corpus_str, 0.05)
            .filter_map(move |line| {
                let chars = line.chars().collect::<Vec<char>>();
                train_token_map.clone().chars_to_ids(&chars)
            })
            .collect()
    };

    let lm = build_or_load_ngram_lm(&lm_output_path, n, VOCAB_SIZE, train_sequences)?;
    println!("N-gram LM ready at {}", lm_output_path.display());

    let eval_token_map = Arc::clone(token_map);
    let eval_sequences: Vec<Vec<usize>> = stream_corpus_lines(&corpus_str, 0.05)
        .filter_map(move |line| {
            let chars = line.chars().collect::<Vec<char>>();
            eval_token_map.clone().chars_to_ids(&chars)
        })
        .take(10000)
        .collect();

    let perplexity = lm.perplexity(Box::new(eval_sequences.into_iter()));
    println!("N-gram LM perplexity on eval set: {:.3}\n", perplexity);
    assert!(perplexity.is_finite(), "LM perplexity ({}) is non-finite", perplexity);

    Ok(())
}



/// Ensures LM exists (if needed), then runs VSRM training.
///
/// ### Params:
/// - `context`: Filesystem context for paths.
/// - `model`: Optional model ID; if `None`, uses default.
/// - `resume`: Optional resume spec; `None` for fresh start, `Some(None)` for latest checkpoint, `Some(Some(epoch))` for specific epoch.
/// - `active_subset`: Optional fraction of dataset to use (e.g. 0.1 for 10%).
/// - `keep_all_checkpoints_cli`: Optional CLI override; `None` = use persisted, `Some(Some("on"))` = keep all checkpoints, `Some(Some("off"))` = keep most recent only.
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or an error on training failure.
fn run_train_vsrm(
    context: &Context,
    model: Option<&str>,
    resume: Option<Option<usize>>,
    active_subset: Option<f32>,
    keep_all_checkpoints_cli: Option<Option<&str>>,
    token_map: &Arc<TokenMap>,
) -> Result<(), Box<dyn Error>> {
    let vocab_size = VOCAB_SIZE;
    let blank_id = BLANK_ID;
    let frame_dims = (50, 100);
    let num_epochs = 100;
    let batch_size = 4;
    let learning_rate = 3e-4;
    let num_workers = 4;
    let accumulation = 8;
    let seed = 42;
    let device = DefaultDevice;
    let dataset_src = DatasetSource::Grid;

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0)));

    let model_id = model
        .map(String::from)
        .unwrap_or_else(|| format!("vsrm_{}", dataset_src.tag()));
    let model_path = context.models_path.join(&model_id);
    let model_config = VsrModelConfig::new()
        .with_frame_dims(frame_dims)
        .with_vocab_size(vocab_size)
        .with_blank_id(blank_id);

    // persist `keep-all-checkpoints`flag and `subset` fraction values across runs
    let keep_all_checkpoints = cli::resolve_keep_all_checkpoints(&model_path, keep_all_checkpoints_cli);
    let active_subset = cli::resolve_active_subset(&model_path, active_subset, seed);

    let learner_config = VsrmLearnerConfig {
        model_id,
        resume_from: resume,
        keep_all_checkpoints,
        frame_dims,
        num_epochs,
        batch_size,
        learning_rate,
        optimizer: optimizer_config,
        num_workers,
        accumulation,
        seed,
        active_subset,
    };

    train::<MyBackend>(
        device,
        context,
        dataset_src,
        model_config,
        learner_config,
        token_map.as_ref().clone(),
    );

    Ok(())
}



// fn run_infer_vsrm<B: Backend>(
//     context: &Context,
//     model: Option<&str>,
//     input: Option<&str>,
//     token_map: &Arc<TokenMap>,
// ) -> Result<> {
//     todo!()
// }



/// Pre-extracts mouth crops to disk for faster training.
///
/// ### Params:
/// - `context`: Filesystem context for paths.
/// - `dataset`: Dataset name; only `"grid"` is supported.
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or an error if dataset is unsupported.
fn run_preprocess(
    context: &Context,
    dataset: &str,
    token_map: &Arc<TokenMap>,
) -> Result<(), Box<dyn Error>> {
    if dataset != "grid" { return Err(format!("Unsupported dataset: {}. Only 'grid' is supported.", dataset).into()); }

    let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
        context.models_path.join("haarcascade_frontalface_alt2.xml"),
        context.models_path.join("haarcascade_mcs_mouth.xml"),
        (50, 100),
    ));

    let grid_dataset = GridDataset::new(
        context,
        token_map.as_ref().clone(),
        Some(tracker_config),
        None,
    );
    grid_dataset.preprocess_all();

    Ok(())
}
