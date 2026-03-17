#![recursion_limit = "2048"]

// create new Rust project with cargo (separate dir name from package name):                                                 cargo new "[dir name]" --name [package_name]
// create new Rust project with cargo (but without auto creating new Git repo):                                              cargo new [dir name] --vcs none

// for big projects:

// compile project with cargo:                                                                                               cargo build
// compile project with cargo with optimizations:                                                                            cargo build --release
// compile and run project with cargo:                                                                                       cargo run
// compile and run all tests (while allowing prints):                                                                        cargo test --nocapture
// compile and run specific unit test(while allowing prints):                                                                cargo test -- [test name] --nocapture

// for small experiments:

// compile single Rust file manually with rustc:                                                                             rustc [file name]
// run compiled binary (in same folder):                                                                                     .\[file name]

// for crate imports:

// import crate with cargo:                                                                                                  cargo add [crate name]

// for this project:

// build N-gram LM (train if missing, else load and eval):                                                                   cargo run -- build-lm --model [lm.bin] --corpus [path/to/corpus] --n [N-gram order]
// preprocess a specific dataset for the VSRM::                                                                              cargo run -- preprocess --dataset [dataset_src]
// train new VSRM with default model ID `vsrm_{dataset_src}` on specified dataset (error if ID alr exists):                  cargo run -- train --dataset [dataset_src]
// train new VSRM with custom model ID on specified dataset (error if ID exists):                                            cargo run -- train --model [model_id] --dataset [dataset_src]
// resume training from latest checkpoint (uses last completed epoch):                                                       cargo run -- train [...] --resume
// resume training from specified checkpoint:                                                                                cargo run -- train [...] --resume [epoch]
// train using a subset of the dataset (e.g. fraction = 0.1 for 10%):                                                        cargo run -- train [...] --subset [fraction]
// toggle keep-all-checkpoints during training (default: keep most recent only):                                             cargo run -- train [...] --keep-all-checkpoints [on|off]
// run inference on a video file (requires --model):                                                                         cargo run -- infer --model [model_id] --input [path/to/video.mpg]
// run real-time live inference from default webcam:                                                                         cargo run -- infer --model [model_id] --live
// run real-time live inference from specified webcam:                                                                       cargo run -- infer --model [model_id] --live --camera [device_id]



// imports
use burn::{
    config::Config,
    backend::{
        Autodiff,
        Wgpu,
        wgpu::WgpuDevice::DefaultDevice,
    },
    grad_clipping::GradientClippingConfig,
    optim::{
        AdamConfig,
        decay::WeightDecayConfig,
    },
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
    io::ErrorKind,
    path::PathBuf,
    sync::Arc,
};



type TrainBackend = Autodiff<Wgpu>;
type InferBackend = Wgpu;



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
    /// Pre-extracts mouth crops to disk for faster training.
    Preprocess {
        #[arg(long)]
        dataset: DatasetSource,
    },
    /// Trains VSRM from scratch or resumes from checkpoint.
    Train {
        #[arg(long)]
        model: Option<String>,

        #[arg(long)]
        dataset: Option<DatasetSource>,

        #[arg(long, num_args = 0..=1, value_name = "EPOCH")]
        resume: Option<Option<usize>>,

        #[arg(long, value_name = "FRACTION")]
        subset: Option<f32>,

        #[arg(long, num_args = 0..=1, value_name = "ON_OFF", default_missing_value = "on", value_parser = clap::builder::PossibleValuesParser::new(["on", "off"]))]
        keep_all_checkpoints: Option<Option<String>>,
    },
    /// Loads trained VSRM and runs inference on a video file or live webcam.
    Infer {
        #[arg(long)]
        model: String,

        #[arg(long, conflicts_with = "live")]
        input: Option<PathBuf>,

        #[arg(long, conflicts_with = "input")]
        live: bool,

        #[arg(long, default_value_t = 0)]
        camera: i32,
    },
}



fn main() -> Result<(), ESS> {
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
        Command::Train { model, dataset, resume, subset, keep_all_checkpoints } => {
            run_train_vsrm(&context, model.as_deref(), dataset.clone(), *resume, *subset, keep_all_checkpoints.as_ref().map(|o| o.as_deref()), &token_map)?;
        }
        Command::Infer { model, input, live, camera } => {
            run_infer_vsrm(&context, &model, input.as_deref(), *live, *camera, &token_map)?;
        }
        Command::Preprocess { dataset } => {
            run_preprocess(&context, *dataset, &token_map)?;
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
) -> Result<(), ESS> {
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



/// Loads a VSRM dataset and runs training with fresh start or resume.
///
/// Resolves `model_id` from `--model` or `--dataset` (default `vsrm_{dataset_src}`).
/// 
/// Validates resume intent against checkpoint state before loading any config.
/// 
/// If resuming, loads persisted `learner_config` and merges with CLI overrides
/// for `keep_all_checkpoints`, `active_subset`, and `dataset_src`.
/// 
/// Builds `model_config` and `learner_config`, then delegates to `train()`.
///
/// ### Params:
/// - `context`: Filesystem context.
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
    dataset: Option<DatasetSource>,
    resume: Option<Option<usize>>,
    active_subset: Option<f32>,
    keep_all_checkpoints_cli: Option<Option<&str>>,
    token_map: &Arc<TokenMap>,
) -> Result<(), ESS> {
    // hyperparameters
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

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0)));

    let model_id = model
        .map(String::from)
        .or_else(|| dataset.map(|d| format!("vsrm_{}", d.tag())))
        .ok_or_else(|| io_err("Train requires `--model` or `--dataset`", ErrorKind::InvalidInput))?;

    let model_path = context.models_path.join(&model_id);
    let model_config = VsrModelConfig::new()
        .with_frame_dims(frame_dims)
        .with_vocab_size(vocab_size)
        .with_blank_id(blank_id);

    // validate resume vs fresh-start intent first (before loading any config)
    // this is the epoch to resume from if not a fresh start
    let resume_epoch = cli::resolve_from_checkpoint(&model_path, resume)
        .map_err(|e| { cli::display_train_cli_help(); e })?; // show help before returning error

    let persisted_config = if resume_epoch.is_some() {
        let config_path = model_path.join("learner_config.json");
        Some(VsrmLearnerConfig::load(&config_path)
            .map_err(|e| io_err(format!("Failed to load config for resume: {}", e), ErrorKind::InvalidData))?)
    } else { None };

    // persist these values across next run
    let keep_all_checkpoints = cli::resolve_keep_all_checkpoints(persisted_config.as_ref(), keep_all_checkpoints_cli)?;
    let active_subset = cli::resolve_active_subset(persisted_config.as_ref(), active_subset, seed)?;
    let dataset_src = cli::resolve_dataset_source(persisted_config.as_ref(), dataset)?;

    let learner_config = VsrmLearnerConfig::new(
        model_id,
        dataset_src,
        optimizer_config,
    )
        .with_resume_from(resume_epoch)
        .with_keep_all_checkpoints(keep_all_checkpoints)
        .with_frame_dims(frame_dims)
        .with_num_epochs(num_epochs)
        .with_batch_size(batch_size)
        .with_learning_rate(learning_rate)
        .with_num_workers(num_workers)
        .with_accumulation(accumulation)
        .with_seed(seed)
        .with_active_subset(active_subset);

    train::<TrainBackend>(
        device,
        context,
        dataset_src,
        model_config,
        learner_config,
        token_map.as_ref().clone(),
    )?;

    Ok(())
}



/// Loads a trained VSRM and runs inference in file mode or live webcam mode.
///
/// Loads `model_config` and `learner_config` from model dir (hard block when missing), builds `predictor_config`
/// with receptive field and frame dims from `learner_config`, then delegates to `infer()`.
///
/// ### Params:
/// - `context`: Filesystem context.
/// - `model_id`: Model directory name (required).
/// - `input`: Video file path for file mode; `None` for live.
/// - `live`: Whether to run live webcam mode.
/// - `camera`: Camera device ID (when live).
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or an error on inference failure.
fn run_infer_vsrm(
    context: &Context,
    model_id: &str,
    input: Option<&std::path::Path>,
    live: bool,
    camera: i32,
    token_map: &Arc<TokenMap>,
) -> Result<(), ESS> {
    if !live && input.is_none() {
        return Err(io_err("Must specify either `--input [path/to/video.mpg]` or `--live` for inference", ErrorKind::InvalidInput));
    }

    let model_path = context.models_path.join(model_id);
    if !model_path.exists() {
        return Err(io_err(format!("Model directory not found: {:?}", model_path), ErrorKind::NotFound));
    }

    let model_config_path = model_path.join("model_config.json");
    let learner_config_path = model_path.join("learner_config.json");
    let norm_stats_path = model_path.join("norm_stats.json");

    // load learner/model configs and norm stats
    let model_config = VsrModelConfig::load(&model_config_path)
        .map_err(|e| io_err(format!("Failed to load model_config.json: {}", e), ErrorKind::InvalidData))?;
    let learner_config = VsrmLearnerConfig::load(&learner_config_path)
        .map_err(|e| io_err(format!("Failed to load learner_config.json: {}", e), ErrorKind::InvalidData))?;
    let norm_stats = load_json(&norm_stats_path)
        .map_err(|e| io_err(format!("Failed to load norm_stats.json: {}", e), ErrorKind::InvalidData))?;

    let model_id = model_id.to_string();
    let frame_dims = learner_config.frame_dims;
    let rf = learner_config.rf;
    let stride = 10;
    let search_type = CtcDecodeType::GreedySearch;
    let device = DefaultDevice;

    let predictor_config = VsrmPredictorConfig::new(model_id.to_string())
        .with_frame_dims(frame_dims)
        .with_rf_window_size(rf)
        .with_rf_window_stride(stride)
        .with_search_type(search_type);

    infer::<InferBackend>(
        device,
        context,
        &model_path,
        model_config,
        predictor_config,
        norm_stats,
        token_map.as_ref().clone(),
        input,
        camera,
    )?;

    Ok(())
}



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
    dataset_src: DatasetSource,
    token_map: &Arc<TokenMap>,
) -> Result<(), ESS> {
    let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
        context.models_path.join("haarcascade_frontalface_alt2.xml"),
        context.models_path.join("haarcascade_mcs_mouth.xml"),
        (50, 100),
    ));

    match dataset_src {
        DatasetSource::Grid => {
            let grid_dataset = GridDataset::new(
                context,
                token_map.as_ref().clone(),
                Some(tracker_config),
                None,
            );
            grid_dataset.preprocess_all();
        }
        // DatasetSource::Lrw => {}  // stubbed for future
    }

    Ok(())
}
