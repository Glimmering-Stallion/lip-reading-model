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

// build N-gram LM (train if missing, else load and eval):                                                                   cargo run -- build-lm --model [lm.bin] --corpus [path/to/corpus] --n [n_gram_order]
// preprocess a specific dataset for the VSRM::                                                                              cargo run -- preprocess --dataset [dataset_src]
// train new VSRM with default model ID `vsrm_<dataset_src>` on specified dataset (error if ID alr exists):                 cargo run -- train --dataset [dataset_src]
// train new VSRM with custom model ID on specified dataset (error if ID exists):                                            cargo run -- train --model [model_id] --dataset [dataset_src]
// resume training from latest checkpoint (uses last completed epoch):                                                       cargo run -- train [...] --resume
// resume training from specified checkpoint:                                                                                cargo run -- train [...] --resume [epoch]
// train using a subset of the dataset (e.g. fraction = 0.1 for 10%):                                                        cargo run -- train [...] --subset [fraction]
// toggle keep-all-checkpoints during training (default: keep most recent only):                                             cargo run -- train [...] --keep-all-checkpoints [on|off]
// run inference on a video file or bundled video-transcript dir (requires --model):                                         cargo run -- infer --model [model_id] --input [path/to/video.mpg|.../bundled_dir]
// run real-time live inference from default webcam:                                                                         cargo run -- infer --model [model_id] --live
// run real-time live inference from a specific camera index (OpenCV device id):                                             cargo run -- infer --model [model_id] --live [device_id]
// export model ONNX and TeX bundle to default output path (exports/<model_id>_export/{onnx,tex}/):                          cargo run -- export --model [model_id]
// export model ONNX and TeX bundle to specified output path:                                                                cargo run -- export --model [model_id] --output [path/to/output]

// embed Info.plist on macOS so AVFoundation uses AVCaptureDeviceTypeContinuityCamera
// and does not emit the AVCaptureDeviceTypeExternal deprecation warning
// (e.g. when using OpenCV camera capture)
#[cfg(target_os = "macos")]
embed_plist::embed_info_plist!("../Info.plist");

// imports
use burn::{
    backend::{
        Autodiff,
        Wgpu,
        wgpu::WgpuDevice::DefaultDevice,
    },
    config::Config,
    grad_clipping::GradientClippingConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
};
use clap::{Parser, Subcommand};
use lrm_rust::{
    cli::eprint_python_export_failure,
    ctc::lm::LanguageModel,
    pipeline::{
        DatasetSource,
        adapters::grid,
        tracker::{HaarTrackerConfig, TrackerConfig},
    },
    prelude::*,
    vocab::SPACE_ID,
};
use std::{
    fs,
    env,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
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

        /// Video file path, or bundled video-transcript directory
        #[arg(long, conflicts_with = "live")]
        input: Option<PathBuf>,

        /// Live webcam inference. Optional device index (default 0). Mutually exclusive with `--input`.
        #[arg(long, conflicts_with = "input", num_args = 0..=1, value_name = "DEVICE_INDEX")]
        live: Option<Option<usize>>,
    },
    /// Export a model ONNX and TeX bundle for vizualization purposes: `onnx/vsrm_export.onnx` + PlotNeuralNet `tex/` (`vsrm_export.tex` + `layers/`).
    Export {
        #[arg(long)]
        model: String,

        /// Bundle root path (default: `exports/<model_id>_export/`). ONNX → `<bundle>/onnx/`, TeX → `<bundle>/tex/`.
        #[arg(long, value_name = "DIR")]
        output: Option<PathBuf>,

        #[arg(long, default_value_t = 17)]
        opset: u32,

        #[arg(long, default_value_t = 96)]
        time_steps: u32,
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
    println!("Blank token ID: {:?}", BLANK_ID);
    println!("Blank token char: {:?}", VOCAB.chars().nth(BLANK_ID).unwrap());
    println!("Space token ID: {}", SPACE_ID);
    println!("Space token char: {:?}\n", VOCAB.chars().nth(SPACE_ID).unwrap());
    assert!(BLANK_ID < VOCAB_SIZE, "blank ID ({}) is out of vocabulary size bounds ({})", BLANK_ID, VOCAB_SIZE);

    // CLI control flow
    match &args.command {
        Command::BuildLm {
            corpus,
            model,
            n,
        } => {
            run_build_lm(
                &context,
                corpus.as_deref(),
                model,
                *n,
                &token_map,
            )?;
        }
        Command::Train {
            model,
            dataset,
            resume,
            subset,
            keep_all_checkpoints,
        } => {
            run_train_vsrm(
                &context,
                model.as_deref(),
                dataset.clone(),
                *resume,
                *subset,
                keep_all_checkpoints.as_ref().map(|o| o.as_deref()),
                &token_map,
            )?;
        }
        Command::Infer {
            model,
            input,
            live,
        } => {
            run_infer_vsrm(
                &context,
                model,
                input.as_deref(),
                *live,
                &token_map,
            )?;
        }
        Command::Preprocess { dataset } => {
            run_preprocess(
                &context,
                *dataset,
                &token_map,
            )?;
        }
        Command::Export {
            model,
            output,
            opset,
            time_steps,
        } => {
            run_export_vsrm(
                &context,
                model,
                output.as_deref(),
                *opset,
                *time_steps,
            )?;
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
/// `Ok(())` on success, or [`ESS`] on I/O or LM build failure.
fn run_build_lm(
    context: &Context,
    corpus: Option<&str>,
    model: &str,
    n: usize,
    token_map: &Arc<TokenMap>,
) -> Result<(), ESS> {
    extract_slr_corpus(&context.rust_root);

    let corpus_path = corpus.map(PathBuf::from).unwrap_or_else(|| {
        context
            .data_path
            .join("librispeech-lm-norm")
            .join("librispeech-lm-norm.txt")
    });

    if !corpus_path.exists()
    { return Err(format!("corpus path {:?} does not exist", corpus_path).into()); }

    let lm_output_path = context.models_path.join(model);
    let corpus_str = corpus_path.to_string_lossy().to_string();
    let train_token_map = Arc::clone(token_map);
    let train_sequences: Vec<Vec<usize>> = if lm_output_path.exists() { vec![] }
    else {
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
/// - `model_id`: Optional model ID; if `None`, uses default.
/// - `resume`: Optional resume spec; `None` for fresh start, `Some(None)` for latest checkpoint, `Some(Some(epoch))` for specific epoch.
/// - `active_subset`: Optional fraction of dataset to use (e.g. 0.1 for 10%).
/// - `keep_all_checkpoints_cli`: Optional CLI override; `None` = use persisted, `Some(Some("on"))` = keep all checkpoints, `Some(Some("off"))` = keep most recent only.
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or [`ESS`] on training failure.
fn run_train_vsrm(
    context: &Context,
    model_id: Option<&str>,
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

    let model_id = model_id
        .map(String::from)
        .or_else(|| dataset.map(|d| format!("vsrm_{}", d.tag())))
        .ok_or_else(|| { io_err("train requires `--model` or `--dataset`", ErrorKind::InvalidInput) })?;

    let model_path = context.models_path.join(&model_id);
    let model_config = VsrModelConfig::new()
        .with_frame_dims(frame_dims)
        .with_vocab_size(vocab_size)
        .with_blank_id(blank_id);

    // validate resume vs fresh-start intent first (before loading any config)
    // this is the epoch to resume from if not a fresh start
    let resume_epoch = resolve_from_checkpoint(&model_path, resume)
        .map_err(|e| { display_train_cli_help(); e })?; // show help before returning error

    let persisted_config = if resume_epoch.is_some() {
        let config_path = model_path.join("learner_config.json");
        Some(VsrmLearnerConfig::load(&config_path)
            .map_err(|e| { io_err(format!("failed to load config for resume: {}", e), ErrorKind::InvalidData) })?)
    } else { None };

    // persist these values across next run
    let keep_all_checkpoints = resolve_keep_all_checkpoints(persisted_config.as_ref(), keep_all_checkpoints_cli)?;
    let active_subset = resolve_active_subset(persisted_config.as_ref(), active_subset, seed)?;
    let dataset_src = resolve_dataset_source(persisted_config.as_ref(), dataset)?;

    let learner_config = VsrmLearnerConfig::new(model_id, dataset_src, optimizer_config)
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
        &model_config,
        &learner_config,
        token_map.as_ref(),
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
/// - `input`: Video file path or bundled GRID utterance directory for file mode; `None` for live.
/// - `live`: `None` for file mode; `Some(None)` for default webcam; `Some(Some(i))` for OpenCV camera index `i`.
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or [`ESS`] on inference failure.
fn run_infer_vsrm(
    context: &Context,
    model_id: &str,
    input: Option<&Path>,
    live: Option<Option<usize>>,
    token_map: &Arc<TokenMap>,
) -> Result<(), ESS> {
    // error if both or neither are provided
    if live.is_some() == input.is_some() { return Err(io_err("must specify either `--input [path/to/bundled_dir]` or `--live [device_index]` for inference", ErrorKind::InvalidInput)); }

    let camera: i32 = match live {
        None => 0,
        Some(None) => 0,
        Some(Some(n)) => i32::try_from(n)
            .map_err(|_| { io_err("camera device index is out of range for this platform", ErrorKind::InvalidInput) })?,
    };

    let model_path = context.models_path.join(model_id);
    if !model_path.exists() { return Err(io_err( format!("model directory not found: {:?}", model_path), ErrorKind::NotFound)); }

    let model_config_path = model_path.join("model_config.json");
    let learner_config_path = model_path.join("learner_config.json");
    let norm_stats_path = model_path.join("norm_stats.json");

    // load learner/model configs and norm stats
    let model_config = VsrModelConfig::load(&model_config_path)
        .map_err(|e| { io_err(format!("failed to load model_config.json: {}", e), ErrorKind::InvalidData) })?;
    let learner_config = VsrmLearnerConfig::load(&learner_config_path)
        .map_err(|e| { io_err(format!("failed to load learner_config.json: {}", e), ErrorKind::InvalidData) })?;
    let norm_stats = load_json(&norm_stats_path)
        .map_err(|e| { io_err(format!("failed to load norm_stats.json: {}", e), ErrorKind::InvalidData) })?;

    let model_id = model_id.to_string();
    let frame_dims = learner_config.frame_dims;
    let rf = learner_config.rf;
    let search_type = CtcDecodeType::GreedySearch;
    let device = DefaultDevice;

    let predictor_config = VsrmPredictorConfig::new(model_id.to_string())
        .with_frame_dims(frame_dims)
        .with_temporal_window(rf)
        .with_search_type(search_type);

    let session = InferenceSession::<InferBackend>::new(
        device,
        &model_path,
        &model_config,
        &predictor_config,
        norm_stats,
        token_map.as_ref().clone(),
    )?;

    infer::<InferBackend>(
        session,
        context,
        &predictor_config,
        input,
        camera,
    )?;

    Ok(())
}



/// Runs all exporters (`export_onnx.py`, `export_tex.py`) into a single bundle directory.
///
/// - ONNX export requires Python with torch and onnx installed (`pip install -r tools/requirements.txt`, or set `PYTHON` to that interpreter).
/// - TeX export requires a vendored [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) clone at `tools/plotneuralnet/`.
///
/// Layout: `<bundle>/onnx/vsrm_export.onnx` and `<bundle>/tex/` (`vsrm_export.tex` + synced PlotNeuralNet `layers/`).
/// Default bundle: `exports/<model_id>_export/`. Optional `--output` sets the bundle root instead.
///
/// ### Params:
/// - `context`: Filesystem context for paths.
/// - `model_id`: Model directory name (required).
/// - `output`: Optional bundle root destination path; if `None`, defaults to `exports/<model_id>_export/`.
/// - `opset`: ONNX opset version (default 17).
/// - `time_steps`: Trace sequence length `T` for ONNX (default 96).
/// 
/// ### Returns:
/// `Ok(())` on success, or [`ESS`] if export fails or prerequisites are not met.
fn run_export_vsrm(
    context: &Context,
    model_id: &str,
    output: Option<&Path>,
    opset: u32,
    time_steps: u32,
) -> Result<(), ESS> {
    let model_path = context.models_path.join(model_id);
    if !model_path.is_dir() { return Err(io_err(format!("model directory not found: {:?}", model_path), ErrorKind::NotFound)); }

    let model_config_path = model_path.join("model_config.json");
    if !model_config_path.is_file() { return Err(io_err(format!("missing model_config.json in {:?}", model_path), ErrorKind::NotFound)); }

    let bundle_dir = if let Some(out) = output {
        let p = if out.is_absolute() { out.to_path_buf() }
        else {
            env::current_dir()
                .map_err(|e| io_err(format!("current_dir: {}", e), ErrorKind::Other))?
                .join(out)
        };
        if p.extension().and_then(|e| e.to_str()).is_some_and(|ext| ext.eq_ignore_ascii_case("onnx")) {
            return Err(io_err(
                "export --output must name a bundle directory, not a .onnx file; onnx is written to bundle/onnx/vsrm_export.onnx",
                ErrorKind::InvalidInput,
            ));
        }
        p
    } else { context.exports_path.join(format!("{model_id}_export")) };

    fs::create_dir_all(&bundle_dir)
        .map_err(|e| { io_err(format!("failed to create export bundle directory {:?}: {}", bundle_dir, e), ErrorKind::Other) })?;

    let onnx_dir = bundle_dir.join("onnx");
    fs::create_dir_all(&onnx_dir)
        .map_err(|e| io_err(format!("failed to create onnx export directory {:?}: {}", onnx_dir, e), ErrorKind::Other))?;
    let onnx_path = onnx_dir.join("vsrm_export.onnx");

    let tex_dir = bundle_dir.join("tex");

    let py = env::var("PYTHON").unwrap_or_else(|_| {
        if cfg!(windows) { "python".to_string() }
        else { "python3".to_string() }
    });

    let model_dir = model_path.canonicalize()
        .map_err(|e| { io_err(format!("model path canonicalize: {}", e), ErrorKind::InvalidInput) })?;

    let mut failures: Vec<String> = Vec::new();

    let onnx_tools = context.rust_root.join("tools/onnx_export");
    let onnx_script = onnx_tools.join("export_onnx.py");
    if !onnx_script.is_file() {
        let msg = format!("export_onnx.py not found: {:?}", onnx_script);
        eprintln!("[export] ONNX: {msg}");
        failures.push(format!("ONNX: {msg}"));
    } else {
        let mut cmd = process::Command::new(&py);
        cmd.current_dir(&onnx_tools);
        cmd.stdout(process::Stdio::piped());
        cmd.stderr(process::Stdio::piped());
        cmd.arg("export_onnx.py");
        cmd.arg("--model-dir");
        cmd.arg(model_dir.as_os_str());
        cmd.arg("--opset");
        cmd.arg(opset.to_string());
        cmd.arg("--time-steps");
        cmd.arg(time_steps.to_string());
        cmd.arg("--output");
        cmd.arg(onnx_path.as_os_str());

        match cmd.output() {
            Ok(out) if out.status.success() => println!("[export] ONNX: ok → {}", onnx_path.display()),
            Ok(out) => {
                eprint_python_export_failure("ONNX", &out);
                let msg = format!("export_onnx.py exited with {}", out.status);
                failures.push(format!("ONNX: {msg}"));
            }
            Err(e) => {
                let msg = format!("failed to spawn {py}: {e}");
                eprintln!("[export] ONNX: {msg}");
                failures.push(format!("ONNX: {msg}"));
            }
        }
    }

    let tex_tools = context.rust_root.join("tools/tex_export");
    let tex_script = tex_tools.join("export_tex.py");
    if !tex_script.is_file() {
        let msg = format!("export_tex.py not found: {:?}", tex_script);
        eprintln!("[export] TeX: {msg}");
        failures.push(format!("TeX: {msg}"));
    } else {
        let mut cmd = process::Command::new(&py);
        cmd.current_dir(&tex_tools);
        cmd.stdout(process::Stdio::piped());
        cmd.stderr(process::Stdio::piped());
        cmd.arg("export_tex.py");
        cmd.arg("--model-dir");
        cmd.arg(model_dir.as_os_str());
        cmd.arg("--output-dir");
        cmd.arg(tex_dir.as_os_str());

        match cmd.output() {
            Ok(out) if out.status.success() => {
                print!("{}", String::from_utf8_lossy(&out.stdout));
                println!(
                    "[export] TeX: ok → {} (vsrm_export.tex + layers/)",
                    tex_dir.display()
                );
            }
            Ok(out) => {
                eprint_python_export_failure("TeX", &out);
                let msg = format!("export_tex.py exited with {}", out.status);
                failures.push(format!("TeX: {msg}"));
            }
            Err(e) => {
                let msg = format!("failed to spawn {py}: {e}");
                eprintln!("[export] TeX: {msg}");
                failures.push(format!("TeX: {msg}"));
            }
        }
    }

    println!("Export bundle: {}", bundle_dir.display());
    println!("ONNX: {}", onnx_path.display());
    println!("TeX: {}", tex_dir.display());

    if failures.is_empty() { Ok(()) }
    else { Err(io_err(format!("export finished with {} error(s): {}", failures.len(), failures.join("; ") ), ErrorKind::Other)) }
}



/// Pre-extracts mouth crops to disk for faster training.
///
/// ### Params:
/// - `context`: Filesystem context for paths.
/// - `dataset`: Dataset name; only `"grid"` is supported.
/// - `token_map`: Bidirectional char-to-ID mapping.
///
/// ### Returns:
/// `Ok(())` on success, or [`ESS`] if dataset is unsupported.
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
            grid::align_grid_directories(context, false)?;
            grid::bundle_grid_utterances(context)?;
            grid::normalize_to_standard_formats(context)?;
            grid::clean_corpus(context, false)?;

            let grid_dataset = GridDataset::new(context, token_map.as_ref(), Some(tracker_config), None);
            grid_dataset.pre_extract_all();
        } // DatasetSource::Lrw => {}  // stubbed for future
    }

    Ok(())
}
