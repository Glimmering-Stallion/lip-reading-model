//! Training CLI resolution helpers.
//!
//! Resolves user intent from CLI flags (resume, keep-all-checkpoints) and persisted config.



use crate::{
    pipeline::{
        io::file_nonempty,
        DatasetSource,
    },
    prelude::*,
    training::VsrmLearnerConfig,
};
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};



/// Finds whether a valid checkpoint exists for the given epoch.
///
/// Burn saves checkpoints under `model_path/checkpoint/`.
///
/// ### Params:
/// - `model_path`: Path to the model directory (parent of `checkpoint/`).
/// - `epoch`: Target epoch of interest.
///
/// ### Returns:
/// A bool for whether a valid checkpoint exists or not.
pub fn checkpoint_epoch_exists(model_path: &Path, epoch: usize) -> bool {
    let checkpoint_dir = model_path.join("checkpoint");

    let model_ok = checkpoint_dir.join(format!("model-{}.mpk.gz", epoch)).exists()
        || checkpoint_dir.join(format!("model-{}.mpk", epoch)).exists()
        || checkpoint_dir.join(format!("model-{}.bin", epoch)).exists();
    let optim_ok = checkpoint_dir.join(format!("optim-{}.mpk.gz", epoch)).exists()
        || checkpoint_dir.join(format!("optim-{}.mpk", epoch)).exists()
        || checkpoint_dir.join(format!("optim-{}.bin", epoch)).exists();
    let sched_ok = checkpoint_dir.join(format!("scheduler-{}.mpk.gz", epoch)).exists()
        || checkpoint_dir.join(format!("scheduler-{}.mpk", epoch)).exists()
        || checkpoint_dir.join(format!("scheduler-{}.bin", epoch)).exists();

    model_ok && optim_ok && sched_ok
}


/// Scans a path to a saved VSRM model for Burn checkpoint files of the latest epoch.
///
/// Burn's FileCheckpointer saves under a `checkpoint/` subdirectory:
/// - model-{epoch}.mpk,
/// - optim-{epoch}.mpk,
/// - scheduler-{epoch}.mpk (or .mpk.gz).
///
/// We require all three to exist for a valid checkpoint.
/// 
/// For example, if training has completed epoch 3, but stopped mid-epoch 4,
/// then resuming training starts from epoch 4.
///
/// ### Params:
/// - `model_path`: Path to the model directory (parent of `checkpoint/`).
///
/// ### Returns:
/// `Some(epoch)` if a valid checkpoint exists.
pub fn find_latest_checkpoint_epoch(model_path: &Path) -> Option<usize> {
    let checkpoint_dir = model_path.join("checkpoint");
    let dir = fs::read_dir(&checkpoint_dir).ok()?;

    let mut max_epoch: u64 = 0;
    for entry in dir.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if let Some(stem) = name.strip_prefix("model-") {
            let epoch_str = stem
                .strip_suffix(".mpk.gz")
                .or_else(|| stem.strip_suffix(".mpk"))
                .or_else(|| stem.strip_suffix(".bin"));
            if let Some(ep) = epoch_str {
                if let Ok(epoch) = ep.parse::<u64>() {
                    if checkpoint_epoch_exists(model_path, epoch as usize)
                    && (epoch > max_epoch) { max_epoch = epoch; }
                }
            }
        }
    }

    if max_epoch > 0 { Some(max_epoch as usize) }
    else { None }
}



/// Resolves which checkpoint epoch to resume from (or `None` for fresh start).
///
/// - **Resume** (`resume_from` is `Some`): Validates model path exists and checkpoint exists; exits on error.
/// - **Fresh start** (`resume_from` is `None`): Validates model path does not exist; exits if it does.
pub fn resolve_from_checkpoint(
    model_path: &Path,
    resume_from: Option<Option<usize>>,
) -> Result<Option<usize>, ESS> {
    // case A: resume flag given --> user explicitly requested a resume
    // - if provided/default model ID exists
    //      - if resume flag carries a value
    //           - if value is valid --> resume from that epoch value
    //           - if value is invalid --> error
    //      - if no value carried --> resume from latest epoch value
    // - if provided/default model ID doesn't exist --> error
    // case B: resume flag not given --> user wants a fresh start
    // - if provided/default model ID exists --> error
    // - if provided/default model ID doesn't exist --> proceed fresh
    match resume_from {
        // --------------- (A) ---------------
        Some(request) => {
            if !model_path.exists() { Err(io_err(format!("cannot resume: model directory {:?} does not exist\n", model_path), ErrorKind::InvalidInput)) }
            else {
                let epoch = match request {
                    Some(epoch) => {
                        if checkpoint_epoch_exists(model_path, epoch) { epoch }
                        else { return Err(io_err(format!("cannot resume: checkpoint for epoch {} not found in {:?}\n", epoch, model_path), ErrorKind::InvalidInput)); }
                    },
                    None => find_latest_checkpoint_epoch(model_path).ok_or_else(|| io_err(format!("cannot resume: no valid checkpoints found in {:?}\n", model_path), ErrorKind::InvalidInput))?,
                };
                println!("Resuming from checkpoint epoch {}\n", epoch);
                Ok(Some(epoch))
            }
        }
        // --------------- (B) ---------------
        None => {
            if !model_path.exists() {
                println!("Training new model: {:?}\n", model_path);
                Ok(None)
            } else {
                Err(io_err(format!(
                    "ERROR: Model directory {:?} already exists!\n
                    To prevent accidental data loss, training has been aborted\n",
                    model_path
                ), ErrorKind::InvalidInput))
            }
        }
    }
}



/// Resolves `keep_all_checkpoints` from CLI and persisted config.
///
/// - **Resume** (model path exists): Loads `learner_config.json`; if CLI is `None`, uses persisted value; otherwise uses CLI value and prints update/same messages.
/// - **Fresh start**: CLI `None` → false; `Some(Some("on"))` or `Some(None)` → true; `Some(Some("off"))` → false.
pub fn resolve_keep_all_checkpoints(
    learner_config: Option<&VsrmLearnerConfig>,
    keep_all_checkpoints: Option<Option<&str>>,
) -> Result<bool, ESS> {
    // resolve provided CLI arg
    let cli_value = match keep_all_checkpoints {
        None => None,                                         // flag not provided
        Some(None) => Some(true),                             // flag provided without value (implicitly true)
        Some(Some(val)) => Some(parse_cli_bool(val)?),  // flag provided with value (parse value validity)
    };

    match learner_config {
        // fresh start case (no config with previous value to persist)
        None => Ok(cli_value.unwrap_or(false)),

        // resume case (persist previous value or update with newly provided CLI arg value)
        Some(config) => {
            let persisted_val = config.keep_all_checkpoints;
            match  cli_value {
                None => Ok(persisted_val),  // implicitly default to previous value
                Some(val) => {        // use newly provided value
                    let status = if val { "on" } else { "off" };
                    if val == persisted_val { println!("'keep-all-checkpoints' flag already {}\n", status); }  // provided value is same as previous value
                    else { println!("'keep-all-checkpoints' flag updated to {}\n", status); }                  // provided value is diff from previous value
                    Ok(val)
                },
            }
        }
    }
}



/// Resolves `dataset_src` from CLI and persisted config.
///
/// - **Resume** (model path exists): Loads `learner_config.json`; if persisted value exists,
///   validates that the CLI value matches it (panics on mismatch). Falls back to CLI value if
///   no persisted config is found.
/// - **Fresh start** (model path doesn't exist): Uses the CLI value directly.
pub fn resolve_dataset_source(
    learner_config: Option<&VsrmLearnerConfig>,
    dataset_src: Option<DatasetSource>,
) -> Result<DatasetSource, ESS> {
    match learner_config {
        // fresh start case (no config with previous value to persist)
        None => dataset_src.ok_or_else(|| io_err("a '--dataset' source is required when starting a new model from scratch", ErrorKind::InvalidInput)),

        // resume case (persist previous value or update with newly provided CLI arg value)
        Some(config) => {
            let persisted_val = config.dataset_src;

            match dataset_src {
                None => Ok(persisted_val),     // implicitly default to previous value
                Some(val) => {  // use newly provided value
                    // provided value is same as previous value
                    if val.tag() == persisted_val.tag() { Ok(persisted_val) }

                    // provided value is diff from previous value
                    else {
                        Err(io_err(format!(
                            "dataset mismatch: model was trained on '{}' but '--dataset {}' was specified",
                            persisted_val.tag(), val.tag(),
                        ), ErrorKind::InvalidInput))
                    }
                },
            }
        }
    }
}



/// Resolves `subset` from CLI and persisted config.
///
/// - **Resume** (model path exists): Loads `learner_config.json`; if CLI is `None`, uses persisted value; otherwise uses CLI value.
/// - **Fresh start**: CLI `None` → None (full dataset); `Some(pct)` → Some((pct, seed)).
pub fn resolve_active_subset(
    learner_config: Option<&VsrmLearnerConfig>,
    subset: Option<f32>,
    seed: u64,
) -> Result<Option<(f32, u64)>, ESS> {
    // resolve provided CLI arg
    let cli_value = subset.map(|pct| (pct, seed));

    match learner_config {
        // fresh start case (no config with previous value to persist)
        None => Ok(subset.map(|pct| (pct, seed))),

        // resume case (persist previous value or update with newly provided CLI arg value)
        Some(config) => {
            let persisted_val = config.active_subset;

            match cli_value {
                None => Ok(persisted_val),  // implicitly default to previous value
                Some(val) => {  // use newly provided value
                    // provided value is same as previous value
                    if Some(val) == persisted_val {
                        println!("'subset' flag remains {:?}\n", val);
                        Ok(persisted_val)
                    }

                    // provided value is diff from previous value
                    else {
                        Err(io_err(format!(
                            "subset mismatch: cannot change data distribution during resume\n
                            persisted: {:?}, requested: {:?}",
                            persisted_val, val,
                        ), ErrorKind::InvalidInput))
                    }
                },
            }
        },
    }
}



/// Resolves an inference `--input` bundle path to the concrete video file contained inside it.
///
/// - `path` must be a **directory** (bundled video-transcript layout), prefers
///   `<sample_dir>/<sample_id>.mp4`, where `sample_id` is the directory name.
///
/// ### Params:
/// - `path`: User-supplied `--input` bundle path.
///
/// ### Returns:
/// Two paths to an existing, non-empty `.mp4` and `.txt` file, or an error.
pub fn resolve_inference_input(path: &Path) -> Result<(PathBuf, PathBuf), ESS> {
    if !path.exists() {
        return Err(io_err(
            format!("inference input bundle path does not exist: {:?}", path),
            ErrorKind::NotFound,
        ));
    }

    if path.is_dir() {
        let stem = path
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| { io_err("inference input directory has an invalid name", ErrorKind::InvalidInput) })?;

        let mp4 = path.join(stem).with_extension("mp4");
        let txt = path.join(stem).with_extension("txt");

        if file_nonempty(&mp4)
        && file_nonempty(&txt)
        { return Ok((mp4, txt)); }

        return Err(io_err(
            format!("expected video file at {:?} and transcript file at {:?}", mp4, txt),
            ErrorKind::NotFound,
        ));
    }

    Err(io_err(
        format!("inference input path is not a bundled video-transcript directory: {:?}", path),
        ErrorKind::InvalidInput,
    ))
}



/// Prints training CLI input options to `stderr` (to guide user intent).
///
/// Note: Checkpoints are saved at the end of each epoch. If training stops mid-epoch N,
/// the last completed checkpoint is epoch (N - 1). Use `--resume` or `--resume [N - 1]` to continue.
pub fn display_train_cli_help() {
    eprintln!(
        "Training options:\n\
        - To resume from latest checkpoint: use '--resume'\n\
        - To resume from specified checkpoint (requires '--keep-all-checkpoints on'): use '--resume [epoch]'\n\
        - To keep all checkpoints (enables resume from earlier epochs): use '--keep-all-checkpoints [on|off]' (default when passed: on)\n\
        - To start fresh: manually delete the model directory or use another model ID in '--model [model ID]'\n"
    );
}



/// Parses given raw CLI strings into a strict boolean, while rejecting invalid inputs.
/// 
/// Considers three case insensitive input string variations for valid bool parsing:
/// - "on"/"off",
/// - "true"/"false",
/// - "1"/"0",
/// - anything else is an error.
/// 
/// ### Params:
/// - `val`: String value to parse.
/// 
/// ## Returns:
/// A result wrapper containing the raw bool value under valid conditions.
fn parse_cli_bool(val: &str) -> Result<bool, ESS> {
    match val.to_lowercase().as_str() {
        ("on" | "true" | "1") => Ok(true),
        ("off" | "false" | "0") => Ok(false),
        _ => Err(io_err(format!(
            "Invalid boolean flag value: '{}'\n
            Expected on/off, true/false, or 1/0.", val
        ), ErrorKind::InvalidInput)),
    }
}
