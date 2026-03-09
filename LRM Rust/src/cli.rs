//! Training CLI resolution helpers.
//!
//! Resolves user intent from CLI flags (resume, keep-all-checkpoints) and persisted config.



use crate::training::VsrmLearnerConfig;
use burn::config::Config;
use std::{fs, path::Path, process};



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
) -> Option<usize> {
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
            if model_path.exists() {
                let epoch = match request {
                    Some(epoch) => {
                        if checkpoint_epoch_exists(model_path, epoch) {
                            epoch
                        } else {
                            eprintln!("Cannot resume: checkpoint for epoch {} not found in {:?}\n", epoch, model_path);
                            display_train_cli_help();
                            process::exit(1);
                        }
                    }
                    None => match find_latest_checkpoint_epoch(model_path) {
                        Some(epoch) => epoch,
                        None => {
                            eprintln!("Cannot resume: no saved checkpoints found in {:?}\n", model_path);
                            display_train_cli_help();
                            process::exit(1);
                        }
                    },
                };
                println!("Resuming from checkpoint epoch {}\n", epoch);
                Some(epoch)
            } else {
                eprintln!("Cannot resume: model directory {:?} does not exist\n", model_path);
                display_train_cli_help();
                process::exit(1);
            }
        }
        // --------------- (B) ---------------
        None => {
            if model_path.exists() {
                eprintln!(
                    "ERROR: Model directory {:?} already exists!\n
                    To prevent accidental data loss, training has been aborted\n",
                    model_path
                );
                display_train_cli_help();
                process::exit(1);
            } else {
                println!("Training new model: {:?}\n", model_path);
                None
            }
        }
    }
}



/// Resolves `keep_all_checkpoints` from CLI and persisted config.
///
/// - **Resume** (model path exists): Loads `learner_config.json`; if CLI is `None`, uses persisted value; otherwise uses CLI value and prints update/same messages.
/// - **Fresh start**: CLI `None` → false; `Some(Some("on"))` or `Some(None)` → true; `Some(Some("off"))` → false.
pub fn resolve_keep_all_checkpoints(
    model_path: &Path,
    keep_all_checkpoints: Option<Option<&str>>,
) -> bool {
    if model_path.exists() {
        let persisted_learner_config = VsrmLearnerConfig::load(model_path.join("learner_config.json")).ok();
        let current_learner_config = persisted_learner_config.as_ref().map(|c| c.keep_all_checkpoints).unwrap_or(false);

        match keep_all_checkpoints {
            None => current_learner_config,
            Some(Some(val)) => {
                let val_bool = matches!(val.to_lowercase().as_str(), "on" | "true" | "1");
                if val_bool == current_learner_config {
                    println!("'keep-all-checkpoints' flag already {}\n", if val_bool { "on" } else { "off" });
                } else {
                    println!("'keep-all-checkpoints' flag updated to {}\n", if val_bool { "on" } else { "off" });
                }
                val_bool
            }
            Some(None) => {
                if !current_learner_config {
                    println!("'keep-all-checkpoints' flag updated to on\n");
                }
                true
            }
        }
    } else {
        match keep_all_checkpoints {
            None => false,
            Some(Some(val)) => matches!(val.to_lowercase().as_str(), "on" | "true" | "1"),
            Some(None) => true,
        }
    }
}



/// Resolves `active_subset` from CLI and persisted config.
///
/// - **Resume** (model path exists): Loads `learner_config.json`; if CLI is `None`, uses persisted value; otherwise uses CLI value.
/// - **Fresh start**: CLI `None` → None (full dataset); `Some(pct)` → Some((pct, seed)).
pub fn resolve_active_subset(
    model_path: &Path,
    subset_cli: Option<f32>,
    seed: u64,
) -> Option<(f32, u64)> {
    if model_path.exists() {
        let persisted = VsrmLearnerConfig::load(model_path.join("learner_config.json")).ok();
        let current = persisted.and_then(|c| c.active_subset);
        match subset_cli {
            None => current,
            Some(pct) => Some((pct, seed)),
        }
    } else {
        subset_cli.map(|pct| (pct, seed))
    }
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
