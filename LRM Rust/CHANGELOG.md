# Change Log Summaries

This document records all modifications to the project for future reference and onboarding.

## 1. Mouth Tracker Robustness ([`src/pipeline/tracker.rs`](src/pipeline/tracker.rs))

**Hierarchical detection pipeline:**

- Face detection → crop to lower half of face ROI → mouth detection → crop mouth ROI to target dims
- Replaces flat detection with a staged, more reliable pipeline

**ROI center smoothing:**

- Min threshold (detection noise): reject tiny movements as jitter
- Max threshold (detection glitch): reject large jumps as detection errors
- Kalman gating combined with EMA (Exponential Moving Average) for stable mouth position across frames

**Fault tolerance (recent fix):**

- When face detection fails: fall back to `prev_center` or frame center
- When mouth detection fails: fall back to `prev_center` (existing logic)
- Prevents single-frame failures from aborting entire video load

## 2. Model Architecture Overhaul

**New module:** [`src/vsrm/residual.rs`](src/vsrm/residual.rs)

- Custom ResBlock frontend layers (replacing vanilla Conv3D frontend)
- ResBlocks use GroupNorm after each Conv3D

**TCN backend:**

- Custom TCN blocks in [`src/vsrm/tcn.rs`](src/vsrm/tcn.rs)
- Added GroupNorm (8 groups default) after each Conv1d in TCN blocks
- Order: `conv → GroupNorm → ReLU → dropout`

**Feature downsampling:**

- Adaptive Average Pooling (AAP) after ResBlock layers
- Projection Linear layer: 8192 → 512 before TCN (was: 8192 → TCN directly)
- Eases spatial-to-temporal transition

**Initialization refinements:**

- FC layer: KaimingUniform → XavierUniform (gain 1.0) for raw logits
- Blank bias: set to +5.0 (or +3.0) to encourage initial blank predictions
- Removed redundant `activation::relu()` wrappers around ResBlock forward calls

## 3. Global Video Pixel Normalization

**Grid dataset** ([`src/pipeline/adapters/grid.rs`](src/pipeline/adapters/grid.rs)):

- `calc_global_stats()`: computes mean and std dev across all video frame pixels in the dataset
- Stats cached to `grid_stats.json` via JSON serialization (avoids ~4.5 hour recomputation)

**Dataset stats** ([`src/pipeline/dataset.rs`](src/pipeline/dataset.rs)):

- `DatasetStats` struct holds `mean` and `std_dev`
- Used for global normalization in the batcher

**IO helpers** ([`src/pipeline/io.rs`](src/pipeline/io.rs)):

- `load_json`, `save_json`: generic JSON serialization/deserialization for stats and configs

**Batcher** ([`src/pipeline/batcher.rs`](src/pipeline/batcher.rs)):

- Uses `DatasetStats` for global frame normalization when available
- Fallback: per-sample normalization (for tests / when stats absent)

## 4. Entropy Regularization (Disabled)

**Location:** [`src/training/learner.rs`](src/training/learner.rs)

- `calc_entropy_penalty()`: penalizes high-confidence (low-entropy) outputs
- Min entropy threshold discourages same-char/blank collapse early in training
- Currently commented out; can be re-enabled for experimentation

## 5. Model Summary Module ([`src/vsrm/summary.rs`](src/vsrm/summary.rs))

- `SummaryVisitor`: displays layer shapes and parameter counts (similar to TF `model.summary()`)
- Invoked at training start to log model architecture stats

## 6. Learning Rate Scheduler ([`src/training/learner.rs`](src/training/learner.rs))

- Replaced `StepLrScheduler` with `ComposedLrSchedulerConfig`
- Linear warmup (0.01 → 1.0 gain over first epoch) × Cosine annealing (lr → lr/10 over total steps)
- `SchedulerReduction::Prod`: multiplies outputs of both schedulers
- Warmup prevents destructive early updates that can cause CTC collapse

## 7. GRID Data Pipeline Robustness ([`src/pipeline/adapters/grid.rs`](src/pipeline/adapters/grid.rs))

**Problem:** Burn's `BatchDataloaderIterator` treats any `None` from `dataset.get()` as end-of-dataset, causing epochs to stop after the first failed item (~37 batches instead of ~2800).

**Fixes:**

- **`try_load(index)`**: Extracted loading logic; returns `None` on any failure (alignment, frames, CTC constraint)
- **`get(index)`**: Fallback loop—tries primary index, then adjacent entries (wrapping) until one succeeds; only returns `None` when `index >= len()`
- **Pre-validation in `new()`**: Parses alignment files upfront, filters entries with empty or invalid transcripts before training
- **Alignment filtering:** Excludes `"sp"` tokens (short pauses) in addition to `"sil"` when parsing `.align` files

**Result:** All ~11200 training items are now processed per epoch; LR warmup and cosine decay behave as intended.

# Change Logs Since Last Git Commit

## 8. CLI Subcommands ([`src/main.rs`](src/main.rs))

**Unified CLI with `clap` subcommands:**

- **`build-lm`**: Builds N-gram LM from corpus or loads existing LM. Options: `--corpus`, `--output`, `--n`. Trains if LM file missing; loads and evaluates perplexity if present.
- **`train`**: Trains VSRM from scratch or resumes from checkpoint. Options: `--model` (optional, defaults to `vsrm_grid`), `--resume` (no value = latest checkpoint, or `--resume N` for epoch N). Errors if model dir exists without `--resume`; errors if resuming but no checkpoint found.
- **`infer`**: Placeholder for inference. Options: `--model`, `--input`, `--output`. Currently a stub; `run_infer_vsrm` and `load_video_with_tracker` are commented out pending integration.

**Usage examples:**

```text
cargo run -- build-lm --corpus data/librispeech-lm-norm/librispeech-lm-norm.txt --output ngram_lm.bin --n 3
cargo run -- train --model vsrm_grid
cargo run -- train --model vsrm_grid --resume
cargo run -- train --model vsrm_grid --resume 5
cargo run -- infer --model vsrm_grid --input path/to/video.mpg --output pred.txt
```

## 9. Train New / Resume Logic ([`src/training/learner.rs`](src/training/learner.rs))

- **Resolve intent:** `resume_from: Some(None)` → latest checkpoint; `Some(Some(epoch))` → specific epoch; `None` → fresh start.
- **Validation:** `checkpoint_epoch_exists()`, `find_latest_checkpoint_epoch()` ensure model/optim/scheduler files exist before resume.
- **Errors:** Clear messages and `display_train_cli_help()` when resume fails or model dir exists without `--resume`.
- **Launch:** `trainer.checkpoint(epoch)` for resume; `fs::create_dir_all` + `trainer` for fresh.

## 10. Inference & Video Module (Stub / Placeholder)

**`src/pipeline/video.rs`** (commented out):

- `load_video_with_tracker()`: Loads video from path, runs LipTracker on each frame, returns grayscale mouth crops as `FramesBuffer`. Intended for inference on arbitrary videos. Not yet wired into pipeline (module not exported).

**`src/inference/predictor.rs`** (commented out):

- `run_infer_vsrm()`: Loads trained VSRM, N-gram LM, runs mouth tracking on input video, builds batch, forward pass, CTC beam decode, returns prediction string. Writes to file if `--output` given.
- `load_learner_config_frame_dims()`: Helper to read `frame_dims` from `learner_config.json`.

**Status:** Inference CLI subcommand exists but delegates to a no-op; predictor and video logic are ready to uncomment and wire once integration is finalized.

## 11. Documentation Improvements

- **`main.rs`**: Added structured doc comments for `run_build_lm()` and `run_train_vsrm()` with Params and Returns sections.
- **`lm.rs`**: Fixed typo "linquistic" → "linguistic" in module doc.
- **General:** Doc comments follow `///` style; module-level `//!` describe purpose. Key public functions document params, returns, and panics where relevant.

## 12. CLI Module and Refactors ([`src/cli.rs`](src/cli.rs))

**New `cli.rs` module** — training CLI resolution helpers:

- `checkpoint_epoch_exists()`: Validates checkpoint files exist for a given epoch.
- `find_latest_checkpoint_epoch()`: Scans model dir for latest checkpoint.
- `resolve_from_checkpoint()`: Resolves resume vs fresh-start intent; exits on invalid state.
- `resolve_keep_all_checkpoints()`: Resolves keep-all-checkpoints from CLI and persisted config.
- `display_train_cli_help()`: Prints training CLI options to stderr.

**keep-all-checkpoints toggleable:**

- `--keep-all-checkpoints` now accepts optional `[on|off]` value.
- On resume: no flag → use persisted value; flag with value → update and print change/same message.
- Persisted in `learner_config.json`.

**Other changes:**

- `levenshtein()` moved from `training/metrics.rs` to `utils.rs`.
- `VsrmLearnerConfig.model_id` changed from `Option<String>` to `String`; default applied once in `run_train_vsrm`.
- Doc comment consistency: `metrics.rs`, `ctc_loss.rs`, `learner.rs` use `### Params:` / `### Returns:` style.
- README Project Tree: added `cli.rs`, fixed tree structure.

## Files Modified (Summary)

| File | Changes |
|------|---------|
| `tracker.rs` | Hierarchical detection, Kalman+EMA smoothing, face-detection fallback |
| `residual.rs` | New ResBlock module |
| `tcn.rs` | GroupNorm in blocks, norm_groups config |
| `vsrm.rs` | ResBlock frontend, AAP+proj, FC init, blank bias, removed double ReLU |
| `grid.rs` | Global stats, try_load/get fallback, pre-validation, sp filtering |
| `dataset.rs` | DatasetStats struct |
| `io.rs` | load_json, save_json |
| `batcher.rs` | Global normalization with DatasetStats |
| `learner.rs` | Composed LR scheduler, train/resume logic, entropy penalty (disabled) |
| `summary.rs` | New SummaryVisitor module |
| `main.rs` | CLI subcommands (build-lm, train, infer), run_build_lm, run_train_vsrm |
| `inference/predictor.rs` | run_infer_vsrm (commented out, stub) |
| `pipeline/video.rs` | load_video_with_tracker (commented out, for inference) |
| `lm.rs` | Doc typo fix (linguistic) |
| `cli.rs` | New module: checkpoint/CLI resolution helpers |
| `utils.rs` | Added levenshtein (moved from metrics) |
| `metrics.rs` | Doc style, typos (aross→across, WEr→WER) |
| `ctc_loss.rs` | Doc style (params/returns → ### Params/### Returns) |
| `learner.rs` | model_id String, Params/Returns colons |
| `README.md` | Project Tree: cli.rs, tree structure fix |
