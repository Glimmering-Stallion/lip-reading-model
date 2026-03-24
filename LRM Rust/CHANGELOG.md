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

**Grid dataset** ([`src/pipeline/adapters/grid/grid.rs`](src/pipeline/adapters/grid/grid.rs)):

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

## 7. GRID Data Pipeline Robustness ([`src/pipeline/adapters/grid/grid.rs`](src/pipeline/adapters/grid/grid.rs))

**Problem:** Burn's `BatchDataloaderIterator` treats any `None` from `dataset.get()` as end-of-dataset, causing epochs to stop after the first failed item (~37 batches instead of ~2800).

**Fixes:**

- **`try_load(index)`**: Extracted loading logic; returns `None` on any failure (alignment, frames, CTC constraint)
- **`get(index)`**: Fallback loop—tries primary index, then adjacent entries (wrapping) until one succeeds; only returns `None` when `index >= len()`
- **Pre-validation in `new()`**: Parses alignment files upfront, filters entries with empty or invalid transcripts before training
- **Alignment filtering:** Excludes `"sp"` tokens (short pauses) in addition to `"sil"` when parsing `.align` files

**Result:** All ~11200 training items are now processed per epoch; LR warmup and cosine decay behave as intended.

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
cargo run -- infer --model vsrm_grid --input path/to/bundled_dir
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

## 13. TCN Causal Normalization ([`src/vsrm/tcn.rs`](src/vsrm/tcn.rs))

- Replaced GroupNorm with per-timestep LayerNorm to preserve strict causality and to maintain coherence in train-inference dynamics.
- LayerNorm applied over channel dimension only: transpose [N,C,T] → [N,T,C], norm over C, transpose back.
- Removed `norm_groups` from `TemporalConvNetConfig` and `TcnBlock`.
- Re-enabled `tcn_is_causal` unit test (now passes).
- **Checkpoint compatibility:** Architecture change; existing checkpoints will not load. Retrain from scratch.
- Chose per-timestep LayerNorm due to:
  - Simplicity of implementation
  - Small channel noise issue countered with sufficiently large number of channels being passed in (using 512 as default)

## 14. Word Separation in Targets ([`src/pipeline/adapters/grid/grid.rs`](src/pipeline/adapters/grid/grid.rs))

- `load_alignment()` now inserts `SPACE_ID` between consecutive words when building target sequences.
- Enables meaningful WER metrics (targets and predictions split by whitespace).
- Vocab already included space; CTC loss and decoder support space tokens without changes.
- **Checkpoint compatibility:** Targets change; retrain required to learn space prediction.

## 15. Tracker Trait Refactor (`src/pipeline/tracker/`)

Refactored the monolithic `tracker.rs` into a trait-based, multi-backend module directory.

**New directory structure:** `pipeline/tracker/`

- **`mod.rs`**: Slim manifest — `pub mod` declarations and `pub use` re-exports only (matches project convention).
- **`backend.rs`**: Backend-agnostic trait, shared types, configuration dispatch, and TLS helpers.
  - `LipTrackerBackend` trait: `process_frame()`, `reset_state()`, `target_dims()`.
  - `TrackerResult`: contains `crop: Mat` (for the model) + `VizMetadata` (for the display fork).
  - `VizMetadata`: `face_rect`, `mouth_rect`, `landmarks`, `stabilized_center` — each tracker populates what it can.
  - `TrackerConfig` enum: dispatch wrapper (`Haar(HaarTrackerConfig)`, future: `MediaPipe(...)`).
  - `with_tracker()`: TLS helper using `Box<dyn LipTrackerBackend>` — replaces old `LipTracker::with_local()`.
- **`haar.rs`**: Existing Haar cascade logic moved and renamed (`LipTracker` → `HaarTracker`, `LipTrackerConfig` → `HaarTrackerConfig`), implementing `LipTrackerBackend`. All doc comments preserved.

**Caller updates:**

- `grid.rs`: Uses `with_tracker(config, |tracker: &mut dyn LipTrackerBackend| { ... })`, extracts `.crop` from `TrackerResult`.
- `learner.rs`, `main.rs`: Construct `TrackerConfig::Haar(HaarTrackerConfig::new(...))`.
- `pipeline/mod.rs`, `lib.rs`: Re-exports updated from `LipTracker`/`LipTrackerConfig` to `LipTrackerBackend`/`TrackerConfig`/`HaarTrackerConfig`.
- `inference/predictor.rs`, `pipeline/video.rs`: Commented-out import paths updated to new names.

**`as_deref_mut` resolution:** TLS helper uses `&mut **opt.as_mut().unwrap()` to dereference `Box<dyn LipTrackerBackend>` without requiring `Deref` on the concrete type.

**No checkpoint impact.** Model weights unchanged; this is a code-organization refactor only.

## 16. Inference Pipeline

**Full inference path from video to prediction:**

- **`InferenceSession`** ([`src/inference/predictor.rs`](src/inference/predictor.rs)): Loads trained VSRM checkpoint and frame batcher. Exposes `predict_file()` for single-video mode and `predict_frames()` for sliding-window inference. Takes pre-loaded configs; no filesystem access except checkpoint.
- **`infer()` free function**: Mirrors `train()`; receives configs, builds session internally, runs file or live loop.
- **`inference/loader.rs`**: `load_video` (video file + tracker → `FramesBuffer`), `load_frame`, `open_camera`, `resolve_inference_video_path` (sample dir → `.mp4`). `pipeline/video.rs` deleted.
- **`FrameAnnotator`** / **`LiveWindow`** ([`src/inference/overlay.rs`](src/inference/overlay.rs)): `FrameAnnotator` draws tracker metadata and prediction text on any `Mat`; `LiveWindow` owns the HighGUI display window and key handling for live inference.
- **`SlidingWindow`**: Buffers frames for live inference; when full, flushes to `FramesBuffer` and runs CTC decode.
- **`run_infer_vsrm`**: Loads configs from model dir (hard block on missing), builds `VsrmPredictorConfig` from `learner_config` (frame_dims, rf), delegates to `infer()`.

## 17. CLI and Config

**Train CLI**

- `--model` and `--dataset`; model ID = `vsrm_{dataset_src}` when `--model` omitted. Train requires `--dataset` for fresh start.
- Config loaded in main only when resuming; resolvers take `Option<&VsrmLearnerConfig>`; `learner_config` built via builder pattern.

**Infer CLI**

- Requires `--model`; exactly one of `--input` or `--live`.
- `--input` may be a bare video file or a bundled video-transcript sample directory (`.../<speaker>/<sample_id>` → loads `<sample_id>.mpg` inside it).
- `--live` accepts an optional camera device index (`--live` = default device `0`; `--live 1` = second camera). OpenCV uses `i32` device indices; the CLI parses a `usize` and rejects values that do not fit.

**Preprocess**

- `--dataset` uses `DatasetSource` (FromStr) instead of `String`.

**VsrmLearnerConfig**

- `dataset_src`, `rf` (receptive field), builder pattern. `rf` persisted; infer reads from `learner_config.json` without model init.
- `train()` returns `Result<(), ESS>`.

**VsrmPredictorConfig**

- `frame_dims`, `rf_window_stride`, `search_type: CtcDecodeType`.

## 18. Batcher and Misc

- **Batcher**: Supports inference (no transcripts): `max_l.max(1)` as placeholder; targets unused during inference.
- **TrainBackend** / **InferBackend** type aliases in main.
- **Error propagation**: `io_err` in utils; `ESS` type alias; resolvers return `Result`; `?` in runners.

## 19. Async Live Inference Mode

**Live webcam inference split into UI + worker threads:**

- Main loop for UI captures frames, runs mouth tracking + overlay rendering, and buffers sliding-window crops.
- A dedicated worker thread owns the inference session and runs the expensive forward pass (`session.predict_frames`).
- Bounded request/response channels (capacity 1) implement “most-recent-only” behavior to avoid latency creep when inference is slower than capture.

## 20. Pass-By-Reference vs. Pass-By-Value Signature Consistency

- Ownership-based thread messaging: `InferenceRequest` owns a `FramesBuffer`, and the worker owns the `InferenceSession`.
- Explicit `Send` bounds enable safe cross-thread execution without shared mutable state.
- The main loop only passes owned buffers into the request channel, keeping borrowing rules simple and preventing accidental cross-thread references.

## 21. Inference Session Struct Initialized Externally to Infer Function

- `infer<B>(session, ...)` now takes a pre-constructed `InferenceSession` rather than building the session internally.
- `infer_file` / `infer_live` accept the session as a parameter, centralizing checkpoint loading and separating model setup from I/O and live orchestration, while also reducing parameter bloat.

## 22. Reformat Error Messages for Clean Composures

- Use `lowercase` and avoid trailing periods so error messages compose cleanly.

- Applied this formatting this project-wide.

## 23. macOS Continuity Camera Info.plist / embed_plist Compatibility

**Suppress macOS AVFoundation camera deprecation warnings:**

- Added an `Info.plist` with `NSCameraUseContinuityCameraDeviceType=true` to opt into `AVCaptureDeviceTypeContinuityCamera`.
- Added `embed_plist` as a macOS-only dependency and embedded the plist into the executable so command-line runs get the expected macOS camera behavior.
- Goal: remove the `AVCaptureDeviceTypeExternal is deprecated for Continuity Cameras` warning when using OpenCV-backed camera capture.

## 24. New Filesystem Formatter for GRID Dataset

- Added a GRID filesystem formatter that bundles `*.mpg` + `*.align` into `data/grid-lr-corpus/<speaker>/<sample_id>/<sample_id>.{mpg,align}` for consistent adapter loading.

## 25. Inference Input Paths and Live CLI

- `--input` for `infer` may be a **bundled video-transcript directory** (path ending at specific input `sample_id`); the binary resolves it to `<sample_id>.mp4` if present, else `<sample_id>.mpg`.
- Removed `--camera`; optional OpenCV device index is now `infer --live [DEVICE_INDEX]` (same pattern as `train --resume [epoch]`). `--live` alone uses device `0`.

**Note:** The static file-mode contract for `--input` was tightened later (see **§27**): the resolver now requires a bundle directory with both non-empty `<stem>.mp4` and `<stem>.txt`—not `.mpg`/`.align` fallback as in training loaders.

## 26. Standardized Corpus Formats Established and Conversion Helpers Implemented for GRID

- Established a new standardized dataset corpus format to adhere to, in the form of sharded video-transcript bundles (with videos as .mp4 files and transcripts as .txt files).
- `grid_adapter`: `convert_to_standard_mp4` (ffmpeg H.264), `convert_to_standard_txt` (word line from `.align`), `normalize_grid_standard_formats`, `clean_corpus` (optional dry-run; drops `.mpg`/`.align` only when `.mp4`/`.txt` exist and are non-empty).
- `preprocess` for GRID runs normalize + clean after bundle; **ffmpeg** must be on `PATH` when `.mpg` files need transcoding.
- `GridDataset` / inference resolution: prefer `.mp4` and `.txt`, fall back to `.mpg` / `.align`.

**Note:** That fallback applies to **dataset loading** and corpus layout; **`infer --input`** bundle rules are stricter and are summarized in **§27**.

## 27. Static File Visualization Overlay

- **Bundle-only file infer:** `infer --input` must be a bundled video_transcript **directory** whose name is the sample stem (`.../<stem>/`). [`resolve_inference_input`](LRM Rust/src/cli.rs) selects non-empty `<stem>.mp4` and `<stem>.txt` beside it.
- **Post-predict artifacts:** After `predict_frames`, results are written under `outputs/<stem>/`: a text file (prediction and reference transcript) and an **annotated MP4** re-encoded from the source video with tracker overlays and caption text ([`annotate_video`](LRM Rust/src/inference/predictor.rs)—second pass, does not run the VSRM).
- **Live mode unchanged:** Webcam path still uses `FrameAnnotator` + `LiveWindow` as before.

# Change Logs Since Last Git Commit

## 28. CTC Loss Forward Optimizations ([`src/ctc/ctc_loss.rs`](src/ctc/ctc_loss.rs), [`src/utils.rs`](src/utils.rs))

- **Pre-gather target log-probs:** A single `gather` over `[N, T, V]` with broadcast indices builds `[N, T, L']` target-aligned log-probs once; the time loop slices by `t_idx` instead of repeating gather/expand each step.
- **Pre-compute time mask:** Build `[N, T]` validity once (frame index vs `input_lengths` via `lower` / comparable ops); each step slices the column for `t_idx` and expands to `[N, L']` instead of recomputing `greater_elem` on lengths every iteration.
- **Log-sum-exp tensors:** [`log_sum_exp_2_tensor`](src/utils.rs) uses `.sub(max.clone())` and `max.add(sum.log())` to avoid an extra `max.clone()` on the final add. [`log_sum_exp_3_tensor`](src/utils.rs) uses a **single** fused path: `max_pair` chain over `a,b,c`, then one `log` after summing shifted exponentials (no nested `log_sum_exp_2_tensor` composition).

## 29. CTC Beam Decode Optimizations ([`src/ctc/ctc_decode.rs`](src/ctc/ctc_decode.rs))

- **Use FxHashMap over HashMap and preallocate with capacity:** Faster hasher for controlled keys; `with_capacity_and_hasher` reduces reallocations during the timestep loop.
- **Quickselect instead of sorting for prefix candidates:** `sort_by` on all surviving prefixes costs $O(P \log P)$; only the top `beam_width` matter. `select_nth_unstable_by` partitions in expected $O(P)$, then `truncate(beam_width)`.
- **Truncate over take:** Prune `next_prefixes_vec` in place instead of allocating a new `Vec` from an iterator chain.
- **BeamPrefix constructor:** `BeamPrefix::new` precomputes `combined_log_prob` (acoustic lse + $\alpha$ LM + $\beta$ length) so pruning compares cached floats instead of recomputing `score()` on every comparator invocation.
- **Single CPU materialization per sample:** After `log_softmax`, one `to_data()` / `Vec<f32>` holds `[T, V]` in row-major order; each timestep indexes `t_idx * V ..` for the frame row. Avoids per-frame GPU `topk` + host sync (major win on Wgpu).
- **CPU top-k per frame:** Reuse a length-`V` buffer of `(token_id, log_prob)` pairs, fill from the current row, then `select_nth_unstable_by(k - 1, …)` and read `top_k_pairs[..k]`; blank log-prob read directly from the row slice.
- **Batch decode:** `log_probs.dims()` without cloning the tensor solely for shape.

## Files Modified (Summary)

| File                      | Changes                                                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `tracker.rs` → `tracker/` | Trait refactor: split into `mod.rs` (manifest), `backend.rs` (trait + TLS), `haar.rs` (Haar impl)              |
| `residual.rs`             | New ResBlock module                                                                                            |
| `tcn.rs`                  | GroupNorm → per-timestep LayerNorm (causal), norm_groups removed, tcn_is_causal re-enabled                     |
| `grid.rs`                 | Word separation: SPACE_ID inserted between words in load_alignment; tracker imports updated to trait-based API |
| `vsrm.rs`                 | ResBlock frontend, AAP+proj, FC init, blank bias, removed double ReLU                                          |
| `grid.rs`                 | Global stats, try_load/get fallback, pre-validation, sp filtering                                              |
| `dataset.rs`              | DatasetStats struct                                                                                            |
| `io.rs`                   | load_json, save_json                                                                                           |
| `batcher.rs`              | Global normalization with DatasetStats; inference support (max_l.max(1))                                       |
| `learner.rs`              | Composed LR scheduler, train/resume logic; VsrmLearnerConfig: rf, dataset_src, builder; train() returns Result |
| `summary.rs`              | New SummaryVisitor module                                                                                      |
| `main.rs`                 | CLI subcommands; infer wired; TrainBackend/InferBackend; Preprocess DatasetSource                              |
| `inference/predictor.rs`  | InferenceSession, infer(), predict_file, predict_frames, SlidingWindow                                         |
| `inference/loader.rs`     | New: load_video, load_frame, open_camera                                                                       |
| `inference/overlay.rs`    | New: FrameAnnotator (draw), LiveWindow (HighGUI)                                                                |
| `pipeline/video.rs`       | Deleted; logic in inference/loader.rs                                                                          |
| `cli.rs`                  | Checkpoint/CLI resolution helpers; pure resolvers                                                              |
| `utils.rs`                | levenshtein, io_err; fused `log_sum_exp_3_tensor`; fewer `max` clones in `log_sum_exp_2_tensor`                 |
| `lib.rs`                  | Tracker re-exports updated                                                                                     |
| `pipeline/mod.rs`         | Tracker re-exports updated                                                                                     |
| `README.md`               | Project tree, CLI examples; CTC module paths and decode/loss optimization notes                                                                 |
| `Info.plist`              | macOS camera key: `NSCameraUseContinuityCameraDeviceType=true`                                                  |
| `Cargo.toml`              | `crossbeam-channel`, macOS `embed_plist`, `rustc-hash` (decoder `FxHashMap`)                                                                   |
| `ctc_loss.rs`             | Pre-gather target log-probs; precomputed `[N,T]` time mask; DP loop slices                                                                     |
| `ctc_decode.rs`           | CPU slab beam acoustics; `FxHashMap`; `select_nth_unstable_by` pruning; `BeamPrefix::new`; reused `(id, log_prob)` buffer                     |
