# Change Log Summary (Since Last Git Commit)

This document records all modifications to the project for future reference and onboarding.

---

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

---

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

---

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

---

## 4. Entropy Regularization (Disabled)

**Location:** [`src/training/learner.rs`](src/training/learner.rs)

- `calc_entropy_penalty()`: penalizes high-confidence (low-entropy) outputs
- Min entropy threshold discourages same-char/blank collapse early in training
- Currently commented out; can be re-enabled for experimentation

---

## 5. Model Summary Module ([`src/vsrm/summary.rs`](src/vsrm/summary.rs))

- `SummaryVisitor`: displays layer shapes and parameter counts (similar to TF `model.summary()`)
- Invoked at training start to log model architecture stats

---

## 6. Learning Rate Scheduler ([`src/training/learner.rs`](src/training/learner.rs))

- Replaced `StepLrScheduler` with `ComposedLrSchedulerConfig`
- Linear warmup (0.01 → 1.0 gain over first epoch) × Cosine annealing (lr → lr/10 over total steps)
- `SchedulerReduction::Prod`: multiplies outputs of both schedulers
- Warmup prevents destructive early updates that can cause CTC collapse

---

## 7. GRID Data Pipeline Robustness ([`src/pipeline/adapters/grid.rs`](src/pipeline/adapters/grid.rs))

**Problem:** Burn's `BatchDataloaderIterator` treats any `None` from `dataset.get()` as end-of-dataset, causing epochs to stop after the first failed item (~37 batches instead of ~2800).

**Fixes:**

- **`try_load(index)`**: Extracted loading logic; returns `None` on any failure (alignment, frames, CTC constraint)
- **`get(index)`**: Fallback loop—tries primary index, then adjacent entries (wrapping) until one succeeds; only returns `None` when `index >= len()`
- **Pre-validation in `new()`**: Parses alignment files upfront, filters entries with empty or invalid transcripts before training
- **Alignment filtering:** Excludes `"sp"` tokens (short pauses) in addition to `"sil"` when parsing `.align` files

**Result:** All ~11200 training items are now processed per epoch; LR warmup and cosine decay behave as intended.

---

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
| `learner.rs` | Composed LR scheduler, entropy penalty (disabled) |
| `summary.rs` | New SummaryVisitor module |
| `main.rs` | recursion_limit, hyperparameters (lr, accumulation, grad clip, etc.) |
