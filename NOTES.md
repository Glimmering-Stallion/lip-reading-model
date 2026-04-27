<!-- This is the file that serves as an engineering notebook (granular decisions and deep implementation details) -->
<!-- Rule of thumb for what goes here: "Is this an implementation detail?" -->

# Documentation Notes

## Project Tree (detailed)

```
Lip Reading Model
├─ LICENSE
├─ docs
│  ├─ assets
│  │  ⋮  (article figures, CTC/diagram exports, pipeline SVGs, media)
│  └─ lrm-portfolio-article.md
├─ LRM Python
│  ├─ application
│  │  ├─ animation.gif
│  │  ├─ general_utils.py
│  │  ├─ lipread.py
│  │  └─ model_utils.py
│  ├─ lipread.ipynb
│  ├─ main.py
│  └─ requirements.txt
├─ LRM Rust
│  ├─ Cargo.lock
│  ├─ Cargo.toml
│  ├─ CHANGELOG.md
│  ├─ data
│  │  ├─ grid-lr-corpus
│  │  │  ├─ cropped_frames
│  │  │  ├─ s1
│  │  │  │  └─ <stem_id>
│  │  │  │  ⋮  ├─ <stem_id>.mp4
│  │  │  │  ⋮  └─ <stem_id>.txt
│  │  │  └─ s34
│  │  │     └─ <stem_id>
│  │  │        ├─ <stem_id>.mp4
│  │  │        └─ <stem_id>.txt
│  │  └─ librispeech-lm-norm
│  │     └─ librispeech-lm-norm.txt
│  ├─ models
│  ├─ outputs
│  ├─ assets
│  ├─ exports
│  ├─ Info.plist
│  ├─ rust-toolchain.toml
│  ├─ rustfmt.toml
│  ├─ scripts
│  ├─ tools
│  │  ├─ requirements.txt
│  │  ├─ plotneuralnet
│  │  ├─ onnx_export
│  │  │  ├─ export_onnx.py
│  │  │  └─ vsrm_twin.py
│  │  └─ tex_export
│  │     └─ export_tex.py
│  ├─ src
│  │  ├─ cli.rs
│  │  ├─ context.rs
│  │  ├─ ctc
│  │  │  ├─ ctc_decode.rs
│  │  │  ├─ ctc_loss.rs
│  │  │  ├─ lm.rs
│  │  │  ├─ mod.rs
│  │  │  └─ viz
│  │  │     ├─ mod.rs
│  │  │     ├─ forward_lattice_viz.rs
│  │  │     └─ prefix_beam_viz.rs
│  │  ├─ inference
│  │  │  ├─ loader.rs
│  │  │  ├─ mod.rs
│  │  │  ├─ overlay.rs
│  │  │  ├─ predictor.rs
│  │  │  └─ speech_gate.rs
│  │  ├─ lib.rs
│  │  ├─ main.rs
│  │  ├─ pipeline
│  │  │  ├─ adapters
│  │  │  │  ├─ grid/
│  │  │  │  │  ├─ grid_dataset.rs
│  │  │  │  │  ├─ grid_adapter.rs
│  │  │  │  │  └─ mod.rs
│  │  │  │  └─ mod.rs
│  │  │  ├─ batcher.rs
│  │  │  ├─ dataset.rs
│  │  │  ├─ io.rs
│  │  │  ├─ mod.rs
│  │  │  └─ tracker
│  │  │     ├─ haar.rs
│  │  │     ├─ mod.rs
│  │  │     └─ tracker.rs
│  │  ├─ training
│  │  │  ├─ learner.rs
│  │  │  ├─ metrics.rs
│  │  │  ├─ mod.rs
│  │  │  └─ trainer.rs
│  │  ├─ utils.rs
│  │  ├─ vocab.rs
│  │  └─ vsrm
│  │     ├─ mod.rs
│  │     ├─ residual.rs
│  │     ├─ summary.rs
│  │     ├─ tcn.rs
│  │     └─ vsrm.rs
│  └─ target
├─ NOTES.md
├─ PLANS.md
└─ README.md
```

## Accomplishments to Date

### Data Ingestion (`pipeline/io.rs` and `pipeline/adapters/grid/grid_dataset.rs`)

- For now, using [GRID](https://zenodo.org/records/3625687) corpus as proof of concept that the VSRM can converge (speaker ("s1", "s2", ..., "s34") data organized into sample bundles under `data/grid-lr-corpus/<speaker>/<sample_id>/`). Each sample folder holds video (`<sample_id>.mp4` preferred after preprocess, else `.mpg`) and transcript (`<sample_id>.txt` preferred after preprocess, else `.align`).
- For GRID, `cargo preprocess --dataset grid` also writes mouth-crop frame tensors as `.bin` files under `data/grid-lr-corpus/cropped_frames/` (`GridDataset::pre_extract_all`), so training can load crops from disk instead of re-decoding video every epoch.
- GRID corpus discovery and bundled-dir listing skip `__MACOSX`, hidden (`.`‑prefixed) entries, and non-`.mpg`/`.align` files so macOS zip metadata does not break normalize/preprocess (`LRM Rust/src/pipeline/adapters/grid/grid_adapter.rs`).
- Will consider using the [Oxford-BBC LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) corpus in the future, for a wider generalization to conversational speech to generalize the VSRM to broader use.
- Built dataset utilities that:
  - Infer file name stems automatically.
  - Pair videos with alignment annotations.
  - Download and extract compressed datasets when missing.
- Implemented a video pipeline using OpenCV in `grid_dataset.rs` that:
  - Decodes video files frame by frame.
  - Converts frames to grayscale.
  - Uses pre-trained Haar Cascade detectors for face and mouth localization (see [Attributions](#attributions)).
  - Crops a dynamic mouth ROI per frame.
  - Flattens pixel data into contiguous `Vec<u8>` tensors.

---

### Data Standardization & Normalization (`adapters/`)

- Implemented dataset adapters that contain source-specific logic to:
  - Transcode src video files into `.mp4`, and write `.txt` from src transcript files, then remove redundant src video/transcript files when safe.
  - Map raw datasets (GRID, LRW, etc.) into a standardized `VsrmItem` format.
  - Rely on the more abstract `DatasetSplit` utility in `pipeline/dataset.rs` for train/val/test partitioning.
- The adapter modules are to reshape a dataset into a dir containing sharded video-transcript bundles, where video files are `.mp4` and transcript files are `.txt` (GRID adapter modules enforces this currently, but other dataset sources are intended to follow the same form).
- In future, will consider FPS standardization to a target FPS (25) as well (frame dropping preferred over interpolation due to simplicity and avoidance of ghost data).

---

### Data Batching (`pipeline/batcher.rs`)

- Developed a custom `VsrmBatcher` that takes a collection of standardized `VsrmItem`, then standardizes and pads its underlying inputs/targets `TensorData` buffers.
- Standardization handled by:
  - Scaling pixel values to [0, 1].
  - Centering pixel values to zero mean and unit variance.
- Padding handled by:
  - Finding longest video-frames/transcript-sequences among a batch of sequences (as `max_t`/`max_l`).
  - Padding variable-length video frames in that batch to `max_t` with $0$.
  - Padding variable-length transcript sequences in that batch to `max_l` with `BLANK_ID`.
- Uses a CPU-to-GPU staging strategy, where tensors are collated on the `NdArray` CPU backend before a single-shot move to the `Wgpu` GPU backend for minimizing PCIe bus latency.

---

### Data Partitioning (`pipeline/dataset.rs`)

- Dataset splitting policy is delegated to a generic and source-agnostic `DatasetSplit` wrapper, to allow any dataset (GRID, LRW, etc.) to be partitioned through index-mapping without modifying the more specialized adapter logic.
- Applies a random but deterministic shuffle to the index-mapping.
- Then partitions dataset instances into train/val/test splits.

---

### Alignment & Vocabulary Handling (`vocab.rs`)

- Implemented parsing of .align files.
- Filters out silence tokens ("sil", "sp").
- Inserts spaces between words in alignment-derived targets (`SPACE_ID`) for WER metrics.
- Converts labels into integer sequences using a bidirectional vocabulary map.
- Designed a character-level vocabulary including:
  - Lowercase letters (a–z)
  - Space (word boundaries)
  - Dedicated CTC blank symbol
- Ensured the blank symbol:
  - Appears only in model outputs.
  - Never appears in training targets.
  - Is removed during decoding.

---

### VSR Model Architecture (`vsrm/vsrm.rs`, `vsrm/residual.rs`, and `vsrm/tcn.rs`)

Implemented an end-to-end **spatiotemporal VSRM** in Rust/Burn (see [Model Summary](#model-summary) for parameter details):

- **Spatial frontend (`residual.rs`) – 3× ResBlock3D:**
  - Two Conv3D layers per block with GroupNorm → ReLU.
  - Strided spatial downsampling in first Conv3D layer (learned downsampling instead of MaxPool3D's naive downsampling).
  - Residual paths use a **1×1×1 projection** before the skip sum when channel/stride geometry requires alignment.
  - **Pyramid-style channel growth** across ResBlock layers with wider later blocks (C → 2C → 4C) vs. an earlier iteration's “diamond” mid-stack squeeze (C → 2C → C/2).
- **Spatial to temporal handoff (`vsrm.rs`) – AAP2D + Linear Proj:**
  - After the ResBlock stack, features are **rearranged** so **Adaptive Average Pooling (AAP2D)** can apply per timestep on H×W.
  - A **linear projection** maps the flattened pooled features into a fixed **hidden dim** before the TCN (stabilizes what the temporal trunk sees vs. flattening huge maps directly).
- **Temporal backend (`tcn.rs`) – 2× TCN:**
  - Three internal TCN Blocks per TCN layer, each TCN Block consisting of two causal, dilated Conv1D layers (dilations 1, 2, 4).
  - Padding in the Conv1D layers enforce causality, which constrains temporal lookahead of model predictions.
  - Per-timestep LayerNorm inside each TCN Block (normalizing over C).
  - Like the ResBlocks, an optional **pointwise projection** inside a TCN block when residual channel alignment is needed.
- **Readout tail – Linear FC:**
  - Final **linear** maps TCN features to **vocabulary logits** each timestep for **CTC** loss/decoding.
  - Training-time utilities include optional **FC bias / init** tuning to mitigate early **blank-collapse** behaviors during CTC optimization (see training notes / hub “blank collapse” narrative).
  - Logit outputs are unnormalized.

---

### Training Pipeline (`training/learner.rs` and `training/trainer.rs`)

- Keeping a legacy `trainer.rs` file implementing a manual training loop to test model convergence on dummy data.
- Implemented a complete training and validation pipeline using Burn's `Learner` API in Rust as `learner.rs`.
- Supports:
  - Batching
  - Epoch-based training
  - Auto-checkpointing
  - Metric logging
  - Train/validation dataset splitting
- Handles dynamic train/eval mode switching implicitly with Burn's `Autodiff` and `Module` traits (which allows gradient tracking).
- Integrated the Adam optimizer with configurable learning rates.
- Implemented a Linear+Cosine type composed LR scheduler with warmup over first epoch.
- Added numerical utility functions (mean, standard deviation, normalization).

---

### Inference Pipeline (`inference/predictor.rs`, `inference/loader.rs`, `inference/overlay.rs`, and `inference/speech_gate.rs`)

- Implemented an inference session pipeline that supports both static-file and live-webcam inference.
- Supports:
  - Static-file mode (`infer_file`) for bundled video-transcript samples
  - Live-camera mode (`infer_live`) with real-time frame capture and overlay rendering
  - Sliding-window temporal buffering for frame-to-frame inference input construction
  - Async worker-thread inference for keeping live UI/render loop responsive
- Uses tracker outputs per frame (`has_lock`, `has_lip_motion`) to gate prediction flow and UI behavior.
- Added a speech-activity hysteresis gate (`SpeechGate`) with configurable on/off frame thresholds to reduce flicker and stale prediction flashes.
- On speech-gate close, clears temporal buffers and drains stale worker responses to avoid displaying outdated predictions.
- Added a visualization overlay system (`FrameAnnotator` + `OverlayLayout`) that renders:
  - Tracker ROI / status metadata
  - Text status block (prediction, tracker lock, speech status)
  - Bottom-right picture-in-picture (PIP) mouth crop inset
- Added annotated video generation for static-file mode, with optional ffmpeg audio muxing fallback behavior for final outputs.

---

### Tracker Pipeline (`pipeline/tracker/tracker.rs` and `pipeline/tracker/haar.rs`)

- Implemented a backend-agnostic tracker interface (`LipTrackerBackend`) to standardize frame processing across tracker implementations.
- The tracker contract exposes:
  - `process_frame` for per-frame mouth ROI extraction and metadata production
  - `has_lock` for reliability state of face/mouth tracking
  - `has_lip_motion` for visual speech-activity proxy
  - `target_dims` / `reset_state` / optional mouth-crop inset hooks for inference overlay integration
- Current implementation uses a hierarchical Haar cascade backend:
  - Face detection in full frame
  - Reduced lower-thirds face search region for mouth detection
  - Stabilized mouth ROI extraction and resize to model target dimensions
- Temporal stabilization combines gating and smoothing (Kalman-style distance gating + EMA) to reduce jitter and reject implausible jumps.
- Lip-motion detection is currently based on temporal absolute-difference of Sobel gradient magnitudes, with inner-vs-periphery activity checks for stronger mouth-region discrimination.
- Future tracking considerations include:
  - Landmark-based backend alternatives (e.g. MediaPipe) under the same backend-agnostic trait with `has_lip_motion` implemented with a Mouth Aspect Ratio (MAR) based system
  - Frame-rate / delta-time normalization for lip-motion thresholds so speech gating behaves more consistently across variable FPS sources (currently not time-normalized, behavior varies with FPS, high FPS input sources result in smaller per-frame deltas and more false negative `has_lip_motion` off-triggers)

---

### Custom CTC Loss (`ctc/ctc_loss.rs`)

- Implemented a custom Connectionist Temporal Classification (CTC) Loss for model's logit alignment to the target sequence (without frame-level labels).
- Goal: Calculate total probability of a specific ground truth sequence
- Operates on log-probability tensors with shape $[T,\ V]$ (or $[N,\ T,\ V]$ batched equivalent), where:
  - $T$ = input timesteps,
  - $V$ = vocabulary size (including blank),
  - targets are length $L$ (padded to max length when batched), expanded to length $L' = 2L + 1$ by interleaving blanks ("cat" → "_c_a_t_").
- Uses a forward-only dynamic programming (DP) pass over the blank-interleaved target:
  - Forward tensor $[L']$ (or $[N,\ L']$ batched equivalent) accumulates valid path's log-probability mass up to current timestep $t$.
  - Combined log-space probability accumulation per timestep using LogSumExp (LSE).
  - Transition actions logic:
    - Same stay transitions,
    - advance-by-1 transitions,
    - advance-by-2 skip transitions (only when symbols differ, preventing invalid repeat merges).
  - Computes final log-probability from last two reachable terminal states (sequence ending in non-blank vs. blank).
  - Invalid states masked to a sentinel value to constrain impossible path transitions in the time-sequence grid.
- Does not implement backward DP as gradient propagation is handled by Burn's `Autodiff` backend decorator.
- Supports variable-length batching with `input_lengths` and `target_lengths`, masking out padded timesteps/states during DP and final aggregation.
- See [CTC Loss Forward Lattice Example Visualization](#ctc-loss-forward-lattice-example-visualization) below.

---

### CTC Loss Visualization (`ctc/viz/forward_lattice_viz.rs`)

- Implemented a forward lattice visualization module that captures the log-alpha DP grid (`N = 1` single sample batch) and per-transition edge fractions into a shared trace struct.
- **ASCII renderer:** monospace heatmap using block-shade characters (`░`, `▒`, `▓`, `█`) mapping log-alpha magnitude, with interleaved target labels on the Y axis and timesteps on the X axis.
- **SVG renderer:** blue-white heatmap cells with DP transition arrows overlaid. Solid green arrows mark edges on at least one complete CTC alignment; dashed red arrows mark incomplete paths. Arrow opacity scales with conditional edge mass. A vertical colorbar legend shows the log-alpha range.
- Style is configurable via `ForwardLatticeSvgTheme` (text color, arrow colors, heatmap border, cell size, font, margins).
- Fixture tests generate diagrams for configurable target sequences (`FIXTURE_SEQS`); ASCII prints to terminal, SVGs export to `LRM Rust/outputs/`.

---

### Custom CTC Decoding & Inference (`ctc/ctc_decode.rs`)

- Implemented custom CTC Decoding (greedy and prefix beam search) for sequence aggregation from model's logit outputs.
- Goal: Interpret most likely final text sequence predicted by model when we don't have a ground truth on what was uttered.
- Two search modes supported:
  - **Greedy decoding** for low-latency inference,
  - **Prefix beam search** for higher-accuracy sequence selection.
- Greedy Search:
  - Takes argmax per timestep.
  - Uses explicit intended vs. non-intended char duplicate + blank handling by:
    - collapsing consecutive repeats ("_ggrrreeee_ettiing_sss" → "_gre_eting_s"),
    - removing blank tokens ("_gre_eting_s" → "greetings").
  - Returns final token sequence.
- Prefix Beam Search:
  - Builds a set of top candidate prefix sequences of "beam width" per timestep.
  - Tracks each prefix sequence's blank and non-blank log-probability masses per timestep.
  - Prefix extension logic (uses implicit intended vs. non-intended char duplicate + blank handling):
    - Blank encountered? Stay on same prefix (prefix: "bal", curr token: "_" → "bal").
    - Same-token encountered?
      - Either extend when last token was blank (prefix: "bal", curr token: "l", last token: "_" → "ball"),
      - or stay when last token was non-blank (prefix: "ball", curr token: "l", last token: "l" → "ball").
    - New-token encountered? Append to prefix (prefix: "ball", curr token: "s", last token: "l" → "balls").
    - Prefix merging when multiple paths map to same collapsed sequence.
  - Combined log-space probability accumulation per timestep using LogSumExp (LSE).
  - Uses beam pruning each timestep (top prefixes by combined score) to bound compute/memory while retaining high-probability hypotheses.
- Decoder architecture is streaming friendly with timestep-local updating and incremental/streaming inference.
- Handles practical edge cases: all-blank outputs, repeated or duplicate char ambiguity (like doubled letters), short/long sequence imbalance, and variable input lengths.
- See [CTC Decode Prefix Beam Example Visualization](#ctc-decode-prefix-beam-example-visualization) below.

---

### CTC Decode Visualization (`ctc/viz/prefix_beam_viz.rs`)

- Implemented a prefix beam DAG visualization module that captures beam search snapshots after each timestep (`N = 1` single sample batch) and derives parent→child edges from CTC prefix structure (stay or one-token extend).
- A shared `GraphLayout` struct feeds both renderers so column packing and rank ordering are identical between ASCII and SVG output.
- **ASCII renderer:** vertical DAG with box-drawing pipe characters (`│`, `├`, `┬`, `┐`, etc.), beam rank left-to-right, time top-to-bottom. Best-path prefix in brackets `[..]` vs other prefix hypotheses `(..)`.
- **SVG renderer:** cubic Bézier edges between prefix-node boxes and top-K emission chips. Node fill/border desaturates green → gray across beam ranks; chip fill/border desaturates lavender → gray across emission order. A lineage palette assigns maximally-separated hues per child lineage; the decode-highlight path uses a distinct bold green.
- Includes a top-positioned legend with two columns (edge types and color swatches) showing Top-W prefixes, Top-K tokens, decode path, and token candidate edge styles.
- Style is configurable via `PrefixBeamSvgTheme` (colors, margins, font sizes, corner radii for nodes vs chips, legend placement).
- Fixture tests use deterministic per-word seeded randomization (`FIXTURE_LOGITS_SEED`) for varied top-K token orderings and a shuffled lineage color palette for visual diversity across generated diagrams.
- SVGs export to `LRM Rust/outputs/`; finalized diagrams are manually copied to `docs/assets/` for the portfolio site.

---

### Language Model Integration (`ctc/lm.rs`)

- Incorporated a dedicated language model interface for CTC decoder's prefix beam search.
- Designed for character-level N-gram scoring.
- Decoder-side LM integration is backend-abstracted so scoring backends are swappable without rewriting the beam-search control flow.
- Uses an enum to support different LM types (N-gram LM, Neural LM, etc.), but will still require concrete adapter/state implementations or tuning.
- Supports configurable:
  - Language model weight (alpha): controls influence of LM over base VSRM's predictions (lower alpha means trusting VSRM over LM more and vice versa for higher alpha).
  - Insertion bonus (beta): counteracts LM's bias toward shorter sequences (adding more tokens makes log-prob score more negative, where beta adds a small positive bonus).
- Currently only a char-level trigram LM is implemented and trained on the [OpenSLR LibriSpeech LM Norm](https://www.openslr.org/11) corpus.
- In future, might consider using a tiny neural LM (char/BPE GRU or small Transformer) with prefix-state caching per beam (running it only on top-K emission symbols each step to bound cost; or use it as an N-best reranker after beam, which mitigates per-frame latency).

---

### Custom WER/CER Metrics (`training/metrics.rs`)

- Implemented custom CTC-aware character and word error metrics for validation-time evaluation.
- Supports:
  - **CER** (Character Error Rate) over decoded predictions vs. reference transcripts
  - **WER** (Word Error Rate) using tokenized word-level comparison after transcript normalization
- Metrics are integrated into Burn `Learner` validation hooks so score tracking is part of standard training runs.
- Built to work with the project’s CTC decode flow (greedy baseline, optional beam + LM), so reported error reflects end-to-end decoding behavior rather than raw logits.
- Added defensive handling for empty/degenerate prediction cases to avoid metric crashes during unstable early epochs.
- Used alongside loss and learning-rate diagnostics to separate optimization progress (loss) from transcription quality (CER/WER).

---

### Model Exporting (`tools/onnx_export/`, and `tools/tex_export/`)

- Not to be confused with model checkpoint saves from training.
- Added a unified `export` CLI path that writes an export bundle under `exports/<model_id>_export/`.
- Export flow currently includes:
  - ONNX export (`onnx/vsrm_export.onnx`) through a lightweight PyTorch twin (`vsrm_twin.py`) because Burn does not natively emit ONNX.
  - TeX architecture exports (`tex/`) for macro VSRM and subcomponent diagrams (TCN / ResBlock), including PlotNeuralNet layer assets.
- Export pipeline supports configurable ONNX opset and trace sequence length.
- Python export failures are surfaced with captured stderr/stdout so dependency errors (e.g., missing `torch`) are visible.

---

### Top-Level CLI Control Flow (`main.rs` and `cli.rs`)

- CLI uses subcommand-driven control flow with explicit runners for:
  - `preprocess` normalizes dataset artifacts and pre-extracts mouth-crop caches
  - `train` builds loaders/scheduler/learner and handles fresh vs resume checkpoint behavior
  - `infer` routes to static-file or live-camera inference mode
  - `export` orchestrates ONNX + TeX generation into a single bundle
  - `build-lm` orchestrates the corpus loading, training, and mini-inference of the N-gram LM artifact for beam prefix decoding
- `main.rs` handles argument parsing and dispatch.
- `cli.rs` centralizes resolution/validation helpers for consistent runtime behavior.
- Control flow emphasizes explicit error propagation and clear user-facing diagnostics.

---

### System Design & Engineering Decisions

- Rust-native core pipeline for emphasis on:
  - Thread Control
  - Parallelism
  - Memory safety
  - Low-latency inference
- Peripheral Python used only for model export API bridging for:
  - ONNX file for standardized model representation.
  - TeX files for model LaTeX/PlotNeuralNet diagram visualizations
- Why GroupNorm (over BatchNorm or LayerNorm) in the VSRM ResBlocks:
  - BatchNorm is brittle with small batches and gets expensive for high-dimensional video if you push batch size to stabilize it.
  - LayerNorm on [N, C, T, H, W] would normalize over too many axes at once for a spatial frontend (channels × space × time together), which tends to erase localized structure you still want the convs to exploit.
  - GroupNorm normalizes within groups of channels per spatial location and timestep, which behaves more predictably than BN at small batches and stays localized compared to a global LN over the whole spatiotemporal volume (a middle ground between InstanceNorm and LayerNorm).
- Why LayerNorm in the VSRM TCN layers:
  - LayerNorm inside each TCN block is applied per timestep to preserve causality (the no lookahead constraint).
  - Note: this is not the same as slapping LN on the full 5D frontend tensor (here it stabilizes the temporal trunk without mixing future frames into the norm stats).
- Adaptive Average Pool 2D (AAP2D) used in VSRM frontend-backend boundary for:
  - Capping spatial variability and fixes H×W to 4×4
  - Decoupling input resolution from parameter/tensor shapes
  - Resolution independent tensor handoff to subsequent TCN layers
- TCN over BiLSTM / BiGRU:
  - Stacked dilated convs give large receptive fields with dense parallel ops over time (good throughput, lower inference latency)
  - Lower recurrent hidden state overhead to init, reset or thread through live inference
  - Simpler tensor-in → tensor-out deployment than stateful RNNs
  - Cost is inaccessibility of future frame timesteps (which doesn't matter due to our live inference goal)
- Char-level over Word-level LM (CTC decode):
  - Currently using char-level decoding with optional char-level N-gram LM
  - Simpler beam state, no lexicon dependency, robust to out-of-vocab (OOV) words
- Why Haar Cascades for the lip-tracking?
  - Fast for live inference
  - Wanted a quick proof of concept before going to more complex methods
  - Chosen as the initial backend with a subsequent lip tracker trait abstraction established for future swapping
  - Tradeoff is not robust to large lip tilt angle orientations (no rotational invariance) and lighting conditions
- Future extensions:
  - Alternative dataset training (such as Oxford BBC's LRW/LRS2/LRS3 dataset)
  - Landmark-based tracking backend (e.g., MediaPipe)
  - FPS / delta-time normalization for lip-motion thresholds
  - Alternate ONNX runtimes for portable deployment (such as Tract)

---

## Current Status

- **I/O and data acquisition:** Video encoding/decoding, mouth ROI extraction utilities, and dataset download/extract helpers are in place.
- **Data pipeline:** Adapter mapping (at least for GRID), preprocessing (including optional on-disk mouth-crop cache in `cropped_frames/`), deterministic splitting, and batching are implemented.
- **CTC Loss:** Custom CTC loss implemented in log-space (forward/backward DP) with vectorized batch support for variable-length sequences.
- **CTC Decoding:** Greedy and prefix beam-search CTC decoding, optionally rescored with the integrated char-level N-gram LM (supports alpha/beta).
- **Training:** Burn `Learner`-based training/validation loop with checkpointing, metrics, and LR scheduling. Uses `create_dataloaders` helper to handle train/val splits, batching, and dataloading for source-specific datasets.
- **Inference:** Using an `InferenceSession` engine, which supports static file inference (as a bundled video-transcript input) with `infer_file` and async live webcam inference with `infer_live` (main thread captures/tracks/overlays; worker thread runs model forward passes).
- **Verification:** Unit tests for CTC loss/decoding, tracker ROI behavior, and sanity checks for model input/output dataflow; training convergence validated via overfit tests.
- **Mouth tracking:** Haar-cascade face/mouth detection with stabilized mouth ROI per frame.
- **CLI:** `build-lm`, `preprocess`, `train` (new/resume), `infer` (static file / live cam), and `export` (bundle `exports/<model_id>_export/` with `onnx/` + `tex/`, optional `--output` bundle root) subcommands are available.
- **Inference viz overlay:** `FrameAnnotator` in `LRM Rust/src/inference/overlay.rs` draws tracker ROIs, stabilized center, a bottom-left status block, and a bottom-right mouth-crop PIP (`draw_mouth_crop_inset`). `OverlayLayout::from_frame` scales margins, text, ROI strokes, and PIP size from the frame’s shorter side (live and annotated export).
- **Live inference speech gating:** When speech gating is enabled, `annotate_video` and `infer_live` both apply `SpeechGate` per frame so the prediction caption matches hysteresis (`LRM Rust/src/inference/speech_gate.rs`).
- **Model exporting:** The export command writes a bundle under exports with an ONNX file plus TeX outputs. ONNX is produced by a small PyTorch twin (since Burn does not emit ONNX itself). Install the Python packages in `requirements.txt` listed under LRM Rust tools. When a Python step fails, the Rust CLI prints its `stderr` so errors like a missing torch install are visible. TeX uses a vendored `PlotNeuralNet` (cloned the upstream repo into this repo's tools once and then customized): macro, TCN detail, and ResBlock diagrams, with optional single-image or multi-frame thumbnails to the left of the input block (this multi-frame art is picked up from a folder next to the TeX export script). To compile and render the generated TeX files for visualization, just upload the whole generated `tex` directory to Overleaf and hit compile.
- **CTC loss visualization:** Forward lattice heatmap renderer (ASCII block-shade + SVG with DP arrows) in `ctc/viz/forward_lattice_viz.rs`. Configurable elements, fixture-driven test generation, SVGs export to `LRM Rust/outputs/`.
- **CTC decode visualization:** Prefix beam DAG renderer (ASCII box-drawing + SVG with Bézier edges, lineage palette, top-K emission chips, legend) in `ctc/viz/prefix_beam_viz.rs`. Configurable elements with decoupled prefix-node/token-chip corner radii, deterministic per-word seeded fixtures, SVGs export to `LRM Rust/outputs/`.

## Pending / Future Work

- **Add Landmark-Based Tracker:** Improve ROI stability and accuracy, plus rotational invariance benefits by adding a landmark/pose-based tracker backend (e.g. MediaPipe) as a separate tracker option to the existing layered Haar cascades tracker.
- **Grad-CAM For Overlay Visualization:** During the forward pass, save the "activations" of the last TCN or Conv layer. Treat those activations as a heatmap. Upscale that heatmap to match the mouth-crop size. Then alpha-blend it (transparent overlay) onto the video.
- **FPS Video Standardization:** Unify potentially varying frame-rates between different video-transcript dataset sources.
- **Normalize Haar Has Lip Motion Output By Time:** Perform delta time normalization between the gradient changes between last and current frames to account for variable frame-rate video/cam inputs.
- **Speech Gate Hysteresis Tweak:** Have a reduced off frames field (or combine on/off frames into a single field) then have an epsilon for the additional extra frames to add to an off condition instead. Then wire the "speech active" state to the true on/off frame conditions, while only have the model inferencing period subject to the hysteresis (where a person might pause a bit with intention to still talk; the speech active state updates responsively to that short pause and says "not talking", while the model inferencing period is not destroyed). In short, apply hysteresis to model inferencing, while keep the speech active state responsive and true to time.
- **Beam Decode LM Merge Rule:** The third field `log_prob_lm` is set in `or_insert` from the first path that creates the entry and is not updated when another path merges into the same key. With `lm = None` / zero LM that doesn’t matter; with a real LM, merged hypotheses might need an explicit LM merge rule (like keep max, or re-score from the shared prefix).

## Full Pipeline

```text
┌──────────────────────────────────────────────────────────────────┐
│  CLI ARGS                                                        │
├──────────────────────────────────────────────────────────────────┤
│  cargo run -- <command>                                          │
│                                                                  │
│  preprocess ──► GridAdapter                                      │
│                 discover → .mp4/.txt bundles → cropped_frames/   │
│                                                                  │
│  build-lm ────► corpus.txt → NgramLM (trigram) → lm.bin          │
│                                                                  │
│  export ──────► Python: onnx_export → exports/<id>/onnx/         │
│                         tex_export  → exports/<id>/tex/          │
└──────────────────────────────────────────────────────────────────┘
           │ train                              │ infer
           ▼                                    ▼
┌─────────────────────┐       ┌────────────────────────────────────┐
│  DATA PIPELINE      │       │  INFERENCE PIPELINE                │
├─────────────────────┤       ├────────────────────────────────────┤
│  Dataset            │       │  loader.rs (file or live cam)      │
│  (video decode      │       │       │                            │
│   → grayscale)      │       │  LipTracker                        │
│       │             │       │  ├─ mouth ROI extraction           │
│  LipTracker         │       │  ├─ ROI stabilization.             │
│  (mouth crop        │       │  └─ lip motion detection           │
│   per frame)        │       │       │                            │
│       │             │       │  SpeechGate                        │
│  DatasetSplit       │       │  (hysteresis: has_lock +           │
│  (train/val/test)   │       │   has_lip_motion → speech_active)  │
│       │             │       │       │                            │
│  VsrmBatcher        │       │  InferenceSession                  │
│  (pad + stage       │       │  (loaded checkpoint +              │
│   CPU → GPU)        │       │   DatasetStats norm)               │
└─────────────────────┘       └────────────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  CORE SPATIOTEMPORAL VSRM                                        │
├──────────────────────────────────────────────────────────────────┤
│  VsrModel  forward pass  [N, T, V] logits                        │
│                                                                  │
│  ResBlock3D × 3  Conv3D + GroupNorm + 1×1 skip proj              │
│  [N, 1, T, H, W] ─────────────────────────► [N, 512, T, H', W']  │
│                                                                  │
│  rearrange → 4×4 AAP2D → flatten                                 │
│  [N, 512, T, H', W'] ─────────────────────────────► [N·T, 8192]  │
│                                                                  │
│  Linear Proj → reshape                                           │
│  [N·T, 8192] ─────────────────────────────────────► [N, 512, T]  │
│                                                                  │
│  TCN × 2  causal dilated Conv1D (dilations 1, 2, 4) + LayerNorm  │
│  [N, 512, T] ─────────────────────────────────────► [N, 512, T]  │
│                                                                  │
│  Linear FC  →  unnormalized logits (per timestep)                │
│  [N, T, 512] ───────────────────────────────────────► [N, T, V]  │
└──────────────────────────────────────────────────────────────────┘
           │ train                              │ infer
           ▼                                    ▼
┌─────────────────────┐       ┌────────────────────────────────────┐
│  CTC LOSS           │       │  CTC DECODE                        │
├─────────────────────┤       ├────────────────────────────────────┤
│  expand target:     │       │  Greedy                            │
│  "cat" → _c_a_t_    │       │  argmax → collapse → strip blank   │
│                     │       │                                    │
│  forward DP (log)   │       │  Prefix Beam Search                │
│  stay / +1 / +2     │       │  top-K tokens per frame (CPU slab) │
│  transitions        │       │  FxHashMap prefix merge (LSE)      │
│  LSE accumulation   │       │  N-gram LM fusion (α/β)            │
│  ── scalar loss     │       │  quickselect pruning (top-W)       │
│       │             │       │       │                            │
│  Burn Autodiff      │       │  text prediction string            │
│  (auto backward)    │       │       │                            │
│       │             │       │  FrameAnnotator + mux_audio        │
│  AdamW + ComposedLR │       │  outputs/<stem>.mp4                │
│  (warmup × cosine)  │       │  (file) or LiveWindow (live)       │
│  CER/WER metrics    │       └────────────────────────────────────┘
│  checkpoint save    │
└─────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  EXPORT PIPELINE  (cargo run -- export)                          │
├──────────────────────────────────────────────────────────────────┤
│  Python subprocess (tools/)                                      │
│                                                                  │
│  onnx_export/export_onnx.py                                      │
│  ├─ reads models/<id>/model_config.json                          │
│  ├─ builds structural PyTorch twin (vsrm_twin.py)                │
│  └─► exports/<id>/onnx/vsrm_export.onnx                          │
│                                                                  │
│  tex_export/export_tex.py  (vendored PlotNeuralNet)              │
│  ├─ macro VSRM diagram + TCN/ResBlock sub-diagrams               │
│  └─► exports/<id>/tex/  (upload to Overleaf to compile)          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  CTC VISUALIZATION  (fixture tests, standalone)                  │
├──────────────────────────────────────────────────────────────────┤
│  forward_lattice_viz  →  ASCII block-shade heatmap (░▒▓█)        │
│                       →  SVG: blue-white heatmap + DP arrows     │
│                                                                  │
│  prefix_beam_viz      →  ASCII box-drawing DAG (│├┬┐)            │
│                       →  SVG: Bézier edges +                     │
│                          top-K chips +                           │
│                          legend                                  │
│                                                                  │
│                       both → LRM Rust/outputs/                   │
│                                    └── (curate) → docs/assets/   │
└──────────────────────────────────────────────────────────────────┘
```

## CLI Usage (from the `LRM Rust` directory – project root)

### Training:
```
# Train new VSRM with default model ID "vsrm_{dataset_src}" (error if ID alr exists):
cargo run -- train --dataset [dataset_src]

# Train new VSRM with custom model ID (error if ID exists; --dataset required for fresh start):
cargo run -- train --model [vsrm_id] --dataset [dataset_src]

# Resume training from latest checkpoint (uses last completed epoch):
cargo run -- train --model [vsrm_id] --resume

# Resume training from specified epoch checkpoint:
cargo run -- train --model [vsrm_id] --resume [epoch]

# Train using a subset of the dataset (e.g. fraction = 0.1 for 10%):
cargo run -- train --model [...] --subset [fraction]

# Keep all checkpoints (default: keep most recent only; enables resume from earlier epochs):
cargo run -- train --model [...] --keep-all-checkpoints [on|off]
```

### Inference:
```
# Inference on a bundled video-transcript directory (predictions printed to stdout):
cargo run -- infer --model [vsrm_id] --input [path/to/dir_id]

# Live inference from default webcam (device index 0):
cargo run -- infer --model [vsrm_id] --live

# Live inference from a specific camera (OpenCV device index):
cargo run -- infer --model [vsrm_id] --live [camera_id]
```

### Other:
```
# Optional build N-gram LM for CTC Beam Decoder (trains if missing, else loads and evaluates perplexity):
cargo run -- build-lm --model [lm_id.bin] --corpus [path/to/corpus.txt] --n [n_gram_order]

# Preprocess a specific dataset for VSRM training:
cargo run -- preprocess --dataset [dataset_src]

# Export VSRM ONNX and TeX bundle:
cargo run -- export --model [vsrm_id]
```

## Model Summary

```text
----------------------------------------------------------------------------------------------------
Layer (Path)                                       | Shape                     |               Count
----------------------------------------------------------------------------------------------------
rb1.conv1.weight                                   | [128, 1, 3, 3, 3]         |         3456 params
rb1.conv1.bias                                     | [128]                     |          128 params
rb1.gn1.gamma                                      | [128]                     |          128 params
rb1.gn1.beta                                       | [128]                     |          128 params
rb1.conv2.weight                                   | [128, 128, 3, 3, 3]       |       442368 params
rb1.conv2.bias                                     | [128]                     |          128 params
rb1.gn2.gamma                                      | [128]                     |          128 params
rb1.gn2.beta                                       | [128]                     |          128 params
rb1.proj.weight                                    | [128, 1, 1, 1, 1]         |          128 params
rb1.proj.bias                                      | [128]                     |          128 params
rb2.conv1.weight                                   | [256, 128, 3, 3, 3]       |       884736 params
rb2.conv1.bias                                     | [256]                     |          256 params
rb2.gn1.gamma                                      | [256]                     |          256 params
rb2.gn1.beta                                       | [256]                     |          256 params
rb2.conv2.weight                                   | [256, 256, 3, 3, 3]       |      1769472 params
rb2.conv2.bias                                     | [256]                     |          256 params
rb2.gn2.gamma                                      | [256]                     |          256 params
rb2.gn2.beta                                       | [256]                     |          256 params
rb2.proj.weight                                    | [256, 128, 1, 1, 1]       |        32768 params
rb2.proj.bias                                      | [256]                     |          256 params
rb3.conv1.weight                                   | [512, 256, 3, 3, 3]       |      3538944 params
rb3.conv1.bias                                     | [512]                     |          512 params
rb3.gn1.gamma                                      | [512]                     |          512 params
rb3.gn1.beta                                       | [512]                     |          512 params
rb3.conv2.weight                                   | [512, 512, 3, 3, 3]       |      7077888 params
rb3.conv2.bias                                     | [512]                     |          512 params
rb3.gn2.gamma                                      | [512]                     |          512 params
rb3.gn2.beta                                       | [512]                     |          512 params
rb3.proj.weight                                    | [512, 256, 1, 1, 1]       |       131072 params
rb3.proj.bias                                      | [512]                     |          512 params
proj.weight                                        | [8192, 512]               |      4194304 params
proj.bias                                          | [512]                     |          512 params
tcn1.tcn_blocks.0.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.0.conv1.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.0.ln1.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.0.ln1.beta                         | [512]                     |          512 params
tcn1.tcn_blocks.0.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.0.conv2.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.0.ln2.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.0.ln2.beta                         | [512]                     |          512 params
tcn1.tcn_blocks.1.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.1.conv1.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.1.ln1.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.1.ln1.beta                         | [512]                     |          512 params
tcn1.tcn_blocks.1.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.1.conv2.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.1.ln2.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.1.ln2.beta                         | [512]                     |          512 params
tcn1.tcn_blocks.2.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.2.conv1.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.2.ln1.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.2.ln1.beta                         | [512]                     |          512 params
tcn1.tcn_blocks.2.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn1.tcn_blocks.2.conv2.bias                       | [512]                     |          512 params
tcn1.tcn_blocks.2.ln2.gamma                        | [512]                     |          512 params
tcn1.tcn_blocks.2.ln2.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.0.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.0.conv1.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.0.ln1.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.0.ln1.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.0.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.0.conv2.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.0.ln2.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.0.ln2.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.1.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.1.conv1.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.1.ln1.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.1.ln1.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.1.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.1.conv2.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.1.ln2.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.1.ln2.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.2.conv1.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.2.conv1.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.2.ln1.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.2.ln1.beta                         | [512]                     |          512 params
tcn2.tcn_blocks.2.conv2.weight                     | [512, 512, 3]             |       786432 params
tcn2.tcn_blocks.2.conv2.bias                       | [512]                     |          512 params
tcn2.tcn_blocks.2.ln2.gamma                        | [512]                     |          512 params
tcn2.tcn_blocks.2.ln2.beta                         | [512]                     |          512 params
fc.weight                                          | [512, 28]                 |        14336 params
fc.bias                                            | [28]                      |           28 params
----------------------------------------------------------------------------------------------------
Total Trainable Parameters: 27551900
Receptive Field: 69 frames
```

## CTC Loss Forward Lattice Example Visualization

```text
=== CTC Loss: ASCII Forward Log-Alpha Heatmap Lattice ===

Target Sequence: "cat"
Interleaved Target Tokens (row order): _, c, _, a, _, t, _
Interleaved Target IDs (row order): 27, 2, 27, 0, 27, 19, 27

Rows = Target Tokens/Indices, Cols = Timesteps
Timesteps = 20, Seq Len = 3, Inter Seq Len = 7

Legend = ['·' = unreachable, ' ' = low, '░' = mid-low, '▒' = mid, '▓' = mid-high, '█' = high]
Bins = [' ': [-47.97, -38.45), '░': [-38.45, -28.94), '▒': [-28.94, -19.43), '▓': [-19.43, -9.91), '█': [-9.91, -0.40]]

     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
  | |█  █  █  █  ▓  ▓  ▓  ▓  ▓  ▒  ▒  ▒  ▒  ▒  ▒  ░  ░           id = 27
  c |█  █  █  █  █  █  ▓  ▓  ▓  ▒  ▒  ▒  ░  ░  ░  ░  ░           id = 2
  | |·  █  █  ▓  █  ▓  █  █  █  ▓  ▓  ▓  ▓  ▓  ▓  ▒  ▒  ░  ░  ░  id = 27
  a |·  ▓  ▓  ▓  █  █  ▓  ▓  ▓  █  █  █  ▓  ▓  ▓  ▒  ▒  ▒  ░  ░  id = 0
  | |·  ·  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  █  █  █  ▓  ▓  ▒  ▒  ▒  id = 27
  t |·  ·  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  █  ▓  ▓  ▓  ▓  id = 19
  | |·  ·  ·  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  ▓  id = 27
```

## CTC Decode Prefix Beam Example Visualization

```text
=== CTC Decode: ASCII Prefix Beam Search Graph ===

Decoded Sequence: "cat"
Greedy Argmax Logits: [_, _, _, c, c, c, _, _, _, a, a, a, _, _, _, t, t, t, _, _]

Rows = Timesteps, Cols = Ranks
Timesteps = 20, Beam width = 4
[..] = best path,  (..) = other hypotheses, ε = empty sequence

          r = 0  r = 1   r = 2   r = 3

t = init  [ε]
           ├──────┬───────┬───────┐
t = 0     [ε]    (m)     (f)     (d)
           ├─────┬│──────┬│──────┐│
           │     ││┌─────││──────│┘
           │     │└│─────││──────│┐
t = 1     [ε]    (d)     (f)     (m)
           ├─────┬│──────┬│──────┐│
t = 2     [ε]    (d)     (f)     (m)
           │      │┌──────┘       │
           │      ││      ┌───────┘
           │      └│──────│───────┐
t = 3     [c]    (fc)    (mc)    (dc)
           │      │┌──────│───────┘
           │      └│──────│───────┐
t = 4     [c]    (dc)    (mc)    (fc)
           │      │┌──────┘       │
           │      └│──────┐       │
t = 5     [c]    (mc)    (dc)    (fc)
           │      │┌──────│───────┘
           │      └│─────┐│
           │       │     │└───────┐
t = 6     [c]    (fc)    (mc)    (dc)
           │      │┌──────┘       │
           │      └│──────┐       │
t = 7     [c]    (mc)    (fc)    (dc)
           │      │       │       │
t = 8     [c]    (mc)    (fc)    (dc)
           │      │       │┌──────┘
           │      │       └│──────┐
t = 9     [ca]   (mca)   (dca)   (fca)
           │      │       │┌──────┘
           │      │       └│──────┐
t = 10    [ca]   (mca)   (fca)   (dca)
           │      │       │       │
t = 11    [ca]   (mca)   (fca)   (dca)
           │      │       │       │
t = 12    [ca]   (mca)   (fca)   (dca)
           │      │       │       │
t = 13    [ca]   (mca)   (fca)   (dca)
           │      │       │       │
t = 14    [ca]   (mca)   (fca)   (dca)
           │      │┌──────┘       │
           │      └│──────┐       │
t = 15    [cat]  (fcat)  (mcat)  (dcat)
           │      │       │       │
t = 16    [cat]  (fcat)  (mcat)  (dcat)
           │      │       │       │
t = 17    [cat]  (fcat)  (mcat)  (dcat)
           │      │       │       │
t = 18    [cat]  (fcat)  (mcat)  (dcat)
           │      │       │       │
t = 19    [cat]  (fcat)  (mcat)  (dcat)
```