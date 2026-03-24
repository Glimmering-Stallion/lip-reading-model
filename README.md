# Project: End-to-End Visual Speech Recognition Model (VSRM) in Rust (Audioless)

Objective:
Build a real-time, audio-free VSRM lip-reading system entirely in Rust, covering data ingestion, model architecture, training, loss computation, decoding, and language model integration, with a long-term goal of live camera inference using a dynamically tracking mouth-cropped ROI.

## Accomplishments to Date

### Data Ingestion (`pipeline/io.rs` and `pipeline/adapters/grid/grid_dataset.rs`)

- For now, using [GRID](https://zenodo.org/records/3625687) corpus as proof of concept that the VSRM can converge (speaker ("s1", "s2", ..., "s34") data organized into sample bundles under `data/grid-lr-corpus/<speaker>/<sample_id>/`). Each sample folder holds video (`<sample_id>.mp4` preferred after preprocess, else `.mpg`) and transcript (`<sample_id>.txt` preferred after preprocess, else `.align`).
- In future, will consider using the [Oxford-BBC LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) corpus in the future, for a broad-term generalization to conversational speech to generalize the VSRM to broader use.
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

### Data Standardization & Normalization (`adapters/`)

- Implemented dataset adapters that contain source-specific logic to:
  - Transcode src video files into `.mp4`, and write `.txt` from src transcript files, then remove redundant src video/transcript files when safe.
  - Map raw datasets (GRID, LRW, etc.) into a standardized `VsrmItem` format.
  - Rely on the more abstract `DatasetSplit` utility in `pipeline/dataset.rs` for train/val/test partitioning.
- The adapter modules are to reshape a dataset into a dir containing sharded video-transcript bundles, where video files are `.mp4` and transcript files are `.txt` (GRID adapter modules enforces this currently, but other dataset sources are intended to follow the same form).
- In future, will consider FPS standardization to a target FPS (25) as well (frame dropping preferred over interpolation due to simplicity and avoidance of ghost data).

### Data Batching (`pipeline/batcher.rs`)

- Developed a custom `VsrmBatcher` that takes a collection of standardized `VsrmItem`  standardizes and pads data.
- Standardization handled by:
  - Scaling pixel values to [0, 1].
  - Centering pixel values to zero mean and unit variance.
- Padding handled by:
  - Finding longest video-frames/transcript-sequences among a batch of sequences (as `max_t`/`max_l`).
  - Padding variable-length video frames in that batch to `max_t` with $0$.
  - Padding variable-length transcript sequences in that batch to `max_l` with `BLANK_ID`.
- Uses a CPU-to-GPU staging strategy, where tensors are collated on the `NdArray` CPU backend before a single-shot move to the `Wgpu` GPU backend for minimizing PCIe bus latency.

### Data Partitioning (`pipeline/dataset.rs`)

- Dataset splitting policy is delegated to a generic and source-agnostic `DatasetSplit` wrapper, to allow any dataset (GRID, LRW, etc.) to be partitioned through index-mapping without modifying the more specialized adapter logic.
- Applies a random but deterministic shuffle to the index-mapping.
- Then partitions dataset instances into train/val/test splits.

### Alignment & Vocabulary Handling (`vocab.rs`)

- Implemented parsing of .align files.
- Filters out silence tokens ("sil", "sp").
- Inserts spaces between words in alignment-derived targets (`SPACE_ID`) for WER metrics.
- Converts labels into integer sequences using a bidirectional vocabulary map.
- Designed a character-level vocabulary including:
  - Lowercase letters
  - Digits
  - Punctuation
  - Space
  - A dedicated CTC blank symbol
- Ensured the blank symbol:
  - Appears only in model outputs.
  - Never appears in training targets.
  - Is removed during decoding.

### VSR Model Architecture (`vsrm/vsrm.rs` and `vsrm/tcn.rs`)

- Implemented a full spatiotemporal VSRM in Rust.
- Uses a 3D convolutional (Conv3D) front-end for joint spatial–temporal feature extraction.
- Gave strided convolutions to Conv3D layers (over 3D maxpooling) for learned rather than naive downsampling.
- Uses GroupNorm following Conv3D layers for mitigating internal covariance shift during forward/backward passes. GroupNorm chosen over:
  - BatchNorm because BatchNorm struggles with small batches and larger batches is memory-heavy against high-dim video data.
  - LayerNorm because LayerNorm globalizes its averaging across all channels, pixels, and timesteps, leading to "washing-out" of localized variations in spatial data.
- Replaced BiLSTMs with Temporal Convolutional Networks (TCN) to improve:
  - Parallelism
  - Inference latency
  - Deployment simplicity
- Built modular TCN blocks featuring:
  - Dilated causal convolutions
  - Per-timestep causal LayerNorm (normalizes over channels only)
  - Residual blocks
  - Dropout and non-linear activations
- Added a projection head mapping features to per-time-step character logits.

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
- Implemented Noam-style learning rate warmup and scheduling.
- Added numerical utility functions (mean, standard deviation, normalization).

### Custom CTC Loss (`ctc/ctc_loss.rs`)

- Implemented custom Connectionist Temporal Classification (CTC) loss.
- Uses forward-backward dynamic programming.
- Performs all computations in log-space for numerical stability.
- Correctly handles blank symbols, repeated labels, and variable-length input/target sequences
- Designed to be framework-agnostic within Burn.

### Custom CTC Decoding & Inference (`ctc/ctc_decode.rs`)

- Implemented custom CTC decoding (greedy and prefix beam search).
- Prefix beam search decoding has:
  - Separate blank and non-blank probability tracking
  - Log-probability accumulation
  - Beam pruning and prefix merging
- Decoder architecture designed to support incremental/streaming inference.

### Language Model Integration (`ctc/lm.rs`)

- Incorporated a dedicated language model interface for CTC decoder's prefix beam search.
- Designed for character-level N-gram scoring.
- In future, might consider a word-level N-gram.
- Uses an enum to support different LM types (N-gram LM, Neural LM, etc.)
- Supports configurable:
  - Language model weight (alpha): controls influence of LM over base VSRM's predictions (lower alpha means trusting VSRM over LM more and vice versa for higher alpha).
  - Insertion bonus (beta): counteracts LM's bias toward shorter sequences (adding more tokens makes log-prob score more negative, where beta adds a small positive bonus).
- Currently implementing an N-gram LM to improve decoding coherence.
- Training on the [OpenSLR LibriSpeech LM Norm](https://www.openslr.org/11) corpus.
- For now, just using self-trained char-level trigram model.
- In future, will consider using pre-trained [trigram ARPA LM](https://www.openslr.org/11) word-level model.
- In future, will also consider using a tiny neural LM (char/BPE GRU or small Transformer) with prefix-state caching per beam (running it only on top-K acoustic symbols each step to bound cost; or use it as an N-best reranker after beam, which mitigates per-frame latency).

### System Design & Engineering Decisions

- Entire pipeline implemented in Rust (no Python, PyTorch, or TensorFlow dependencies) for emphasis on:
  - Determinism
  - Parallelism
  - Memory safety
  - Low-latency inference
- Architecture explicitly designed to support future extensions:
  - Real-time webcam input
  - Sliding-window streaming inference
  - Dynamic mouth tracking via face detection and landmarks
  - Portable deployment via ONNX or native Rust runtimes (e.g., Tract)

## Current Status

- **I/O and data acquisition:** Video encoding/decoding, mouth ROI extraction utilities, and dataset download/extract helpers are in place.
- **Data pipeline:** Adapter mapping (at least for GRID), preprocessing, deterministic splitting, and batching are implemented.
- **Loss:** Custom CTC loss implemented in log-space (forward/backward DP) with vectorized batch support for variable-length sequences.
- **Decoding:** Greedy and prefix beam-search CTC decoding, optionally rescored with the integrated char-level N-gram LM (supports alpha/beta).
- **Training:** Burn `Learner`-based training/validation loop with checkpointing, metrics, and LR scheduling. Uses `create_dataloaders` helper to handle train/val splits, batching, and dataloading for source-specific datasets.
- **Inference:** Using an `InferenceSession` engine, which supports static file inference (as a bundled video-transcript input) with `infer_file` and async live webcam inference with `infer_live` (main thread captures/tracks/overlays; worker thread runs model forward passes).
- **Verification:** Unit tests for CTC loss/decoding, tracker ROI behavior, and sanity checks for model input/output dataflow; training convergence validated via overfit tests.
- **Mouth tracking:** Haar-cascade face/mouth detection with stabilized mouth ROI per frame.
- **CLI:** `build-lm`, `preprocess`, `train` (new/resume), and `infer` (static file / live cam input types) subcommands are available.
- **Inference Viz Overlay:** Inference pipeline's visualization overlay for both static file and live camera inference modes is implemented.

## Pending / Future Work

- **Add Landmark-Based Tracker:** Improve ROI stability and accuracy, plus rotational invariance benefits by adding a landmark/pose-based tracker backend (e.g. MediaPipe) as a separate tracker option to the existing layered Haar cascades tracker.
- **Grad-CAM For Overlay Visualization:** During the forward pass, save the "activations" of the last TCN or Conv layer. Treat those activations as a heatmap. Upscale that heatmap to match the mouth-crop size. Then alpha-blend it (transparent overlay) onto the video.
- **FPS video standardization:** Unify potentially varying frame-rates between different video-transcript dataset sources.
- **Word-Level N-gram vs. Char Level Decoder Incongruity:** Current decoding uses character-level LM scoring; evaluate unifying with a word-level LM/tokenization or retraining the LM to match the decoder’s output unit.
- **Talking vs. Non-Talking States:** Live inference still runs the lip-reading model whenever the camera stream is active. Add a **gating or state layer** (audio-free): e.g. mouth-ROI motion / frame-difference energy, tracker confidence (face/mouth found), or optional logits-based confidence, to classify **active visual speech** vs **idle/silent** and suppress, clear, or hold the displayed prediction accordingly.

## CLI Usage

From the `LRM Rust` directory (project root):

```
# Build N-gram LM (trains if missing, else loads and evaluates perplexity)
cargo run -- build-lm --model [my_lm.bin] --corpus [path/to/corpus.txt] --n [n_gram_order]

# Preprocess a specific dataset for the VSRM:
cargo run -- preprocess --dataset [dataset_src]

# Train new VSRM with default model ID "vsrm_{dataset_src}" (error if ID alr exists):
cargo run -- train --dataset [dataset_src]

# Train new VSRM with custom model ID (error if ID exists; --dataset required for fresh start):
cargo run -- train --model [my_vsrm] --dataset [dataset_src]

# Resume training from latest checkpoint (uses last completed epoch):
cargo run -- train --model [my_vsrm] --resume

# Resume training from specified epoch checkpoint:
cargo run -- train --model [my_vsrm] --resume [epoch]

# Train using a subset of the dataset (e.g. fraction = 0.1 for 10%):
cargo run -- train --model [...] --subset [fraction]

# Keep all checkpoints (default: keep most recent only; enables resume from earlier epochs):
cargo run -- train --model [...] --keep-all-checkpoints [on|off]

# Inference on a bundled video-transcript directory (predictions printed to stdout):
cargo run -- infer --model [my_vsrm] --input [path/to/dir_id]

# Live inference from default webcam (device index 0):
cargo run -- infer --model [my_vsrm] --live

# Live inference from a specific camera (OpenCV device index):
cargo run -- infer --model [my_vsrm] --live [my_camera]
```

## Attributions

### Face and Mouth Detection (Haar Cascades)

Mouth tracking uses pre-trained Haar cascade classifiers:

- **Face detection:** `haarcascade_frontalface_alt2.xml`
- **Mouth detection:** `haarcascade_mcs_mouth.xml`

These cascade files are obtained from [opencv-processing/cascade-files](https://github.com/atduskgreg/opencv-processing/tree/master/lib/cascade-files).

**Research use:** If you use these detectors or related ideas, please cite one of the following papers:

- Castrillón Santana, M., Déniz Suárez, O., Hernández Tejera, M., & Guerra Artal, C. (2007). **ENCARA2: Real-time Detection of Multiple Faces at Different Resolutions in Video Streams.** _Journal of Visual Communication and Image Representation_, 18(2), 130–140.
- Castrillón Santana, M., Déniz Suárez, O., Hernández Sosa, D., & Lorenzo Navarro, J. (2007). **Using Incremental Principal Component Analysis to Learn a Gender Classifier Automatically.** _1st Spanish Workshop on Biometrics_, Girona, Spain.
- Castrillón-Santana, M., Déniz-Suárez, O., Antón-Canalís, L., & Lorenzo-Navarro, J. (2008). **Face and Facial Feature Detection Evaluation.** _Third International Conference on Computer Vision Theory and Applications (VISAPP)_.

## References / Further Reading

Resources that informed the implementation of concepts in this project:

- **CTC loss:** [Sequence Modeling with CTC](https://distill.pub/2017/ctc/) (Hannun, 2017, Distill)
- **CTC (original):** Graves et al. (2006). Connectionist Temporal Classification. ICML. [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- **N-gram language models, smoothing (Witten-Bell, etc.):** Jurafsky, D. & Martin, J. H. _Speech and Language Processing_ (3rd ed.), Ch. 3. [PDF](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- **LipNet:** Assael et al. (2016). End-to-End Sentence-level Lipreading. [arXiv](https://arxiv.org/abs/1611.01599)
- **Temporal Convolutional Networks (original):** Lea et al. (2016). Temporal Convolutional Networks for Action Segmentation and Detection. [arXiv](https://arxiv.org/abs/1611.05267)
- **TCNs for sequence modeling (popularized):** Bai et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. [arXiv](https://arxiv.org/abs/1803.01271)
- **AdamW optimizer** Loshchilov, I. & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. [arXiv](https://arxiv.org/abs/1711.05101)

## Project Tree

```
Lip Reading Model
├─ LICENSE
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
│  ├─ data
│  │  ├─ grid-lr-corpus
│  │  │  ├─ s1
│  │  │  │  └─ <stem_id>
│  │  │  │  ⋮  ├─ <stem_id>.mp4
│  │  │  │  ⋮  └─ <stem_id>.text
│  │  │  └─ s34
│  │  │     └─ <stem_id>
│  │  │        ├─ <stem_id>.mp4
│  │  │        └─ <stem_id>.text
│  │  └─ librispeech-lm-norm
│  │     └─ librispeech-lm-norm.txt
│  ├─ models
│  ├─ outputs
│  ├─ rust-toolchain.toml
│  ├─ rustfmt.toml
│  ├─ src
│  │  ├─ cli.rs
│  │  ├─ context.rs
│  │  ├─ ctc
│  │  │  ├─ ctc_decode.rs
│  │  │  ├─ ctc_loss.rs
│  │  │  ├─ lm.rs
│  │  │  └─ mod.rs
│  │  ├─ inference
│  │  │  ├─ loader.rs
│  │  │  ├─ mod.rs
│  │  │  ├─ overlay.rs
│  │  │  └─ predictor.rs
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
└─ README.md
```
