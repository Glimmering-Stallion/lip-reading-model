<!-- This is the file that serves as the entry point (what it is how to run, what works now, where to go next) -->
<!-- Rule of thumb for what goes here: "Would a new visitor need this in 60 seconds?" -->

# End-to-End Visual Speech Recognition Model (VSRM) in Rust (Audioless)

## Objective

Build a real-time, audio-free VSRM lip-reading system entirely in Rust, covering data ingestion, model architecture, training, loss computation, decoding, and language model integration, with a long-term goal of live camera inference using a dynamically tracking mouth-cropped ROI.

## Motivations

- Deep learning is currently heavily Python-dominated, despite being well established. For **real-time** systems, Python stacks often imply heavier runtimes and more moving parts.
- My goal with this project was to test what was possible with Rust for ML by building an **audio-free end-to-end Visual Speech Recognition Model (VSRM)** for lip-reading tasks, using **Burn** for the deep learning side, and **OpenCV** for the computer vision side.

- I have two layers of success established:
  - I started with the long-term goal of wanting broad-corpus training (like Oxford BBC's LRW or LRS2/3) to enable generalized live model inferencing (deferred for now, as a phase 2 future ordeal).
  - The current checkpoint represents a milestone in a working data ingestion system (source-agnostic), training framework, with model demonstrating palpable loss convergence on the GRID Audio-Visual Speech Corpus, and a working inference engine using a swappable mouth tracker.

## Why Rust?

- Rust provides tighter thread and runtime control.
- Native compilation should help keep inference overhead low.
- Rust's ownership/borrow system eliminates entire classes of runtime errors common in other low-level system languages (like dangling pointers, double-frees, null dereferences, etc.).
- Unlike Python with its Global Interpreter Lock, Rust's concurrent nature allows multi-core parallelism (in my case, Burn's ```DataLoader``` spawning multi-worker threads to fetch/process frames, or ```VsrmBatcher``` concurrently collating data on CPU before moving to GPU).
- Rust compiles projects into one lightweight deployable binary which mitigates container bloat / deployment sizes (compared with Python + CUDA + framework heavy stacks).

## Stack

The following are the section-by-section **Deep dives** found in the [**LRM portfolio article**](docs/lrm-portfolio-article.md). Each link below jumps to the matching *Part*.

- **Data pipeline** — GRID normalization/bundling, adapters, batching, norm stats, optional `cropped_frames/` cache.  
  [Part 1 — Data pipeline](docs/lrm-portfolio-article.md#part-1--data-pipeline)

- **Model architecture** — Conv3D/ResBlock frontend, GroupNorm, TCN backend, CTC head.  
  [Part 2 — Neural architecture](docs/lrm-portfolio-article.md#part-2--neural-architecture)

- **Training framework** — Burn `Learner`, LR scheduling, CTC loss/decode, optional char n-gram LM.  
  [Part 3 — DL training framework](docs/lrm-portfolio-article.md#part-3--dl-training-framework)

- **Inference framework** — Haar tracker, overlays, speech gating, file and live inference paths.  
  [Part 4 — CV inference framework](docs/lrm-portfolio-article.md#part-4--cv-inference-framework)

- **CLI & export** — `build-lm`, `preprocess`, `train`, `infer`, `export` (ONNX + TeX).  
  [Part 5 — CLI design and usage](docs/lrm-portfolio-article.md#part-5--cli-design-and-usage)

## Quick Start

Note: Currently verified end-to-end on GRID dataset only; other datasets like LRW/LRS are planned extensions, not plug-and-play yet.

### Setup

```
# 1. Install Rust toolchain manager (rustup)

# macOS (zsh)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Windows (PowerShell)
winget install Rustlang.Rustup
$env:Path += ";$env:USERPROFILE\.cargo\bin"
# Note: USERPROFILE is your home folder on Windows (e.g., C:\Users\YourName)

# check versions
rustup --version
cargo --version

# 2. Install OpenCV system libs

# macOS (zsh)
brew install opencv

# Windows (PowerShell)
winget install OpenCV.OpenCV

# 3. Clone and enter Rust Crate
git clone https://github.com/Glimmering-Stallion/lip-reading-model.git
cd "lip-reading-model/LRM Rust"

# 4. Build and compile local package and dependencies
cargo build
```

### Data layout (required)

The CLI expects local datasets under `LRM Rust/data/` with these paths:

```text
data/
├─ grid-lr-corpus/
│  └─ <speaker>/<sample_id>/
│     ├─ <sample_id>.mp4   # preferred (normalized video)
│     └─ <sample_id>.txt   # preferred (normalized transcript)
└─ librispeech-lm-norm/
   └─ librispeech-lm-norm.txt
```

### First Run

```
# 1. Preprocess on GRID dataset (only GRID supported, also may take a while as it runs for entire dataset)
cargo run -- preprocess --dataset grid

# 2. Train on GRID dataset
cargo run -- train --model my_vsrm --dataset grid

# 3. Infer on a GRID video-transcript sample
cargo run -- infer --model my_vsrm --input path/to/<bundle_id>
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

## Project Tree (abridged)

Top-level layout and main Rust module directories. For every file and nested path, see [**NOTES**](./NOTES.md#project-tree-detailed) (*Project Tree (detailed)*).

```
Lip Reading Model
├─ LICENSE
├─ docs
│  ├─ assets
│  └─ lrm-portfolio-article.md
├─ LRM Python
│  ├─ application
│  ├─ lipread.ipynb
│  ├─ main.py
│  └─ requirements.txt
├─ LRM Rust
│  ├─ Cargo.toml
│  ├─ Cargo.lock
│  ├─ CHANGELOG.md
│  ├─ data
│  ├─ models
│  ├─ outputs
│  ├─ exports
│  ├─ tools
│  │  ├─ plotneuralnet
│  │  ├─ onnx_export
│  │  └─ tex_export
│  └─ src
│     ├─ lib.rs
│     ├─ main.rs
│     ├─ utils.rs
│     ├─ vocab.rs
│     ├─ pipeline
│     ├─ vsrm
│     ├─ ctc
│     ├─ training
│     └─ inference
├─ NOTES.md
├─ PLANS.md
└─ README.md
```
