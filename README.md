# Project: End-to-End Visual Speech Recognition Model (VSRM) in Rust (Audioless)

Objective:
Build a real-time, audio-free VSRM lip-reading system entirely in Rust, covering data ingestion, model architecture, training, loss computation, decoding, and language model integration, with a long-term goal of live camera inference using a dynamically tracking mouth-cropped ROI.

---

## Accomplishments to Date

### Data Ingestion (```pipeline/io.rs``` and ```pipeline/adapters/grid.rs```)

- For now, using [GRID](https://zenodo.org/records/3625687) corpus as proof of concept that the VSRM can converge (speaker ("s1", "s2", ..., "s34") data organized into "frames" and "alignments" directories under a self created "data/grid-lr-corpus").
- In future, will consider using the [Oxford-BBC LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) corpus in the future, for a broad-term generalization to conversational speech to generalize the VSRM to broader use.
- Built dataset utilities that:
  - Infer file name stems automatically.
  - Pair videos with alignment annotations.
  - Download and extract compressed datasets when missing.
- Implemented a video pipeline using OpenCV in ```io.rs``` that:
  - Decodes video files frame by frame.
  - Converts frames to grayscale.
  - Crops a fixed mouth region of interest (ROI).
  - Flattens pixel data into contiguous ```Vec<u8>``` tensors.
  - In future, might consider using pre-trained Haar Cascade or a DNN-based Face Detector to find face, then estimate mouth region.

### Data Standardization (```adapters/```)

- Implemented dataset adapter ```adapters/``` that:
  - Contains source-specific logic to map raw datasets (GRID, LRW, etc.) into a standardized ```VsrmItem``` format.
  - Scales 8-bit grayscale pixel values to within $[0, 1]$ for normalization.
- In future, will consider FPS standardization to a target FPS (25) as well (frame dropping preferred over interpolation due to simplicity and avoidance of ghost data).

### Data Batching (```pipeline/batcher.rs```)

- Developed a custom ```VsrmBatcher``` that:
  - Scales pixel values to [0, 1].
  - Centers pixel values to zero mean and unit variance.
- Padding handled by:
  - Finding longest video-frames/transcript-sequences among a batch of sequences (as ```max_t```/```max_l```).
  - Padding variable-length video frames in that batch to ```max_t``` with $0$.
  - Padding variable-length transcript sequences in that batch to ```max_l``` with ```BLANK_ID```.
- Uses a CPU-to-GPU staging strategy, where tensors are collated on the ```NdArray``` CPU backend before a single-shot move to the ```Wgpu``` GPU backend for minimizing PCIe bus latency.

### Alignment & Vocabulary Handling (```vocab.rs```)

- Implemented parsing of .align files.
- Filters out silence tokens ("sil").
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

### VSR Model Architecture (```vsrm/vsrm.rs``` and ```vsrm/tcn.rs```)

- Implemented a full spatiotemporal VSRM in Rust.
- Uses a 3D convolutional (Conv3D) front-end for joint spatial–temporal feature extraction.
- Gave strided convolutions to Conv3D layers (over 3D maxpooling) for learned rather than naive downsampling.
- Uses GroupNorm following Conv3D layers for mitigating internal covariance shift during forward/backward passes. GroupNorm chosen over:
  - BatchNorm because BatchNorm struggles with small batches and larger batches is memory-heavy against high-dim video data.
  - LayerNorm because LayerNorm globalizes its averaging across all channels, pixels, and timesteps, leading to "washing-out" of localized variations in spatial data.
- Replaced BiLSTMs with a Temporal Convolutional Network (TCN) to improve:
  - Parallelism
  - Inference latency
  - Deployment simplicity
- Built modular TCN blocks featuring:
  - Dilated causal convolutions
  - Residual connections
  - Dropout and non-linear activations
- Added a projection head mapping features to per-time-step character logits.

### Training Pipeline (```training/learner.rs``` and ```training/trainer.rs```)

- Keeping a legacy ```trainer.rs``` file implementing a manual training loop to test model convergence on dummy data.
- Implemented a complete training and validation pipeline using Burn's ```Learner``` API in Rust as ```learner.rs```.
- Supports:
  - Batching
  - Epoch-based training
  - Auto-checkpointing
  - Metric logging
  - Train/validation dataset splitting
- Handles dynamic train/eval mode switching implicitly with Burn's ```Autodiff``` and ```Module``` traits (which allows gradient tracking).
- Integrated the Adam optimizer with configurable learning rates.
- Implemented Noam-style learning rate warmup and scheduling.
- Added numerical utility functions (mean, standard deviation, normalization).

### Custom CTC Loss (```ctc/ctc_loss.rs```)

- Implemented Connectionist Temporal Classification (CTC) loss from scratch.
- Uses forward-backward dynamic programming.
- Performs all computations in log-space for numerical stability.
- Correctly handles blank symbols, repeated labels, and variable-length input/target sequences
- Designed to be framework-agnostic within Burn.

### Custom CTC Decoding & Inference (```ctc/ctc_decoder.rs```)

- Implemented CTC decoding (greedy and prefix beam search methods) from scratch.
- Prefix beam search decoding has:
  - Separate blank and non-blank probability tracking
  - Log-probability accumulation
  - Beam pruning and prefix merging
- Decoder architecture designed to support incremental/streaming inference.

### Language Model Integration (```ctc/lm.rs```)

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

---

## Current Status

- I/O and data aqcuisition helper functions are finished.
- Data adapting (for GRID), preprocessing, and batching pipeline is officially done.
- Offline lip-reading inference and language model integration is functionally complete.
- Greedy decoding and Beam Search decoding with a char-level N-gram language model both work end-to-end.
- Training and validation pipeline is implemented and wired into Burn's learner framework.
- Pending VSRM train/eval.
- Dynamic mouth tracking planned but not implemented yet.
- Planning FPS video standardization to unify varying video-transcript datasets.

---

## Summary

This project represents a full end-to-end implementation of a VSRM lip-reading system in Rust, covering data processing, model architecture, training, loss computation, and decoding. The system is designed with real-time deployment in mind and avoids reliance on Python-based ML frameworks, emphasizing performance, safety, and extensibility.

## Project Tree

```
Lip Reading Model
├─ LICENSE
├─ LRM Python
│  ├─ application
│  │  ├─ animation.gif
│  │  ├─ general_utils.py
│  │  ├─ lipread.py
│  │  ├─ model_utils.py
│  │  └─ test_video.mp4
│  ├─ lipread.ipynb
│  ├─ main.py
│  └─ requirements.txt
├─ LRM Rust
│  ├─ Cargo.lock
│  ├─ Cargo.toml
│  ├─ data
│  │  ├─ grid-lr-corpus
│  │  │  ├─ alignments
│  │  │  │  ├─ s1
│  │  │  │  │  ⋮
│  │  │  │  └─ s34
│  │  │  └─ frames
│  │  │     ├─ s1
│  │  │     │  ⋮
│  │  │     └─ s34
│  │  └─ librispeech-lm-norm
│  │     └─ librispeech-lm-norm.txt
│  ├─ models
│  │  └─ ⋯
│  ├─ rust-toolchain.toml
│  ├─ rustfmt.toml
│  ├─ src
│  │  ├─ ctc
│  │  │  ├─ ctc_decode.rs
│  │  │  ├─ ctc_loss.rs
│  │  │  ├─ lm.rs
│  │  │  └─ mod.rs
│  │  ├─ inference
│  │  │  ├─ mod.rs
│  │  │  └─ predictor.rs
│  │  ├─ lib.rs
│  │  ├─ main.rs
│  │  ├─ pipeline
│  │  │  ├─ batcher.rs
│  │  │  ├─ dataset.rs
│  │  │  ├─ io.rs
│  │  │  ├─ mod.rs
│  │  │  └─ adapters
│  │  │     ├─ grid.rs
│  │  │     └─ mod.rs
│  │  ├─ training
│  │  │  ├─ learner.rs
│  │  │  ├─ metrics.rs
│  │  │  ├─ mod.rs
│  │  │  └─ trainer.rs
│  │  ├─ utils.rs
│  │  ├─ vocab.rs
│  │  └─ vsrm
│  │     ├─ mod.rs
│  │     ├─ tcn.rs
│  │     └─ vsrm.rs
│  └─ target
│     ├─ .rustc_info.json
│     ├─ debug
│     └─ flycheck0
│        ├─ stderr
│        └─ stdout
├─ NOTES.md
├─ README.md
└─ papers
   ├─ 2006-Graves-CTC.pdf
   └─ 2016-Assael-LipNet.pdf
```