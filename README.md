# Project: End-to-End Lip Reading System in Rust (No Audio)

Objective:
Build a real-time, audio-free lip-reading system entirely in Rust, covering data ingestion, model architecture, training, and decoding, with a long-term goal of live camera inference using a dynamically tracking mouth-cropped ROI.

---

## Accomplishments to Date

### Data Ingestion & Preprocessing
- Implemented a Rust-based video pipeline using OpenCV.
- Decodes video files frame by frame.
- Converts frames to grayscale.
- Crops a fixed mouth region of interest (ROI).
- Flattens pixel data into contiguous Vec<f32> tensors.
- Applies per-sample Z-score normalization (zero mean, unit variance).
- Built dataset utilities that:
  - Infer file name stems automatically.
  - Pair videos with alignment annotations.
  - Download and extract compressed datasets when missing.
- For now, using GRID dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset) (make sure to unzip, rename from "data" to "grid-lr-dataset", and place under project's dedicated "data" dir).

### Alignment & Vocabulary Handling
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

### Model Architecture (Rust, Burn)
- Implemented a full spatiotemporal neural network in Rust.
- Uses a 3D convolutional front-end for joint spatial–temporal feature extraction.
- Replaced BiLSTMs with a Temporal Convolutional Network (TCN) to improve:
  - Parallelism
  - Inference latency
  - Deployment simplicity
- Built modular TCN blocks featuring:
  - Dilated causal convolutions
  - Residual connections
  - Dropout and non-linear activations
- Added a projection head mapping features to per-time-step character logits.

### Training Pipeline
- Implemented a complete training loop in Rust.
- Supports batching and epoch-based training.
- Handles explicit train and eval mode switching.
- Integrated the Adam optimizer with configurable learning rates.
- Implemented Noam-style learning rate warmup and scheduling.
- Added numerical utility functions (mean, standard deviation, normalization).

### Custom CTC Loss
- Implemented Connectionist Temporal Classification (CTC) loss from scratch.
- Uses forward-backward dynamic programming.
- Performs all computations in log-space for numerical stability.
- Correctly handles:
  - Blank symbols
  - Repeated labels
  - Variable-length input and target sequences
- Designed to be framework-agnostic within Burn.

### Decoding & Inference
- Implemented greedy CTC decoding.
- Implemented prefix beam search decoding, including:
  - Separate blank and non-blank probability tracking
  - Log-probability accumulation
  - Beam pruning and prefix merging
- Decoder architecture designed to support incremental and streaming inference.

### Language Model Integration (In Progress)
- Added a dedicated language model interface for prefix beam search.
- Designed for character-level n-gram scoring.
- Supports configurable:
  - Language model weight (alpha)
  - Insertion bonus (beta)
- Currently implementing the LM logic to improve decoding coherence.
- Training on the OpenSLR LibriSpeech LM Corpus dataset.

### System Design & Engineering Decisions
- Entire pipeline implemented in Rust.
- No Python, PyTorch, or TensorFlow dependencies.
- Emphasis on:
  - Determinism
  - Memory safety
  - Low-latency inference
- Architecture explicitly designed to support future extensions:
  - Real-time webcam input
  - Sliding-window streaming inference
  - Dynamic mouth tracking via face detection and landmarks
  - Portable deployment via ONNX or native Rust runtimes (e.g., Tract)

---

## Current Status
- Offline lip-reading inference is functionally complete, pending language model integration.
- Real-time inference pipeline is architecturally planned but not yet implemented.
- Greedy decoding works end-to-end; beam search decoding nearing completion.

---

## Summary
This project represents a full end-to-end implementation of a lip-reading system in Rust, covering data processing, model architecture, training, loss computation, and decoding. The system is designed with real-time deployment in mind and avoids reliance on Python-based ML frameworks, emphasizing performance, safety, and extensibility.
