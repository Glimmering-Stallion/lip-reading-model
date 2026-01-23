# Project: End-to-End Visual Speech Recognition Model (VSRM) in Rust (Audioless)

Objective:
Build a real-time, audio-free VSRM lip-reading system entirely in Rust, covering data ingestion, model architecture, training, and decoding, with a long-term goal of live camera inference using a dynamically tracking mouth-cropped ROI.

---

## Accomplishments to Date

### Data Ingestion & Preprocessing

- For now, using [GRID](https://zenodo.org/records/3625687) corpus as proof of concept that the VSRM can converge (make sure to unzip, place speaker ("s1", "s2", ..., "s34") and "alignments" files under a self created "grid-lr-corpus", and place under project's dedicated "data" dir).
- In future, will consider using the [Oxford-BBC LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) corpus in the future, for a broad-term generalization to conversational speech to generalize the VSRM to broader use.
- Built dataset utilities that:
  - Infer file name stems automatically.
  - Pair videos with alignment annotations.
  - Download and extract compressed datasets when missing.
- Implemented a video pipeline using OpenCV in ```io.rs``` that:
  - Decodes video files frame by frame.
  - Converts frames to grayscale.
  - Crops a fixed mouth region of interest (ROI).
  - In future, might consider using pre-trained Haar Cascade or a DNN-based Face Detector to find face, then estimate mouth region.
- Implemented a custom preprocessor ```grid.rs``` that:
  - Flattens pixel data into contiguous ```Vec<f32>``` tensors.
  - Scales pixel values to within $[0, 1]$ for normalization.

### Data Batching

- Developed a custom ```VsrmBatcher``` that handles padding by:
  - Padding all video sequences in a batch to the maximum found sequence length (timesteps) within that batch with zeros.
  - Synchronizing variable-length transcript sequences by padding with ```BLANK_ID```.
- This batcher uses a CPU-to-GPU staging strategy, where tensors are collated on the ```NdArray``` CPU backend before a single-shot move to the compute device for minimizing PCIe bus latency.

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

### Training Pipeline

- Implemented a complete training loop in Rust.
- Supports batching and epoch-based training.
- Handles dynamic train/eval mode switching implicitly with Burn's ```Autodiff``` and ```Module``` traits (which allows gradient tracking).
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

### Custom CTC Decoding & Inference

- Implemented greedy CTC decoding from scratch.
- Implemented prefix beam search decoding, including:
  - Separate blank and non-blank probability tracking
  - Log-probability accumulation
  - Beam pruning and prefix merging
- Decoder architecture designed to support incremental and streaming inference.

### Language Model Integration (In Progress)

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
  - Memory safety
  - Low-latency inference
- Architecture explicitly designed to support future extensions:
  - Real-time webcam input
  - Sliding-window streaming inference
  - Dynamic mouth tracking via face detection and landmarks
  - Portable deployment via ONNX or native Rust runtimes (e.g., Tract)

---

## Current Status

- Offline lip-reading inference and language model integration is functionally complete, pending VSRM train/eval.
- Real-time inference pipeline is architecturally planned but not yet implemented.
- Greedy decoding and Beam Search decoding with a char-level N-gram language model both work end-to-end.

---

## Summary

This project represents a full end-to-end implementation of a VSRM lip-reading system in Rust, covering data processing, model architecture, training, loss computation, and decoding. The system is designed with real-time deployment in mind and avoids reliance on Python-based ML frameworks, emphasizing performance, safety, and extensibility.
