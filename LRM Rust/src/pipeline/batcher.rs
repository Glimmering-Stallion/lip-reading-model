// Source-agnostic data handler pipeline for VSRM data ingestion (dataset item, batching strategies, and tensor collation)



// custom imports
use crate::{vocab::BLANK_ID};

// imports
use burn::{
    backend::ndarray::NdArray,
    prelude::Int,
    data::dataloader::batcher::Batcher,
    tensor::{
        backend::Backend,
        Tensor,
        TensorData,
    },
};



pub type CpuB = NdArray<f32>; // CPU backend for data staging area



// standardized container for a batch of VSRM data samples (after collation and padding)
#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 5>,               // [N, C, T_max, H, W]  padded frames
    pub targets: Tensor<B, 2, Int>,         // [N, L_max]           padded sequences
    pub input_lengths: Tensor<B, 1, Int>,   // [N]                  pre-padded frame lengths
    pub target_lengths: Tensor<B, 1, Int>,  // [N]                  pre-padded sequence lengths
}



// standardized container for any VSRM dataset sample (GRID, LRW, etc...)
#[derive(Clone, Debug)]
pub struct VsrmItem {
    pub frames: TensorData,          // [C, T, H, W]  frames of the video (as TensorData to avoid Backend binding)
    pub transcript_ids: Vec<usize>,  // [L]           sequence IDs corresponding to speech in video
    pub item_id: String,             // ID of data sample (perhaps useful for debugging failed samples)
}



#[derive(Clone, Debug)]
pub struct VsrmBatcher<B: Backend> {
    pub device: B::Device,
}



impl<B: Backend> VsrmBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}



// stack data on CPU tensor first, then move that singular final Batch Tensor to GPU for model ingestion
// (cheaper to move one big tensor to GPU than many small tensors)
impl<B: Backend> Batcher<B, VsrmItem, Batch<B>> for VsrmBatcher<B> {
    /// create a batch from a list of dataset items
    /// handles dynamic padding for both video frames and transcript sequences
    /// params:
    /// - items: list of [C, T, H, W] frames and sequence IDs
    /// - device: backend device to load final tensors onto
    /// returns: batch containing padded inputs [N, C, max_T, H, W] and targets [N, max_L]
    fn batch(&self, items: Vec<VsrmItem>, device: &B::Device) -> Batch<B> {
        assert!(!items.is_empty(), "VsrmBatcher received an empty batch");

        // analyze batch to find max video timesteps length and max transcript sequence length
        let max_t = items
            .iter()
            .map(|item| item.frames.shape[1])
            .max()
            .unwrap_or(0);
        let max_l = items
            .iter()
            .map(|item| item.transcript_ids.len())
            .max()
            .unwrap_or(0);

        assert!(max_t > 0, "Max time dimension is zero");
        assert!(max_l > 0, "Max transcript length is zero");

        // padded frames and sequence targets
        let mut padded_frames_container: Vec<Tensor<CpuB, 4>> = Vec::with_capacity(items.len());
        let mut padded_sequences_container: Vec<Tensor<CpuB, 1, Int>> = Vec::with_capacity(items.len());

        // non-padded lengths of input frames and sequence targets
        let mut input_lengths: Vec<i32> = Vec::with_capacity(items.len());
        let mut target_lengths: Vec<i32> = Vec::with_capacity(items.len());

        // part A: pad frames
        // part B: pad sequences
        // part C: one shot stack
        for item in items {
            // isolate curr frame's dims
            let (c, t, h, w, l) = (
                item.frames.shape[0],
                item.frames.shape[1],
                item.frames.shape[2],
                item.frames.shape[3],
                item.transcript_ids.len(),
            );
            assert!(c == 1, "VSRM assumes grayscale frame inputs: expected single-channel input, got {}", c);
            assert!(t >= (2 * l), "CTC Constraint Violated: Video frames ({}) for item {} must be greater than transcript length ({})", t, item.item_id, l);
            assert!(h > 0 && w > 0, "Invalid frame dimensions {}x{}", h, w);

            // --------------- (A) ---------------

            let frames: Tensor<CpuB, 4> = Tensor::from_data(item.frames, &Default::default()); // [C, T, H, W] frames of the video

            // preprocess frames
            // 1. scale pixel values to [0, 1] range
            // 2. calc mean and var with var_mean_bias on reshaped frames to get single mean and var across all pixels in video; reshape back to [C, T, H, W] for broadcasting
            // 3. calc st dev from var, add small epsilon for numerical stability
            // 4. standardize frames (by centering to zero mean and scaling to unit variance)
            let frames = frames.div_scalar(255.0);
            let (var, mean) = frames.clone().reshape([1, c * t * h * w]).var_mean_bias(1);
            let st_dev = var.sqrt().add_scalar(1e-7);
            let frames = frames.sub(mean.reshape([c, 1, 1, 1])).div(st_dev.reshape([c, 1, 1, 1]));

            // if curr item's num frames are shorter than max timesteps, pad it
            let padded_frames = if t < max_t {
                let pad_amount = max_t - t;

                // init tensor of zeros of padding amount t, then concat with frames along dim T
                let zeros: Tensor<CpuB, 4> = Tensor::zeros([c, pad_amount, h, w], &Default::default());
                Tensor::cat(vec![frames, zeros], 1)
            } else { frames };
            debug_assert!(padded_frames.shape()[1] == max_t, "Frame padding failed: expected T = {}, got {}", max_t, padded_frames.shape()[1]);

            padded_frames_container.push(padded_frames);

            // --------------- (B) ---------------

            let mut sequence = item.transcript_ids.clone();
            assert!(item.transcript_ids.iter().all(|&id| id < BLANK_ID), "Sequence contains out-of-range token in item {}", item.item_id);

            // if curr item's sequence length is shorter than max length, pad it
            sequence.resize(max_l, BLANK_ID);

            // convert padded sequence from vec to tensor
            let padded_sequence: Tensor<CpuB, 1, Int> = Tensor::from_ints(&sequence[..], &Default::default());
            debug_assert!(padded_sequence.shape()[0] == max_l, "Sequence padding failed: expected L = {}, got {}", max_l, padded_sequence.shape()[0]);

            padded_sequences_container.push(padded_sequence);

            // -----------------------------------

            // track original non-padded lengths
            input_lengths.push(t as i32);
            target_lengths.push(item.transcript_ids.len() as i32);
        }

        // --------------- (C) ---------------

        // stack list of [C, T, H, W] (4D) tensors into [N, C, T, H, W] (5D) tensors
        let inputs: Tensor<B, 5> = Tensor::<B, 5>::from_data(
            Tensor::stack::<5>(padded_frames_container, 0).into_data(),
            device
        );

        // stack list of [L] (1D) tensors into [N, L] (2D) tensors
        let targets: Tensor<B, 2, Int> = Tensor::<B, 2, Int>::from_data(
            Tensor::stack::<2>(padded_sequences_container, 0).into_data(),
            device
        );

        // make sure stacking/padding worked for inputs/targets
        debug_assert!(inputs.shape()[0] > 0 && inputs.shape()[1] == 1 && inputs.shape()[2] == max_t);
        debug_assert!(targets.shape()[0] == inputs.shape()[0] && targets.shape()[1] == max_l);

        let input_lengths: Tensor<B, 1, Int> = Tensor::from_ints(&input_lengths[..], device);
        let target_lengths: Tensor<B, 1, Int> = Tensor::from_ints(&target_lengths[..], device);

        // make sure batch sizes are aligned between inputs/targets and input/target lengths
        assert_eq!(input_lengths.dims()[0], inputs.shape()[0], "Inputs/lengths batch size mismatch");
        assert_eq!(target_lengths.dims()[0], targets.shape()[0], "Targets/lengths batch size mismatch");

        Batch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
        }
    }
}
