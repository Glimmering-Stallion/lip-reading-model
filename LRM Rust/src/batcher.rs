// Data handler pipeline for VSRM data ingestion (dataset item, batching strategies, and tensor collation)



// custom imports
use crate::{vocab::BLANK_ID};

// imports
use burn::{
    backend::ndarray::NdArray,
    prelude::Int,
    data::dataloader::batcher::Batcher,
    tensor::{
        backend::Backend,
        ElementConversion,
        Tensor,
        TensorData,
    },
};



pub type CpuB = NdArray<f32>; // CPU backend for data staging area



#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 5>,               // [N, C, T, H, W]
    pub targets: Tensor<B, 2, Int>,         // [N, L] (L padded to max target length in batch)
    pub input_lengths: Tensor<B, 1, Int>,   // [N]
    pub target_lengths: Tensor<B, 1, Int>,  // [N]
}



// standardized container for any VSRM dataset sample (GRID, LRW, etc...)
#[derive(Clone, Debug)]
pub struct VsrmItem {
    // frames: [C, T, H, W]
    pub frames: TensorData, // frames of the video (as TensorData to avoid Backend binding)
    pub transcript_ids: Vec<usize>, // sequence ids corresponding to speech in video
    pub item_id: String,  // ID of data sample (perhaps useful for debugging failed samples)
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
impl<B: Backend> Batcher<B, VsrmItem, Batch<B>> for VsrmBatcher<B> {
    /// create a batch from a list of dataset items
    /// handles dynamic padding for both video frames and transcript sequences
    /// params:
    /// - items: list of [C, T, H, W] frames and sequence IDs
    /// - device: backend device to load final tensors onto
    /// returns: batch containing padded inputs [N, C, max_T, H, W] and targets [N, max_L]
    fn batch(&self, items: Vec<VsrmItem>, device: &B::Device) -> Batch<B> {
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
            let (c, t, h, w) = (
                item.frames.shape[0],
                item.frames.shape[1],
                item.frames.shape[2],
                item.frames.shape[3],
            );

            // --------------- (A) ---------------

            let frames: Tensor<CpuB, 4> = Tensor::from_data(item.frames, &Default::default());

            // if curr item's num frames are shorter than max timesteps, pad it
            let padded_frames = if t < max_t {
                let pad_amount = max_t - t;

                // init tensor of zeros of padding amount t, then concat with frames along dim T
                let zeros: Tensor<CpuB, 4> = Tensor::zeros([c, pad_amount, h, w], &Default::default());
                Tensor::cat(vec![frames, zeros], 1)
            } else {
                frames
            };
            padded_frames_container.push(padded_frames);

            // --------------- (B) ---------------

            let mut sequence = item.transcript_ids.clone();

            // if curr item's sequence length is shorter than max length, pad it
            sequence.resize(max_l, BLANK_ID);

            // convert padded sequence from vec to tensor
            let padded_sequence: Tensor<CpuB, 1, Int> = Tensor::from_ints(&sequence[..], &Default::default());
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

        let input_lengths: Tensor<B, 1, Int> = Tensor::from_ints(&input_lengths[..], device);
        let target_lengths: Tensor<B, 1, Int> = Tensor::from_ints(&target_lengths[..], device);

        Batch {
            inputs,
            targets,
            input_lengths,
            target_lengths,
        }
    }
}
