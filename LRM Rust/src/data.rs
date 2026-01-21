// Data handler pipeline for VSRM data ingestion (dataset item, batching strategies, and tensor collation)



use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{
        ElementConversion,
        backend::Backend,
        TensorData,
        Tensor,
    }
};

use crate::train::Batch;



// standardized container for any VSRM dataset sample (GRID, LRW, etc...)
#[derive(Clone, Debug)]
pub struct VsrmItem {
    // frames: [C, T, H, W]
    pub frames: TensorData,         // frames of the video (as TensorData to avoid Backend binding)
    pub transcript_ids: Vec<usize>, // sequence ids corresponding to speech in video
    pub sample_id: String,          // item ID (perhaps useful for debugging failed samples)
}



#[derive(Clone)]
pub struct VsrmBatcher<B: Backend> {
    pub device: B::Device,
}



impl<B: Backend> Batcher<B, VsrmItem, Batch<B>> for VsrmBatcher<B> {
    fn batch(&self, items: Vec<VsrmItem>, device: &B::Device) -> Batch<B> {
        // analyze batch to find max video timesteps length and max transcript sequence length
        let max_t = items.iter().map(|item| item.frames.shape[1]).max().unwrap_or(0);
        let max_l = items.iter().map(|item| item.transcript_ids.len()).max().unwrap_or(0);

        // non-padded lengths of inputs/targets
        let input_lengths: Vec<i32> = Vec::with_capacity(items.len());
        let target_lengths: Vec<i32> = Vec::with_capacity(items.len());
        todo!()
    }
}