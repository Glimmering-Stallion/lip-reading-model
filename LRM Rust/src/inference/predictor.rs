//! Inference predictor for loading trained VSRM and running prediction on video input.

// use crate::{
//     context::Context,
//     ctc::{
//         ctc_decode::{CtcDecodeType, CtcDecoderConfig},
//         lm::{LanguageModelConfig, NgramConfig},
//     },
//     pipeline::{
//         batcher::{Batch, VsrmBatcher, VsrmItem},
//         dataset::DatasetStats,
//         io::load_json,
//         tracker::TrackerConfig,
//         video::load_video_with_tracker,
//     },
//     vocab::{TokenMap, BLANK_ID, VOCAB_SIZE},
//     vsrm::{VsrModel, VsrModelConfig},
// };
// use burn::{
//     backend::{Wgpu, wgpu::WgpuDevice::DefaultDevice},
//     data::dataloader::batcher::Batcher,
//     module::Module,
//     record::CompactRecorder,
// };
// use burn::tensor::TensorData;
// use std::{error::Error, fs, path::Path};
