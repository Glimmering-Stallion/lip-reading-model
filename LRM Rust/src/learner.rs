// VSRM dedicated training pipeline using Burn (handles auto-checkpointing, logging, and validation sets)



// custom imports
use crate::{
    vocab::{TokenMap, BLANK_ID},
    preprocessors::grid::GridDataset,
    DatasetSplit,
    batcher::{
        Batch,
        VsrmBatcher,
    },
    model::{
        VsrModel,
        VsrModelConfig,
    },
    ctc::ctc_loss::CtcLossConfig,
};

// imports
use burn::{
    config::Config,
    module::Module,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::noam::NoamLrSchedulerConfig,
    optim::{
        AdamConfig,
        GradientsParams,
    },
    record::CompactRecorder,
    tensor::{
        Tensor,
        backend::{
            AutodiffBackend,
            Backend,
        },
    },
    train::{
        SupervisedTraining,
        LearningComponentsMarker,
        Learner,
        TrainStep,
        InferenceStep,
        TrainOutput,
        metric::{
            Adaptor,
            LossInput,
            LossMetric,
        },
        ClassificationOutput,
    }
};
use std::{
    sync::{atomic, Arc},
    error::Error,
    path::Path,
    env,
    fs,
    // io::{self, BufRead},
};



impl<B: AutodiffBackend> TrainStep for VsrModel<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.inputs);

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths,
                batch.target_lengths
            );

        let grads = loss.backward();

        // hack: reshape to satisfy ClassificationOutput shape requirements (loss, [N * T, V], [N])
        let [n, t, v] = logits.dims();
        let flattened_logits = logits.reshape([n * t, v]);
        let flattened_targets = batch.targets.flatten(0, 1);

        let output = ClassificationOutput {
            loss,
            output: flattened_logits,
            targets: flattened_targets,
        };

        TrainOutput::new(self, grads, output)
    }
}



impl<B: Backend> InferenceStep for VsrModel<B> {
    type Input = Batch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Batch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.inputs);

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths,
                batch.target_lengths,
            );

        // hack: same reshape logic to satisfy ClassificationOutput shape requirements (loss, [N * T, V], [N])
        let [n, t, v] = logits.dims();
        let flattened_logits = logits.reshape([n * t, v]);
        let flattened_targets = batch.targets.flatten(0, 1);

        ClassificationOutput {
            loss,
            output: flattened_logits,
            targets: flattened_targets,
        }
    }
}



#[derive(Config, Debug)]
pub struct LearnerConfig {
    #[config(default = 50)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-4)]
    pub learning_rate: f64,

    pub optimizer: AdamConfig,
}



pub fn train<B, P>(
    device: B::Device,
    model_config: VsrModelConfig,
    learner_config: LearnerConfig,
    token_map: TokenMap,
    output_path: P,
)
where
    B: AutodiffBackend,
    P: AsRef<Path>,
    ClassificationOutput<B>: Adaptor<LossInput<B>>,
    ClassificationOutput<B::InnerBackend>: Adaptor<LossInput<B::InnerBackend>>,
{
    let output_path = output_path.as_ref();

    // (train/validation boundary at 80% of data, validation/test boundary at 90% of data)
    // data:            |---------------------------train---------------------------|-valid-|--test--|
    // train/valid:     |----------------------------80%--------------------------->|<---------------|
    // valid/test:      |--------------------------------90%------------------------------->|<-------|
    let split_thresholds = (0.8, 0.9);

    // train/validation dataset instances (GRID corpus)
    let train_dataset = GridDataset::new(
        "data",
        DatasetSplit::Train,
        split_thresholds,
        token_map.clone()
    );
    let valid_dataset = GridDataset::new(
        "data",
        DatasetSplit::Val,
        split_thresholds,
        token_map.clone()
    );

    // train/validation data batcher instances (train uses Autodiff B, while valid uses InnerBackend raw B)
    let train_batcher = VsrmBatcher::<B>::new(device.clone());
    let valid_batcher = VsrmBatcher::<B::InnerBackend>::new(device.clone());

    // train/validation data loader instances
    let train_dataloader = DataLoaderBuilder::new(train_batcher)
        .batch_size(learner_config.batch_size)
        .shuffle(learner_config.seed)
        .num_workers(learner_config.num_workers)
        .build(train_dataset);
    let valid_dataloader = DataLoaderBuilder::new(valid_batcher)
        .batch_size(learner_config.batch_size)
        .shuffle(learner_config.seed)
        .num_workers(learner_config.num_workers)
        .build(valid_dataset);

    // find warmup steps for scheduler
    let num_items = train_dataloader.num_items();
    let num_batches = num_items.div_ceil(learner_config.batch_size);
    let total_steps = num_batches * learner_config.num_epochs;
    let warmup_steps = (0.2_f64 * total_steps as f64).floor() as usize;

    // learning rate scheduler (Noam scheduler)
    let scheduler = NoamLrSchedulerConfig::new(learner_config.learning_rate)
        .with_warmup_steps(warmup_steps)
        .with_model_size(model_config.out_channels)
        .init()
        .expect("Failed to initialize Noam scheduler");

    let optimizer = learner_config.optimizer.init();
    let model = model_config.init::<B>(&device);

    // learner instance (and move model to device)
    let mut learner = Learner::new(
        model,
        optimizer,
        scheduler,
    );
    learner.fork(&device);

    // trainer instance
    let trained_model = SupervisedTraining::new(
        output_path,
        train_dataloader,
        valid_dataloader,
    )
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(CompactRecorder::new())
    .num_epochs(learner_config.num_epochs)
    .launch(learner);

    // save final model weights
    trained_model
        .model
        .save_file(output_path.join("vsrm"), &CompactRecorder::new())
        .expect("Failed to save trained model");

    println!("Training complete, weights saved to {:?}", output_path);
}