// VSRM dedicated training pipeline and orchestrator using Burn (handles auto-checkpointing, logging, and validation sets)



// custom imports
use crate::{
    vocab::{VOCAB_SIZE, BLANK_ID, TokenMap},
    pipeline::{
        batcher::{
            Batch,
            VsrmBatcher,
        },
        dataset::{
            DatasetSource,
            DatasetSplit,
        },
        preprocessors::grid::GridDataset,
    },
    training::{
        metrics::{
            CtcCharErrorRate,
            CtcWordErrorRate,
            VsrmStepOutput,
        },
    },
    vsrm::{
        VsrModel,
        VsrModelConfig,
    },
    ctc::{
        ctc_loss::CtcLossConfig,
        ctc_decode::{
            CtcDecodeType,
            CtcDecoderConfig,
        },
        lm::{
            LanguageModelConfig,
            NgramConfig,
        },
    },
};

// imports
use burn::{
    config::Config,
    data::dataloader::{DataLoader, DataLoaderBuilder},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module, optim::AdamConfig,
    record::CompactRecorder,
    data::dataset::Dataset,
    tensor::{
        ElementConversion,
        backend::{
            AutodiffBackend,
            Backend,
        },
    },
    train::{
        InferenceStep, Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            Adaptor,
            LossInput,
            LossMetric,
        }
    }
};
use std::{
    sync::Arc,
    path::Path,
    fs::create_dir_all,
    // io::{self, BufRead},
};



// type aliases for the train/validation dataloader creator return type
type TrainLoader<B> = Arc<dyn DataLoader<B, Batch<B>>>;
type ValidLoader<B> = Arc<dyn DataLoader<<B as AutodiffBackend>::InnerBackend, Batch<<B as AutodiffBackend>::InnerBackend>>>;



#[derive(Config, Debug)]
pub struct VsrmLearnerConfig  {
    #[config(default = 50)]
    pub num_epochs: usize,

    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 1e-4)]
    pub learning_rate: f64,

    pub optimizer: AdamConfig,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,
}



impl<B: AutodiffBackend> TrainStep for VsrModel<B> {
    type Input = Batch<B>;
    type Output = VsrmStepOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<VsrmStepOutput<B>> {
        let logits = self.forward(batch.inputs);

        let [n, t, v] = logits.dims();
        assert!(n > 0 && t > 0 && v > 0);

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );
        if !loss.clone().is_finite().all().into_scalar().elem::<bool>() { panic!("Loss contains non-finite values"); }

        let grads = loss.backward();

        let output = VsrmStepOutput {
            loss,
            outputs: logits,
            targets: batch.targets,
            output_lengths: batch.input_lengths,
            target_lengths: batch.target_lengths,
        };

        TrainOutput::new(self, grads, output)
    }
}



impl<B: Backend> InferenceStep for VsrModel<B> {
    type Input = Batch<B>;
    type Output = VsrmStepOutput<B>;

    fn step(&self, batch: Batch<B>) -> VsrmStepOutput<B> {
        let logits = self.forward(batch.inputs);

        let [n, t, v] = logits.dims();
        assert!(n > 0 && t > 0 && v > 0);

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );
        if !loss.clone().is_finite().all().into_scalar().elem::<bool>() { panic!("Loss contains non-finite values"); }

        VsrmStepOutput {
            loss,
            outputs: logits,
            targets: batch.targets,
            output_lengths: batch.input_lengths,
            target_lengths: batch.target_lengths,
        }
    }
}



pub fn train<B, PR, PO>(
    device: B::Device,
    dataset_src: DatasetSource,
    model_config: VsrModelConfig,
    learner_config: VsrmLearnerConfig ,
    token_map: TokenMap,
    root_path: PR,
    output_path: PO,
)
where
    B: AutodiffBackend,
    PR: AsRef<Path>,
    PO: AsRef<Path>,
    VsrmStepOutput<B>: Adaptor<LossInput<B>>,
    VsrmStepOutput<B::InnerBackend>: Adaptor<LossInput<B::InnerBackend>>,
{
    let root_path = root_path.as_ref();
    let output_path = output_path.as_ref();
    assert!(root_path.exists(), "Root path {:?} does not exist", root_path);
    assert!(output_path.exists(), "Output path {:?} does not exist", output_path);

    // create model experiment/artifacts directory
    let model_dir = format!("vsrm_{}", dataset_src.tag());
    let model_path = output_path.join(&model_dir);
    create_dir_all(&model_path).expect("Failed to create trained vsrm artifacts directory");

    assert!(learner_config.num_epochs > 0, "Number of epochs must be > 0, got {}", learner_config.num_epochs);
    assert!(learner_config.batch_size > 0, "Batch size must be > 0, got {}", learner_config.batch_size);
    assert!(learner_config.learning_rate > 0.0, "Learning rate must be > 0, got {}", learner_config.learning_rate);
    assert!(learner_config.num_workers <= 64, "Exceeded reasonable worker limit ({})", learner_config.num_workers);
    if learner_config.num_workers == 0 { println!("Running with 0 workers: data loading will be synchronous"); }

    // save hyperparams
    learner_config.save(model_path.join("learner_config.json")).expect("Failed to save config");

    // ------------------------------------ Dataset batching and loading ------------------------------------

    // obtain train/validation data loader instances
    let (train_dataloader, valid_dataloader) = create_dataloaders(
        &device,
        dataset_src,
        &learner_config,
        token_map.clone(),
        &root_path,
    );

    // ------------------------ Training learner, LR scheduling, and optimizer setup ------------------------

    // find warmup steps for scheduler
    let num_items = train_dataloader.num_items();
    let num_batches = num_items.div_ceil(learner_config.batch_size);
    let total_steps = num_batches * learner_config.num_epochs;
    let warmup_steps = (0.2_f64 * total_steps as f64).floor() as usize;

    assert!(num_batches > 0, "Computed 0 batches for training");
    assert!(total_steps > 0, "Total training steps is 0");
    assert!(warmup_steps < total_steps, "Warmup steps ({}) must be less than total steps ({})", warmup_steps, total_steps);

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

    // -------------------------------------- CTC decoder and LM setup --------------------------------------

    let ngram_lm_path = Path::new(&root_path)
        .join("models")
        .join("ngram_lm.bin");
    assert!(ngram_lm_path.exists(), "N-gram LM at {:?} does not exist", ngram_lm_path);

    // N-gram LM instance
    let ngram_lm = NgramConfig::new()
        .with_n(3)
        .with_vocab_size(VOCAB_SIZE)
        .with_path(ngram_lm_path.to_str().map(|s| s.to_string()));

    let beam_width = 5;
    let lm_alpha = 2.0;
    let lm_beta = 2.0;

    // init CTC Beam decoder with N-gram LM (slower)
    let beam_decoder = CtcDecoderConfig::new()
        .with_search_type(CtcDecodeType::BeamSearch)
        .with_beam_width(beam_width)
        .with_blank_id(BLANK_ID)
        .with_lm(Some(LanguageModelConfig::Ngram(ngram_lm)))
        .with_lm_alpha(lm_alpha)
        .with_lm_beta(lm_beta)
        .init();

    // init CTC Greedy decoder (faster)
    let greedy_decoder = CtcDecoderConfig::new()
        .with_search_type(CtcDecodeType::GreedySearch)
        .with_blank_id(BLANK_ID)
        .init();

    // -------------------------------------- VSRM training and saving --------------------------------------

    println!("Training model {}", model_dir);

    // trainer instance
    let trained_model = SupervisedTraining::new(
        &model_path,
        train_dataloader,
        valid_dataloader,
    )
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(CtcCharErrorRate::new(greedy_decoder.clone()))
        .metric_valid_numeric(CtcWordErrorRate::new(greedy_decoder.clone(), token_map))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(learner_config.num_epochs)
        .launch(learner);

    // save final model weights
    trained_model
        .model
        .save_file(model_path.join(format!("{}_final_weights", model_dir)), &CompactRecorder::new())
        .expect("Failed to save trained model");

    println!("Training complete: model weights saved to {:?}", output_path);
}



fn create_dataloaders<B, P>(
    device: &B::Device,
    dataset_src: DatasetSource,
    learner_config: &VsrmLearnerConfig ,
    token_map: TokenMap,
    root_path: &P,
) -> (
    TrainLoader<B>,
    ValidLoader<B>,
)
where
    B: AutodiffBackend,
    P: AsRef<Path>,
{
    let root_path = root_path.as_ref();
    assert!(root_path.exists(), "Root path {:?} does not exist", root_path);

    // (train/validation boundary, validation/test boundary)
    // total data:                  |---------------------------train---------------------------|-valid-|--test--|
    // train/valid split point:     |----------------------------80%--------------------------->|<---------------|
    // valid/test split point:      |--------------------------------90%------------------------------->|<-------|
    let split_thresholds = (0.8, 0.9);

    match dataset_src {
        DatasetSource::Grid => {
            // train/validation dataset instances
            let train_dataset = GridDataset::new(root_path, DatasetSplit::Train, split_thresholds, token_map.clone());
            let valid_dataset = GridDataset::new(root_path, DatasetSplit::Val, split_thresholds, token_map.clone());

            assert!(train_dataset.len() > 0, "Training dataset is empty");
            assert!(valid_dataset.len() > 0, "Validation dataset is empty");

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

            assert!(train_dataloader.num_items() > 0, "Training dataloader has 0 items");
            assert!(valid_dataloader.num_items() > 0, "Validation dataloader has 0 items");

            (train_dataloader, valid_dataloader)
        }
        // DatasetSource::Lrw => { // similar logic for future LRW dataset },
    }
}
