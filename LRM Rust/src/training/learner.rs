//! VSRM dedicated training orchestrator and lifecycle management.
//! 
//! This module defines the training and inference steps for the Burn framework, which
//! provides auto-checkpointing and logging. It manages the model's forward pass,
//! CTC-loss calculation, and gradient backpropagation. The module serves as the bridge
//! between the dataset loaders, the VSRM architecture, and the training loop.



// custom imports
use crate::{
    prelude::{io_err, ESS},
    context::Context,
    ctc::{
        ctc_decode::{
            CtcDecodeType,
            CtcDecoderConfig,
        },
        ctc_loss::CtcLossConfig,
    },
    pipeline::{
        adapters::grid::GridDataset,
        batcher::{
            Batch,
            VsrmBatcher,
        },
        dataset::{
            DatasetSource,
            DatasetSplit,
            DatasetStats,
        },
        io::{
            load_json,
            save_json,
        },
        tracker::{
            HaarTrackerConfig,
            TrackerConfig,
        },
    },
    training::metrics::{
        CtcCharErrorRate,
        CtcWordErrorRate,
        VsrmStepOutput,
    },
    vocab::{
        TokenMap,
        BLANK_ID,
        VOCAB,
    },
    vsrm::{
        SummaryVisitor,
        VsrModel,
        VsrModelConfig,
    }
};

// imports
use burn::{
    Tensor,
    config::Config,
    data::{
        dataloader::{
            DataLoader,
            DataLoaderBuilder
        },
        dataset::Dataset,
    },
    lr_scheduler::{
        composed::{
            ComposedLrSchedulerConfig,
            SchedulerReduction,
        },
        cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::Module,
    optim::{
        AdamConfig,
    },
    record::CompactRecorder,
    tensor::{
        ElementConversion,
        activation::log_softmax,
        backend::{
            AutodiffBackend,
            Backend,
        },
    },
    train::{
        checkpoint::KeepLastNCheckpoints,
        InferenceStep,
        Learner,
        SupervisedTraining,
        TrainOutput,
        TrainStep,
        metric::{
            Adaptor,
            LearningRateMetric,
            LossInput,
            LossMetric,
        },
    }
};
use std::{
    fs,
    io::{self, Write},
    sync::Arc,
    thread,
    time::Duration
};
use log;



/// Type alias for the train dataloader creator return type
type TrainLoader<B> = Arc<dyn DataLoader<B, Batch<B>>>;
/// Type alias for the validation dataloader creator return type
type ValidLoader<B> = Arc<dyn DataLoader<<B as AutodiffBackend>::InnerBackend, Batch<<B as AutodiffBackend>::InnerBackend>>>;



/// Serializable configuration hyperparameters and metadata for training the VSR model.
#[derive(Config, Debug)]
pub struct VsrmLearnerConfig  {
    pub model_id: String,                    // model name to be saved as

    pub dataset_src: DatasetSource,          // dataset src that model will be trained on

    // None = not specified, Some(None) = use latest, Some(Some(e)) = use epoch e
    #[config(default = "None")]
    pub resume_from: Option<usize>,          // which checkpoint to resume training model from (runtime-only, not persisted to learner_config.json)

    #[config(default = false)]
    pub keep_all_checkpoints: bool,          // flag/toggle for checkpoint save strategy (if true, keep all checkpoints; else keep most recent only)

    #[config(default = "(50, 100)")]
    pub frame_dims: (usize, usize),          // input video frame dimensions (height, width)

    #[config(default = 0)]
    pub rf: usize,                           // receptive field (RF) of the model (populated on initialization)

    #[config(default = 50)]
    pub num_epochs: usize,                   // number of times the model has processed through entire training dataset

    #[config(default = 4)]
    pub batch_size: usize,                   // number of training samples to process per iteration

    #[config(default = 1e-4)]
    pub learning_rate: f64,                  // peak learning rate that the scheduler reaches (controls step size towards minimum of CTC loss function)

    pub optimizer: AdamConfig,               // Adam optimizer for weight updates

    #[config(default = 8)]
    pub num_workers: usize,                  // number of background threads to process dataset in parallel (too low --> GPU idle, too high --> CPU contention)

    #[config(default = 1)]
    pub accumulation: usize,                 // number of mini-batches that gets processed by the model before weight updates

    #[config(default = 42)]
    pub seed: u64,                           // seed value for deterministic RNG on dataset shuffling

    #[config(default = "None")]
    pub active_subset: Option<(f32, u64)>,   // fraction of entire dataset to use for training (e.g. Some((0.1, 69) = 10% with seed 69, None = use full dataset)
}



impl<B: AutodiffBackend> TrainStep for VsrModel<B> {
    type Input = Batch<B>;
    type Output = VsrmStepOutput<B>;

    fn step(&self, batch: Batch<B>) -> TrainOutput<VsrmStepOutput<B>> {
        let logits = self.forward(batch.inputs);

        let [n, t, v] = logits.dims();
        assert!(n > 0 && t > 0 && v > 0);

        // -------------------------------------- Debugging --------------------------------------

        let [n, t, _] = logits.dims();
        let token_map = TokenMap::new(VOCAB);
        let indices = logits.clone().argmax(2).reshape([n, t]);

        // get first target sequence in batch
        let first_len = batch.target_lengths.to_data().to_vec::<i32>().unwrap()[0] as usize;
        let targ_ids = batch.targets.to_data().to_vec::<i32>().unwrap();
        let first_targ_ids: Vec<usize> = targ_ids[0..first_len].iter().map(|&id| id as usize).collect();
        let first_targ_chars = TokenMap::new(VOCAB).ids_to_chars(&first_targ_ids);

        // get first prediction sequence in batch (Greedy argmax)
        let pred_ids = indices.to_data().to_vec::<i32>().unwrap();
        let first_pred_ids: Vec<usize> = pred_ids[0..t].iter().map(|&x| x as usize).collect();
        let first_pred_chars = token_map.ids_to_chars(&first_pred_ids);

        // log to experiment.log
        log::info!("=== TRAIN SAMPLE ===");
        if let Some(chars) = first_targ_chars { log::info!("  Target: {:?}", chars); }
        if let Some(chars) = first_pred_chars { log::info!("  Preds : {:?}", chars); }

        // ---------------------------------------------------------------------------------------

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );
        if !loss.clone().is_finite().all().into_scalar().elem::<bool>() {
            println!("DUMPING BATCH: Loss is NaN/Inf!");
            println!("Batch lengths: {:?}", batch.input_lengths);
            panic!("loss contains non-finite values");
        }

        // ------------------------------- Entropy Regularization --------------------------------

        // let lambda = 20.0;                 // penalty scaling factor
        // let min_entropy_threshold = 2.6;  // min threshold before entropy starts to penalize
        // let penalty = calc_entropy_penalty(
        //     logits.clone(),
        //     lambda,
        //     min_entropy_threshold,
        // );
        // let loss = loss.add(penalty);

        // ---------------------------------------------------------------------------------------

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

        // -------------------------------------- Debugging --------------------------------------

        let [n, t, _] = logits.dims();
        let token_map = TokenMap::new(VOCAB);
        let indices = logits.clone().argmax(2).reshape([n, t]);

        // get first target sequence in batch
        let first_len = batch.target_lengths.to_data().to_vec::<i32>().unwrap()[0] as usize;
        let targ_ids = batch.targets.to_data().to_vec::<i32>().unwrap();
        let first_targ_ids: Vec<usize> = targ_ids[0..first_len].iter().map(|&id| id as usize).collect();
        let first_targ_chars = TokenMap::new(VOCAB).ids_to_chars(&first_targ_ids);

        // get first prediction sequence in batch (Greedy argmax)
        let pred_ids = indices.to_data().to_vec::<i32>().unwrap();
        let first_pred_ids: Vec<usize> = pred_ids[0..t].iter().map(|&x| x as usize).collect();
        let first_pred_chars = token_map.ids_to_chars(&first_pred_ids);

        // log to experiment.log
        log::info!("=== VALIDATION SAMPLE ===");
        if let Some(chars) = first_targ_chars { log::info!("  Target: {:?}", chars); }
        if let Some(chars) = first_pred_chars { log::info!("  Pred  : {:?}", chars); }

        // ---------------------------------------------------------------------------------------

        let loss = CtcLossConfig::new()
            .with_blank_id(BLANK_ID)
            .init()
            .forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );
        if !loss.clone().is_finite().all().into_scalar().elem::<bool>() { panic!("loss contains non-finite values"); }

        VsrmStepOutput {
            loss,
            outputs: logits,
            targets: batch.targets,
            output_lengths: batch.input_lengths,
            target_lengths: batch.target_lengths,
        }
    }
}



pub fn train<B>(
    device: B::Device,
    context: &Context,
    model_config: &VsrModelConfig,
    learner_config: &VsrmLearnerConfig,
    token_map: &TokenMap,
) -> Result<(), ESS>
where
    B: AutodiffBackend,
    VsrmStepOutput<B>: Adaptor<LossInput<B>>,
    VsrmStepOutput<B::InnerBackend>: Adaptor<LossInput<B::InnerBackend>>,
{
    // create output/model paths
    let output_path = context.models_path.clone();
    let model_path = output_path.join(&learner_config.model_id);
    fs::create_dir_all(&model_path)?;

    assert!(learner_config.num_epochs > 0, "number of epochs must be > 0, got {}", learner_config.num_epochs);
    assert!(learner_config.batch_size > 0, "batch size must be > 0, got {}", learner_config.batch_size);
    assert!(learner_config.learning_rate > 0.0, "learning rate must be > 0, got {}", learner_config.learning_rate);
    assert!(learner_config.num_workers <= 64, "exceeded reasonable worker limit ({})", learner_config.num_workers);
    if learner_config.num_workers == 0 { println!("Running with 0 workers: data loading will be synchronous"); }

    // ------------------------------------ Dataset batching and loading ------------------------------------

    // obtain train/validation data loader instances
    let (train_dataloader, valid_dataloader) = create_dataloaders(
        &device,
        context,
        learner_config.dataset_src,
        learner_config,
        token_map,
    );

    // ------------------------ Training learner, LR scheduling, and optimizer setup ------------------------

    let num_items = train_dataloader.num_items();
    let num_batches = num_items.div_ceil(learner_config.batch_size);
    let total_steps = num_batches * learner_config.num_epochs;
    let warmup_steps = num_batches; // warmup over first epoch

    assert!(num_batches > 0, "computed 0 batches for training");
    assert!(total_steps > 0, "total training steps is 0");
    assert!(warmup_steps < total_steps, "warmup steps ({}) must be less than total steps ({})", warmup_steps, total_steps);

    let lr = learner_config.learning_rate;
    let scheduler = ComposedLrSchedulerConfig::new()
        .linear(LinearLrSchedulerConfig::new(0.01, 1.0, warmup_steps))
        .cosine(CosineAnnealingLrSchedulerConfig::new(lr, total_steps).with_min_lr(lr / 10.0))
        .with_reduction(SchedulerReduction::Prod)
        .init()
        .expect("failed to initialize composed scheduler");

    // init optimizer and model
    let optimizer = learner_config.optimizer.init::<B, VsrModel<B>>();
    let model = model_config.init::<B>(&device);
    SummaryVisitor::summarize(&model);

    // learner instance (and move model to device)
    let mut learner = Learner::new(
        model.clone(),
        optimizer,
        scheduler,
    );
    learner.fork(&device);

    // ------------------------------------------ CTC decoder setup -----------------------------------------

    // init CTC Greedy decoder (faster)
    let greedy_decoder = CtcDecoderConfig::new()
        .with_search_type(CtcDecodeType::GreedySearch)
        .with_blank_id(BLANK_ID)
        .init();

    // -------------------------------------- VSRM training and saving --------------------------------------

    // ------------------------ Diagnostics ------------------------
    println!("=== Scheduler Diagnostics ===");
    println!("  Num Items:      {}",      num_items);
    println!("  Num Batches:    {}",      num_batches);
    println!("  Total Steps:    {}",      total_steps);
    println!("  Warmup Steps:   {}",      warmup_steps);
    println!("  Target LR:      {:.9}",   learner_config.learning_rate);
    println!("  Accumulation:   {}",      learner_config.accumulation);
    println!("=============================\n");
    // -------------------------------------------------------------

    // resolve user intent for checkpointing behavior:
    // - if `keep_all_checkpoints` on --> set checkpoints to keep as total epochs
    // - if `keep_all_checkpoints` off --> set checkpoints to keep as just one
    let keep_n_checkpoints = if learner_config.keep_all_checkpoints
    { learner_config.num_epochs } else { 1 };

    // trainer instance (Burn's `FileCheckpointer` creates model dir on init)
    let trainer = SupervisedTraining::new(
        &model_path,
        train_dataloader,
        valid_dataloader,
    )
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_valid_numeric(CtcCharErrorRate::new(greedy_decoder.clone()))
        .metric_valid_numeric(CtcWordErrorRate::new(greedy_decoder.clone(), token_map.clone()))
        .with_file_checkpointer(CompactRecorder::new())
        .with_checkpointing_strategy(KeepLastNCheckpoints::new(keep_n_checkpoints))
        .num_epochs(learner_config.num_epochs)
        .grads_accumulation(learner_config.accumulation);

    // resume training on existing model at specified checkpoint or train new model
    let training = if let Some(epoch) = learner_config.resume_from
    { trainer.checkpoint(epoch) } else { trainer };

    let mut persisted_learner_config = learner_config.clone()
        .with_rf(model.total_receptive_field()); // update model receptive field value for learner config
    persisted_learner_config.resume_from = None; // clear `resume_from` value (runtime-only)

    // save learner and model configs
    persisted_learner_config.save(model_path.join("learner_config.json"))
        .map_err(|e| io_err(format!("failed to save learner config: {}", e), io::ErrorKind::Other))?;
    model_config.save(model_path.join("model_config.json"))
        .map_err(|e| io_err(format!("failed to save model config: {}", e), io::ErrorKind::Other))?;

    // launch training and save final model weights
    let trained_model = training.launch(learner);
    trained_model.model
        .save_file(model_path.join(format!("{}_final_weights", learner_config.model_id)), &CompactRecorder::new())
        .map_err(|e| io_err(format!("failed to save trained model: {}", e), io::ErrorKind::Other))?;

    // small pause for Learner dashboard TUI cleanup, then training loop confirmation
    io::stdout().flush().unwrap();
    thread::sleep(Duration::from_millis(100));
    println!("Training complete: model weights saved to {:?}\n", output_path);
    Ok(())
}



/// Creates train and validation dataloaders for the given dataset source.
///
/// Dispatches to dataset-specific creators (e.g. `create_grid_dataloaders` for GRID).
/// 
/// Returns a tuple of `(train_dataloader, valid_dataloader)` configured with
/// batch size, shuffling, and worker count from `learner_config`.
///
/// ### Params:
/// - `device`: Backend device for tensor placement.
/// - `context`: Application context (paths, config).
/// - `dataset_src`: Which dataset to use (Grid, Lrw, etc.).
/// - `learner_config`: Training config (batch size, workers, frame dims, seed).
/// - `token_map`: Token-to-ID mapping for transcript encoding.
///
/// ### Returns:
/// `(TrainLoader<B>, ValidLoader<B>)` — train and validation dataloaders.
fn create_dataloaders<B>(
    device: &B::Device,
    context: &Context,
    dataset_src: DatasetSource,
    learner_config: &VsrmLearnerConfig,
    token_map: &TokenMap,
) -> (TrainLoader<B>, ValidLoader<B>)
where B: AutodiffBackend,
{
    match dataset_src {
        DatasetSource::Grid => create_grid_dataloaders::<B>(device, context, learner_config, token_map),
        // DatasetSource::Lrw => create_lrw_dataloaders::<B>(...),
    }
}



/// Creates train and validation dataloaders for the GRID corpus.
///
/// Loads or computes global pixel stats for normalization, splits the dataset
/// (80% train / 10% valid / 10% test), and builds Burn dataloaders with
/// `VsrmBatcher`.
/// 
/// Uses pre-extracted mouth crops from `cropped_frames/` when
/// available; otherwise decodes video and runs `LipTracker` on demand.
///
/// ### Params:
/// - `device`: Backend device for tensor placement.
/// - `context`: Application context (paths, config).
/// - `learner_config`: Training config (batch size, workers, frame dims, seed).
/// - `token_map`: Token-to-ID mapping for transcript encoding.
///
/// ### Returns:
/// `(TrainLoader<B>, ValidLoader<B>)` — train and validation dataloaders.
fn create_grid_dataloaders<B>(
    device: &B::Device,
    context: &Context,
    learner_config: &VsrmLearnerConfig,
    token_map: &TokenMap,
) -> (TrainLoader<B>, ValidLoader<B>)
where
    B: AutodiffBackend,
{
    // (train/validation boundary, validation/test boundary)
    // for example:
    // total data:                  |---------------------------train---------------------------|-valid-|--test--|
    // train/eval split point:      |----------------------------80%--------------------------->|<---------------|
    // valid/test split point:      |-----------------------------------------------------------|--10%-->|<------|

    let split_thresholds = (0.8, 0.1);

    let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
        context.models_path.join("haarcascade_frontalface_alt2.xml"),
        context.models_path.join("haarcascade_mcs_mouth.xml"),
        learner_config.frame_dims,
    ));

    // GRID dataset instance
    let dataset = Arc::new(GridDataset::new(
        context,
        token_map,
        Some(tracker_config),
        learner_config.active_subset.clone(),
    ));

    let norm_stats_path = context.models_path
    .join(&learner_config.model_id)
    .join("norm_stats.json");

    // find global mean and std dev stats of all video frame pixels in GRID dataset (to use as input normalization)
    let grid_stats: DatasetStats = if norm_stats_path.exists() {
        load_json(&norm_stats_path).expect("failed to load cached GRID global mean and std dev stats")
    } else {
        let (mean, std_dev) = dataset.calc_global_stats();
        let stats = DatasetStats::new(mean, std_dev);
        save_json(&norm_stats_path, &stats).expect("failed to cache GRID global mean and std dev stats");
        stats
    };

    // train/validation dataset instances
    let (train_dataset, valid_dataset, _) = DatasetSplit::split(
        dataset,
        split_thresholds.0,
        split_thresholds.1,
        learner_config.seed,
    );

    assert!(train_dataset.len() > 0, "training dataset is empty");
    assert!(valid_dataset.len() > 0, "validation dataset is empty");

    println!("Train dataset has {} samples", train_dataset.len());
    println!("Valid dataset {} samples\n", valid_dataset.len());

    // train/validation data batcher instances (train uses Autodiff B, while valid uses InnerBackend raw B)
    let train_batcher = VsrmBatcher::<B>::new(device.clone(), Some(grid_stats));
    let valid_batcher = VsrmBatcher::<B::InnerBackend>::new(device.clone(), Some(grid_stats));

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

    assert!(train_dataloader.num_items() > 0, "training dataloader has 0 items");
    assert!(valid_dataloader.num_items() > 0, "validation dataloader has 0 items");

    (train_dataloader, valid_dataloader)
}




/// Applies entropy regularization to loss to improve model generalization.
/// 
/// Goal: penalize model when it becomes too confident too fast (such as collapsing to repeating chars/blanks).
/// 
/// Formula: loss + lambda * max(0, (threshold - entropy)).
///
/// ### Params:
/// - `logits`: Raw unnormalized model output scores over vocabulary.
/// - `lambda`: Penalty scaling factor.
/// - `min_entropy_threshold`: Min allowable entropy value before penalty increases overall loss.
///
/// ### Returns:
/// The final net penalty contribution towards model's output loss.
fn calc_entropy_penalty<B: Backend>(
    logits: Tensor<B, 3>,
    lambda: f32,
    min_entropy_threshold: f32,
) -> Tensor<B, 1> {
    // find entropy (or rather negative entropy in this case)
    // rule of thumb: (low entropy --> overconfident model --> higher penalty)
    let log_probs = log_softmax(logits.clone(), 2); // negative log-probs
    let probs = log_probs.clone().exp();                        // standard (0, 1] probs
    let neg_entropy = (probs * log_probs).sum_dim(2).mean();

    // find net entropy penalty
    let penalty = neg_entropy
        .add_scalar(min_entropy_threshold)     // (-entropy + threshold)
        .clamp(0.0, f32::MAX)             // max(0, (-entropy + threshold))
        .mul_scalar(lambda);                               // lambda * max(0, (-entropy + threshold))

    // debugging: Log entropy stats
    let log_probs = log_softmax(logits.clone(), 2);
    let probs = log_probs.clone().exp();
    let entropy_raw = (probs * log_probs).sum_dim(2).neg();           // [N, T] entropy per timestep
    let entropy_mean = entropy_raw.clone().mean().into_scalar().elem::<f32>(); // average entropy (positive)
    let entropy_min = entropy_raw.clone().min().into_scalar().elem::<f32>();   // min entropy (most confident)
    let entropy_max = entropy_raw.clone().max().into_scalar().elem::<f32>();   // max entropy (least confident)
    let penalty_val = penalty.clone().into_scalar().elem::<f32>();
    log::info!("Entropy: mean = {:.3}, min = {:.3}, max = {:.3}, penalty = {:.4}", entropy_mean, entropy_min, entropy_max, penalty_val);

    penalty
}



#[cfg(test)]
mod tests {
    use crate::{
        context::Context,
        vocab::VOCAB_SIZE,
    };

    use super::*;
    use burn::{
        tensor::{
            Tensor,
            Int,
            Distribution,
        },
        backend::{
            Autodiff,
            NdArray,
            ndarray::NdArrayDevice,
            wgpu::{Wgpu, WgpuDevice},
        },
        data::dataloader::batcher::Batcher,
        optim::{
            AdamConfig,
            GradientsParams,
            Optimizer,
        },
    };
    use rand::{
        Rng,
        SeedableRng,
        rngs::StdRng,
    };

    type TestBackend = Autodiff<Wgpu<f32, i32>>;
    // type TestBackend = Autodiff<NdArray>;

    const SEED: u64 = 69;

    // these unit tests are sanity checks to ensure the training loop, loss function, and backward pass are implemented correctly
    // and can successfully optimize model on a small dataset (without this, we might have silent bugs that prevent learning but don't cause crashes)

    #[test]
    #[ignore = "heavy computation: 300 steps overfit on synthetic data"]
    fn test_overfit_synthetic_sample() {
        // test if a randomly initialized model can overfit a sample of dummy data (loss should drop significantly after 100 training steps)

        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = WgpuDevice::default();
        // let device = NdArrayDevice::Cpu;

        let lr = 1e-3;
        let steps = 300;
        let mut initial_loss = 0.0;
        let mut current_loss = 0.0;
        let loss_threshold = 0.1;
        let frame_dims = (50, 150);

        println!("Device used for testing: {:?}", device);

        // create dummy dims for a batch
        let (n, c, t, h, w, l) = (
            1,            // batch size
            1,            // channels (grayscale)
            50,           // input length (frames)
            frame_dims.0, // frame height
            frame_dims.1, // frame width
            10,           // target length (chars)
        );

        // create dummy inputs data
        // (generate data between 0.0 and 1.0 as normalized pixel values)
        let inputs: Tensor<TestBackend, 5> = Tensor::random(
            [n, c, t, h, w],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        // create dummy targets data
        // (just a repeating sequence of indices [1, 2, ... 10])
        let targets: Tensor<TestBackend, 2, Int> = Tensor::<TestBackend, 1, Int>::from_ints(
            (0..l as i32).cycle().take(l * n).collect::<Vec<i32>>().as_slice(),
            &device,
        ).reshape([n, l]);

        // establish input/target lengths tensors
        let input_lengths = Tensor::from_ints([t as i32], &device);
        let target_lengths = Tensor::from_ints([l as i32], &device);

        // init optimizer
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        // init VSR model
        let mut model: VsrModel<TestBackend> = VsrModelConfig::new()
            .with_frame_dims(frame_dims)
            .with_vocab_size(vocab_size)
            .with_blank_id(blank_id)
            .init(&device);

        // init CTC loss
        let ctc_loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .init();

        // init CTC decoder
        let ctc_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        println!("=== OVERFITTING SYNTHETIC SAMPLE ===\n");

        // ------------------------- Debugging: Print Out Target Sequence ------------------------

        let targ_sequences = targets.to_data().to_vec::<i32>().unwrap()
            .chunks(l)
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        println!("Actual sequence(s): {:?}\n", targ_sequences);

        // ---------------------------------------------------------------------------------------

        // training loop
        for i in 0..steps {
            // forward pass and loss calculation
            let logits = model.forward(inputs.clone());
            let loss = ctc_loss
                .forward(
                    logits.clone(),
                    targets.clone(),
                    input_lengths.clone(),
                    target_lengths.clone(),
                );
            
            // check for explosion
            let loss_val = loss.clone().into_scalar().elem();
            if i == 0 { initial_loss = loss_val; }
            if i % 10 == 0 {
                println!("Step {}/{}: Loss = {:.4}", i, steps, loss_val);

                // ------------------------ Debugging: Print Out Predicted Sequence ----------------------

                let pred_sequences = ctc_decoder.forward(logits)
                    .iter()
                    .map(|seq| {
                        let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                        token_map.ids_to_chars(&ids).unwrap()
                    })
                    .collect::<Vec<Vec<char>>>();
                println!("Predicted sequence(s): {:?}\n", pred_sequences);

                // ---------------------------------------------------------------------------------------
            }

            // backward pass and optimizer
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            // success condition
            if loss_val < loss_threshold {
                println!("SUCCESS: Model overfit the batch! Loss dropped from {:.4} to {:.4}", initial_loss, current_loss);
                return;
            }

            current_loss = loss_val;
        }

        panic!("loss did not drop significantly: started at {}, ended at {}", initial_loss, current_loss);
    }

    #[test]
    #[ignore = "heavy computation: 300 steps overfit on real GRID sample with lip tracking"]
    fn test_overfit_real_sample() {
        // test if a randomly initialized model can overfit a sample of real data (loss should drop significantly after 100 training steps)

        let context = Context::new();
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = WgpuDevice::default();
        // let device = NdArrayDevice::Cpu;

        let mut rng = StdRng::seed_from_u64(SEED);
        let lr = 1e-3;
        let steps = 300;
        let mut initial_loss = 0.0;
        let mut current_loss = 0.0;
        let loss_threshold = 0.1;
        let frame_dims = (50, 100);

        // init mouth tracker
        let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
            context.models_path.join("haarcascade_frontalface_alt2.xml"),
            context.models_path.join("haarcascade_mcs_mouth.xml"),
            frame_dims,
        ));

        // init GRID dataset instance
        let dataset = GridDataset::new(
            &context,
            &token_map,
            Some(tracker_config),
            None,
        );

        // grab single real sample from our actual dataset (GRID)
        let dataset_item = dataset
            .get(rng.random_range(0..dataset.len()))
            .expect("failed to get first item from dataset");
        let batch = VsrmBatcher::<TestBackend>::new(device.clone(), None)
            .batch(vec![dataset_item], &device.clone());

        // init optimizer
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        // init VSR model
        let mut model: VsrModel<TestBackend> = VsrModelConfig::new()
            .with_frame_dims(frame_dims)
            .with_vocab_size(vocab_size)
            .with_blank_id(blank_id)
            .init(&device);

        // init CTC loss
        let ctc_loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .init();

        // init CTC decoder
        let ctc_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        println!("=== OVERFITTING REAL SAMPLE ===\n");

        // check normalization on the fly
        let min = batch.inputs.clone().min().into_scalar().elem::<f32>();
        let max = batch.inputs.clone().max().into_scalar().elem::<f32>();
        println!("Real frame pixels range: [{:.2} to {:.2}]\n", min, max);

        // ------------------------- Debugging: Print Out Target Sequence ------------------------

        let targ_sequences = batch.targets.to_data().to_vec::<i32>().unwrap()
        .chunks(batch.target_lengths.clone().into_scalar().elem::<i32>() as usize)
        .map(|seq| {
            let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
            token_map.ids_to_chars(&ids).unwrap()
        })
        .collect::<Vec<Vec<char>>>();
        println!("Actual sequence(s): {:?}\n", targ_sequences);

        // ---------------------------------------------------------------------------------------

        for i in 0..steps {
            // forward pass and loss calculation
            let logits = model.forward(batch.inputs.clone());
            let loss = ctc_loss.forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );

            // check for explosion
            let loss_val = loss.clone().into_scalar().elem();
            if i == 0 { initial_loss = loss_val; }
            if i % 10 == 0 {
                println!("Step {}/{}: Loss = {:.4}", i, steps, loss_val);

                // ------------------------ Debugging: Print Out Predicted Sequence ----------------------

                let pred_sequences = ctc_decoder.forward(logits)
                    .iter()
                    .map(|seq| {
                        let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                        token_map.ids_to_chars(&ids).unwrap()
                    })
                    .collect::<Vec<Vec<char>>>();
                println!("Predicted sequence(s): {:?}\n", pred_sequences);

                // ---------------------------------------------------------------------------------------
            }

            // backward pass and optimizer
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            // success condition
            if loss_val < loss_threshold {
                println!("SUCCESS: Model overfit the batch! Loss dropped from {:.4} to {:.4}", initial_loss, current_loss);
                return;
            }

            current_loss = loss_val;
        }

        panic!("loss did not drop significantly: started at {}, ended at {}", initial_loss, current_loss);
    }

    #[test]
    #[ignore = "heavy computation: 300 steps overfit on batch of 8 real GRID samples"]
    fn test_overfit_real_batch() {
        // test if a randomly initialized model can overfit a batch of real data

        let context = Context::new();
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = WgpuDevice::default();
        // let device = NdArrayDevice::Cpu;

        let mut rng = StdRng::seed_from_u64(SEED);
        let n = 8; // batch size
        let lr = 1e-3;
        let steps = 300;
        let mut initial_loss = 0.0;
        let mut current_loss = 0.0;
        let loss_threshold = 0.5;
        let frame_dims = (50, 150);

        // init mouth tracker
        let tracker_config = TrackerConfig::Haar(HaarTrackerConfig::new(
            context.models_path.join("haarcascade_frontalface_alt2.xml"),
            context.models_path.join("haarcascade_mcs_mouth.xml"),
            frame_dims,
        ));

        // init GRID dataset instance
        let dataset = GridDataset::new(
            &context,
            &token_map,
            Some(tracker_config),
            None,
        );

        // grab N real samples from our actual dataset (GRID again)
        let mut items = Vec::with_capacity(n);
        while items.len() < n {
            let idx = rng.random_range(0..dataset.len());
            if let Some(valid_item) = dataset.get(idx) { items.push(valid_item); }
            else { println!("Skipped invalid dataset item at index {}", idx) }
        }
        let batch = VsrmBatcher::<TestBackend>::new(device.clone(), None)
            .batch(items, &device.clone());

        // init optimizer
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        // init VSR model
        let mut model: VsrModel<TestBackend> = VsrModelConfig::new()
            .with_frame_dims(frame_dims)
            .with_vocab_size(vocab_size)
            .with_blank_id(blank_id)
            .init(&device);

        // init CTC loss
        let ctc_loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .init();

        // init CTC decoder
        let ctc_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        println!("=== OVERFITTING REAL BATCH (16 SAMPLES) ===\n");

        // check normalization on the fly
        let min = batch.inputs.clone().min().into_scalar().elem::<f32>();
        let max = batch.inputs.clone().max().into_scalar().elem::<f32>();
        println!("Real frame pixels range: [{:.2} to {:.2}]\n", min, max);

        // ------------------------- Debugging: Print Out Target Sequences -----------------------

        // print out actual target sequences in batch for debugging
        let flat_lengths = batch.target_lengths
            .to_data()
            .to_vec::<i32>()
            .unwrap();

        let flat_targets = batch.targets
            .to_data()
            .to_vec::<i32>()
            .unwrap();

        let mut offset = 0;
        let padded_len = batch.targets.dims()[1];
        let targ_sequences: Vec<Vec<char>> = flat_lengths
            .into_iter()
            .map(|actual_len| {
                let slice = &flat_targets[offset..offset + actual_len as usize];
                offset += padded_len;

                let ids: Vec<usize> = slice.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect();

        println!("Actual sequence(s):");
        for i in 0..targ_sequences.len() { println!("Sample {}: {:?}", (i + 1), &targ_sequences[i]); }
        println!();

        // ---------------------------------------------------------------------------------------

        for i in 0..steps {
            // forward pass and loss calculation
            let logits = model.forward(batch.inputs.clone());
            let loss = ctc_loss.forward(
                logits.clone(),
                batch.targets.clone(),
                batch.input_lengths.clone(),
                batch.target_lengths.clone(),
            );

            // check for explosion
            let loss_val = loss.clone().into_scalar().elem();
            if i == 0 { initial_loss = loss_val; }
            if i % 20 == 0 {
                println!("Step {}/{}: Loss = {:.4}", i, steps, loss_val);

                // ----------------------- Debugging: Print Out Predicted Sequences ----------------------

                let pred_sequences = ctc_decoder.forward(logits)
                    .iter()
                    .map(|seq| {
                        let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                        token_map.ids_to_chars(&ids).unwrap()
                    })
                    .collect::<Vec<Vec<char>>>();

                println!("Predicted sequence(s):");
                for i in 0..pred_sequences.len() { println!("Sample {}: {:?}", (i + 1), &pred_sequences[i]); }
                println!();

                // ---------------------------------------------------------------------------------------
            }

            // backward pass and optimizer
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(lr, model, grads);

            // success condition
            if loss_val < loss_threshold {
                println!("SUCCESS: Model overfit the batch! Loss dropped from {:.4} to {:.4}", initial_loss, current_loss);
                return;
            }

            current_loss = loss_val;
        }

        panic!("loss did not drop significantly: started at {}, ended at {}", initial_loss, current_loss);
    }
}
