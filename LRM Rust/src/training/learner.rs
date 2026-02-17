// VSRM dedicated training pipeline and orchestrator using Burn (handles auto-checkpointing, logging, and validation sets)



// custom imports
use crate::{
    vocab::{VOCAB, VOCAB_SIZE, BLANK_ID, TokenMap},
    pipeline::{
        batcher::{
            Batch,
            VsrmBatcher,
        },
        dataset::{
            DatasetSplit,
            DatasetSource,
        },
        adapters::grid::GridDataset,
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
    module::Module,
    optim::AdamConfig,
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
    },
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

    #[config(default = 1)]
    pub accumulation: usize,

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
        if !loss.clone().is_finite().all().into_scalar().elem::<bool>() {
            println!("DUMPING BATCH: Loss is NaN/Inf!");
            println!("Batch lengths: {:?}", batch.input_lengths);
            panic!("Loss contains non-finite values");
        }

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

        // debugging: print out character-level predictions for first item in batch
        let indices = logits.clone().argmax(2).squeeze::<2>();

        // convert indices to &[usize]
        let indices_vec: Vec<usize> = indices.clone().slice([0..1, 0..t])
            .to_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .unwrap()
            .into_iter()
            .map(|x| x as usize)
            .collect();

        // use token map to convert indices to characters (for first item in batch)
        let token_map = TokenMap::new(VOCAB); // replace with actual token map if needed
        let predicted_tokens: Vec<char> = token_map.ids_to_chars(&indices_vec).unwrap();

        println!("Predicted tokens for first item in batch: {:?}", predicted_tokens);

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
        .grads_accumulation(learner_config.accumulation)
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
    // train/eval split point:      |----------------------------80%--------------------------->|<---------------|
    // valid/test split point:      |-----------------------------------------------------------|--10%-->|<------|
    let split_thresholds = (0.8, 0.1);

    match dataset_src {
        DatasetSource::Grid => {
            // dataset instance
            let dataset = Arc::new(GridDataset::new(root_path, token_map));

            // train/validation dataset instances
            let (train_dataset, valid_dataset, _) = DatasetSplit::split(
                dataset,
                split_thresholds.0,
                split_thresholds.1,
                learner_config.seed,
            );

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



#[cfg(test)]
mod tests {
    use crate::pipeline::dataset;

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
    use std::path::PathBuf;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    type TestBackend = Autodiff<Wgpu<f32, i32>>;
    // type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_overfit_single_synthetic_batch() {
        // test if a randomly initialized model can overfit a single batch of dummy data (loss should drop significantly after 100 training steps)
        // this is a sanity check to ensure the training loop, loss function, and backward pass are implemented correctly
        // and can successfully optimize the model on a small dataset (without this, we might have silent bugs that prevent learning but don't cause crashes)

        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = WgpuDevice::default();
        // let device = NdArrayDevice::Cpu;

        let lr = 1e-3;
        let steps = 200;
        let mut initial_loss = 0.0;
        let mut current_loss = 0.0;
        let loss_threshold = 0.1;
        let frame_dims = (50, 150);

        println!("Device used for testing: {:?}", device);

        // create dummy dims for a batch
        let (n, c, t, h, w, l) = (
            2,            // batch size
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
        let input_lengths = Tensor::from_ints([t as i32, t as i32], &device);
        let target_lengths = Tensor::from_ints([l as i32, l as i32], &device);

        // init optimizer
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        // init VSR model
        let mut model: VsrModel<TestBackend> = VsrModelConfig::new(frame_dims)
            .with_vocab_size(vocab_size)
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

        println!("--- STARTING SINGLE BATCH OVERFIT TEST ---\n");

        // print out actual target sequences for debugging
        let targ_sequences = targets.to_data().to_vec::<i32>().unwrap()
            .chunks(l)
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        println!("Actual sequences: {:?}\n", targ_sequences);

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

                // print out decoded predictions for debugging
                let pred_sequences = ctc_decoder.forward(logits)
                    .iter()
                    .map(|seq| {
                        let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                        token_map.ids_to_chars(&ids).unwrap()
                    })
                    .collect::<Vec<Vec<char>>>();
                println!("Predicted sequences: {:?}\n", pred_sequences);
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

        panic!("FAILURE: Loss did not drop significantly. Started at {}, ended at {}.", initial_loss, current_loss);
    }

    #[test]
    fn test_overfit_single_real_batch() {
        // test if a randomly initialized model can overfit a single batch of real data (loss should drop significantly after 100 training steps)

        let root_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = WgpuDevice::default();
        // let device = NdArrayDevice::Cpu;

        let seed = 69;
        let mut rng = StdRng::seed_from_u64(seed);
        let lr = 1e-3;
        let steps = 200;
        let mut initial_loss = 0.0;
        let mut current_loss = 0.0;
        let loss_threshold = 0.1;
        let frame_dims = (50, 150);

        // grab single real batch from our actual dataset (GRID)
        let dataset = GridDataset::new(root_path, token_map.clone());
        let dataset_item = dataset
            .get(rng.random_range(0..dataset.len()))
            .expect("Failed to get first item from dataset");
        let batch = VsrmBatcher::<TestBackend>::new(device.clone())
            .batch(vec![dataset_item], &device.clone());

        // init optimizer
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        // init VSR model
        let mut model: VsrModel<TestBackend> = VsrModelConfig::new(frame_dims)
            .with_vocab_size(vocab_size)
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

        println!("--- OVERFITTING REAL BATCH ---\n");
        
        // check normalization on the fly
        let min = batch.inputs.clone().min().into_scalar().elem::<f32>();
        let max = batch.inputs.clone().max().into_scalar().elem::<f32>();
        println!("Real frame pixels range: [{:.2} to {:.2}]\n", min, max);

        // print out actual target sequences for debugging
        let targ_sequences = batch.targets.to_data().to_vec::<i32>().unwrap()
            .chunks(batch.target_lengths.clone().into_scalar().elem::<i32>() as usize)
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        println!("Actual sequences: {:?}\n", targ_sequences);

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

                // decode predictions for debugging
                let pred_sequences = ctc_decoder.forward(logits)
                    .iter()
                    .map(|seq| {
                        let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                        token_map.ids_to_chars(&ids).unwrap()
                    })
                    .collect::<Vec<Vec<char>>>();
                println!("Predicted sequences: {:?}\n", pred_sequences);
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

        panic!("FAILURE: Loss did not drop significantly. Started at {}, ended at {}.", initial_loss, current_loss);
    }
}