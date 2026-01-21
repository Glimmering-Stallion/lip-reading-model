// Model training loop



// custom imports
use crate::ctc::ctc_loss::CtcLossConfig;
use crate::ctc::ctc_decode::CtcDecoderConfig;

// imports
use crate::model::LRModel;
use burn::{
    backend::ndarray::NdArray,
    grad_clipping::GradientClippingConfig,
    lr_scheduler::LrScheduler,
    nn::loss::Reduction,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Int,
    tensor::{backend::Backend, Tensor},
};
use burn_autodiff::Autodiff;



pub type B0 = NdArray<f32>; // backend type
pub type AD = Autodiff<B0>; // autodiff backend



#[derive(Clone)]
pub struct Batch<B: Backend> {
    inputs: Tensor<B, 5>,       // [N, C, T, H, W]
    targets: Tensor<B, 2, Int>, // [N, L] (L padded to max target length in batch)

    input_lengths: Tensor<B, 1, Int>,  // [N]
    target_lengths: Tensor<B, 1, Int>, // [N]
}



pub fn train_epoch<S: LrScheduler>(
    mut model: LRModel<AD>,
    optimizer: &mut impl Optimizer<LRModel<AD>, AD>,
    loader: &mut impl Iterator<Item = Batch<AD>>,
    scheduler: &mut S,
    blank_id: usize,
) -> (LRModel<AD>, f64) {
    let mut total_loss = 0.0f64;
    let mut steps = 0usize;
    let reduction = Reduction::Mean;

    let ctc = CtcLossConfig::new()
        .with_blank_id(blank_id)
        .with_reduction(reduction)
        .init();

    for Batch {
        inputs,
        targets,
        input_lengths,
        target_lengths,
    } in loader
    {
        // forward pass
        let logits = model.forward(inputs); // [N, T, Vocab] (these are the raw, unnormalized predictions)

        // loss from CTC
        let loss = ctc.forward(logits.clone(), targets, input_lengths, target_lengths);

        // bail backprop if loss is non-finite
        let l = loss.clone().to_data().to_vec::<f32>().unwrap()[0];
        if !l.is_finite() {
            eprintln!("\n[skip] non-finite loss: {l}\n");
            continue;
        }

        // backpropagation
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // update model params
        let learning_rate = scheduler.step();
        model = optimizer.step(learning_rate, model, grads);

        // accumulate loss per batch
        total_loss += loss.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0] as f64;
        steps += 1;
    }

    // batch-wise average loss
    let avg_loss = if steps == 0 {
        0.0
    } else {
        total_loss / steps as f64
    };
    (model, avg_loss)
}



pub fn train_loop<S, F, L>(
    mut model: LRModel<AD>,
    epochs: usize,
    // learning_rate: f64,
    scheduler: &mut S,
    mut make_loader: F,
    blank_id: usize,
) -> (LRModel<AD>, Vec<f64>)
where
    S: LrScheduler,
    L: Iterator<Item = Batch<AD>>,
    F: FnMut() -> L,
{
    let clipper = GradientClippingConfig::Norm(0.05);
    let mut optimizer = AdamConfig::new()
        .with_epsilon(1e-6)
        .with_grad_clipping(Some(clipper))
        .init();
    let mut losses = Vec::with_capacity(epochs); // history of losses for each epoch

    for epoch in 0..epochs {
        let mut loader = make_loader(); // new loader each epoch
        let (m, l) = train_epoch(model, &mut optimizer, &mut loader, scheduler, blank_id);
        losses.push(l);
        model = m;
        println!("Epoch {}/{} | Loss {:.4}", epoch + 1, epochs, l);
    }

    (model, losses)
}



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::TrainEval, utils::mean};
    use burn::{
        backend::ndarray::NdArray,
        lr_scheduler::noam::NoamLrSchedulerConfig,
        prelude::Int,
        tensor::{backend::Backend, Distribution, Tensor},
    };
    use burn_autodiff::Autodiff;

    // backends
    type B = NdArray<f32>;
    type AD = Autodiff<B>;

    // fn pick_norm_group(out_channels: usize) -> usize {
    //     for g in (1..=out_channels).rev() {
    //         if out_channels % g == 0 { return g.min(32);  /* 8 group cap */ }
    //     }
    //     1
    // }

    #[test]
    fn test_train_epoch() {
        // (batch size, channels, timesteps, height, width, sequence length), where t ≥ 2l - 1
        let (n, c, t, h, w, l) = (1, 1, 6, 16, 16, 3);
        let out_channels = 10;
        let vocab_size = 41;
        let blank_id = vocab_size - 1; // last index is blank token
        let epochs = 25;
        let batches = 2; // num batches per epoch
        let norm_groups = 5;
        let in_dist = Distribution::Uniform(-0.5, 0.5);
        let tgt_dist = Distribution::Uniform(0.0, (vocab_size - 2) as f64);
        B::seed(69);

        let total_steps = epochs * batches; // batch-wise steps
        let scale_factor = 3e-3;
        let warmup_steps = (0.2 * total_steps as f64).floor() as usize; // num warmup steps before decay steps
        let model_size = 75; // hidden feature dim

        // noam learning rate scheduler (linear warmup and inverse-sqrt decay)
        // lr peak: factor * (d_model * warmup_steps)^(-0.5)
        let mut noam_lr = NoamLrSchedulerConfig::new(scale_factor)
            .with_warmup_steps(warmup_steps)
            .with_model_size(model_size)
            .init()
            .unwrap();

        // dummy model
        let device = Default::default();
        let mut model =
            LRModel::<AD>::new(c, out_channels, (h, w), norm_groups, vocab_size, &device);
        model.eval(); // disable TCN dropout for more determinism in this unit test

        // debugging: inspect model's layers' shapes
        println!("\nModel layer shapes:");
        model.inspect_shapes_once(Tensor::<AD, 5>::random([n, c, t, h, w], in_dist, &device));

        // fixed-value random inputs
        let (inputs, targets, in_len, tgt_len) = (
            Tensor::<AD, 5>::random([n, c, t, h, w], in_dist, &device), // random pixel values per frame
            Tensor::<AD, 2, Int>::random([n, l], tgt_dist, &device),    // random symbol ID
            Tensor::<AD, 1, Int>::from_ints([t as i64], &device),
            Tensor::<AD, 1, Int>::from_ints([l as i64], &device),
        );

        let make_loader = {
            move || {
                let mut v = Vec::new();
                for _ in 0..batches {
                    v.push(Batch {
                        inputs: inputs.clone(),
                        targets: targets.clone(),
                        input_lengths: in_len.clone(),
                        target_lengths: tgt_len.clone(),
                    });

                    // // debugging: print input/target tensor values (floats/ints)
                    // let in_data = inputs.clone().to_data().to_vec::<f32>().unwrap();
                    // let tgt_data = targets.clone().to_data().to_vec::<i64>().unwrap();
                    // println!("\nInputs data (first 10/{}) = {:?}", in_data.len(), &in_data[..in_data.len().min(10)]);
                    // println!("Targets data (first 10/{}) = {:?}\n", tgt_data.len(), &tgt_data[..tgt_data.len().min(10)]);
                }
                v.into_iter()
            }
        };

        // run training loop
        println!("\nRunning training loop with {} epochs\n", epochs);
        let (_, losses) = train_loop(model, epochs, &mut noam_lr, make_loader, blank_id);

        let n = 5;
        let first_n = losses[0..n].to_vec();
        let last_n = losses[(losses.len() - n)..losses.len()].to_vec();

        // sanity checks
        println!("\nLosses = {:.4?}\n", losses);
        assert!(losses.iter().all(|x| x.is_finite())); // all losses should be finite
        assert!(losses.len() == epochs); // should have one loss per epoch
        assert!(mean(&last_n) < mean(&first_n)); // losses should have a downward trend
        assert!(t as i64 >= 2 * l as i64 - 1, "CTC needs T ≥ 2L - 1");
    }
}
