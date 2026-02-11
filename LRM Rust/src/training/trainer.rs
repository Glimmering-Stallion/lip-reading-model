// VSRM manual training loop for low-level control (like custom gradient inspection, and debugging CTC-specific edge cases)



// custom imports
use crate::{
    ctc::ctc_loss::{CtcLoss, CtcLossConfig},
    pipeline::batcher::Batch,
    vsrm::VsrModel,
};

// imports
use burn::{
    backend::{
        Autodiff,
        wgpu::Wgpu,
        ndarray::NdArray,
    },
    tensor::Tensor,
    grad_clipping::GradientClippingConfig,
    lr_scheduler::LrScheduler,
    optim::{
        AdamConfig,
        GradientsParams,
        Optimizer,
    },
};



pub type B = B2;               // select active backend type
pub type B1 = NdArray<f32>;    // backend type (CPU)
pub type B2 = Wgpu<f32, i32>;  // backend type (GPU)
pub type AD = Autodiff<B>;     // autodiff backend



pub fn train_epoch<S: LrScheduler>(
    mut model: VsrModel<AD>,
    optimizer: &mut impl Optimizer<VsrModel<AD>, AD>,
    loader: &mut impl Iterator<Item = Batch<AD>>,
    ctc_loss: &CtcLoss,
    scheduler: &mut S,
) -> (VsrModel<AD>, f64) {
    let mut steps = 0usize;
    let mut total_loss: Option<Tensor<AD, 1>> = None;

    for batch in loader {
        let (inputs, targets, input_lengths, target_lengths) = (
            batch.inputs,
            batch.targets,
            batch.input_lengths,
            batch.target_lengths,
        );

        // forward pass
        // [N, T, Vocab] (these are the raw, unnormalized predictions)
        let logits = model.forward(inputs);

        // loss from CTC
        // [N] (these are scores for how well the model aligned with the targets)
        let loss = ctc_loss.forward(logits.clone(), targets, input_lengths, target_lengths);

        // backpropagation
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // update model params
        let learning_rate = scheduler.step();
        model = optimizer.step(learning_rate, model, grads);

        // accumulate loss per batch
        let loss_val = loss.clone().detach();
        if let Some(acc) = total_loss { total_loss = Some(acc.add(loss_val)); }
        else { total_loss = Some(loss_val); }

        steps += 1;
    }

    // batch-wise average loss
    let avg_loss = if let Some(acc) = total_loss {
        let total = acc.to_data().convert::<f64>().as_slice::<f64>().unwrap()[0];
        if steps == 0 { 0.0 } else { total / steps as f64 }
    } else { 0.0 };

    (model, avg_loss)
}



pub fn train_loop<S, F, L>(
    mut model: VsrModel<AD>,
    epochs: usize,
    scheduler: &mut S,
    mut make_loader: F,
    blank_id: usize,
) -> (VsrModel<AD>, Vec<f64>)
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

    // history of losses per epoch
    let mut losses = Vec::with_capacity(epochs);

    let ctc_loss = CtcLossConfig::new()
        .with_blank_id(blank_id)
        .init();

    for epoch in 0..epochs {
        let mut loader = make_loader(); // new loader each epoch
        let (m, l) = train_epoch(
            model,
            &mut optimizer,
            &mut loader,
            &ctc_loss,
            scheduler,
        );
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
    use crate::{
        utils::mean, vocab::{
            BLANK_ID, VOCAB_SIZE
        }, vsrm::VsrModelConfig
    };
    use burn::{
        backend::Autodiff,
        lr_scheduler::noam::NoamLrSchedulerConfig,
        prelude::Int,
        tensor::{
            backend::Backend,
            Distribution,
            Tensor,
        },
    };

    pub type B = Wgpu<f32, i32>;  // backend type (GPU)
    pub type AD = Autodiff<B>;    // autodiff backend

    #[test]
    fn test_train_epoch() {
        // (batch size, channels, timesteps, height, width, sequence length), where t ≥ 2l - 1
        let (n, c, t, h, w, l) = (1, 1, 6, 40, 40, 3);
        let out_channels = 32;
        let epochs = 25;
        let batches = 2; // num batches per epoch
        let norm_groups = 4;
        let in_dist = Distribution::Uniform(-0.5, 0.5);
        let tgt_dist = Distribution::Uniform(0.0, (VOCAB_SIZE - 2) as f64);
        let device = Default::default();
        B::seed(&device, 69);

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
        let model = VsrModelConfig::new((h, w))
            .with_in_channels(c)
            .with_out_channels(out_channels)
            .with_norm_groups(norm_groups)
            .with_vocab_size(VOCAB_SIZE)
            .init(&device);

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
        let (_, losses) = train_loop(
            model,
            epochs,
            &mut noam_lr,
            make_loader,
            BLANK_ID,
        );

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
