// model training loop



// imports
use crate::model::LRModel;
use burn::{
    backend::ndarray::NdArray,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::Backend, Tensor},
    LearningRate, // just an f64 alias
};
use burn_autodiff::Autodiff;



pub type B0 = NdArray<f32>; // backend type
pub type AD = Autodiff<B0>; // autodiff backend



#[derive(Clone)]
pub struct Batch<B: Backend> {
    // shapes: x = [N, C, T, H, W], Y = [N, T, Vocab]
    x: Tensor<B, 5>,
    y: Tensor<B, 3>,
}



pub fn train_epoch(
    mut model: LRModel<AD>,
    optimizer: &mut impl Optimizer<LRModel<AD>, AD>,
    loader: &mut impl Iterator<Item = Batch<AD>>,
    learning_rate: LearningRate,
) -> (LRModel<AD>, f64) {
    let mut total_loss = 0.0f64;
    let mut steps = 0usize;

    for Batch { x, y: _ } in loader {
        // forward pass
        let logits = model.forward(x.require_grad());
        // let loss = ctc_loss(&logits, &y); // TODO: implement CTC loss function (change batch to include targets, target_lengths, and input_lengths)
        let loss = (logits.clone() * logits).mean(); // placeholder L2-Norm loss function

        // backpropagation
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        // update model params
        model = optimizer.step(learning_rate, model, grads);

        // accumulate loss
        total_loss += loss.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0] as f64;
        steps += 1;
    }

    // batch-wise average loss
    let avg_loss = if steps == 0 { 0.0 } else { total_loss / steps as f64 };
    (model, avg_loss)
}



pub fn train_loop<F, L>(
    mut model: LRModel<AD>,
    epochs: usize,
    learning_rate: f64,
    mut make_loader: F,
) -> (LRModel<AD>, Vec<f64>)
where
    L: Iterator<Item = Batch<AD>>,
    F: FnMut() -> L,
{
    let mut optimizer = AdamConfig::new().init();
    let mut losses = Vec::with_capacity(epochs); // history of losses for each epoch

    for epoch in 0..epochs {
        let mut loader = make_loader(); // new loader each epoch
        let (m, l) = train_epoch(model, &mut optimizer, &mut loader, learning_rate);
        losses.push(l);
        model = m;
        println!(
            "Epoch {}/{} | Loss {:.4}",
            epoch + 1,
            epochs,
            l
        );
    }

    (model, losses)
}



// if we eventually have a DataLoader type, give it an iter() method:
// impl DataLoader {
//     pub fn iter(&self) -> impl Iterator<Item = Batch<AD>> { /* ... */ }
// }
// then the factory is just: || data_loader.iter()



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::ndarray::NdArray,
        tensor::{Distribution, Tensor},
    };
    use burn_autodiff::Autodiff;

    // backends
    type B = NdArray<f32>;
    type AD = Autodiff<B>;

    #[test]
    fn test_train_epoch() {
        let (n, c, t, h, w) = (1, 1, 8, 16, 16);
        let out_channels = 8;
        let vocab_size = 41;
        let epochs = 10;
        let learning_rate = 0.001;
        let batches = 2; // per epoch
        let distribution = Distribution::Uniform(-0.5, 0.5);

        // dummy model
        let device = Default::default();
        let model = LRModel::<AD>::new(c, out_channels, (h, w), vocab_size, &device);

        // inspect model shapes
        model.inspect_shapes_once(Tensor::<AD, 5>::random([n, c, t, h, w], distribution, &device),);

        let make_loader = {
            move || {
                let mut v = Vec::new();
                for _ in 0..batches {
                    v.push(Batch {
                        x: Tensor::<AD, 5>::random([n, c, t, h, w], distribution, &device),
                        y: Tensor::<AD, 3>::random([n, t, vocab_size], distribution, &device),
                    });
                }
                v.into_iter()
            }
        };

        // run training loop
        println!(
            "\nRunning training loop with {} epochs and learning rate {}",
            epochs, learning_rate
        );
        let (_, losses) = train_loop(model, epochs, learning_rate, make_loader);

        // sanity checks
        println!("\nlosses = {:?}\n", losses);
        assert!(losses.iter().all(|x| x.is_finite())); // all losses should be finite
        assert!(losses.len() == epochs); // should have one loss per epoch
        assert!(losses.last().unwrap() < losses.first().unwrap()); // losses should have a downward trend
    }
}
