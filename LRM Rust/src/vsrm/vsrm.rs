// Lip Reading Model architecture implementation



// imports
use burn::{
    backend::Autodiff,
    config::Config,
    module::{Module, ParamId},
    nn::{
        conv::{
            Conv3d,
            Conv3dConfig,
        },
        GroupNorm,
        GroupNormConfig,
        Initializer,
        Linear,
        LinearConfig,
        PaddingConfig3d,
    },
    optim::GradientsParams,
    tensor::{
        activation,
        backend::Backend,
        Shape,
        Tensor,
    },
};



#[cfg(test)]
use std::sync::Once;
use crate::vsrm::tcn::{TemporalConvNet, TemporalConvNetConfig};

#[cfg(test)]
static PRINT_ONCE: Once = Once::new();



#[derive(Config, Debug)]
pub struct VsrModelConfig {
    #[config(default = 1)]
    pub in_channels: usize,

    #[config(default = 128)]
    pub out_channels: usize,

    // #[config(default = (50, 150))] // default should be (50, 150)
    pub frame_dims: (usize, usize), // (height, width)
    
    #[config(default = 8)]
    pub norm_groups: usize,
    
    #[config(default = 26)]
    pub vocab_size: usize,
}



impl VsrModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VsrModel<B> {
        VsrModel::new(
            self.in_channels,
            self.out_channels,
            self.frame_dims,
            self.norm_groups,
            self.vocab_size,
            device,
        )
    }
}



// define model architecture
#[derive(Module, Debug)]
pub struct VsrModel<B: Backend> {
    conv1: Conv3d<B>, gn1: GroupNorm<B>,
    conv2: Conv3d<B>, gn2: GroupNorm<B>,
    conv3: Conv3d<B>, gn3: GroupNorm<B>,

    tcn1: TemporalConvNet<B>,
    tcn2: TemporalConvNet<B>,

    fc: Linear<B>,
}



// helper for model layer tensor stat debugging
#[cfg(test)]
fn stats_any<B: burn::tensor::backend::Backend, const D: usize>(
    name: &str,
    t: &burn::tensor::Tensor<B, D>,
) {
    let data = t.clone().to_data().convert::<f32>();
    let s = data.as_slice::<f32>().unwrap();
    let (mut min, mut max, mut sum, mut nans, mut infs) =
        (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0, 0);
    for &v in s {
        if v.is_nan() {
            nans += 1;
        }
        if v.is_infinite() {
            infs += 1;
        }
        if v.is_finite() {
            if v < min {
                min = v
            }
            if v > max {
                max = v
            }
            sum += v;
        }
    }
    let mean = sum / (s.len().max(1) as f32);
    println!("{name} | mean = {mean:.6} min = {min:.6} max = {max:.6} NaNs = {nans} Infs = {infs} len = {}", s.len());
}
#[cfg(not(test))]
fn stats_any<B: burn::tensor::backend::Backend, const D: usize>(
    _name: &str,
    _t: &burn::tensor::Tensor<B, D>,
) {
}



// helper for model gradient debugging
#[cfg(test)]
fn print_grad_stats<B: Backend>(label: &str, t: &Tensor<B, 1>) {
    let v = t.clone().to_data().convert::<f32>();
    let s = v.as_slice::<f32>().unwrap();
    let (mut min, mut max, mut sum, mut n_nan, mut n_inf) =
        (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0, 0);
    for &x in s {
        if x.is_nan() {
            n_nan += 1;
        }
        if x.is_infinite() {
            n_inf += 1;
        }
        if x.is_finite() {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
            sum += x;
        }
    }
    let mean = sum / (s.len().max(1) as f32);
    println!(
        "{label} | mean={mean:.6} min={min:.6} max={max:.6} NaNs={n_nan} Infs={n_inf} len={}",
        s.len()
    );
}



// default input parameters: input_channels = 1, output_channels = 128, frame_dims = (50, 150), vocab_size = 41
impl<B: Backend> VsrModel<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        frame_dims: (usize, usize),
        norm_groups: usize,
        vocab_size: usize,
        device: &B::Device,
    ) -> Self {
        
        let conv1_out = out_channels;       // 128 (default)
        let conv2_out = out_channels * 2;   // 256 (default)
        let conv3_out = out_channels / 2;   // 64  (default)
        
        let (h0, w0) = frame_dims;
        let (h1, w1) = ((h0 + 2 - 3) / 2 + 1, (w0 + 2 - 3) / 2 + 1);
        let (h2, w2) = ((h1 + 2 - 3) / 2 + 1, (w1 + 2 - 3) / 2 + 1);
        let (h3, w3) = ((h2 + 2 - 3) / 2 + 1, (w2 + 2 - 3) / 2 + 1);

        let conv1 = Conv3dConfig::new([in_channels, conv1_out], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn1 = GroupNormConfig::new(norm_groups, conv1_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let conv2 = Conv3dConfig::new([conv1_out, conv2_out], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn2 = GroupNormConfig::new(norm_groups, conv2_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let conv3 = Conv3dConfig::new([conv2_out, conv3_out], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn3 = GroupNormConfig::new(norm_groups, conv3_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let tcn1 = TemporalConvNetConfig::new([(conv3_out * h3 * w3), out_channels], 3)
            .with_layers(6)
            .with_dropout(0.5)
            .init(device);

        let tcn2 = TemporalConvNetConfig::new([out_channels, 75], 3)
            .with_layers(6)
            .with_dropout(0.5)
            .init(device);

        let fc = LinearConfig::new(75, vocab_size)
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .with_bias(true)
            .init(device);

        Self {
            conv1, gn1,
            conv2, gn2,
            conv3, gn3,
            tcn1,
            tcn2,
            fc,
        }
    }

    /// forward pass of VSRM architecture
    /// processes raw video frames into raw unnormalized character scores (logits)
    /// params:
    /// - input: [N, C, T, H, W] batch of video frames
    /// returns: [N, T, Vocab] logits for each timestep
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 3> {
        // note: N is samples per batch (batch size), C is channels, T is timesteps (number of frames), H is height (frame dim), W is width (frame dim)

        // three 3D convolutional layers with ReLU activation and strided downsampling
        // input shape: (batch, channels, timesteps, height, width)
        // output shape: (batch, channels, timesteps, height/(2^3), width/(2^3))
        let x = activation::relu(self.gn1.forward(self.conv1.forward(input)));
        let x = activation::relu(self.gn2.forward(self.conv2.forward(x)));
        let x = activation::relu(self.gn3.forward(self.conv3.forward(x)));

        // reshape input to rank 3 as NCT format for TCN layers (bringing timesteps to last dim)
        // input shape: (batch, channels, timesteps, height/(2^3), width/(2^3))
        // output shape: (batch, channels * height/(2^3) * width/(2^3), timesteps)
        let [batch, channels, timesteps, height, width] = x.dims();
        let x = x.reshape(Shape::new([batch, channels * height * width, timesteps]));

        // two custom TCN layers with ReLU activation
        // input shape: (batch, channels * height/(2^3) * width/(2^3), timesteps)
        // output shape: (batch, channels * height/(2^3) * width/(2^3), timesteps)
        let x: Tensor<B, 3> = activation::relu(self.tcn1.forward(x));
        let x: Tensor<B, 3> = activation::relu(self.tcn2.forward(x));

        // reshape input to NTC format for FC layer (bringing features to last dim)
        // input shape: (batch, channels * height/(2^3) * width/(2^3), timesteps)
        // output shape: (batch, timesteps, channels * height/(2^3) * width/(2^3))
        let x = x.swap_dims(1, 2);

        // single FC layer
        // input shape: (batch, timesteps, channels * height/(2^3) * width/(2^3))
        // output shape: (batch, timesteps, vocab_size)
        let y = self.fc.forward(x);

        y
    }
}



#[cfg(test)]
pub struct ParamIds {
    pub conv1_w: ParamId,
    pub conv2_w: ParamId,
    pub conv3_w: ParamId,
    pub fc_w: ParamId,
}



#[cfg(test)]
impl<B0: Backend> VsrModel<Autodiff<B0>> {
    pub fn param_ids(&self) -> ParamIds {
        ParamIds {
            conv1_w: self.conv1.weight.id,
            conv2_w: self.conv2.weight.id,
            conv3_w: self.conv3.weight.id,
            fc_w: self.fc.weight.id,
        }
    }

    pub fn debug_print_grads(&self, grads: &GradientsParams) {
        let ids = self.param_ids();
        // tiny helper
        fn print_grad_stats<B: burn::tensor::backend::Backend, const D: usize>(
            name: &str,
            t: &burn::tensor::Tensor<B, D>,
        ) {
            let data = t.clone().to_data().convert::<f32>();
            let s = data.as_slice::<f32>().unwrap();
            let (mut min, mut max, mut sum, mut nans, mut infs) =
                (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0, 0);
            for &v in s {
                if v.is_nan() {
                    nans += 1;
                }
                if v.is_infinite() {
                    infs += 1;
                }
                if v.is_finite() {
                    if v < min {
                        min = v
                    }
                    if v > max {
                        max = v
                    }
                    sum += v;
                }
            }
            let mean = sum / (s.len().max(1) as f32);
            println!(
                "{name} | mean={mean:.6} min={min:.6} max={max:.6} NaNs={nans} Infs={infs} len={}",
                s.len()
            );
        }

        if let Some(g) = grads.get::<B0, 5>(ids.conv1_w) {
            print_grad_stats("grad conv1.weight", &g);
        }
        if let Some(g) = grads.get::<B0, 5>(ids.conv2_w) {
            print_grad_stats("grad conv2.weight", &g);
        }
        if let Some(g) = grads.get::<B0, 5>(ids.conv3_w) {
            print_grad_stats("grad conv3.weight", &g);
        }
        if let Some(g) = grads.get::<B0, 2>(ids.fc_w) {
            print_grad_stats("grad fc.weight", &g);
        }
    }

    #[cfg(test)]
    pub fn inspect_shapes_once(&self, input: Tensor<Autodiff<B0>, 5>) {
        PRINT_ONCE.call_once(|| {
            println!("IN (N, C, T, H, W): {:?}", input.dims());

            let x = activation::relu(self.conv1.forward(input));
            println!("C1 (N, C, T, H, W): {:?}", x.dims());

            let x = activation::relu(self.conv2.forward(x));
            println!("C2 (N, C, T, H, W): {:?}", x.dims());

            let x = activation::relu(self.conv3.forward(x));
            println!("C3 (N, C, T, H, W): {:?}", x.dims());

            let [batch, channels, timesteps, height, width] = x.dims();
            let x = x.reshape(Shape::new([batch, channels * height * width, timesteps]));
            println!("RS (N, C_feat, T): {:?}", x.dims());

            let x = activation::relu(self.tcn1.forward(x));
            println!("TCN1 (N, C_feat, T): {:?}", x.dims());

            let x = activation::relu(self.tcn2.forward(x));
            println!("TCN2 (N, C_feat, T): {:?}", x.dims());

            let x = x.swap_dims(1, 2);
            println!("SWP (N, T, C_feat): {:?}", x.dims());

            let y = self.fc.forward(x);
            println!("OUT (N, T, Vocab): {:?}", y.dims());
        });
    }
}



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VOCAB_SIZE;
    use burn::{
        backend::ndarray::NdArray,
        tensor::Tensor,
    };

    // choosing backend for testing
    type B = NdArray<f32>;

    #[test]
    fn model_input_shapes_data_flow_small() {
        // let (n, c, t, h, w) = (1, 1, 8, 16, 16);
        // let out_channels = 10;
        // let norm_groups = 5;

        let (n, c, t, h, w) = (1, 1, 75, 50, 150); // Real GRID dimensions
        let out_channels = 128;
        let norm_groups = 8;

        let device = Default::default();
        // let model = VsrModel::<B>::new(c, out_channels, (h, w), norm_groups, VOCAB_SIZE, &device);
        let model = VsrModel::<B>::new(c, out_channels, (h, w), norm_groups, VOCAB_SIZE, &device);

        let input = Tensor::<B, 5>::zeros([n, c, t, h, w], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [n, t, VOCAB_SIZE]); // expected output shape
    }
}
