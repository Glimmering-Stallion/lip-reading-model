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
    
    #[config(default = 27)] // 0-25 for alphabet, 26 for space, 27 for blank ID
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



/// helper that computes and prints basic distribution statistics for a given tensor
/// used for identifying vanishing/exploding gradients or activations
/// params:
/// - name: label for console output
/// - t: tensor to analyze [D-dimensional]
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



/// specific helper for logging gradient magnitudes during backpropagation
/// params:
/// - label: name of the parameter layer
/// - t: 1D tensor containing flattened gradient values
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



impl<B: Backend> VsrModel<B> {
    /// initializes full VSRM architecture with frontend CNN and backend TCN
    /// params:
    /// - in_channels: input video channels (usually always 1 for grayscale)
    /// - out_channels: base feature width (determines hidden sizes of TCN)
    /// - frame_dims: tuple of (height, width) for spatial input
    /// - norm_groups: number of groups for GroupNorm (must divide channel counts)
    /// - vocab_size: total number of character classes for output
    /// - device: backend device for initialization
    /// returns: initialized VSR model with precomputed receptive field metadata
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        frame_dims: (usize, usize),
        norm_groups: usize,
        vocab_size: usize,
        device: &B::Device,
    ) -> Self {
        // number of Conv3D/GroupNorm layers
        let frontend_layers = 3;

        // Conv3D kernel size values: [temporal, height, width]
        let (k_t, k_h, k_w) = (3, 3, 3);
        let kernel_size = [k_t, k_h, k_w];

        // Conv3D stride length values: [temporal, height, width]
        let (s_t, s_h, s_w) = (1, 2, 2);
        let stride = [s_t, s_h, s_w];

        // Conv3D padding values: [temporal, height, width]
        let (p_t, p_h, p_w) = (1, 1, 1);

        // Conv3D out channel values for each layer
        let conv1_out = out_channels;       // 128 (default)
        let conv2_out = out_channels * 2;   // 256 (default)
        let conv3_out = out_channels / 2;   // 64  (default)

        // precompute spatial dim downsample output sizes after each Conv3D layer
        let downsample = |size: usize, stride: usize, kernel: usize, pad: usize| {
            ((size + 2 * pad - kernel) / stride) + 1
        };
        let (h0, w0) = frame_dims; // frame dims need to be at least 40
        let (h1, w1) = ( // downsamp out needs to at least be 13
            downsample(h0, s_h, k_h, p_h),
            downsample(w0, s_w, k_w, p_w),
        );
        let (h2, w2) = ( // downsamp out needs to at least be 4
            downsample(h1, s_h, k_h, p_h),
            downsample(w1, s_w, k_w, p_w),
        );
        let (h3, w3) = ( // downsamp out needs to at least be 1
            downsample(h2, s_h, k_h, p_h),
            downsample(w2, s_w, k_w, p_w),
        );

        assert!(conv1_out.is_multiple_of(norm_groups), "First Conv3D layer output ({}) must be divisible by Norm Groups ({})", conv1_out, norm_groups);
        assert!(conv2_out.is_multiple_of(norm_groups), "Second Conv3D layer output ({}) must be divisible by Norm Groups ({})", conv2_out, norm_groups);
        assert!(conv3_out.is_multiple_of(norm_groups), "Third Conv3D layer output ({}) must be divisible by Norm Groups ({})", conv3_out, norm_groups);

        assert!(frame_dims.0 >= 40 && frame_dims.1 >= 40, "Frame dimensions must be >= 40, got H = {}, W = {}", frame_dims.0, frame_dims.1);
        if h3 < 4 || w3 < 4 { eprintln!("Warning: downsampled feature map is very small ({}x{}); representation quality may suffer", h3, w3); }

        assert!(out_channels >= 32, "Out channels ({}) must be >= 32", out_channels);
        if conv3_out < 64 { eprintln!("Warning: third Conv3D channels ({}) is very small; representation quality may suffer", conv3_out); }


        let conv1 = Conv3dConfig::new([in_channels, conv1_out], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig3d::Explicit(p_t, p_h, p_w))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn1 = GroupNormConfig::new(norm_groups, conv1_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let conv2 = Conv3dConfig::new([conv1_out, conv2_out], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig3d::Explicit(p_t, p_h, p_w))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn2 = GroupNormConfig::new(norm_groups, conv2_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let conv3 = Conv3dConfig::new([conv2_out, conv3_out], kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig3d::Explicit(p_t, p_h, p_w))
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);
        let gn3 = GroupNormConfig::new(norm_groups, conv3_out)
            .with_epsilon(1e-5)
            .with_affine(true)
            .init(device);

        let tcn1 = TemporalConvNetConfig::new([(conv3_out * h3 * w3), out_channels])
            .with_layers(4)
            .with_dropout_prob(0.5)
            .init(device);

        let tcn2 = TemporalConvNetConfig::new([out_channels, 75])
            .with_layers(4)
            .with_dropout_prob(0.5)
            .init(device);

        let fc = LinearConfig::new(75, vocab_size)
            .with_initializer(Initializer::KaimingUniform {
                gain: 1.0,
                fan_out_only: false,
            })
            .with_bias(true)
            .init(device);

        // compute total receptive field from:
        // - frontend (3 Conv3D + GroupNorm) layers  RF = 1 + Σ_{i = (0..layers-1)} (k_t - 1) * s_t^i
        // - backend (2 TCN) layers                  RF = 1 + 2 * (k - 1) * (2^layers - 1)
        let rf_frontend = 1 + (0..frontend_layers).map(|i| (k_t - 1) * s_t.pow(i as u32)).sum::<usize>();
        let rf_backend = (tcn1.receptive_field() - 1) + (tcn2.receptive_field() - 1); // Note: why minus 1 each time? sequential temporal modules stack additively minus overlapping center frame
        let total_rf = rf_frontend + rf_backend;

        let min_viable_rf = 25;
        let max_viable_rf = 3 * 75;

        println!("Model initialized with temporal receptive field of {} frames", total_rf);
        debug_assert!(total_rf > min_viable_rf, "Vision too narrow: {} frames", total_rf);
        debug_assert!(total_rf < max_viable_rf, "Vision too wide: {} frames", total_rf);

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
    /// - input: [N, C, T, H, W]  batch of video frames
    /// returns: [N, T, Vocab]    logits for each timestep
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 3> {
        // note: N is samples per batch (batch size), C is channels, T is timesteps (number of frames), H is height (frame dim y), W is width (frame dim x)

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

        debug_assert!(channels * height * width > 0);
        debug_assert!(timesteps > 0);

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
    /// maps parameter IDs for weight inspection in autodiff mode
    /// returns: container of IDs for Conv3D and Linear layers
    pub fn param_ids(&self) -> ParamIds {
        ParamIds {
            conv1_w: self.conv1.weight.id,
            conv2_w: self.conv2.weight.id,
            conv3_w: self.conv3.weight.id,
            fc_w: self.fc.weight.id,
        }
    }

    /// prints statistical summaries of current gradients across major layers
    /// params:
    /// - grads: gradient container from current training iteration
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

    /// logs exact tensor shapes at every stage of the forward pass
    /// execution is guarded by PRINT_ONCE to prevent console flooding
    /// params:
    /// - input: [N, C, T, H, W] sample input tensor
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
