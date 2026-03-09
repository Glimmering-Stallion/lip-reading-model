//! Visual Speech Recognition Model (VSRM) architecture.
//! 
//! This module implements the core neural network, which consists of:
//! - 3D ResBlock Frontend: Spatio-temporal convolutional layers for feature extraction.
//! - TCN Backend: Stacked temporal Conv1D blocks for sequence modeling.
//! - Iteration Counter: Custom Atomic iteration tracker for training progress.



// imports
use burn::{
    backend::Autodiff,
    config::Config,
    module::{
        AutodiffModule,
        Content,
        Module,
        ModuleDisplay,
        ModuleDisplayDefault,
        ModuleVisitor,
        Param,
        ParamId,
    },
    nn::{
        GroupNorm,
        GroupNormConfig,
        Initializer,
        Linear,
        LinearConfig,
        PaddingConfig3d,
        conv::{
            Conv3d,
            Conv3dConfig,
        },
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
        }
    },
    optim::GradientsParams,
    prelude::TensorData,
    tensor::{
        Shape,
        Tensor,
        activation,
        backend::{
            AutodiffBackend,
            Backend,
        }
    }
};



#[cfg(test)]
use std::sync::Once;
use std::sync::{Arc, atomic::AtomicU64};
use crate::vsrm::{
    residual::{
        ResidualBlock,
        ResidualBlockConfig,
    },
    tcn::{
        TemporalConvNet,
        TemporalConvNetConfig,
    }
};

#[cfg(test)]
static PRINT_ONCE: Once = Once::new();



// -------------------------------- Internal Metadata Tracking For Training Iterations --------------------------------

// since Burn's Module system is strictly designed for Tensors,
// for sake of tracking an iteration counter without triggering constant GPU-to-CPU syncs,
// or breaking serialization (Record) system,
// wrap an AtomicU64 in a dummy module

#[derive(Default, Debug)]
pub struct IterationCounter(pub Arc<AtomicU64>);

/// prevents Burn logger from crashing when printing AtomicU64 iteration value
impl ModuleDisplay for IterationCounter {}
impl ModuleDisplayDefault for IterationCounter {
    fn content(&self, _content: Content) -> Option<Content> {
        let mut new_content = Content::new(_content.display_settings);
        new_content.top_level_type = Some("IterationCounter".to_string());
        Some(new_content)
    }
}

/// AtomicU64 can't be cloned, so manual Clone implementation to return fresh counter
impl Clone for IterationCounter {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

/// satisfy Autodiff requirement so this can live inside a training model
impl<B: AutodiffBackend> AutodiffModule<B> for IterationCounter {
    type InnerModule = IterationCounter;
    fn valid(&self) -> Self::InnerModule { self.clone() }
}

/// dummy Module implementation for IterationCounter
impl<B: Backend> Module<B> for IterationCounter {
    type Record = (); // tells Burn "nothing to save to disk"

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> { devices } // no tensors, so no new devices to add
    fn fork(self, _device: &B::Device) -> Self { self } // atomics are CPU-bound, so ignore device forking
    fn to_device(self, _device: &B::Device) -> Self { self } // stay on CPU
    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {}
    fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self { self } // nothing to map (no weights/biases)
    fn load_record(self, _record: Self::Record) -> Self { self }
    fn into_record(self) -> Self::Record { () } // save nothing to disk
}

// --------------------------------------------------------------------------------------------------------------------



// ------------------------------------------ Debugging For Model Gradients -------------------------------------------

pub struct ParamIds {
    pub rb1_w: ParamId,
    pub rb2_w: ParamId,
    pub rb3_w: ParamId,
    pub fc_w: ParamId,
}

/// Helper that computes and prints basic distribution statistics for a given tensor.
/// 
/// Used for identifying vanishing/exploding gradients or activations.
///
/// ### Params:
/// - `name`: Label for console output.
/// - `t`: Tensor to analyze [D-dimensional].
#[cfg(test)]
fn stats_any<B: Backend, const D: usize>(
    name: &str,
    t: &Tensor<B, D>,
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
fn stats_any<B: Backend, const D: usize>(
    _name: &str,
    _t: &Tensor<B, D>,
) {}

/// Specific helper for logging gradient magnitudes during backpropagation.
///
/// ### Params:
/// - `label`: Name of the parameter layer.
/// - `t`: 1D tensor containing flattened gradient values.
#[cfg(test)]
fn print_grad_stats<B: Backend, const D: usize>(label: &str, t: &Tensor<B, D>) {
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

// --------------------------------------------------------------------------------------------------------------------



#[derive(Config, Debug)]
pub struct VsrModelConfig {
    #[config(default = 1)]
    pub in_channels: usize,           // initial channel width input for the first ResBlock layer (typically 1 for grayscale)

    #[config(default = 128)]
    pub out_channels: usize,          // base channel width output for frontend 3D-CNN layers (the feature extractors)

    #[config(default = 512)]
    pub hidden_dim: usize,            // latent dimension for backend TCN layers (the sequence processors)

    #[config(default = "(50, 100)")]
    pub frame_dims: (usize, usize),   // initial frame dimensions used to precompute 3D-CNN downsampled spatial sizes (height, width)
    
    #[config(default = 8)]
    pub norm_groups: usize,           // number of groups for GroupNorm
    
    #[config(default = 28)]           // default assumes 0-25 for alphabet, 26 for space, 27 for blank ID
    pub vocab_size: usize,

    #[config(default = 27)]           // default assumes blank ID is at last position in char vocab
    pub blank_id: usize,
}



impl VsrModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> VsrModel<B> {
        VsrModel::new(
            self.in_channels,
            self.out_channels,
            self.hidden_dim,
            self.frame_dims,
            self.norm_groups,
            self.vocab_size,
            self.blank_id,
            device,
        )
    }
}



// define model architecture
#[derive(Module, Debug)]
pub struct VsrModel<B: Backend> {
    rb1: ResidualBlock<B>,
    rb2: ResidualBlock<B>,
    rb3: ResidualBlock<B>,

    aap: AdaptiveAvgPool2d,
    proj: Linear<B>,

    tcn1: TemporalConvNet<B>,
    tcn2: TemporalConvNet<B>,

    fc: Linear<B>,

    #[module(skip)] // treat this field not as a tensor/param
    pub iterations: IterationCounter,
}



impl<B: Backend> VsrModel<B> {
    /// Initializes full VSRM architecture with frontend CNN and backend TCN.
    ///
    /// ### Params:
    /// - `in_channels`: Input video channels (usually 1 for grayscale).
    /// - `out_channels`: Base feature width (determines hidden sizes of TCN).
    /// - `hidden_dim`: Latent dimension for TCN block layers.
    /// - `frame_dims`: Tuple of (height, width) for spatial input.
    /// - `norm_groups`: Number of groups for GroupNorm (must divide channel counts).
    /// - `vocab_size`: Total number of character classes for output.
    /// - `device`: Backend device for initialization.
    ///
    /// ### Returns:
    /// An initialized VSR model with precomputed receptive field metadata.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_dim: usize,
        frame_dims: (usize, usize),
        norm_groups: usize,
        vocab_size: usize,
        blank_id: usize,
        device: &B::Device,
    ) -> Self {
        let iterations = IterationCounter::default();
        // ResBlock internal Conv3D kernel size values: [temporal, height, width]
        let (k_t, k_h, k_w) = (3, 3, 3);
        let kernel_size = [k_t, k_h, k_w];

        // ResBlock internal Conv3D stride length values: [temporal, height, width]
        let (s_t, s_h, s_w) = (1, 2, 2);
        let stride = [s_t, s_h, s_w];

        // ResBlock internal Conv3D padding values: [temporal, height, width]
        let (p_t, p_h, p_w) = (1, 1, 1);
        let padding = [p_t, p_h, p_w];

        // ResBlock output channel values for each layer
        let rb1_out = out_channels;       // 128 (default)
        let rb2_out = out_channels * 2;   // 256 (default)
        let rb3_out = out_channels * 4;   // 512  (default)

        // AdaptiveAvgPool2D output size: [height, width]
        let (aap_h, aap_w) = (4, 4);

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

        assert!(rb1_out.is_multiple_of(norm_groups), "First Conv3D layer output ({}) must be divisible by Norm Groups ({})", rb1_out, norm_groups);
        assert!(rb2_out.is_multiple_of(norm_groups), "Second Conv3D layer output ({}) must be divisible by Norm Groups ({})", rb2_out, norm_groups);
        assert!(rb3_out.is_multiple_of(norm_groups), "Third Conv3D layer output ({}) must be divisible by Norm Groups ({})", rb3_out, norm_groups);

        assert!(frame_dims.0 >= 40 && frame_dims.1 >= 40, "Frame dimensions must be >= 40, got H = {}, W = {}", frame_dims.0, frame_dims.1);
        if h3 < 4 || w3 < 4 { eprintln!("Warning: downsampled feature map is very small ({}x{}); representation quality may suffer", h3, w3); }

        assert!(out_channels >= 32, "Out channels ({}) must be >= 32", out_channels);
        if rb3_out < 64 { eprintln!("Warning: third Conv3D channels ({}) is very small; representation quality may suffer", rb3_out); }

        let rb1 = ResidualBlockConfig::new([in_channels, rb1_out], kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_norm_groups(norm_groups)
            .init(device);
        let rb2 = ResidualBlockConfig::new([rb1_out, rb2_out], kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_norm_groups(norm_groups)
            .init(device);
        let rb3 = ResidualBlockConfig::new([rb2_out, rb3_out], kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_norm_groups(norm_groups)
            .init(device);

        let aap = AdaptiveAvgPool2dConfig::new([aap_h, aap_w])
            .init();

        let proj = LinearConfig::new((rb3_out * aap_h * aap_w), hidden_dim)
            .with_initializer(Initializer::KaimingUniform {
                gain: 2.0f64.sqrt(),
                fan_out_only: false,
            })
            .with_bias(true)
            .init(device);

        let tcn1 = TemporalConvNetConfig::new([hidden_dim, hidden_dim])
            .with_layers(3)
            .with_dropout_prob(0.1)
            .init(device);
        let tcn2 = TemporalConvNetConfig::new([hidden_dim, hidden_dim])
            .with_layers(3)
            .with_dropout_prob(0.1)
            .init(device);

        let mut fc = LinearConfig::new(hidden_dim, vocab_size)
            .with_initializer(Initializer::XavierUniform { gain: 1.0 }) // possibly need to init final layer even smaller (like 0.1 or 0.01)
            .with_bias(true)
            .init(device);

        // apply negative biasing for blank ID
        if let Some(bias_param) = &fc.bias {
            // FC layer outputs logits shaped [N, T, Vocab]
            // with external softmax, we have probabilites:
            // P(blank) = exp(bias) / (Vocab * exp(0) + exp(bias))
            // P(char) = exp(0) / (Vocab * exp(0) + exp(bias))

            // example, with Vocab (chars and space only) = 27:
            // bias (for blank) = 0.0: P(blank) = 1/28 ≈ 3.6%
            // bias (for blank) = 1.0: P(blank) = e^1/(27 + e^1) ≈ 9.1%
            // bias (for blank) = 2.0: P(blank) = e^2/(27 + e^2) ≈ 21.4%
            // bias (for blank) = 3.0: P(blank) = e^3/(27 + e^3) ≈ 42.6%
            // bias (for blank) = 4.0: P(blank) = e^4/(27 + e^4) ≈ 66.9%

            let mut data = bias_param.val().to_data();
            
            // optionally tweak initial blank prob up/down here so other chars can breathe
            if let Ok(values) = data.as_mut_slice::<f32>() { values[blank_id] = 5.0; }
            
            // re-upload to device and update layer
            let new_bias = Tensor::<B, 1>::from_data(data, device);
            fc.bias = Some(Param::from_tensor(new_bias));
        }

        let min_viable_rf = 25;
        let max_viable_rf = 3 * 75;
        let mut total_rf = 1;

        // sum contributions from frontend ResBlocks
        total_rf += rb1.receptive_field_contribution();
        total_rf += rb2.receptive_field_contribution();
        total_rf += rb3.receptive_field_contribution();

        // sum contributions from backend TCNs
        total_rf += tcn1.receptive_field_contribution() - 1;
        total_rf += tcn2.receptive_field_contribution() - 1;

        println!("Model initialized with temporal receptive field of {} frames\n", total_rf);
        debug_assert!(total_rf > min_viable_rf, "Vision too narrow: {} frames", total_rf);
        debug_assert!(total_rf < max_viable_rf, "Vision too wide: {} frames", total_rf);

        Self {
            rb1, rb2, rb3,
            aap, proj,
            tcn1, tcn2,
            fc,
            iterations,
        }
    }

    /// Forward pass of VSRM architecture.
    /// 
    /// Processes raw video frames into raw unnormalized character scores (logits).
    ///
    /// ### Params:
    /// - `input`: [N, C, T, H, W] batch of video frames.
    ///
    /// ### Returns:
    /// [N, T, Vocab] logits for each timestep.
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 3> {
        // note: N is samples per batch (batch size), C is channels, T is timesteps (number of frames), H is height (frame dim y), W is width (frame dim x)

        // three custom ResBlock3D layers (with strided downsampling and ReLU internally applied)
        let x = self.rb1.forward(input);     // [N, C, T, (H / 2), (W / 2)]
        let x = self.rb2.forward(x);  // [N, C, T, (H / 4), (W / 4)]
        let x = self.rb3.forward(x);  // [N, C, T, (H / 8), (W / 8)]

        // reshape input to rank 4 for spatial Adaptive Average Pooling (AAP)
        let [n, c, t, h, w] = x.dims();
        let x = x
            .swap_dims(1, 2)           // [N, T, C, H, W]
            .reshape([n * t, c, h, w]);                 // [(N * T), C, H, W]

        // AAP layer for downsampling spatial dims, then reshape input to rank 3
        let x = self.aap
            .forward(x)                     // [(N * T), C, 4, 4]
            .reshape([n, t, c * 4 * 4]);                // [N, T, (C * 4 * 4)]

        // projection layer right before TCN layers
        // for compressing large flattened features to hidden dim
        // reshape input to NDT format for subsequent TCN layers
        let x = activation::relu(self.proj.forward(x));  // [N, T, D]
        let x = x.swap_dims(1, 2);                          // [N, D, T]

        // two custom TCN layers (ReLU internally applied)
        let x: Tensor<B, 3> = self.tcn1.forward(x);        // [N, D, T]
        let x: Tensor<B, 3> = self.tcn2.forward(x);        // [N, D, T]

        // reshape input to NTD format for FC layer (bringing features to last dim)
        let x = x.swap_dims(1, 2);   // [N, T, D]
        let y = self.fc.forward(x);      // [N, T, V]

        y
    }

    /// Helper that computes total receptive field of VSR model.
    ///
    /// ### Returns:
    /// Total number of temporal frames the model can see.
    pub fn total_receptive_field(&self) -> usize {
        let mut total_rf = 1;

        // sum contributions from frontend ResBlocks
        total_rf += self.rb1.receptive_field_contribution();
        total_rf += self.rb2.receptive_field_contribution();
        total_rf += self.rb3.receptive_field_contribution();

        // sum contributions from backend TCNs
        // Note: why minus 1 each time?
        // sequential temporal modules stack additively
        // minus overlapping center frame
        total_rf += self.tcn1.receptive_field_contribution() - 1;
        total_rf += self.tcn2.receptive_field_contribution() - 1;

        total_rf
    }

    /// Maps parameter IDs for weight inspection in autodiff mode.
    ///
    /// ### Returns:
    /// Container of IDs for ResBlock and Linear layers.
    pub fn param_ids(&self) -> ParamIds {
        ParamIds {
            rb1_w: self.rb1.primary_weight_id(),
            rb2_w: self.rb2.primary_weight_id(),
            rb3_w: self.rb3.primary_weight_id(),
            fc_w: self.fc.weight.id,
        }
    }
}



#[cfg(test)]
impl<B0: Backend> VsrModel<Autodiff<B0>> {
    /// Prints statistical summaries of current gradients across major layers.
    ///
    /// ### Params:
    /// - `grads`: Gradient container from current training iteration.
    pub fn debug_print_grads(&self, grads: &GradientsParams) {
        let ids = self.param_ids();
        if let Some(g) = grads.get::<B0, 5>(ids.rb1_w) { print_grad_stats("grad rb1.weight", &g); }
        if let Some(g) = grads.get::<B0, 5>(ids.rb2_w) { print_grad_stats("grad rb2.weight", &g); }
        if let Some(g) = grads.get::<B0, 5>(ids.rb3_w) { print_grad_stats("grad rb3.weight", &g); }
        if let Some(g) = grads.get::<B0, 2>(ids.fc_w) { print_grad_stats("grad fc.weight", &g); }
    }

    /// Logs exact tensor shapes at every stage of the forward pass.
    /// 
    /// Execution is guarded by `PRINT_ONCE` to prevent console flooding.
    ///
    /// ### Params:
    /// - `input`: [N, C, T, H, W] sample input tensor.
    pub fn inspect_shapes_once(&self, input: Tensor<Autodiff<B0>, 5>) {
        PRINT_ONCE.call_once(|| {
            println!("IN (N, C, T, H, W): {:?}", input.dims());

            let x = activation::relu(self.rb1.forward(input));
            println!("C1 (N, C, T, H, W): {:?}", x.dims());

            let x = activation::relu(self.rb2.forward(x));
            println!("C2 (N, C, T, H, W): {:?}", x.dims());

            let x = activation::relu(self.rb3.forward(x));
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
    use crate::vocab::{BLANK_ID, VOCAB_SIZE};
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
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let out_channels = 128;
        let norm_groups = 8;

        let device = Default::default();
        let model = VsrModelConfig::new()
            .with_frame_dims((h, w))
            .with_in_channels(c)
            .with_out_channels(out_channels)
            .with_norm_groups(norm_groups)
            .with_vocab_size(vocab_size)
            .with_blank_id(blank_id)
            .init(&device);

        let input = Tensor::<B, 5>::zeros([n, c, t, h, w], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [n, t, VOCAB_SIZE]); // expected output shape
    }
}
