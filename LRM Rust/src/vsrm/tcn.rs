//! Temporal Convolutional Network (TCN) implementation.
//! 
//! This module provides causal, dilated 1D convolutions that provides the
//! temporal receptive field for the VSRM.
//! Features of the TCN include:
//! - Causality: Padding logic makes sure the output at present timestep only depends on inputs from previous timesteps.
//! - Exponential Dilation: Increases lookback range that doubles with each layer.
//! - Residual Blocks: The dual-stack dilated causal convolution layers that enable the large receptive field.



// imports
use burn::{
    module::Module,
    config::Config,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig,
    },
    tensor::{
        activation,
        backend::Backend,
        Tensor,
    },
};



#[derive(Module, Debug)]
pub struct TcnBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    padding: usize,
    dropout: Dropout,
    proj: Option<Conv1d<B>>,
}



#[derive(Config, Debug)]
pub struct TemporalConvNetConfig {
    /// [in channels, out channels]
    pub channels: [usize; 2],

    #[config(default = 3)]
    pub kernel_size: usize,

    #[config(default = 4)]
    pub layers: usize,

    #[config(default = 0.0)]
    pub dropout_prob: f64,
}



#[derive(Module, Debug)]
pub struct TemporalConvNet<B: Backend> {
    tcn_blocks: Vec<TcnBlock<B>>,

    #[module(ignored)]
    pub kernel_size: usize,

    #[module(ignored)]
    pub layers: usize,
}



impl<B: Backend> TcnBlock<B> {
    /// initializes a residual block with two dilated causal convolutions
    /// params:
    /// - in_channels: number of input features
    /// - out_channels: number of output features (and internal width)
    /// - kernel_size: temporal width of convolution window
    /// - dilation: spacing between kernel elements (controls lookback range)
    /// - dropout_prob: probability for dropout layers between convolutions
    /// - device: the backend device to initialize weights on
    /// returns: a block containing dual convolutions and an optional projection for residuals
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        assert!(in_channels > 0 && out_channels > 0, "TCN channels must be > 0");
        assert!(kernel_size > 0, "TCN kernel size must be > 0");
        assert!(dilation > 0, "TCN dilation must be > 0");
        assert!((0.0..=1.0).contains(&dropout_prob), "TCN dropout probability must be in [0, 1]");

        let conv1 = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_dilation(dilation)
            .init(device);

        let conv2 = Conv1dConfig::new(out_channels, out_channels, kernel_size)
            .with_dilation(dilation)
            .init(device);

        let padding = (kernel_size - 1) * dilation;

        let dropout = DropoutConfig::new(dropout_prob).init();

        let proj = if in_channels != out_channels {
            Some(Conv1dConfig::new(in_channels, out_channels, 1).init(device))
        } else { None };

        Self {
            conv1,
            conv2,
            padding,
            dropout,
            proj,
        }
    }

    /// forward pass of single TCN block
    /// applies dilated causal convolution with residual connection/dropout
    /// params:
    /// - input: [N, C, T] feature sequence
    /// returns: [N, C, T] processed sequence with preserved temporal length
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // manually apply causal left padding to time dimension of input
        // default case:    (left, right, top, bottom) = (dim -1, dim -2)
        // our case:        (timesteps_left, timesteps_right, channels_left, channels_right)
        // resulting input: (padding, 0, 0, 0)

        debug_assert_eq!(input.dims().len(), 3);
        debug_assert!(input.dims()[2] > 0, "Temporal dimension must be > 0");

        let x = input.clone().pad((self.padding, 0, 0, 0), 0.0); // first left-padding
        let x = activation::relu(self.conv1.forward(x));
        let x = self.dropout.forward(x);

        let x = x.pad((self.padding, 0, 0, 0), 0.0); // second left-padding
        let x = activation::relu(self.conv2.forward(x));
        let x = self.dropout.forward(x);

        let residual = match &self.proj {
            Some(p) => p.forward(input),
            None => input,
        };

        x + residual
    }
}



impl TemporalConvNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TemporalConvNet<B> {
        TemporalConvNet::new(
            self.channels,
            self.kernel_size,
            self.layers,
            self.dropout_prob,
            device,
        )
    }
}



impl<B: Backend> TemporalConvNet<B> {
    /// builds a stack of TCN blocks with exponentially increasing dilation
    /// params:
    /// - channels: array of [in channels, out channels]
    /// - kernel_size: size of the 1D temporal kernel
    /// - layers: number of residual blocks to stack (dilation = 2^layers)
    /// - dropout_prob: dropout rate applied within each residual block
    /// - device: backend device for weight allocation
    /// returns: a network with a receptive field determined by kernel and depth
    pub fn new(
        channels: [usize; 2],
        kernel_size: usize,
        layers: usize,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
        assert!(layers > 0, "TCN must have at least one layer");
        assert!(layers < 12, "TCN layers too large ({}), dilation will explode", layers);
        assert!(kernel_size > 0);
        assert!(channels[0] > 0 && channels[1] > 0);

        let mut tcn_blocks = Vec::with_capacity(layers);
        let mut current_in_channels = channels[0];
        let out_channels = channels[1];

        for i in 0..layers {
            // exponential dilation: 1, 2, 4, 8, ...
            let dilation = 1 << i; // bitwise left shift here (2^i)
            let tcn_block = TcnBlock::new(
                current_in_channels,
                out_channels,
                kernel_size,
                dilation,
                dropout_prob,
                device,
            );
            tcn_blocks.push(tcn_block);
            current_in_channels = out_channels;
        }

        Self {
            tcn_blocks,
            kernel_size,
            layers,
        }
    }

    /// sequential forward pass through stack of residual blocks
    /// params:
    /// - x: [N, C_in, T] input tensor from frontend/embedding
    /// returns: [N, C_out, T] tensor with deep temporal features
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for tcn_block in &self.tcn_blocks {
            x = tcn_block.forward(x);
        }
        x
    }

    /// calculates model's temporal lookback range in frames
    /// formula: 1 + 2 * (kernel_size - 1) * (2^layers - 1)
    /// returns: total number of past frames a single output point can see
    pub fn receptive_field(&self) -> usize {
        1 + 2 * (self.kernel_size - 1) * ((1 << self.layers) - 1)
    }
}



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::ndarray::{
            NdArray,
            NdArrayDevice,
        },
        tensor::Tensor
    };

    type B = NdArray<f32>;

    #[test]
    fn tcn_block_preserves_length_and_sets_channels() {
        let device = Default::default();
        let (n, c_in, l) = (2, 16, 32);
        let c_out = 32;

        let block = TcnBlock::<B>::new(
            c_in, c_out, 3,   // kernel
            2,   // dilation
            0.1, // dropout
            &device,
        );

        let x = Tensor::<B, 3>::zeros([n, c_in, l], &device);
        let y = block.forward(x);

        assert_eq!(y.dims(), [n, c_out, l]);
    }

    #[test]
    fn tcn_is_causal() {
        let device: NdArrayDevice = Default::default();
        let (n, c, t) = (2, 4, 32); // 2 batches to compare two cases
        let t0 = 10; // current timestep

        // causal TCN with no dropout for determinism
        let tcn: TemporalConvNet<B> = TemporalConvNetConfig::new([c, c])
            .with_layers(3)
            // .with_dropout(0.0)
            .init(&device);

        // input tensor with two batches
        // first batch: only prefix (control)
        // second batch: prefix + suffix (test)
        let x = Tensor::<B, 3>::zeros([n, c, t], &device);

        // make identical prefix (past) for both batches: fill [:, :, 0..=t0] with a constant
        const ARBITRARY_PREFIX_OFFSET: f64 = 0.5;
        let prefix = Tensor::<B, 3>::zeros([n, c, t0 + 1], &device) + ARBITRARY_PREFIX_OFFSET;
        let x = x.slice_assign([0..n, 0..c, 0..(t0 + 1)], prefix);

        // perturb suffix (future) for second batch: set [1, :, t0+1..] to another constant
        const ARBITRARY_SUFFIX_OFFSET: f64 = 3.15;
        let suffix = Tensor::<B, 3>::zeros([1, c, t - (t0 + 1)], &device) + ARBITRARY_SUFFIX_OFFSET;
        let x = x.slice_assign([1..2, 0..c, (t0 + 1)..t], suffix);

        // run TCN
        let y = tcn.forward(x);

        // compare outputs up to t0: y[0, :, ..=t0] vs y[1, :, ..=t0]
        let y0 = y.clone().slice([0..1, 0..c, 0..(t0 + 1)]);
        let y1 = y.clone().slice([1..2, 0..c, 0..(t0 + 1)]);
        let diff = (y0 - y1).abs().sum().into_scalar();

        // suffix should not affect prefix
        assert!(diff < 1e-6, "Non-causal behavior detected: diff={diff}");
    }
}
