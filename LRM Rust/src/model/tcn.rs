// Temporal Convolutional Network (TCN) implementation in Rust using Burn framework



// imports
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    optim::Adam,
    tensor::{Tensor, activation, activation::log_softmax, backend::Backend},
};



#[derive(Module, Debug)]
pub struct TcnBlock<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    padding: usize,
    dropout: Dropout,
    proj: Option<Conv1d<B>>,
}



impl<B: Backend> TcnBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
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
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            padding,
            dropout,
            proj,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // manually apply causal left padding to time dimension of input
        // default case:    (left, right, top, bottom) = (dim -1, dim -2)
        // our case:        (timesteps_left, timesteps_right, channels_left, channels_right)
        // resulting input: (padding, 0, 0, 0)

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



#[derive(Module, Debug)]
pub struct TemporalConvNet<B: Backend> {
    tcn_blocks: Vec<TcnBlock<B>>,
}



pub struct TemporalConvNetConfig {
    pub channels: [usize; 2],
    pub kernel_size: usize,
    pub layers: usize,
    pub dropout_prob: f64,
}



impl<B: Backend> TemporalConvNet<B> {
    pub fn new(
        channels: [usize; 2],
        kernel_size: usize,
        layers: usize,
        dropout_prob: f64,
        device: &B::Device,
    ) -> Self {
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

        Self { tcn_blocks }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for tcn_block in &self.tcn_blocks {
            x = tcn_block.forward(x);
        }
        x
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

    pub fn new(channels: [usize; 2], kernel_size: usize) -> Self {
        Self {
            channels,
            kernel_size,
            layers: 4,
            dropout_prob: 0.1,
        }
    }

    pub fn with_layers(mut self, layers: usize) -> Self {
        self.layers = layers;
        self
    }

    pub fn with_dropout(mut self, dropout_prob: f64) -> Self {
        self.dropout_prob = dropout_prob;
        self
    }
}



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::ndarray::NdArray, tensor::Tensor};
    use burn_ndarray::NdArrayDevice;

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
        // type B = NdArray<f32>;

        let device: NdArrayDevice = Default::default();
        let (n, c, t) = (2, 4, 32); // 2 batches to compare two cases
        let t0 = 10; // current timestep

        // causal TCN with no dropout for determinism
        let tcn: TemporalConvNet<B> = TemporalConvNetConfig::new([c, c], 3)
            .with_layers(3)
            .with_dropout(0.0)
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
