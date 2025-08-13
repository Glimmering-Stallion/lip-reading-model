// Lip Reading Model architecture implementation



// imports
mod tcn;
use burn::{
    module::Module,
    nn::{
        conv::{Conv3d, Conv3dConfig},
        Linear, LinearConfig, PaddingConfig3d,
    },
    tensor::{activation, backend::Backend, Shape, Tensor},
};
use tcn::{TemporalConvNet, TemporalConvNetConfig};



#[cfg(test)]
use std::sync::Once;

#[cfg(test)]
static PRINT_ONCE: Once = Once::new();



// define model architecture
#[derive(Module, Debug)]
pub struct LRModel<B: Backend> {
    conv1: Conv3d<B>,
    conv2: Conv3d<B>,
    conv3: Conv3d<B>,

    tcn1: TemporalConvNet<B>,
    tcn2: TemporalConvNet<B>,

    fc: Linear<B>,
}



// default input parameters: input_channels = 1, output_channels = 128, input_dims = (50, 150), vocab_size = 41
impl<B: Backend> LRModel<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        input_dims: (usize, usize),
        vocab_size: usize,
        device: &B::Device,
    ) -> Self {
        let (height, width) = input_dims;

        let conv1 = Conv3dConfig::new([in_channels, out_channels], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);

        let conv2 = Conv3dConfig::new([out_channels, out_channels * 2], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);

        let conv3 = Conv3dConfig::new([out_channels * 2, 75], [3, 3, 3])
            .with_stride([1, 2, 2])
            .with_padding(PaddingConfig3d::Explicit(1, 1, 1))
            .init(device);

        let tcn1 = TemporalConvNetConfig::new([75 * (height / 8) * (width / 8), out_channels], 3)
            .with_layers(6)
            .with_dropout(0.5)
            .init(device);

        let tcn2 = TemporalConvNetConfig::new([out_channels, 75], 3)
            .with_layers(6)
            .with_dropout(0.5)
            .init(device);

        let fc = LinearConfig::new(75, vocab_size).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            tcn1,
            tcn2,
            fc,
        }
    }

    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 3> {
        // note: N is samples per batch (batch size), C is channels, T is timesteps (number of frames), H is height (frame dim), W is width (frame dim)

        // three 3D convolutional layers with ReLU activation and strided downsampling
        // input shape: (batch, channels, timesteps, height, width)
        // output shape: (batch, channels, timesteps, height/(2^3), width/(2^3))
        let x = activation::relu(self.conv1.forward(input));
        let x = activation::relu(self.conv2.forward(x));
        let x = activation::relu(self.conv3.forward(x));

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

    #[cfg(test)]
    pub fn inspect_shapes_once(&self, input: Tensor<B, 5>) {
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

            let x: Tensor<B, 3> = activation::relu(self.tcn1.forward(x));
            println!("TCN1 (N, C_feat, T): {:?}", x.dims());

            let x: Tensor<B, 3> = activation::relu(self.tcn2.forward(x));
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
    use burn::{backend::ndarray::NdArray, tensor::Tensor};

    // choosing backend for testing
    type B = NdArray<f32>;

    #[test]
    fn model_input_shapes_data_flow_small() {
        let (n, c, t, h, w) = (1, 1, 8, 16, 16);
        let out_channels = 8;
        let vocab_size = 41;

        let device = Default::default();
        let model = LRModel::<B>::new(c, out_channels, (h, w), vocab_size, &device);

        let input = Tensor::<B, 5>::zeros([n, c, t, h, w], &device);
        let output = model.forward(input);

        assert_eq!(output.dims(), [n, t, vocab_size]); // expected output shape
    }
}
