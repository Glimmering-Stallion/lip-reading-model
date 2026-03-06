//! Frontend 3D Residual Block implementation.
//! 
//! This module provides 3D spatio-temporal residual blocks for the VSRM frontend.
//! These blocks allow deeper architectures by mitigating vanishing/exploding gradients
//! through identity/projection.



// imports
use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{
        GroupNorm, GroupNormConfig, Initializer, PaddingConfig3d, conv::{
            Conv3d,
            Conv3dConfig,
        }
    },
    tensor::{
        Tensor,
        activation,
        backend::Backend,
    },
};



#[derive(Config, Debug)]
pub struct ResidualBlockConfig {
    pub channels: [usize; 2],     // input and output channel numbers [C_in, C_out]
    pub kernel_size: [usize; 3],  // spatio-temporal kernel size for first and second Conv3D layers [T, H, w]

    #[config(default = "[1, 1, 1]")]
    pub stride: [usize; 3],       // stride for first Conv3D layer spatial downsampling [T, H, W]
    
    #[config(default = "[1, 1, 1]")]
    pub padding: [usize; 3],      // spatio-temporal padding for first and second Conv3D layers [T, H, W]

    #[config(default = 8)]
    pub norm_groups: usize,       // number of groups for GroupNorm (must be divisible by channels)
}



#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv3d<B>, gn1: GroupNorm<B>,
    conv2: Conv3d<B>, gn2: GroupNorm<B>,
    proj: Option<Conv3d<B>>,
}



impl ResidualBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualBlock<B> {
        let [in_channels, out_channels] = self.channels;
        let [p_t, p_h, p_w] = self.padding;

        assert!(in_channels > 0 && out_channels > 0, "Channels must be positive");
        assert!(out_channels.is_multiple_of(self.norm_groups), "Output channels ({}) must be divisible by norm_groups ({})", out_channels, self.norm_groups);

        // handles downsample transitioning
        let conv1 = Conv3dConfig::new(self.channels, self.kernel_size)
            .with_stride(self.stride)
            .with_padding(PaddingConfig3d::Explicit(p_t, p_h, p_w))
            .with_initializer(Initializer::KaimingNormal {gain: 2.0f64.sqrt(), fan_out_only: false})
            .init(device);
        let gn1 = GroupNormConfig::new(self.norm_groups, out_channels).init(device);

        // handles channel refinement (maintains spatial dims)
        let conv2 = Conv3dConfig::new([out_channels, out_channels], self.kernel_size)
            .with_padding(PaddingConfig3d::Explicit(p_t, p_h, p_w))
            .with_initializer(Initializer::KaimingNormal {gain: 2.0f64.sqrt(), fan_out_only: false})
            .init(device);
        let mut gn2 = GroupNormConfig::new(self.norm_groups, out_channels).init(device);

        // zero out gamma weight parameter
        let zero_gamma = Tensor::<B, 1>::zeros([out_channels], device);
        gn2.gamma = Some(Param::from_tensor(zero_gamma));

        // perform identity projection if needed, to align input to output channels for residual operation
        let projection_needed = in_channels != out_channels || self.stride.iter().any(|&s| s > 1);
        let proj = if projection_needed {
            Some(
                Conv3dConfig::new([in_channels, out_channels], [1, 1, 1])
                    .with_stride(self.stride)
                    .with_initializer(Initializer::KaimingNormal {gain: 1.0, fan_out_only: false})
                    .init(device)
            )
        } else { None };

        ResidualBlock {
            conv1, gn1,
            conv2, gn2,
            proj,
        }
    }
}



impl<B: Backend> ResidualBlock<B> {
    /// forward pass of a 3D Residual Block
    /// applies: Conv3D --> GN --> ReLU --> Conv3D --> GN --> Res Sum --> ReLU
    /// params:
    /// - input: [N, C_in, T, H, W] the spatio-temporal tensor input
    /// returns: [N, C_out, T', H', W'] where T', H', W' depend on stride S
    pub fn forward(&self, input: Tensor<B, 5>) -> Tensor<B, 5> {
        let residual = match &self.proj {
            Some(path) => path.forward(input.clone()),
            None => input.clone(),
        };

        // [N, C, (T / S), (H / S), (W / S)]
        let x = self.conv1.forward(input);
        let x = self.gn1.forward(x);
        let x = activation::relu(x);

        // [N, C, (T / S), (H / S), (W / S)]
        let x = self.conv2.forward(x);
        let x = self.gn2.forward(x);
        let x = activation::relu(x.add(residual));

        x // post activation residual sum
    }

    /// calculate how many additional frames of temporal context this ResBlock adds
    /// since each block has two Conv3D layers with temporal stride 1:
    /// formula: (k1_t - 1) + (k2_t - 1)
    /// returns: temporal context as total number of frames
    pub fn receptive_field_contribution(&self) -> usize {
        let k1 = self.conv1.kernel_size[0];
        let k2 = self.conv2.kernel_size[0];

        (k1 - 1) + (k2 - 1)
    }

    /// returns: ID of the primary convolutional weight
    /// useful for gradient tracking and per-layer optim stats
    pub fn primary_weight_id(&self) -> ParamId { self.conv1.weight.id.clone() }
}