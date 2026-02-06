// Utility functions



// imports
use burn::tensor::{backend::Backend, Tensor};
use num_traits::Float;



// #[inline]
// pub fn mean(data: &Vec<f32>) -> f32 {
//     let count = data.len() as f32;
//     let sum: f32 = data.iter().sum();
//     sum / count
// }

// #[inline]
// pub fn std_dev(data: &Vec<f32>) -> f32 {
//     let count = data.len();
//     let mean = mean(data);
//     let variance: f32 = data.iter().map(|x| x - mean).map(|x| x * x).sum::<f32>() / count as f32;
//     variance.sqrt()
// }



#[inline]
pub fn mean<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let sum = data.iter().copied().fold(T::zero(), |acc, x| acc + x);
    sum / count
}

#[inline]
pub fn std_dev<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let mean = mean(data);
    let variance = data
        .iter()
        .copied()
        .fold(T::zero(), |acc, x| acc + ((x - mean) * (x - mean)))
        / count;
    variance.sqrt()
}



#[inline]
pub fn log_sum_exp_2_scalar(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let sum = (a - max).exp() + (b - max).exp();
    let lse = max + sum.ln();

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    if lse.is_nan() { f32::NEG_INFINITY } else { lse }
}

#[inline]
pub fn log_sum_exp_3_scalar(a: f32, b: f32, c: f32) -> f32 {
    log_sum_exp_2_scalar(log_sum_exp_2_scalar(a, b), c)
}



#[inline]
pub fn log_sum_exp_2_tensor<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
) -> Tensor<B, D> {
    let max = a.clone().max_pair(b.clone()); // element-wise maxxing
    let sum = (a - max.clone()).exp().add((b - max.clone()).exp());
    let lse = max.clone().add(sum.log());

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    let nan_mask = lse.clone().is_nan();
    lse.mask_fill(nan_mask, f32::NEG_INFINITY)
}

#[inline]
pub fn log_sum_exp_3_tensor<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
    c: Tensor<B, D>,
) -> Tensor<B, D> {
    log_sum_exp_2_tensor(log_sum_exp_2_tensor(a, b), c)
}
